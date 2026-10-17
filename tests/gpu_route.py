"""Fused MoE routing (experts.route, csrc/route.cu) against the exact routing: fp64 router logits rounded once to bf16,
then vLLM's softmax / top-k / renormalisation. vLLM's own router (bf16 GEMM logits, not correctly rounded) is reported
for comparison only.

usage: python tests/gpu_route.py --ckpt CKPT [--layers 0,20,39] [--rows 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True)
ap.add_argument("--layers", default="0,20,39")
ap.add_argument("--rows", type=int, default=2000)
a = ap.parse_args()

from vllm import _custom_ops as ops        # noqa: E402
from vllm_espalier import experts as G          # noqa: E402
from vllm_espalier.matrix import EspMatrix      # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), flush=True)
wm = json.load(open(a.ckpt / "model.safetensors.index.json"))["weight_map"]
handles = {}
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def tensor(key):
    f = wm[key]
    if f not in handles:
        handles[f] = safe_open(str(a.ckpt / f), "pt", device="cuda")
    return handles[f].get_tensor(key)


def mat(base):
    keys = [x for x in ("esp_trellis", "suh", "svh", "mcg", "mul1", "esp_grid") if f"{base}.{x}" in wm]
    return EspMatrix({x: tensor(f"{base}.{x}") for x in keys})


def vllm_topk(logits, k):
    M = logits.shape[0]
    w = torch.empty(M, k, dtype=torch.float32, device=dev)
    i = torch.empty(M, k, dtype=torch.int32, device=dev)
    tei = torch.empty(M, k, dtype=torch.int32, device=dev)
    ops.topk_softmax(w, i, tei, logits, True)
    return w, i


P = "model.language_model.layers"
k = 8
for li in [int(v) for v in a.layers.split(",")]:
    experts = [tuple(mat(f"{P}.{li}.mlp.experts.{e}.{p}_proj") for p in ("gate", "up", "down")) for e in range(256)]
    shared = tuple(mat(f"{P}.{li}.mlp.shared_expert.{p}_proj") for p in ("gate", "up", "down"))
    gate_w = tensor(f"{P}.{li}.mlp.shared_expert_gate.weight").to(torch.bfloat16)
    W = tensor(f"{P}.{li}.mlp.gate.weight").to(torch.bfloat16).contiguous()
    L = G.prepare(experts, shared, gate_w)
    s = G.scratch(L, 32, k + 1, dev)
    del experts
    n_rows = 0
    same = {"ours": 0, "vllm": 0}
    dw = {"ours": 0.0, "vllm": 0.0}
    bad_ties = {"ours": 0, "vllm": 0}
    group_ok = shared_ok = True
    for start in range(0, a.rows, 32):
        R = min(32, a.rows - start)
        # hidden states after RMSNorm look like unit-RMS rows with a few large channels
        x = (torch.randn(R, W.shape[1], device=dev) * (1 + 3 * (torch.rand(W.shape[1], device=dev) < 0.01))).to(torch.bfloat16)
        G.route(x, W, k, L, s)
        P_ = R * (k + 1)
        pk = s.pick[:P_].view(R, k + 1).clone()
        wt = s.wts[:P_].view(R, k + 1).clone()
        ids, members = s.window(R)
        cnt = int(s.count.item())
        g_ids, g_mem = ids[:cnt].clone(), members[:cnt].clone()
        lg = F.linear(x, W)                                       # bf16 logits, as vLLM's router GEMV
        vw, vi = vllm_topk(lg, k)
        el = (x.double() @ W.double().T).to(torch.bfloat16)      # exact logits, rounded once
        ew, ei = vllm_topk(el, k)
        for r in range(R):
            n_rows += 1
            e_set = set(ei[r].tolist())
            for name, ids_r, w_r in (("ours", pk[r, :k], wt[r, :k]), ("vllm", vi[r], vw[r])):
                r_set = set(ids_r.tolist())
                if r_set == e_set:
                    same[name] += 1
                    order = {int(e): j for j, e in enumerate(ids_r.tolist())}
                    dw[name] = max(dw[name], max(abs(float(w_r[order[int(e)]]) - float(ew[r, j]))
                                                 for j, e in enumerate(ei[r].tolist())))
                else:
                    mine, ref = sorted(r_set - e_set), sorted(e_set - r_set)
                    lv = el[r].float()
                    gap = abs(float(lv[mine].min()) - float(lv[ref].max()))
                    bad_ties[name] += gap > float(lv[ref].abs().max()) * 2 ** -7
        # grouping == group_shared on our own picks
        G.ext().group_shared(pk[:, :k].contiguous(), wt[:, :k].contiguous(), x, x.stride(0), L.gate_w, L.shared_id,
                             s.pick[:P_], s.wts[:P_], ids, s.count, members, R, k, L.E)
        group_ok &= int(s.count.item()) == cnt and torch.equal(ids[:cnt], g_ids) and torch.equal(members[:cnt], g_mem)
        shared_ok &= float((s.wts[:P_].view(R, k + 1)[:, k] - wt[:, k]).abs().max()) < 1e-6
    check(same["ours"] >= 0.995 * n_rows and bad_ties["ours"] == 0,
          f"layer {li}: selected sets equal the exact routing in {same['ours']}/{n_rows} rows, "
          f"non-tie mismatches {bad_ties['ours']}")
    check(dw["ours"] < 2e-3, f"layer {li}: max |dw| where the sets match {dw['ours']:.2e}")
    print(f"INFO layer {li}: vLLM's router against the exact routing: sets equal {same['vllm']}/{n_rows}, "
          f"non-tie mismatches {bad_ties['vllm']}, max |dw| {dw['vllm']:.2e}", flush=True)
    check(group_ok and shared_ok, f"layer {li}: grouping == group_shared, shared weight within 1e-6")
    # graph == eager and row invariance
    x = (torch.randn(8, W.shape[1], device=dev)).to(torch.bfloat16)
    G.route(x, W, k, L, s)
    ref = s.wts[:72].clone(), s.pick[:72].clone()
    rows_ok = True
    for r in range(8):
        G.route(x[r:r + 1].contiguous(), W, k, L, s)
        rows_ok &= torch.equal(s.wts[:9], ref[0][r * 9:(r + 1) * 9]) and torch.equal(s.pick[:9], ref[1][r * 9:(r + 1) * 9])
    g = torch.cuda.CUDAGraph()
    G.route(x, W, k, L, s)
    with torch.cuda.graph(g):
        G.route(x, W, k, L, s)
    s.wts.zero_()
    g.replay()
    torch.cuda.synchronize()
    check(rows_ok and torch.equal(s.wts[:72], ref[0]) and torch.equal(s.pick[:72], ref[1]),
          f"layer {li}: rows alone == window, graph == eager")
    del L

print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
sys.exit(1 if fails else 0)
