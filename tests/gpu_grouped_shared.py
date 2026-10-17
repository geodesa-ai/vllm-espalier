"""Grouped experts with the shared expert folded in as one more slot, bf16 in and out (vllm_espalier/experts.py routed2).

Gates (predeclared before the first run):
  1. Without a shared slot, routed2 on bf16 x equals the fp32 routed() output rounded to bf16, bit for bit.
  2. With the shared slot: routed output and shared output each within 5e-4 of the bf16 rounding floor (max abs error /
     max |ref| against a float32 reference, minus the same for the reference rounded to bf16; amended from 2e-3 raw); the shared reference is sigmoid(x . gate_w) * down(silu(gate(x)) * up(x)) from the checkpoint.
  3. Rows alone == rows in the window (R = 2, 8, 17, 32), both outputs, bit for bit.
  4. CUDA graph replay == eager (R = 1, 8, 32), both outputs, bit for bit.
Then speed: one MoE layer with the shared expert, grouped+folded vs grouped + three separate exllamav3 GEMMs for the
shared expert (the previous serving path), both replayed from CUDA graphs.

usage: python tests/gpu_grouped_shared.py --ckpt CKPT --layers 0,20,39
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True)
ap.add_argument("--layers", default="0,20,39")
ap.add_argument("--iters", type=int, default=200)
a = ap.parse_args()

from vllm_espalier import experts as G          # noqa: E402
from vllm_espalier import ops                   # noqa: E402,F401  (registers espalier::matmul)
from vllm_espalier.matrix import EspMatrix      # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), flush=True)
wm = json.load(open(a.ckpt / "model.safetensors.index.json"))["weight_map"]
TS = next(s for s in (".esp_trellis",) if any(k.endswith(s) for k in wm))   # trellis key suffix
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


def find(pat):
    return next(k[:-len(TS)] for k in wm if k.endswith(TS) and re.search(pat, k))


H = None


def had(x):
    global H
    if H is None:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        H = (h / 128 ** 0.5).to(dev)
    return (x.view(*x.shape[:-1], -1, 128) @ H).view(x.shape)


def mlp_ref(xf, mats):
    g, u, d = mats

    def lin(v, m):
        return had(had(v * m.suh.float()) @ m.weight().float()) * m.svh.float()
    return lin(F.silu(lin(xf, g)) * lin(xf, u), d)


def graph_time(fn, iters):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000, g


def picks(R, E, k=8):
    pick = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(R)]).to(torch.int32)
    return pick.contiguous(), torch.softmax(torch.randn(R, k, device=dev), -1).contiguous()


speed = []
for li in [int(v) for v in a.layers.split(",")]:
    experts = [tuple(mat(find(rf"layers\.{li}\.mlp\.experts\.{e}\.{p}_proj\.")) for p in ("gate", "up", "down"))
               for e in range(256)]
    shared = tuple(mat(find(rf"layers\.{li}\.mlp\.shared_expert\.{p}_proj\.")) for p in ("gate", "up", "down"))
    gate_w = tensor(f"model.language_model.layers.{li}.mlp.shared_expert_gate.weight").to(torch.bfloat16)
    print(f"layer {li}: shared expert K{shared[0].K} grid{shared[0].cb}, gate {tuple(gate_w.shape)}", flush=True)
    W_ref_mats = experts
    L0 = G.prepare(experts)                     # no shared slot
    Ls = G.prepare(experts, shared, gate_w)     # shared slot at index 256 (rebinds the same stacks' rows again)
    s0 = G.scratch(L0, 32, 8, dev)
    s1 = G.scratch(Ls, 32, 9, dev)
    D = L0.D
    # 1. bf16 output == fp32 routed rounded
    for R in (1, 8, 32):
        x = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16)
        pick, wts = picks(R, 256)
        ref32 = G.routed(x, pick, wts, L0, G.scratch(L0, 32, 8, dev)).clone()
        out, sh = G.routed2(x, pick, wts, L0, s0)
        check(sh is None and out.dtype == torch.bfloat16 and torch.equal(out, ref32.to(torch.bfloat16)),
              f"layer {li} R={R}: routed2 bf16 == fp32 routed rounded (no shared slot)")
    # 2. shared slot accuracy
    for R in (1, 8, 32):
        x = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16)
        pick, wts = picks(R, 256)
        out, sh = G.routed2(x, pick, wts, Ls, s1)
        xf = x.float()
        sref = torch.sigmoid(xf @ gate_w.float().reshape(-1, 1)) * mlp_ref(xf, shared)
        rref = torch.zeros_like(xf)
        for r in range(R):
            for j in range(8):
                rref[r] += float(wts[r, j]) * mlp_ref(xf[r:r + 1], experts[int(pick[r, j])])[0]
        er = float((out.float() - rref).abs().max() / rref.abs().max())
        es = float((sh.float() - sref).abs().max() / sref.abs().max())
        # the floor of a bf16 output: the fp32 reference itself rounded to bf16
        fr = float((rref.to(torch.bfloat16).float() - rref).abs().max() / rref.abs().max())
        fs = float((sref.to(torch.bfloat16).float() - sref).abs().max() / sref.abs().max())
        # amended criterion (accepted 2026-10-16): the error
        # above the bf16 output-rounding floor, not the raw error, against 5e-4
        check(er - fr < 5e-4 and es - fs < 5e-4, f"layer {li} R={R}: routed rel err {er:.2e} (bf16 floor {fr:.2e}), "
                                       f"shared rel err {es:.2e} (bf16 floor {fs:.2e})")
    # 3. row invariance
    for R in (2, 8, 17, 32):
        x = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16)
        pick, wts = picks(R, 256)
        o, sh = (t.clone() for t in G.routed2(x, pick, wts, Ls, s1))
        oa = torch.cat([G.routed2(x[r:r + 1], pick[r:r + 1], wts[r:r + 1], Ls, s1)[0].clone() for r in range(R)])
        sa = torch.cat([G.routed2(x[r:r + 1], pick[r:r + 1], wts[r:r + 1], Ls, s1)[1].clone() for r in range(R)])
        check(torch.equal(o, oa) and torch.equal(sh, sa), f"layer {li} R={R}: rows alone == rows in the window")
    # 4. graphs + speed vs the previous path (grouped routed + 3 exllamav3 GEMMs with fp16 casts for the shared expert)
    for R in (1, 8, 32):
        x = (torch.randn(R, D, device=dev) * 0.5).to(torch.bfloat16)
        pick, wts = picks(R, 256)
        out = torch.empty((R, D), dtype=torch.bfloat16, device=dev)
        osh = torch.empty_like(out)
        eo, es = (t.clone() for t in G.routed2(x, pick, wts, Ls, s1))
        t_new, _ = graph_time(lambda: G.routed2(x, pick, wts, Ls, s1, out, osh), a.iters)
        check(torch.equal(out, eo) and torch.equal(osh, es), f"layer {li} R={R}: graph replay == eager")

        def old():
            r = G.routed(x, pick, wts, L0, s0).to(torch.bfloat16)
            xh = x.half()
            g = torch.ops.espalier.matmul(xh, shared[0].trellis, shared[0].suh, shared[0].svh, shared[0].K, shared[0].cb)
            u = torch.ops.espalier.matmul(xh, shared[1].trellis, shared[1].suh, shared[1].svh, shared[1].K, shared[1].cb)
            h = (F.silu(g.float()) * u.float()).half()
            d = torch.ops.espalier.matmul(h, shared[2].trellis, shared[2].suh, shared[2].svh, shared[2].K, shared[2].cb).to(torch.bfloat16)
            return r + torch.sigmoid(x @ gate_w.reshape(-1, 1)) * d
        t_old, _ = graph_time(old, a.iters)
        speed.append((li, R, t_new, t_old))
        print(f"layer {li} R={R}: folded {t_new:.1f} us, previous {t_old:.1f} us", flush=True)

print("\nlayer  R  folded_us  previous_us  speedup")
for li, R, tn, to in speed:
    print(f"{li:5d} {R:2d} {tn:10.1f} {to:12.1f} {to / tn:7.2f}x")
print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
sys.exit(1 if fails else 0)
