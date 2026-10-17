"""Prefill in 32-row windows (ESPALIER_PREFILL_MODE=chunk): row invariance across window boundaries.

Gate P1 (predeclared before the first run): at
T = 33, 100, 517 rows, the dense op (every dense layer kind and the head) and the MoE layer (routed experts + shared
slot, with vLLM's picks and with fused routing) equal, bit for bit, the same rows run one at a time.

usage: python tests/gpu_prefill.py --ckpt CKPT [--layers 0,3,39]
"""
import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True)
ap.add_argument("--layers", default="0,3,39")
ap.add_argument("--rows", default="33,100,517")
a = ap.parse_args()

from vllm_espalier import dense as D            # noqa: E402
from vllm_espalier import experts as G          # noqa: E402
from vllm_espalier import ops                   # noqa: E402
from vllm_espalier.matrix import EspMatrix      # noqa: E402

assert ops.PREFILL == "chunk", "run with ESPALIER_PREFILL_MODE unset or chunk"
torch.manual_seed(0)
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), flush=True)
wm = json.load(open(a.ckpt / "model.safetensors.index.json"))["weight_map"]
TS = next(s for s in (".esp_trellis",) if any(k.endswith(s) for k in wm))
P = "model.language_model.layers"
handles = {}
fails = []
ROWS = [int(v) for v in a.rows.split(",")]


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
    keys = [x for x in ("esp_trellis", "suh", "svh", "mcg", "mul1", "esp_grid")
            if f"{base}.{x}" in wm]
    return EspMatrix({x: tensor(f"{base}.{x}") for x in keys})


def find(pat):
    return next(k[:-len(TS)] for k in wm if k.endswith(TS) and re.search(pat, k))


def has(base):
    return f"{base}{TS}" in wm


# dense layers: the op as vLLM calls it (espalier::dense, registered Dense) vs row by row
layers = []
for li in [int(v) for v in a.layers.split(",")]:
    if has(f"{P}.{li}.linear_attn.out_proj"):
        layers.append((f"{li}.in_proj_qkvz", [mat(f"{P}.{li}.linear_attn.in_proj_qkv"),
                                              mat(f"{P}.{li}.linear_attn.in_proj_z")]))
        layers.append((f"{li}.linear_out", [mat(f"{P}.{li}.linear_attn.out_proj")]))
    else:
        layers.append((f"{li}.qkv", [mat(f"{P}.{li}.self_attn.{p}_proj") for p in ("q", "k", "v")]))
        layers.append((f"{li}.o", [mat(f"{P}.{li}.self_attn.o_proj")]))
    layers.append((f"{li}.shared_gate_up", [mat(f"{P}.{li}.mlp.shared_expert.gate_proj"),
                                            mat(f"{P}.{li}.mlp.shared_expert.up_proj")]))
layers.append(("lm_head", [mat("lm_head")]))
for name, mats in layers:
    d = D.Dense(mats)
    key = next(ops.DENSE_KEYS)
    ops.DENSE_LAYERS[key] = d
    t = [m.trellis for m in mats]
    s = [m.suh for m in mats]
    v = [m.svh for m in mats]
    for T in ROWS if name != "lm_head" else ROWS[:2]:
        x = (torch.randn(T, d.K, device=dev) * 0.5).to(torch.bfloat16)
        y = torch.ops.espalier.dense(x, t, s, v, key)
        y1 = torch.cat([d.forward(x[i:i + 1]) for i in range(T)])
        check(torch.equal(y, y1), f"{name} T={T}: dense op == rows one at a time")
    # single-matrix op (espalier::matmul, fp16 in/out) above 32 rows
    m = mats[0]
    x = (torch.randn(ROWS[0], m.in_features, device=dev) * 0.5).half()
    y = torch.ops.espalier.matmul(x, m.trellis, m.suh, m.svh, m.K, m.cb)
    y1 = torch.cat([torch.ops.espalier.matmul(x[i:i + 1], m.trellis, m.suh, m.svh, m.K, m.cb) for i in range(ROWS[0])])
    check(torch.equal(y, y1), f"{name}[0] T={ROWS[0]}: matmul op == rows one at a time")
    del ops.DENSE_LAYERS[key]

# MoE layers: experts.windows (what moe.apply runs above 32 tokens) vs routed2 row by row
for li in [int(v) for v in a.layers.split(",")]:
    experts = [tuple(mat(find(rf"layers\.{li}\.mlp\.experts\.{e}\.{p}_proj\.")) for p in ("gate", "up", "down"))
               for e in range(256)]
    shared = tuple(mat(find(rf"layers\.{li}\.mlp\.shared_expert\.{p}_proj\.")) for p in ("gate", "up", "down"))
    gate_w = tensor(f"{P}.{li}.mlp.shared_expert_gate.weight").to(torch.bfloat16)
    router = tensor(f"{P}.{li}.mlp.gate.weight").to(torch.bfloat16).contiguous()
    L = G.prepare(experts, shared, gate_w)
    s = G.scratch(L, 32, 9, dev)
    for T in ROWS:
        x = (torch.randn(T, L.D, device=dev) * 0.5).to(torch.bfloat16)
        pick = torch.stack([torch.randperm(256, device=dev)[:8] for _ in range(T)]).to(torch.int32).contiguous()
        wts = torch.softmax(torch.randn(T, 8, device=dev), -1).contiguous()
        out, sh = G.windows(x, pick, wts, L, s)
        o1, s1 = zip(*[G.routed2(x[i:i + 1], pick[i:i + 1], wts[i:i + 1], L, s) for i in range(T)])
        check(torch.equal(out, torch.cat(o1)) and torch.equal(sh, torch.cat(s1)),
              f"layer {li} T={T}: MoE windows == rows one at a time (vLLM picks)")
        out, sh = G.windows(x, None, None, L, s, k=8, gate=router)
        o1, s1 = [], []
        for i in range(T):
            G.route(x[i:i + 1], router, 8, L, s)
            o, h = G.routed2(x[i:i + 1], None, None, L, s, k=8)
            o1.append(o)
            s1.append(h)
        check(torch.equal(out, torch.cat(o1)) and torch.equal(sh, torch.cat(s1)),
              f"layer {li} T={T}: MoE windows == rows one at a time (fused routing)")
    del experts, L, s
    torch.cuda.empty_cache()

print("ALL PASS" if not fails else f"{len(fails)} FAILED")
