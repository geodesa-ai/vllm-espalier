"""Grouped experts (routed + folded shared expert) at decode, cold: every layer of the checkpoint in one CUDA graph, so
expert weights stream from DRAM as in serving (one layer replayed alone stays in a 48 MB L2). Times one MoE layer for
each (n tiles, warps, K splits, tiles in flight) of the gate+up and the down launch, against the byte floor.

usage: python tests/gpu_grouped_sweep.py --ckpt CKPT [--rows 1] [--layers 40]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True)
ap.add_argument("--rows", type=int, default=1)
ap.add_argument("--layers", type=int, default=40)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--peak-gbs", type=float, default=504.0)
ap.add_argument("--force-cb", type=int, default=-1, help="experiment: decode every expert with this kernel codebook "
                "(wrong outputs, same bytes) to see how much of the time is decode arithmetic")
ap.add_argument("--no-sweep", action="store_true")
ap.add_argument("--impl", default="routed2")
a = ap.parse_args()

from vllm_espalier import experts as G          # noqa: E402
from vllm_espalier.matrix import EspMatrix      # noqa: E402

IMPL = getattr(G, a.impl)
torch.manual_seed(0)
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), flush=True)
wm = json.load(open(a.ckpt / "model.safetensors.index.json"))["weight_map"]
handles = {}


def tensor(key):
    f = wm[key]
    if f not in handles:
        handles[f] = safe_open(str(a.ckpt / f), "pt", device="cuda")
    return handles[f].get_tensor(key)


def mat(base):
    keys = [x for x in ("esp_trellis", "suh", "svh", "mcg", "mul1", "esp_grid") if f"{base}.{x}" in wm]
    return EspMatrix({x: tensor(f"{base}.{x}") for x in keys})


P = "model.language_model.layers"
Ls, picks, byts = [], [], []
R, k = a.rows, 8
for li in range(a.layers):
    experts = [tuple(mat(f"{P}.{li}.mlp.experts.{e}.{p}_proj") for p in ("gate", "up", "down")) for e in range(256)]
    shared = tuple(mat(f"{P}.{li}.mlp.shared_expert.{p}_proj") for p in ("gate", "up", "down"))
    gate_w = tensor(f"{P}.{li}.mlp.shared_expert_gate.weight").to(torch.bfloat16)
    pick = torch.stack([torch.randperm(256, device=dev)[:k] for _ in range(R)]).to(torch.int32)
    used = sorted(set(pick.flatten().tolist()))
    byts.append(sum(m.trellis.numel() * 2 for e in used for m in experts[e]) +
                sum(m.trellis.numel() * 2 for m in shared))
    Ls.append(G.prepare(experts, shared, gate_w))
    if a.force_cb >= 0:
        for c in Ls[-1].cb:
            c.fill_(a.force_cb)
    picks.append(pick.contiguous())
    del experts
    print(f"layer {li} loaded", end="\r", flush=True)
wts = torch.softmax(torch.randn(R, k, device=dev), -1).contiguous()
floor_us = sum(byts) / len(byts) / a.peak_gbs / 1e3
L0 = Ls[0]
print(f"\n{len(Ls)} layers, R={R}: mean bytes a layer {sum(byts) / len(byts) / 1e6:.2f} MB, floor {floor_us:.1f} us")
xs = [(torch.randn(R, L0.D, device=dev) * 0.5).to(torch.bfloat16) for _ in Ls]
outs = [torch.empty((R, L0.D), dtype=torch.bfloat16, device=dev) for _ in Ls]
shs = [torch.empty_like(o) for o in outs]
s = G.scratch(L0, 32, k + 1, dev)


def graph_time(fn, iters):
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters * 1000


def run():
    for L, x, p, o, sh in zip(Ls, xs, picks, outs, shs):
        IMPL(x, p, wts, L, s, o, sh)


def timed(cfg_gu, cfg_d):
    s.cfg_gu, s.cfg_d = cfg_gu, cfg_d
    need = max(2 * cfg_gu[2] * L0.I, cfg_d[2] * L0.D) * 32 * (k + 1)
    if s.z.numel() < need:
        s.z = torch.zeros(need, dtype=torch.float32, device=dev)
    return graph_time(run, a.iters) / len(Ls)


base = (tuple(s.cfg_gu), tuple(s.cfg_d))
t0 = timed(*base)
print(f"current gu {base[0]} d {base[1]}: {t0:.1f} us a layer ({100 * floor_us / t0:.0f}% of peak)", flush=True)
TILES = [(8, 4, 1), (8, 4, 2), (4, 4, 2)]


def valid(K, N, nt, w, sk):
    return K % (16 * sk * w) == 0 and N % (16 * nt) == 0


if a.no_sweep:
    sys.exit(0)
res_gu = []
for nt, w, pf in TILES:
    for sk in (1, 2, 4, 8, 16):
        if valid(L0.D, L0.I, nt, w, sk):
            res_gu.append((timed((nt, w, sk, pf), base[1]), (nt, w, sk, pf)))
res_gu.sort()
print("gate+up (down at current): " + ", ".join(f"{t:.1f}{c}" for t, c in res_gu[:6]), flush=True)
best_gu = res_gu[0][1]
res_d = []
for nt, w, pf in TILES:
    for sk in (1, 2, 4, 8):
        if valid(L0.I, L0.D, nt, w, sk):
            res_d.append((timed(best_gu, (nt, w, sk, pf)), (nt, w, sk, pf)))
res_d.sort()
print(f"down (gate+up at {best_gu}): " + ", ".join(f"{t:.1f}{c}" for t, c in res_d[:6]), flush=True)
t1 = res_d[0][0]
print(f"best: gu {best_gu} d {res_d[0][1]}: {t1:.1f} us a layer ({100 * floor_us / t1:.0f}% of peak), "
      f"current {t0:.1f} us, floor {floor_us:.1f} us")
sys.exit(0)
