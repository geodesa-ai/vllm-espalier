"""Grouped routed experts (vllm_espalier/experts.py, TensorFold's pipeline + Espalier codebooks) on real checkpoint layers.

Gates (predeclared before the first run):
  1. dequant == exllamav3 reconstruct, bit for bit, for a random trellis of every compiled (codebook, width) pair
     (only when ESPALIER_TEST_EXLLAMAV3_EXT points at an exllamav3 extension patched with the Espalier codebooks; skipped otherwise).
  2. routed vs a float32 reference (reconstruct, fp32 Hadamards and matmuls, fp32 SwiGLU): max abs error over max
     |ref| < 2e-3 at every R.
  3. row invariance: each row of an R-row window equals the same row alone, bit for bit (R = 2, 8, 17, 32).
  4. CUDA graph replay equals the eager call, bit for bit (R = 1, 2, 4, 8, 32).
Then speed: one MoE layer per call, replayed from CUDA graphs.

usage: python tests/gpu_grouped.py --ckpt CKPT --layers 0,20,39   [ESPALIER_TEST_EXLLAMAV3_EXT=... for gate 1]
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
from vllm_espalier.matrix import EspMatrix      # noqa: E402

torch.manual_seed(0)
dev = torch.device("cuda")
print(torch.cuda.get_device_name(), flush=True)
wm = json.load(open(a.ckpt / "model.safetensors.index.json"))["weight_map"]
TS = next(s for s in (".esp_trellis",) if any(k.endswith(s) for k in wm))   # trellis key suffix
fails = []


def check(ok, msg):
    print(("PASS " if ok else "FAIL ") + msg, flush=True)
    if not ok:
        fails.append(msg)


def load_layer(li):
    pat = re.compile(rf"layers\.{li}\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)" + re.escape(TS) + "$")
    found = {}
    for k in wm:
        m = pat.search(k)
        if m:
            found[(int(m.group(1)), m.group(2))] = k[:-len(TS)]
    E = 1 + max(e for e, _ in found)
    handles = {}
    mats = []
    for e in range(E):
        row = []
        for p in ("gate_proj", "up_proj", "down_proj"):
            base = found[(e, p)]
            f = wm[base + TS]
            if f not in handles:
                handles[f] = safe_open(str(a.ckpt / f), "pt", device="cuda")
            s = handles[f]
            keys = [x for x in ("esp_trellis", "suh", "svh", "mcg", "mul1", "esp_grid") if f"{base}.{x}" in s.keys()]
            row.append(EspMatrix({x: s.get_tensor(f"{base}.{x}") for x in keys}))
        mats.append(tuple(row))
    return mats


H = None


def had(x):
    """x @ blockdiag(H128) / sqrt(128) in fp32 (Sylvester, symmetric)."""
    global H
    if H is None:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        H = (h / 128 ** 0.5).to(dev)
    return (x.view(*x.shape[:-1], -1, 128) @ H).view(x.shape)


def ref_moe(x, pick, wts, mats, W):
    out = torch.zeros(x.shape, dtype=torch.float64, device=dev)
    xf = x.float()
    for r in range(x.shape[0]):
        for j in range(pick.shape[1]):
            e = int(pick[r, j])
            g, u, d = mats[e]
            Wg, Wu, Wd = W[e]

            def lin(v, m, Wm):
                return had(had(v * m.suh.float()) @ Wm) * m.svh.float()
            h = F.silu(lin(xf[r], g, Wg)) * lin(xf[r], u, Wu)
            out[r] += float(wts[r, j]) * lin(h, d, Wd).double()
    return out.float()


def picks(R, E, k=8):
    pick = torch.stack([torch.randperm(E, device=dev)[:k] for _ in range(R)]).to(torch.int32)
    wts = torch.softmax(torch.randn(R, k, device=dev), -1)
    return pick.contiguous(), wts.contiguous()


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
    return st.elapsed_time(en) / iters * 1000


# 1. synthetic: every compiled (codebook, width) pair, against exllamav3's reconstruct when it is available
X = None
if __import__("os").environ.get("ESPALIER_TEST_EXLLAMAV3_EXT"):
    import importlib.machinery
    import importlib.util
    _l = importlib.machinery.ExtensionFileLoader("exllamav3_ext", __import__("os").environ["ESPALIER_TEST_EXLLAMAV3_EXT"])
    _x = importlib.util.module_from_spec(importlib.util.spec_from_loader("exllamav3_ext", _l))
    _l.exec_module(_x)

    class X:
        @staticmethod
        def reconstruct(t, K, mcg, grid, i, o):
            w = torch.empty((i, o), dtype=torch.half, device=t.device)
            _x.reconstruct(w, t, K, mcg, grid)
            return w
grid_of = {v: k for k, v in G.KERNEL_CB.items()}
bad = []
for cb, k2 in sorted(G.COMPILED):
    K = k2 // 2
    t = torch.randint(-32768, 32767, (256 // 16, 256 // 16, 16 * K), dtype=torch.int16, device=dev)
    got = G.dequant(t, K, grid_of[cb])
    want = X.reconstruct(t, K, True, grid_of[cb], 256, 256) if X else got
    if not torch.equal(got, want):
        bad.append((cb, k2, int((got != want).sum())))
if X:
    check(not bad, f"dequant == reconstruct on random trellises for {len(G.COMPILED)} (codebook, width) pairs {bad}")
else:
    print("SKIP gate 1: set ESPALIER_TEST_EXLLAMAV3_EXT to compare against exllamav3's reconstruct", flush=True)

speed = []
for li in [int(v) for v in a.layers.split(",")]:
    mats = load_layer(li)
    E = len(mats)
    combos = sorted({(m.K, m.cb) for row in mats for m in row})
    print(f"layer {li}: {E} experts, (K, grid) {combos}", flush=True)
    W = [[m.weight().float() for m in row] for row in mats]
    L = G.prepare(mats)
    s = G.scratch(L, 32, 8, dev)
    D = L.D
    # 2. accuracy
    for R in (1, 4, 8, 32):
        x = (torch.randn(R, D, device=dev) * 0.5).half()
        pick, wts = picks(R, E)
        ref = ref_moe(x, pick, wts, mats, W)
        got = G.routed(x, pick, wts, L, s)
        err = float((got - ref).abs().max() / ref.abs().max())
        check(err < 2e-3, f"layer {li} R={R}: grouped rel err {err:.2e}")
    # 3. row invariance
    for R in (2, 8, 17, 32):
        x = (torch.randn(R, D, device=dev) * 0.5).half()
        pick, wts = picks(R, E)
        win = G.routed(x, pick, wts, L, s).clone()
        alone = torch.cat([G.routed(x[r:r + 1], pick[r:r + 1], wts[r:r + 1], L, s).clone() for r in range(R)])
        check(torch.equal(win, alone), f"layer {li} R={R}: rows alone == rows in the window")
    # 4. graphs, then speed
    for R in (1, 2, 4, 8, 16, 32):
        x = (torch.randn(R, D, device=dev) * 0.5).half()
        pick, wts = picks(R, E)
        out = torch.empty((R, D), dtype=torch.float32, device=dev)
        eager = G.routed(x, pick, wts, L, s).clone()
        t_new = graph_time(lambda: G.routed(x, pick, wts, L, s, out), a.iters)
        if R in (1, 2, 4, 8, 32):
            check(torch.equal(out, eager), f"layer {li} R={R}: graph replay == eager")
        speed.append((li, R, t_new))
        print(f"layer {li} R={R}: grouped {t_new:.1f} us", flush=True)

print("\nlayer  R  grouped_us  (one layer alone: L2-warm; tests/gpu_grouped_sweep.py times it cold)")
for li, R, tn in speed:
    print(f"{li:5d} {R:2d} {tn:10.1f}")
print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
sys.exit(1 if fails else 0)
