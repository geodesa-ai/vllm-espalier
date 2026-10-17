"""Dense EXL3 layers in one launch (vllm_espalier/dense.py, csrc/dense.cu): gates and speed.

Gates (predeclared before the first run):
  1. per matrix, M = 1, 8, 32: max abs err / max |ref| vs float32 H(H(x suh) W) svh, minus the bf16 floor, < 5e-4
  2. the merged launch's columns == each matrix launched alone, bit for bit
  3. rows alone == rows in the window (M = 2, 8, 17, 32), bit for bit
  4. CUDA graph replay == eager (M = 1, 8, 32), bit for bit
Speed: each layer at M = 1 (and 8) from CUDA graphs vs the unmerged layer (one launch per matrix on fp16 x, casts, cat),
with % of DRAM peak, against the unmerged layer (one launch per matrix plus casts and cat). --sweep times every (K splits, warps, tiles in flight) at M = 1 instead.

usage: python tests/gpu_dense.py --ckpt CKPT [--layers 0,3,20,39] [--sweep]
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
ap.add_argument("--layers", default="0,3,20,39")
ap.add_argument("--iters", type=int, default=300)
ap.add_argument("--peak-gbs", type=float, default=504.0)
ap.add_argument("--sweep", action="store_true")
ap.add_argument("--no-head", action="store_true")
ap.add_argument("--cold", action="store_true", help="sweep with every layer of a shape in one graph (weights stream "
                "from DRAM, as in decode; one layer replayed alone stays in the 48 MB L2 of an RTX 4070 Ti)")
a = ap.parse_args()

from vllm_espalier import dense as D            # noqa: E402
from vllm_espalier import ops                   # noqa: E402,F401  (registers espalier::matmul for the unmerged baseline)
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


H = None


def had(x):
    global H
    if H is None:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        H = (h / 128 ** 0.5).to(dev)
    return (x.view(*x.shape[:-1], -1, 128) @ H).view(x.shape)


def ref(xf, m, W):
    return had(had(xf * m.suh.float()) @ W) * m.svh.float()


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
    for _ in range(10):
        g.replay()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters * 1000, g


def old_path(x, mats):
    """The same layer unmerged: one espalier::matmul per matrix on fp16 x, cast back and concatenated."""
    xh = x.half()
    ys = [torch.ops.espalier.matmul(xh, m.trellis, m.suh, m.svh, m.K, m.cb).to(x.dtype) for m in mats]
    return ys[0] if len(ys) == 1 else torch.cat(ys, -1)


P = "model.language_model.layers"
layers = []
for li in [int(v) for v in a.layers.split(",")]:
    if any(f"{P}.{li}.linear_attn.out_proj.{t}" in wm for t in ("esp_trellis",)):
        layers.append((f"{li}.in_proj_qkvz", [mat(f"{P}.{li}.linear_attn.in_proj_qkv"),
                                              mat(f"{P}.{li}.linear_attn.in_proj_z")]))
        layers.append((f"{li}.linear_out", [mat(f"{P}.{li}.linear_attn.out_proj")]))
    else:
        layers.append((f"{li}.qkv", [mat(f"{P}.{li}.self_attn.{p}_proj") for p in ("q", "k", "v")]))
        layers.append((f"{li}.o", [mat(f"{P}.{li}.self_attn.o_proj")]))
if not a.no_head:
    layers.append(("lm_head", [mat("lm_head")]))
sms = torch.cuda.get_device_properties(dev).multi_processor_count


def nbytes(mats):
    return sum(m.trellis.numel() * 2 + (m.suh.numel() + m.svh.numel()) * 2 for m in mats)


if a.cold:
    groups = {}
    for li in range(64):
        if any(f"{P}.{li}.linear_attn.out_proj.{t}" in wm for t in ("esp_trellis",)):
            groups.setdefault("in_proj_qkvz", []).append([mat(f"{P}.{li}.linear_attn.in_proj_qkv"),
                                                          mat(f"{P}.{li}.linear_attn.in_proj_z")])
            groups.setdefault("linear_out", []).append([mat(f"{P}.{li}.linear_attn.out_proj")])
        elif any(f"{P}.{li}.self_attn.o_proj.{t}" in wm for t in ("esp_trellis",)):
            groups.setdefault("qkv", []).append([mat(f"{P}.{li}.self_attn.{p}_proj") for p in ("q", "k", "v")])
            groups.setdefault("o", []).append([mat(f"{P}.{li}.self_attn.o_proj")])
    print(f"\ncold sweep at M=1: every layer of a shape in one graph ({sms} SMs, peak {a.peak_gbs:.0f} GB/s)")
    tot_new = tot_old = tot_floor = 0.0
    for name, lst in groups.items():
        ds = [D.Dense(m) for m in lst]
        K, N = ds[0].K, ds[0].N
        x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
        outs = [torch.empty((1, d.n_out), dtype=x.dtype, device=dev) for d in ds]
        byt = sum(nbytes(m) for m in lst) / len(lst)
        res = []
        for sk in (1, 2, 4, 8, 16, 32):
            for w in (4, 8, 16):
                for pf in (1, 2, 4, 8):
                    if K % (128 * sk) or (K // 16 // sk) % w or (w == 8 and pf == 8) or (w == 16 and pf > 2):
                        continue
                    for d in ds:
                        d.plans = [(sk, w, pf)] * len(d.groups)
                    D.scratch(dev, sk * D.MAX_ROWS * N, 4 * D.MAX_ROWS * K)
                    t, _ = graph_time(lambda: [d.forward(x, out=o) for d, o in zip(ds, outs)], 50)
                    res.append((t / len(ds), sk, w, pf))
        res.sort()
        pl = D.plan(K, N, sms)
        t_plan = next(t for t, *p in res if tuple(p) == pl)
        t_old, _ = graph_time(lambda: [old_path(x, m) for m in lst], 50)
        t_old /= len(lst)
        floor = byt / a.peak_gbs / 1e3
        tot_new += t_plan * len(lst); tot_old += t_old * len(lst); tot_floor += floor * len(lst)
        print(f"{name:13s} x{len(lst)} K={K} N={N}: best {res[0][0]:6.1f} us (sk={res[0][1]} w={res[0][2]} "
              f"pf={res[0][3]}), plan {pl} {t_plan:6.1f} us ({100 * floor / t_plan:3.0f}% peak), old {t_old:6.1f} us, "
              f"floor {floor:5.1f} us", flush=True)
        print("    top: " + ", ".join(f"{t:.1f}({sk},{w},{pf})" for t, sk, w, pf in res[:6]), flush=True)
    print(f"per token, dense layers without the head: plan {tot_new / 1e3:.2f} ms, old {tot_old / 1e3:.2f} ms, "
          f"floor {tot_floor / 1e3:.2f} ms")
    sys.exit(0)

if a.sweep:
    print(f"\nsweep at M=1 ({sms} SMs, peak {a.peak_gbs:.0f} GB/s)")
    seen = set()
    for name, mats in layers:
        K, N = mats[0].in_features, sum(m.out_features for m in mats)
        key = (K, N, tuple((m.K, m.cb) for m in mats))
        if key in seen:
            continue
        seen.add(key)
        d = D.Dense(mats)
        x = (torch.randn(1, K, device=dev) * 0.5).to(torch.bfloat16)
        res = []
        for sk in (1, 2, 4, 8, 16, 32):
            for w in (4, 8):
                for pf in (1, 2):
                    kt = K // 16
                    if K % (128 * sk) or (kt // sk) % w:
                        continue
                    d.plans = [(sk, w, pf)] * len(d.groups)
                    d.counters = torch.zeros(2 * (N // 128), dtype=torch.int32, device=dev)
                    D.scratch(dev, sk * D.MAX_ROWS * N, 4 * D.MAX_ROWS * K)
                    out = torch.empty((1, d.n_out), dtype=x.dtype, device=dev)
                    t, _ = graph_time(lambda: d.forward(x, out=out), a.iters)
                    res.append((t, sk, w, pf))
        res.sort()
        t_old, _ = graph_time(lambda: old_path(x, mats), a.iters)
        gbs = nbytes(mats) / res[0][0] / 1e3
        print(f"{name:14s} K={K} N={N}: best {res[0][0]:7.1f} us (sk={res[0][1]} w={res[0][2]} pf={res[0][3]}, "
              f"{100 * gbs / a.peak_gbs:4.0f}% peak), plan {D.plan(K, N, sms)} -> "
              f"{next(t for t, *p in res if tuple(p) == D.plan(K, N, sms)):7.1f} us, old {t_old:7.1f} us", flush=True)
        print("    top: " + ", ".join(f"{t:.1f}({sk},{w},{pf})" for t, sk, w, pf in res[:6]), flush=True)
    sys.exit(0)

speed = []
for name, mats in layers:
    d = D.Dense(mats)
    K = d.K
    Ws = [m.weight().float() for m in mats]
    print(f"{name}: K={K} N={d.N} matrices {[(m.out_features, m.K, m.cb) for m in mats]} plans {d.plans}", flush=True)
    # 1. accuracy
    for M in (1, 8, 32):
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        y = d.forward(x).clone()
        xf = x.float()
        msg = []
        ok = True
        for m, W, c in zip(mats, Ws, d.col0):
            r = ref(xf, m, W)
            e = float((y[:, c:c + m.out_features].float() - r).abs().max() / r.abs().max())
            fl = float((r.to(torch.bfloat16).float() - r).abs().max() / r.abs().max())
            ok &= e - fl < 5e-4
            msg.append(f"{e:.2e}(floor {fl:.2e})")
        check(ok, f"{name} M={M}: rel err {' '.join(msg)}")
    # 2. merged == alone
    if len(mats) > 1:
        x = (torch.randn(8, K, device=dev) * 0.5).to(torch.bfloat16)
        y = d.forward(x).clone()
        same = True
        for i, (m, c) in enumerate(zip(mats, d.col0)):
            di = D.Dense([m])
            di.plans = [d.plans[0]]                 # the same K split as in the merged launch
            same &= torch.equal(di.forward(x), y[:, c:c + m.out_features])
        check(same, f"{name}: merged launch == each matrix alone")
    # 3. row invariance
    for M in (2, 8, 17, 32):
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        y = d.forward(x).clone()
        ya = torch.cat([d.forward(x[r:r + 1]).clone() for r in range(M)])
        check(torch.equal(y, ya), f"{name} M={M}: rows alone == rows in the window")
    # 4. graphs + speed
    for M in (1, 8, 32):
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        out = torch.empty((M, d.n_out), dtype=x.dtype, device=dev)
        e = d.forward(x).clone()
        t_new, _ = graph_time(lambda: d.forward(x, out=out), a.iters)
        check(torch.equal(out, e), f"{name} M={M}: graph replay == eager")
        if M <= 8:
            t_old, _ = graph_time(lambda: old_path(x, mats), a.iters)
            speed.append((name, M, t_new, t_old, nbytes(mats) / t_new / 1e3))
    del Ws

print(f"\nlayer            M     new_us   old_us  speedup  new %peak")
for name, M, tn, to, gbs in speed:
    print(f"{name:14s} {M:3d} {tn:10.1f} {to:8.1f} {to / tn:7.2f}x {100 * gbs / a.peak_gbs:7.0f}%")
print(f"\n{'ALL PASS' if not fails else f'{len(fails)} FAILED'}")
sys.exit(1 if fails else 0)
