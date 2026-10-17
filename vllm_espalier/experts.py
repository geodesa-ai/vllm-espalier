"""Grouped routed experts: one MoE layer for up to ROWS tokens in six launches, whatever the experts' widths.

The kernels are TensorFold's universal EXL3 expert pipeline (csrc/, Apache-2.0, vendored at ed78d6f) with the Espalier
codebooks added and a codebook per expert matrix:

  group_shared     the pick/weight tables with the shared expert as one more slot (its weight: sigmoid of its
                   gate, computed here), then the device histogram -> distinct experts and member rows (no host sync)
  rot_in           H(x * suh) for gate and up of every slot, read straight from the model's bf16/fp16 x
  grouped (mats=2) gate and up of all distinct experts in one launch, W decoded into mma fragments
  gateup_epilogue  splits summed, H, * svh, SwiGLU in fp32, * suh_d, H -> the down input
  grouped (mats=1) down
  down_combine2    splits summed, H, * svh_d, then the routing-weighted sum over the routed slots (slot order),
                   and the gated shared slot, both written in the model's dtype (no cast kernels)

Every sum runs in a fixed order, so a token's output never depends on the other tokens in the window, and nothing
allocates, so CUDA graphs capture it. The trellises are read in place through pointer tables; suh/svh are stacked
per projection and each EspMatrix's suh/svh become views of the stacks, so nothing is duplicated.

The extension is built from csrc/ on first use (torch JIT, cached under TORCH_EXTENSIONS_DIR); ESPALIER_PREBUILT_EXTENSION points at a
prebuilt .so instead.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch
from ._env import env

# Espalier grid code (EspMatrix.cb) -> the kernels' codebook id (exllamav3's: 1 mcg, 3 E2M1, 6 int5, 7 int6)
KERNEL_CB = {0: 1, 1: 3, 3: 6, 4: 7}
# (codebook id, half-bits) pairs compiled (experts_grouped.cuh ESP_CASES)
COMPILED = {(1, k2) for k2 in range(2, 17, 2)} | {(3, 2), (3, 4), (3, 6), (6, 4), (6, 6), (6, 8), (7, 6), (7, 8),
                                                   (7, 10)}
ACT_F32 = 1
CSRC = Path(__file__).resolve().parent / "csrc"


@lru_cache(1)
def ext():
    path = os.environ.get("ESPALIER_PREBUILT_EXTENSION")
    if path:
        import importlib.machinery
        import importlib.util
        loader = importlib.machinery.ExtensionFileLoader("espalier_ext", path)
        spec = importlib.util.spec_from_loader("espalier_ext", loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        return mod
    from torch.utils.cpp_extension import load
    return load(name="espalier_ext", sources=[str(CSRC / "experts.cpp"), str(CSRC / "experts.cu"), str(CSRC / "dense.cu"),
                                             str(CSRC / "route.cu")],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=False)


def tile_config(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """(n tiles a block, warps, K splits, tiles in flight) for a K -> N projection; the shape's alone, so a row's bits
    never depend on the batch. First choice from a cold sweep on the RTX 4070 Ti (tests/gpu_grouped_sweep.py: one MoE
    layer at R=1 53.5 -> 46.8 us, at R=8 no slower); TensorFold's settings follow as fallbacks."""
    cands = [(4, 4, 4, 2) if gateup else (4, 4, 1, 2), (8, 4, 4, 1) if gateup else (8, 4, 1, 1), (8, 4, 2, 1),
             (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


@dataclass
class Layer:
    """One layer's routed experts as the kernels read them."""

    ptr: tuple            # int64 [E] trellis pointers: gate, up, down
    k2: tuple             # int32 [E] half-bits: gate, up, down
    cb: tuple             # int32 [E] kernel codebook ids: gate, up, down
    suh_g: torch.Tensor   # fp16 [E, D]
    suh_u: torch.Tensor
    svh_g: torch.Tensor   # fp16 [E, I]
    svh_u: torch.Tensor
    suh_d: torch.Tensor   # fp16 [E, I]
    svh_d: torch.Tensor   # fp16 [E, D]
    E: int                # matrices in the tables: routed experts, plus the shared expert at index E-1 if any
    D: int
    I: int
    k2_gu: tuple          # (min, max) over gate and up
    k2_d: tuple
    shared_id: int = -1   # index of the shared expert in the tables, -1 without one
    gate_w: torch.Tensor | None = None   # the shared expert's sigmoid-gate weight [D], the model's dtype
    keep: list = field(default_factory=list, repr=False)


def prepare(experts, shared=None, gate_w=None) -> Layer:
    """experts: [(gate, up, down) EspMatrix] in expert order; shared: the shared expert's (gate, up, down), appended as
    the last table entry, with gate_w its sigmoid-gate weight [D]. Rebinds each matrix's suh/svh to stack rows."""
    if shared is not None:
        experts = list(experts) + [tuple(shared)]
    E = len(experts)
    g0 = experts[0][0]
    D, I = g0.in_features, g0.out_features
    dev = g0.trellis.device

    def tables(j, k, n):
        ptrs, k2s, cbs = [], [], []
        for e, mats in enumerate(experts):
            m = mats[j]
            if (m.in_features, m.out_features) != (k, n):
                raise ValueError(f"expert {e} projection {j}: {m.in_features}x{m.out_features}, expected {k}x{n}")
            if m.cb not in KERNEL_CB:
                raise NotImplementedError(f"expert {e}: grid code {m.cb} has no grouped kernel")
            k2, cb = 2 * m.K, KERNEL_CB[m.cb]
            if (cb, k2) not in COMPILED:
                raise NotImplementedError(f"expert {e}: no grouped instance for codebook {cb} at {m.K} bits")
            assert m.trellis.is_contiguous() and m.trellis.dtype == torch.int16
            ptrs.append(m.trellis.data_ptr())
            k2s.append(k2)
            cbs.append(cb)
        return (torch.tensor(ptrs, dtype=torch.int64, device=dev), torch.tensor(k2s, dtype=torch.int32, device=dev),
                torch.tensor(cbs, dtype=torch.int32, device=dev), k2s)

    gp, gk, gc, gks = tables(0, D, I)
    up, uk, uc, uks = tables(1, D, I)
    dp, dk, dc, dks = tables(2, I, D)

    def stack(j, attr, n):
        out = torch.empty((E, n), dtype=torch.float16, device=dev)
        for e, mats in enumerate(experts):
            out[e].copy_(getattr(mats[j], attr).reshape(-1))
            setattr(mats[j], attr, out[e])
        return out

    return Layer((gp, up, dp), (gk, uk, dk), (gc, uc, dc), stack(0, "suh", D), stack(1, "suh", D),
                 stack(0, "svh", I), stack(1, "svh", I), stack(2, "suh", I), stack(2, "svh", D), E, D, I,
                 (min(gks + uks), max(gks + uks)), (min(dks), max(dks)),
                 E - 1 if shared is not None else -1,
                 None if gate_w is None else gate_w.reshape(-1).contiguous(),
                 [m.trellis for mats in experts for m in mats])


class Scratch:
    """Buffers for up to `rows` tokens of `slots` picks, shared by every layer of one shape (layers run in turn)."""

    def __init__(self, E: int, D: int, I: int, rows: int, slots: int, device):
        self.cfg_gu = tile_config(D, I, True)
        self.cfg_d = tile_config(I, D, False)
        P = rows * slots
        self.xg = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, I), dtype=torch.float16, device=device)
        self.z = torch.zeros((max(2 * self.cfg_gu[2] * I, self.cfg_d[2] * D) * P,), dtype=torch.float32,
                             device=device)
        self.y = torch.zeros((P, D), dtype=torch.float32, device=device)
        self.route_counter = torch.zeros((1,), dtype=torch.int32, device=device)       # route_group's arrivals
        self.logits = torch.zeros((rows * E,), dtype=torch.float32, device=device)     # route_group's logits
        maxu = min(P, E)
        self.pick = torch.zeros((P,), dtype=torch.int32, device=device)      # with the shared slot
        self.wts = torch.zeros((P,), dtype=torch.float32, device=device)
        self.ids = torch.zeros((maxu,), dtype=torch.int32, device=device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.members = torch.full((maxu * rows,), -1, dtype=torch.int32, device=device)
        self.rows, self.slots, self.E = rows, slots, E

    def window(self, R: int):
        maxu = min(R * self.slots, self.E)
        return self.ids[:maxu], self.members[:maxu * R].view(maxu, R)


_SCRATCH: dict = {}


def scratch(layer: Layer, rows: int, slots: int, device) -> Scratch:
    """slots counts the shared slot when the layer has one."""
    key = (layer.E, layer.D, layer.I, rows, slots, str(device))
    if key not in _SCRATCH:
        _SCRATCH[key] = Scratch(layer.E, layer.D, layer.I, rows, slots, device)
    return _SCRATCH[key]


def routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor, L: Layer, s: Scratch,
           out: torch.Tensor | None = None) -> torch.Tensor:
    """x [R, D] fp16/bf16, pick int32 [R, slots], wts fp32 [R, slots] -> fp32 [R, D], the weighted sum of the
    routed experts' outputs (picks outside [0, E) are skipped)."""
    e = ext()
    R, slots = pick.shape
    D, I, E = L.D, L.I, L.E
    P = R * slots
    if R > s.rows or slots != s.slots:
        raise ValueError(f"{R}x{slots} picks but the scratch holds {s.rows}x{s.slots}")
    ids, members = s.window(R)
    e.group(pick, ids, s.count, members, R, slots, E)
    e.rot_in(x, x.stride(0), pick, L.suh_g, L.suh_u, s.xg, s.xu, R, D, slots, E)
    nt, w, sk, pf = s.cfg_gu
    e.grouped(s.xg, s.xu, L.ptr[0], L.ptr[1], L.k2[0], L.k2[1], L.cb[0], L.cb[1], ids, s.count, members, s.z, 2,
              D, I, P, sk, slots, nt, w, pf, L.k2_gu[0], L.k2_gu[1])
    e.gateup_epilogue(s.z, pick, L.svh_g, L.svh_u, L.suh_d, s.xd, R, P, I, sk, slots, E, math.inf, ACT_F32)
    nt, w, sk, pf = s.cfg_d
    e.grouped(s.xd, s.xd, L.ptr[2], L.ptr[2], L.k2[2], L.k2[2], L.cb[2], L.cb[2], ids, s.count, members, s.z, 1,
              I, D, P, sk, slots, nt, w, pf, L.k2_d[0], L.k2_d[1])
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    e.down_combine(s.z, pick, L.svh_d, s.y, wts, out, R, P, D, sk, slots, E)
    return out


def route(x: torch.Tensor, gate: torch.Tensor, k: int, L: Layer, s: Scratch):
    """Router logits (x @ gate.T, plus the shared gate), softmax / top-k / renormalisation and grouping in one launch
    (csrc/route.cu), into the scratch's pick / wts / ids / count / members; then routed2(..., grouped=True)."""
    R = x.shape[0]
    ids, members = s.window(R)
    ext().route_group(x, gate, L.gate_w, k, L.shared_id, s.logits, s.route_counter, s.pick[:R * s.slots],
                      s.wts[:R * s.slots], ids, s.count, members)


def windows(x: torch.Tensor, pick: torch.Tensor | None, wts: torch.Tensor | None, L: Layer, s: Scratch,
            k: int | None = None, gate: torch.Tensor | None = None):
    """routed2 over windows of s.rows tokens (prefill), each window as in decode; with gate (and pick=None) route()
    first in each window. Returns (routed sum, shared expert output or None)."""
    R = x.shape[0]
    out = torch.empty_like(x)
    sh = torch.empty_like(x) if L.shared_id >= 0 else None
    for r in range(0, R, s.rows):
        w = slice(r, r + s.rows)
        if gate is not None:
            route(x[w], gate, k, L, s)
        routed2(x[w], None if pick is None else pick[w], None if wts is None else wts[w], L, s, out=out[w],
                out_shared=None if sh is None else sh[w], k=k)
    return out, sh


def routed2(x: torch.Tensor, pick: torch.Tensor | None, wts: torch.Tensor | None, L: Layer, s: Scratch,
            out: torch.Tensor | None = None, out_shared: torch.Tensor | None = None, k: int | None = None):
    """x [R, D] in the model's dtype, pick int32 [R, k] (vLLM's routed ids), wts fp32 [R, k]. Returns (routed sum,
    shared expert output or None), both [R, D] in x's dtype; the shared output is gated by sigmoid(x . gate_w).
    pick=None: the scratch already holds the grouping (route()), for k routed slots a row."""
    e = ext()
    R, k = (x.shape[0], k) if pick is None else pick.shape
    D, I, E = L.D, L.I, L.E
    slots = k + (L.shared_id >= 0)
    P = R * slots
    if R > s.rows or slots != s.slots:
        raise ValueError(f"{R}x{slots} slots but the scratch holds {s.rows}x{s.slots}")
    ids, members = s.window(R)
    pk, wt = s.pick[:P], s.wts[:P]
    if pick is not None:
        e.group_shared(pick, wts, x, x.stride(0), L.gate_w, L.shared_id, pk, wt, ids, s.count, members, R, k, E)
    e.rot_in(x, x.stride(0), pk, L.suh_g, L.suh_u, s.xg, s.xu, R, D, slots, E)
    nt, w, sk, pf = s.cfg_gu
    e.grouped(s.xg, s.xu, L.ptr[0], L.ptr[1], L.k2[0], L.k2[1], L.cb[0], L.cb[1], ids, s.count, members, s.z, 2,
              D, I, P, sk, slots, nt, w, pf, L.k2_gu[0], L.k2_gu[1])
    e.gateup_epilogue(s.z, pk, L.svh_g, L.svh_u, L.suh_d, s.xd, R, P, I, sk, slots, E, math.inf, ACT_F32)
    nt, w, sk, pf = s.cfg_d
    e.grouped(s.xd, s.xd, L.ptr[2], L.ptr[2], L.k2[2], L.k2[2], L.cb[2], L.cb[2], ids, s.count, members, s.z, 1,
              I, D, P, sk, slots, nt, w, pf, L.k2_d[0], L.k2_d[1])
    if out is None:
        out = torch.empty((R, D), dtype=x.dtype, device=x.device)
    if L.shared_id >= 0 and out_shared is None:
        out_shared = torch.empty((R, D), dtype=x.dtype, device=x.device)
    e.down_combine2(s.z, pk, L.svh_d, s.y, wt, out, out_shared if L.shared_id >= 0 else None, R, P, D, sk, slots,
                    E, k)
    return out, (out_shared if L.shared_id >= 0 else None)


def dequant(trellis: torch.Tensor, K: int, grid: int) -> torch.Tensor:
    """W [in, out] fp16 through the grouped kernels' lane decode (for tests against exllamav3's reconstruct)."""
    out = torch.empty((trellis.shape[0] * 16, trellis.shape[1] * 16), dtype=torch.float16, device=trellis.device)
    ext().dequant(trellis.contiguous(), out, 2 * K, KERNEL_CB[grid])
    return out
