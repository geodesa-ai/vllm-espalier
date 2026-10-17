"""Dense EXL3 layers in one launch (csrc/dense.cu): every matrix of a vLLM layer, x in the model's dtype, output written
into each matrix's columns, no casts or cat. Up to 32 rows (decode); above that linear.py's per-matrix path.

Each matrix keeps its own suh/svh. All matrices of one launch must share K and codebook (true of every dense layer in
our checkpoints); a layer that mixes them runs one launch per matrix into the same output. The K split, warps and
prefetch depth (`plan`) depend on the layer's shape alone, so a row's bits never depend on the batch.
"""
from __future__ import annotations

import os

import torch

from .experts import KERNEL_CB, ext
from ._env import env

MAX_ROWS = 32
ROT_SMEM = int(os.environ.get("ESPALIER_DENSE_ROTATION_SMEM", 32 * 1024))   # fuse the input rotation up to this much smem
_scratch: dict = {}


def plan(K: int, N: int, sms: int) -> tuple[int, int, int]:
    """(K splits, warps, tiles in flight) for a K -> N layer of 128-column blocks, from its shape alone. From cold
    sweeps on the RTX 4070 Ti (tests/gpu_dense.py --cold, best in every repeat): very wide layers (the head) one split
    of 4 warps; wide layers (>= 1.5 blocks an SM) one split of 8 warps; narrower ones 4 warps and the fewest K splits
    that give 2 programs an SM, keeping at least 2 k tiles a warp."""
    forced = env("ESPALIER_DENSE_TILE_PLAN")
    if forced:
        sk, w, pf = (int(v) for v in forced.split(","))
        return sk, w, pf
    nb, kt = N // 128, K // 16
    if nb >= 8 * sms:
        return 1, 4, 1
    if 2 * nb >= 3 * sms and kt % (8 * 2) == 0:
        return 1, 8, 1
    w, sk = 4, 1
    while nb * sk < 2 * sms and kt % (2 * sk * w) == 0 and kt // (2 * sk * w) >= 2:
        sk *= 2
    return sk, w, 1


def scratch(device, z_floats: int, xh_halves: int):
    """Shared scratch, grown while layers are prepared (before any graph capture), never during a forward."""
    s = _scratch.get(device)
    if s is None or s[0].numel() < z_floats or s[1].numel() < xh_halves:
        z = max(z_floats, s[0].numel() if s else 0, 1)
        h = max(xh_halves, s[1].numel() if s else 0, 1)
        s = (torch.empty(z, dtype=torch.float32, device=device), torch.empty(h, dtype=torch.half, device=device))
        _scratch[device] = s
    return s


class Dense:
    MAX_ROWS = MAX_ROWS

    def __init__(self, mats, n_out: int | None = None):
        self.mats = list(mats)
        m0 = self.mats[0]
        self.K = m0.in_features
        assert all(m.in_features == self.K for m in self.mats)
        self.groups = []                       # launches: (matrix indices, k2, kernel cb)
        for i, m in enumerate(self.mats):
            key = (2 * m.K, KERNEL_CB[m.cb])
            if self.groups and (self.groups[-1][1], self.groups[-1][2]) == key and len(self.groups[-1][0]) < 4:
                self.groups[-1][0].append(i)
            else:
                self.groups.append(([i], *key))
        self.col0 = []
        c = 0
        for m in self.mats:
            self.col0.append(c)
            c += m.out_features
        self.N = c
        self.n_out = n_out or c
        dev = m0.trellis.device
        sms = torch.cuda.get_device_properties(dev).multi_processor_count
        self.plans, nbt_max, z = [], 0, 0
        for idx, _, _ in self.groups:
            n = sum(self.mats[i].out_features for i in idx)
            p = plan(self.K, n, sms)
            self.plans.append(p)
            nbt_max = max(nbt_max, n // 128)
            z = max(z, p[0] * MAX_ROWS * n)
        self.counters = torch.zeros(2 * nbt_max, dtype=torch.int32, device=dev)
        scratch(dev, z, 4 * MAX_ROWS * self.K)

    def forward(self, x, trellis=None, suh=None, svh=None, out=None):
        """x [M, K] (M <= 32) -> [M, n_out] in x's dtype; trellis/suh/svh lists override the matrices' own tensors
        (module buffers under torch.compile)."""
        M = x.shape[0]
        trellis = trellis or [m.trellis for m in self.mats]
        suh = suh or [m.suh for m in self.mats]
        svh = svh or [m.svh for m in self.mats]
        y = out if out is not None else torch.empty((M, self.n_out), dtype=x.dtype, device=x.device)
        if self.n_out > self.N:
            y[:, self.N:].zero_()
        z, xh = scratch(x.device, 0, 0)
        if x.stride(-1) != 1:
            x = x.contiguous()
        for (idx, k2, cb), (sk, w, pf) in zip(self.groups, self.plans):
            ext().dense(x, [trellis[i] for i in idx], [suh[i] for i in idx], [svh[i] for i in idx],
                        [self.col0[i] for i in idx], y, xh, z, self.counters, k2, cb, sk, w, pf, ROT_SMEM)
        return y
