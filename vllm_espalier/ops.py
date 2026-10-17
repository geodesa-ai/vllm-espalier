"""torch custom ops around the extension, so torch.compile and CUDA-graph capture see opaque calls.

espalier::matmul(x, trellis, suh, svh, K, cb) = H(H(x * suh) @ W) * svh for one EXL3 matrix, x fp16 [m, in] contiguous ->
fp16 [m, out]: the dense kernel (csrc/dense.cu) for the one matrix, 32 rows a launch. The fake implementation only
gives the shape.

espalier::dense(x, trellis[], suh[], svh[], key) = every matrix of one vLLM layer, [m, in] in the model's dtype -> [m, out]
in the same dtype: one launch of csrc/dense.cu (dense.Dense, registered under `key`) per 32 rows.

Above 32 rows (prefill) the kernels run in 32-row windows (ESPALIER_PREFILL_MODE=chunk, default), so a row gets the bits it gets in
decode. ESPALIER_PREFILL_MODE=gemm instead decodes each matrix's weight (trellis decode kernel), applies the Hadamards with the had
kernel and multiplies with cuBLAS fp16, per matrix (the head always runs in windows: its weight would need ~1 GB).
"""
from __future__ import annotations

import torch

from ._env import env

PREFILL = env("ESPALIER_PREFILL_MODE", "chunk")
assert PREFILL in ("chunk", "gemm"), f"ESPALIER_PREFILL_MODE={PREFILL}: chunk or gemm"
_SINGLE: dict = {}           # trellis data_ptr -> dense.Dense for one matrix (<= 32 rows)


def _windows(d, x, *args):
    """d.forward over 32-row windows of x into one output."""
    if x.shape[0] <= d.MAX_ROWS:
        return d.forward(x, *args)
    y = x.new_empty((x.shape[0], d.n_out))
    for r in range(0, x.shape[0], d.MAX_ROWS):
        d.forward(x[r:r + d.MAX_ROWS], *args, out=y[r:r + d.MAX_ROWS])
    return y


def _matmul(x, trellis, suh, svh, K: int, cb: int):
    from . import dense, experts
    if x.shape[0] <= dense.MAX_ROWS or PREFILL == "chunk":
        d = _SINGLE.get(trellis.data_ptr())
        if d is None:
            from types import SimpleNamespace
            m = SimpleNamespace(trellis=trellis, suh=suh, svh=svh, K=K, cb=cb, in_features=suh.numel(),
                                out_features=svh.numel())
            d = _SINGLE[trellis.data_ptr()] = dense.Dense([m])
        return _windows(d, x)
    e = experts.ext()
    w = experts.dequant(trellis, K, cb)
    xh = torch.empty_like(x)
    e.had(x, suh, None, xh)
    y = xh @ w
    e.had(y, None, svh, y)
    return y


@torch.library.custom_op("espalier::matmul", mutates_args=())
def matmul(x: torch.Tensor, trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor,
           K: int, cb: int) -> torch.Tensor:
    return _matmul(x, trellis, suh, svh, K, cb)


DENSE_LAYERS: dict = {}       # key -> dense.Dense
DENSE_KEYS = __import__("itertools").count()
HUGE = 1 << 28                 # weights above which a layer never reconstructs (K * N; the head is 2048 x 248320)


@torch.library.custom_op("espalier::dense", mutates_args=())
def dense(x: torch.Tensor, trellis: list[torch.Tensor], suh: list[torch.Tensor], svh: list[torch.Tensor],
          key: int) -> torch.Tensor:
    d = DENSE_LAYERS[key]
    if x.shape[0] <= d.MAX_ROWS or PREFILL == "chunk" or d.K * d.N >= HUGE:
        # (the head: reconstructing its fp16 weight for hgemm would need ~1 GB)
        return _windows(d, x.contiguous(), trellis, suh, svh)
    xh = x.half().contiguous()
    ys = [matmul(xh, t, s, v, m.K, m.cb) for t, s, v, m in zip(trellis, suh, svh, d.mats)]
    y = (ys[0] if len(ys) == 1 else torch.cat(ys, dim=-1)).to(x.dtype)
    if d.n_out > d.N:
        y = torch.nn.functional.pad(y, (0, d.n_out - d.N))
    return y


@dense.register_fake
def _(x, trellis, suh, svh, key):
    return x.new_empty((x.shape[0], DENSE_LAYERS[key].n_out))


@matmul.register_fake
def _(x, trellis, suh, svh, K, cb):
    return x.new_empty((x.shape[0], svh.shape[0]), dtype=torch.half)
