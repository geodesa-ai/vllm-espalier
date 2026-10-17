"""One Espalier matrix: its checkpoint tensors and its forward, matching exllamav3's LinearEXL3.

Tensors per matrix (checkpoint suffixes): esp_trellis (int16 [in/16, out/16, 16K]), suh (fp16
[in]), svh (fp16 [out]), mcg (marker), and optionally mul1 (E2M1-snap codebook) or esp_grid
(int32 code: 1 E2M1, 2 E4M3, 3 int5, 4 int6). Codebook argument, as in the snap-v4 LinearEXL3:
esp_grid's code if present, else 1 if mul1 is present, else 0 (stock mcg).

forward(x) = H(H(x * suh) @ W) * svh, through the espalier::matmul custom op (ops.py). Nothing persistent beyond
the packed tensors.
"""
from __future__ import annotations

import torch


SUFFIXES = ("esp_trellis", "suh", "svh", "mcg", "mul1", "esp_grid")


class EspMatrix:
    def __init__(self, tensors: dict[str, torch.Tensor]):
        missing = [s for s in ("esp_trellis", "suh", "svh", "mcg") if s not in tensors]
        if missing:
            raise ValueError(f"Espalier matrix is missing {missing}")
        self.trellis = tensors["esp_trellis"].contiguous()
        assert self.trellis.dtype == torch.int16 and self.trellis.dim() == 3, self.trellis.shape
        self.suh = tensors["suh"].half().contiguous()
        self.svh = tensors["svh"].half().contiguous()
        self.in_features = self.suh.numel()
        self.out_features = self.svh.numel()
        self.K = self.trellis.shape[-1] // 16
        assert self.trellis.shape[0] * 16 == self.in_features, (self.trellis.shape, self.in_features)
        assert self.trellis.shape[1] * 16 == self.out_features, (self.trellis.shape, self.out_features)
        if "esp_grid" in tensors:
            self.cb = int(tensors["esp_grid"].flatten()[0].item())
            assert self.cb in (1, 2, 3, 4), f"esp_grid code {self.cb}"
        else:
            self.cb = 1 if "mul1" in tensors else 0

    def to(self, device):
        self.trellis, self.suh, self.svh = (t.to(device) for t in (self.trellis, self.suh, self.svh))
        return self

    def weight(self) -> torch.Tensor:
        """W [in, out] fp16 (before the Hadamards), from the trellis decode kernel."""
        from .experts import dequant
        return dequant(self.trellis, self.K, self.cb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., in] (any float dtype) -> [..., out] in x's dtype."""
        return self.forward_with(x, self.trellis, self.suh, self.svh)

    def forward_with(self, x, trellis, suh, svh) -> torch.Tensor:
        """forward with this matrix's tensors passed in (module buffers under torch.compile)."""
        shape = x.shape[:-1]
        x2 = x.reshape(-1, self.in_features).half().contiguous()
        y = torch.ops.espalier.matmul(x2, trellis, suh, svh, self.K, self.cb)
        return y.view(*shape, self.out_features).to(x.dtype)
