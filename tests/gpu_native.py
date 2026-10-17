"""espalier::matmul against a float32 reference, both branches: <= 32 rows (dense kernel, one matrix) and > 32 rows (trellis
decode + had kernel + cuBLAS). Passes when the error above the fp16 output-rounding floor is below 5e-4.

usage: python tests/gpu_native.py --ckpt CKPT
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", type=Path, required=True)
a = ap.parse_args()
from vllm_espalier import ops                     # noqa: E402
from vllm_espalier.experts import dequant         # noqa: E402
from vllm_espalier.matrix import EspMatrix        # noqa: E402

dev = torch.device("cuda")
wm = json.load(open(a.ckpt / "model.safetensors.index.json"))["weight_map"]
hs, fails = {}, []


def t(k):
    f = wm[k]
    if f not in hs:
        hs[f] = safe_open(str(a.ckpt / f), "pt", device="cuda")
    return hs[f].get_tensor(k)


def mat(b):
    return EspMatrix({x: t(f"{b}.{x}") for x in ("esp_trellis", "suh", "svh", "mcg", "mul1", "esp_grid") if f"{b}.{x}" in wm})


H = None


def had(x):
    global H
    if H is None:
        h = torch.ones(1, 1)
        while h.shape[0] < 128:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        H = (h / 128 ** 0.5).to(dev)
    return (x.view(*x.shape[:-1], -1, 128) @ H).view(x.shape)


P = "model.language_model.layers"
cases = [f"{P}.0.linear_attn.in_proj_qkv", f"{P}.0.linear_attn.out_proj", f"{P}.3.self_attn.k_proj",
         f"{P}.5.mlp.experts.7.gate_proj", f"{P}.5.mlp.experts.7.down_proj", f"{P}.5.mlp.shared_expert.up_proj"]
torch.manual_seed(0)
for b in cases:
    m = mat(b)
    W = dequant(m.trellis, m.K, m.cb).float()
    for M in (1, 17, 32, 33, 200):
        x = (torch.randn(M, m.in_features, device=dev) * 0.5).half()
        y = torch.ops.espalier.matmul(x, m.trellis, m.suh, m.svh, m.K, m.cb).float()
        ref = had(had(x.float() * m.suh.float()) @ W) * m.svh.float()
        e = float((y - ref).abs().max() / ref.abs().max())
        fl = float((ref.half().float() - ref).abs().max() / ref.abs().max())
        msg = f"{b.split('layers.')[1]} M={M}: rel err {e:.2e} (fp16 floor {fl:.2e})"
        ok = e - fl < 5e-4
        print(("PASS " if ok else "FAIL ") + msg, flush=True)
        if not ok:
            fails.append(msg)
print("ALL PASS" if not fails else f"{len(fails)} FAILED")
sys.exit(1 if fails else 0)
