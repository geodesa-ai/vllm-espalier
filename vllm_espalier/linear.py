"""EXL3 linear method for dense layers and the LM head.

A vLLM layer may fuse several checkpoint matrices (q/k/v, gate/up, in_proj_qkv/z). Each keeps
its own suh/svh (its own input and output transform), so they cannot be concatenated into one
trellis: the layer holds one EspMatrix per shard and concatenates their outputs in vLLM's shard
order. Matrix shapes (K varies per matrix) are only known from the checkpoint, so the
parameters registered here are empty placeholders whose loader keeps each arriving tensor.

With DENSE (default; ESPALIER_FUSED_DENSE=0 turns it off) the whole layer is one espalier::dense call: up to 32 rows one launch of
csrc/dense.cu for all its matrices, input and output in the model's dtype (dense.py).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Parameter

import os

from vllm.model_executor.layers.linear import LinearMethodBase

from .matrix import SUFFIXES, EspMatrix
from ._env import env

_QKV = {"q": 0, "k": 1, "v": 2}
DENSE = env("ESPALIER_FUSED_DENSE", "1") != "0"


def shard_order(shard) -> int:
    if shard is None:
        return 0
    if isinstance(shard, str):
        return _QKV[shard]
    if isinstance(shard, (tuple, list)):
        return min(shard)
    return int(shard)


def single_process():
    from vllm.distributed import get_tensor_model_parallel_world_size
    if get_tensor_model_parallel_world_size() != 1:
        raise NotImplementedError("vllm_espalier milestone 1 supports tensor parallel size 1 only")


class EspLinearMethod(LinearMethodBase):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes, input_size,
                       output_size, params_dtype, **extra_weight_attrs):
        single_process()
        layer._esp_parts = {}                          # shard -> {suffix: tensor}
        layer._esp_in = input_size_per_partition
        layer._esp_out = sum(output_partition_sizes)

        def loader(param, loaded_weight, shard_id=None, *args, **kwargs):
            shard = getattr(loaded_weight, "shard_id", shard_id)
            key = tuple(shard) if isinstance(shard, list) else shard
            dev = param.device
            layer._esp_parts.setdefault(key, {})[param._esp_suffix] = loaded_weight.to(dev).clone()

        for s in SUFFIXES:
            p = Parameter(torch.empty(0), requires_grad=False)
            p._esp_suffix = s
            p.weight_loader = loader
            layer.register_parameter(s, p)

    def process_weights_after_loading(self, layer) -> None:
        if not layer._esp_parts:
            if self.prefix.endswith("lm_head"):
                # an MTP drafter's head, deliberately not loaded: vLLM replaces it with the target's (__init__.py)
                layer._esp_mats, layer._esp_dense, layer._esp_pad = [], None, 0
                return
            raise RuntimeError(f"{self.prefix}: no EXL3 tensors were loaded")
        layer._esp_mats = [EspMatrix(t) for _, t in
                           sorted(layer._esp_parts.items(), key=lambda kv: shard_order(kv[0]))]
        del layer._esp_parts
        # the packed tensors become module buffers, so torch.compile passes them in as inputs
        for i, m in enumerate(layer._esp_mats):
            layer.register_buffer(f"_esp_t{i}", m.trellis, persistent=False)
            layer.register_buffer(f"_esp_s{i}", m.suh, persistent=False)
            layer.register_buffer(f"_esp_v{i}", m.svh, persistent=False)
        ins = {m.in_features for m in layer._esp_mats}
        out = sum(m.out_features for m in layer._esp_mats)
        if ins != {layer._esp_in} or out > layer._esp_out:
            raise RuntimeError(f"{self.prefix}: EXL3 shapes in {ins} out {out} do not fit "
                               f"the layer (in {layer._esp_in}, out {layer._esp_out})")
        layer._esp_pad = layer._esp_out - out          # vocab padding of the LM head
        layer._esp_dense = None
        if DENSE and all(m.in_features % 128 == 0 and m.out_features % 128 == 0 for m in layer._esp_mats):
            import weakref
            from . import dense, ops
            key = next(ops.DENSE_KEYS)
            ops.DENSE_LAYERS[key] = dense.Dense(layer._esp_mats, layer._esp_out)
            layer._esp_dense = key
            # a layer vLLM discards after loading (an MTP drafter's own lm_head, replaced by the target's) must not
            # stay alive through the registry
            weakref.finalize(layer, ops.DENSE_LAYERS.pop, key, None)

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if layer._esp_dense is not None:
            n = len(layer._esp_mats)
            shape = x.shape[:-1]
            y = torch.ops.espalier.dense(x.reshape(-1, x.shape[-1]),
                                    [getattr(layer, f"_esp_t{i}") for i in range(n)],
                                    [getattr(layer, f"_esp_s{i}") for i in range(n)],
                                    [getattr(layer, f"_esp_v{i}") for i in range(n)], layer._esp_dense)
            y = y.view(*shape, y.shape[-1])
            return y + bias if bias is not None else y
        ys = [m.forward_with(x, getattr(layer, f"_esp_t{i}"), getattr(layer, f"_esp_s{i}"),
                             getattr(layer, f"_esp_v{i}")) for i, m in enumerate(layer._esp_mats)]
        y = ys[0] if len(ys) == 1 else torch.cat(ys, dim=-1)
        if layer._esp_pad:
            y = F.pad(y, (0, layer._esp_pad))
        if bias is not None:
            y = y + bias
        return y
