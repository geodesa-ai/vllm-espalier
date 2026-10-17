"""EXL3 routed experts.

Up to GROUPED_ROWS tokens (default; ESPALIER_GROUPED_EXPERTS=0 turns it off): the grouped pipeline of experts.py, the whole layer
for all tokens in six launches (after TensorFold's kernels, with the Espalier codebooks). The shared expert runs inside it
as one more slot: the method reports mk_can_overlap_shared_experts, so vLLM's runner leaves the shared expert to
apply(), which writes its gated output where the runner reads it (SharedExperts._output). Inputs and outputs stay in
the model's dtype; vLLM is asked for int32 topk ids, so no cast kernels run around the MoE.

With ESPALIER_FUSED_ROUTING=1 the method is monolithic: up to GROUPED_ROWS tokens the router GEMV, softmax / top-k /
renormalisation and the grouping run as one launch (csrc/route.cu) instead of three.

More tokens (prefill): the same pipeline in windows of GROUPED_ROWS tokens (ESPALIER_PREFILL_MODE=chunk, default), so a token
gets the bits it gets in decode; ESPALIER_PREFILL_MODE=gemm (or ESPALIER_GROUPED_EXPERTS=0) the expert-by-expert loop, each matrix through
espalier::matmul.

Each expert has its own gate/up/down EspMatrix, each with its own K and codebook (bit width set per expert), so nothing
is stacked. The layer's own load_weights is replaced: vLLM's expert loader treats any 3-D tensor as a pre-fused
all-experts tensor, and an EXL3 trellis is 3-D per expert ([in/16, out/16, 16K]).
"""
from __future__ import annotations

import re
import types

import torch
import torch.nn.functional as F
from torch.nn import Parameter

from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExpertsOrder

import os

from . import experts as grouped
from .linear import single_process
from .matrix import SUFFIXES, EspMatrix
from ._env import env
from .ops import PREFILL

GROUPED = env("ESPALIER_GROUPED_EXPERTS", "1") != "0"
GROUPED_ROWS = 32
DEBUG_FOLD = env("ESPALIER_DEBUG_SHARED_FOLD", "0") != "0"
# routing fused into one launch with the router GEMV (route.cu): the method is monolithic and the runner's gate passes
# the hidden states through. Its expert choices match exact routing (fp64 logits rounded to bf16); vLLM's bf16 router
# GEMM differs from it in ~2% of rows on near-ties. ESPALIER_FUSED_ROUTING=0 uses vLLM's router.
ROUTE = GROUPED and env("ESPALIER_FUSED_ROUTING", "1") != "0"

_NAME = re.compile(r"(?:^|\.)(\d+)\.(gate_proj|up_proj|down_proj)\.(\w+)$")


class _PassThroughGate(torch.nn.Module):
    """Stands in for the runner's router gate under ROUTE: returns the hidden states as the "logits", so the plugin's
    monolithic apply computes the GEMV inside route_group. Keeps the original module (and its weight)."""

    def __init__(self, gate):
        super().__init__()
        self.inner = gate
        self.weight = gate.weight

    def forward(self, x):
        return x, None


class EspMoEMethod(FusedMoEMethodBase):
    @property
    def topk_indices_dtype(self):
        return torch.int32

    @property
    def is_monolithic(self) -> bool:
        return ROUTE

    @property
    def mk_can_overlap_shared_experts(self) -> bool:
        return GROUPED              # apply() computes the shared expert (folded into the grouped launch)

    def create_weights(self, layer, num_experts, hidden_size, intermediate_size_per_partition,
                       params_dtype, **extra_weight_attrs):
        single_process()
        layer._esp_num_experts = num_experts
        layer._esp_parts = {}                  # (expert, proj) -> {suffix: tensor}
        for w in ("w13", "w2"):
            for s in SUFFIXES:
                layer.register_parameter(f"{w}_{s}", Parameter(torch.empty(0), requires_grad=False))
        dev = layer.w13_esp_trellis.device

        def load_weights(self_, weights):
            for name, t in weights:
                m = _NAME.search(name)
                if m is None:
                    raise ValueError(f"{self_.layer_name}: unexpected expert tensor {name!r}")
                e, proj, suffix = int(m.group(1)), m.group(2), m.group(3)
                if suffix not in SUFFIXES:
                    raise ValueError(f"{self_.layer_name}: unknown EXL3 suffix in {name!r}")
                self_._esp_parts.setdefault((e, proj), {})[suffix] = t.to(dev).clone()
                yield f"{'w2' if proj == 'down_proj' else 'w13'}_{suffix}"

        layer.load_weights = types.MethodType(load_weights, layer)

    def process_weights_after_loading(self, layer) -> None:
        E = layer._esp_num_experts
        mats = {}
        for (e, proj), t in layer._esp_parts.items():
            mats[(e, proj)] = EspMatrix(t)
        missing = [(e, p) for e in range(E) for p in ("gate_proj", "up_proj", "down_proj")
                   if (e, p) not in mats]
        if missing:
            raise RuntimeError(f"{layer.layer_name}: {len(missing)} expert matrices missing, "
                               f"e.g. {missing[:3]}")
        layer._esp_experts = [(mats[(e, "gate_proj")], mats[(e, "up_proj")], mats[(e, "down_proj")])
                              for e in range(E)]
        del layer._esp_parts
        act = str(getattr(layer, "activation", "silu")).lower()
        if "silu" not in act:
            raise NotImplementedError(f"{layer.layer_name}: activation {act}")
        # grouped: built at the first apply(), when the shared expert is known (grouped.prepare rebinds every
        # suh/svh to rows of per-projection stacks)
        layer._esp_grouped = None

    def get_fused_moe_quant_config(self, layer):
        return None

    def apply(self, layer, x, topk_weights, topk_ids, shared_experts=None, shared_experts_input=None):
        T, k = topk_ids.shape
        internal = (shared_experts is not None and shared_experts._determine_shared_experts_order(
            shared_experts_input) == SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
        if GROUPED and (T <= GROUPED_ROWS or PREFILL == "chunk"):
            if layer._esp_grouped is None:
                layer._esp_same_input = self._runner_has_no_input_transform(layer)
                layer._esp_grouped = self._prepare_grouped(
                    layer, shared_experts if internal and layer._esp_same_input else None)
            L = layer._esp_grouped
            # the shared slot reads x: valid when the runner has no routed-input transform, since then it passes the
            # same value as x and as shared_experts_input (compiled graphs hand them over in different buffers)
            fold = internal and L.shared_id >= 0 and layer._esp_same_input
            if DEBUG_FOLD and not getattr(layer, "_esp_dbg", False):
                layer._esp_dbg = True
                order = (shared_experts._determine_shared_experts_order(shared_experts_input)
                         if shared_experts is not None else None)
                print(f"[espalier fold] {getattr(layer, 'layer_name', '?')} T={T} shared_experts={shared_experts is not None} "
                      f"order={order} internal={internal} shared_id={L.shared_id} "
                      f"same_ptr={None if shared_experts_input is None else shared_experts_input.data_ptr() == x.data_ptr()} "
                      f"x={tuple(x.shape)}/{x.stride()} se_in={None if shared_experts_input is None else (tuple(shared_experts_input.shape), shared_experts_input.stride())} "
                      f"fold={fold}", flush=True)
            s = grouped.scratch(L, GROUPED_ROWS, k + (L.shared_id >= 0), x.device)
            xc, pick, wts = x.contiguous(), topk_ids.to(torch.int32).contiguous(), topk_weights.float().contiguous()
            if T <= GROUPED_ROWS:
                out, sh = grouped.routed2(xc, pick, wts, L, s)
            else:
                out, sh = grouped.windows(xc, pick, wts, L, s)
            if fold:
                shared_experts._output[shared_experts._output_idx] = sh
            elif internal:
                shared_experts.forward(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
            return out
        if internal:
            shared_experts.forward(shared_experts_input, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        ids = topk_ids.reshape(-1)
        tok = torch.arange(T, device=x.device).repeat_interleave(k)
        w = topk_weights.reshape(-1).float()
        for e in ids.unique().tolist():
            sel = ids == e
            t = tok[sel]
            gate, up, down = layer._esp_experts[e]
            xs = x[t]
            h = F.silu(gate.forward(xs).float()) * up.forward(xs).float()
            out.index_add_(0, t, down.forward(h.to(x.dtype)).float() * w[sel, None])
        return out.to(x.dtype)

    def apply_monolithic(self, layer, x, router_logits, input_ids=None):
        """Routing inside the plugin (ROUTE). The runner's gate is replaced on the first call by a pass-through (its
        weight kept here), so router_logits is then the hidden states; up to GROUPED_ROWS tokens route_group does
        GEMV + top-k + grouping in one launch, more tokens use the gate and vLLM's router as before."""
        from vllm.forward_context import get_forward_context
        runner = get_forward_context().no_compile_layers[layer.layer_name]
        if getattr(layer, "_esp_gate_w", None) is None:
            r = runner.router
            assert getattr(r, "renormalize", True) and getattr(r, "scoring_func", "softmax") == "softmax", \
                "route_group implements softmax top-k with renormalisation only"
            layer._esp_gate_w = runner.gate.weight.detach()
            runner.gate = _PassThroughGate(runner.gate)
        shared = runner._shared_experts
        # real logits (E wide) on the first call, before the pass-through is installed; the hidden states afterwards
        logits_given = router_logits.shape[-1] == layer._esp_num_experts != x.shape[-1]
        T = x.shape[0]
        k = runner.router.top_k if hasattr(runner.router, "top_k") else layer.top_k
        internal = shared is not None and shared._determine_shared_experts_order(x) == \
            SharedExpertsOrder.MK_INTERNAL_OVERLAPPED
        if (T <= GROUPED_ROWS or PREFILL == "chunk") and not logits_given:
            if layer._esp_grouped is None:
                layer._esp_same_input = self._runner_has_no_input_transform(layer)
                layer._esp_grouped = self._prepare_grouped(layer, shared if internal and layer._esp_same_input else None)
            L = layer._esp_grouped
            fold = internal and L.shared_id >= 0 and layer._esp_same_input
            s = grouped.scratch(L, GROUPED_ROWS, k + (L.shared_id >= 0), x.device)
            xc = x.contiguous()
            if T <= GROUPED_ROWS:
                grouped.route(xc, layer._esp_gate_w, k, L, s)
                out, sh = grouped.routed2(xc, None, None, L, s, k=k)
            else:
                out, sh = grouped.windows(xc, None, None, L, s, k=k, gate=layer._esp_gate_w)
            if fold:
                shared._output[shared._output_idx] = sh
            elif internal:
                shared.forward(x, SharedExpertsOrder.MK_INTERNAL_OVERLAPPED)
            return out
        logits = router_logits if logits_given else torch.nn.functional.linear(x, layer._esp_gate_w)
        topk_weights, topk_ids = runner.router.select_experts(hidden_states=x, router_logits=logits,
                                                              topk_indices_dtype=self.topk_indices_dtype,
                                                              input_ids=input_ids)
        return self.apply(layer, x, topk_weights, topk_ids, shared, x)

    @staticmethod
    def _runner_has_no_input_transform(layer) -> bool:
        """True when this layer's MoE runner applies no routed-input transform (latent MoE), so the routed input and
        the shared experts' input are the same value (MoERunner.apply_routed_input_transform)."""
        try:
            from vllm.forward_context import get_forward_context
            runner = get_forward_context().no_compile_layers.get(layer.layer_name)
        except Exception:
            return False
        return runner is not None and getattr(runner, "routed_input_transform", "missing") is None

    @staticmethod
    def _prepare_grouped(layer, shared_experts):
        """Pointer/width/codebook tables for the grouped kernels; the shared expert (if vLLM hands it to apply)
        becomes the last entry, its gate weight the sigmoid gate of that slot."""
        shared = gate_w = None
        if shared_experts is not None:
            mlp = shared_experts._layer
            gu, dn = getattr(mlp.gate_up_proj, "_esp_mats", None), getattr(mlp.down_proj, "_esp_mats", None)
            gate = getattr(mlp, "expert_gate", None)
            if gu and dn and len(gu) == 2 and gate is not None and gate.weight.shape[0] == 1:
                shared, gate_w = (gu[0], gu[1], dn[0]), gate.weight.detach()
        return grouped.prepare(layer._esp_experts, shared, gate_w)
