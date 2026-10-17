"""vLLM quantization config for Espalier checkpoints (quant_method "espalier").

Which modules are EXL3 is read from the checkpoint itself (every key ending in .esp_trellis),
in maybe_update_config, which vLLM calls with the model path. A vLLM layer is EXL3 if every
checkpoint matrix it fuses (qkv_proj <- q/k/v, gate_up_proj <- gate/up, in_proj_qkvz <-
in_proj_qkv/z, in_proj_ba <- in_proj_b/a) is; a partly-EXL3 fused layer is an error. Everything
else (router, in_proj_a/b, norms, embeddings, vision tower) stays unquantized.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

NAME = "espalier"
TRELLIS = (".esp_trellis",)
# vLLM fused layer -> the checkpoint matrices it holds, in output order
PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
}


def text_key(name: str) -> str | None:
    """Checkpoint or vLLM module name -> 'layers.N.<...>', 'mtp.layers.N.<...>' or 'lm_head' for the text model, else
    None."""
    if "visual" in name or "vision" in name:
        return None
    if name.endswith("lm_head") or ".lm_head." in name + ".":
        return "lm_head"
    m = re.search(r"(?:^|\.)(mtp\.)?(layers\.\d+\..+)$", name)
    # the MTP head's layers ("mtp.layers.N.*" in the checkpoint and as vLLM prefixes) keep their own namespace, so
    # they never alias the decoder's layer N
    return (m.group(1) or "") + m.group(2) if m else None


def checkpoint_dir(model_name: str, revision: str | None = None) -> Path:
    p = Path(model_name)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_name, revision=revision,
                                  allow_patterns=["*.json"]))


@register_quantization_config(NAME)
class EspalierConfig(QuantizationConfig):
    def __init__(self, raw: dict[str, Any] | None = None):
        super().__init__()
        self.raw = dict(raw or {})
        self.exl3: set[str] = set()          # text_key of every EXL3 matrix in the checkpoint
        self.packed_modules_mapping = dict(PACKED)

    def get_name(self) -> str:
        return NAME

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EspalierConfig":
        if str(config.get("quant_method", "")).lower() != NAME:
            raise ValueError(f"not an {NAME} checkpoint: {config.get('quant_method')!r}")
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
        if str((hf_quant_cfg or {}).get("quant_method", "")).lower() == NAME:
            return NAME
        return None

    def maybe_update_config(self, model_name: str, hf_config=None, revision: str | None = None):
        d = checkpoint_dir(model_name, revision)
        wmap = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        for k in wmap:
            suf = next((s for s in TRELLIS if k.endswith(s)), None)
            if suf:
                t = text_key(k[: -len(suf)])
                if t is not None:
                    self.exl3.add(t)
        if not self.exl3:
            raise ValueError(f"{d}: no .esp_trellis tensors found")
        self.checkpoint = d

    def is_exl3(self, prefix: str) -> bool:
        t = text_key(prefix)
        if t is None:
            return False
        if t == "lm_head":
            return "lm_head" in self.exl3
        head, _, leaf = t.rpartition(".")
        parts = [f"{head}.{c}" for c in PACKED.get(leaf, [leaf])]
        hit = [p in self.exl3 for p in parts]
        if any(hit) and not all(hit):
            raise ValueError(f"{prefix}: fused layer is only partly EXL3: {dict(zip(parts, hit))}")
        return all(hit)

    def experts_exl3(self, prefix: str) -> bool:
        t = text_key(prefix)
        return t is not None and any(k.startswith(t + ".") for k in self.exl3)

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        from vllm.model_executor.layers.fused_moe import RoutedExperts
        from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
        from .linear import EspLinearMethod
        from .moe import EspMoEMethod
        if isinstance(layer, RoutedExperts):
            if not self.experts_exl3(prefix):
                raise ValueError(f"{prefix}: routed experts without EXL3 tensors")
            return EspMoEMethod(layer.moe_config)
        if isinstance(layer, ParallelLMHead):
            return EspLinearMethod(prefix) if self.is_exl3(prefix) else None
        if isinstance(layer, LinearBase):
            return EspLinearMethod(prefix) if self.is_exl3(prefix) else UnquantizedLinearMethod()
        return None
