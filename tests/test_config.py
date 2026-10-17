"""Which vLLM layers the plugin treats as EXL3, from a checkpoint index (CPU only)."""
import json

import pytest

from vllm_espalier.config import EspalierConfig, text_key

KEYS = [f"model.language_model.layers.0.linear_attn.{m}.esp_trellis" for m in ("in_proj_qkv", "in_proj_z", "out_proj")] + \
       [f"model.language_model.layers.3.self_attn.{m}.esp_trellis" for m in ("q_proj", "k_proj", "v_proj", "o_proj")] + \
       [f"model.language_model.layers.0.mlp.shared_expert.{m}.esp_trellis" for m in ("gate_proj", "up_proj", "down_proj")] + \
       ["model.language_model.layers.0.mlp.experts.7.down_proj.esp_trellis", "lm_head.esp_trellis",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight", "model.visual.blocks.0.attn.qkv.weight"]


@pytest.fixture
def cfg(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "x" for k in KEYS}}))
    c = EspalierConfig.from_config({"quant_method": "espalier"})
    c.maybe_update_config(str(tmp_path))
    return c


def test_text_key():
    assert text_key("language_model.model.layers.3.self_attn.qkv_proj") == "layers.3.self_attn.qkv_proj"
    assert text_key("language_model.lm_head") == "lm_head"
    assert text_key("visual.blocks.0.attn.qkv") is None


def test_fused_and_plain_layers(cfg):
    assert cfg.is_exl3("language_model.model.layers.3.self_attn.qkv_proj")
    assert cfg.is_exl3("language_model.model.layers.3.self_attn.o_proj")
    assert cfg.is_exl3("language_model.model.layers.0.linear_attn.in_proj_qkvz")
    assert cfg.is_exl3("language_model.model.layers.0.mlp.shared_expert.gate_up_proj")
    assert not cfg.is_exl3("language_model.model.layers.0.linear_attn.in_proj_ba")
    assert not cfg.is_exl3("language_model.model.layers.0.mlp.gate")
    assert not cfg.is_exl3("visual.blocks.0.attn.qkv")
    assert cfg.is_exl3("language_model.lm_head")
    assert cfg.experts_exl3("language_model.model.layers.0.mlp.experts")


def test_partly_exl3_fused_layer_is_an_error(tmp_path):
    keys = ["model.language_model.layers.3.self_attn.q_proj.esp_trellis"]
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "x" for k in keys}}))
    c = EspalierConfig.from_config({"quant_method": "espalier"})
    c.maybe_update_config(str(tmp_path))
    with pytest.raises(ValueError):
        c.is_exl3("language_model.model.layers.3.self_attn.qkv_proj")


def test_new_names():
    """Checkpoints after the rename: .esp_trellis keys and quant_method "espalier"."""
    keys = ["model.language_model.layers.3.self_attn.q_proj.esp_trellis"]
    c = EspalierConfig.from_config({"quant_method": "espalier"})
    assert c.get_name() == "espalier"
    assert text_key(keys[0][: -len(".esp_trellis")]) == "layers.3.self_attn.q_proj"
