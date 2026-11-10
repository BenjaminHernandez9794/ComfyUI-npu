import torch
import pytest

from comfy.text_encoders.minimax import _minimax_h3_tp_state_dict


def _checkpoint():
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(32).reshape(8, 4),
        "model.layers.0.self_attn.o_proj.weight": torch.arange(32).reshape(4, 8),
        "model.layers.0.self_attn.o_proj.bias": torch.arange(4),
        "model.layers.0.mlp.gate_proj.weight": torch.arange(64).reshape(16, 4),
        "model.layers.0.mlp.down_proj.weight": torch.arange(64).reshape(4, 16),
        "model.layers.0.input_layernorm.weight": torch.arange(4),
        "model.embed_tokens.weight": torch.arange(40).reshape(10, 4),
        "visual.patch_embed.weight": torch.arange(16).reshape(4, 4),
    }


def test_tp_checkpoint_shards_projection_dimensions():
    source = _checkpoint()
    shard = _minimax_h3_tp_state_dict(source, world_size=2, rank=1, keep_frontend=False)

    assert torch.equal(shard["model.layers.0.self_attn.q_proj.weight"], source["model.layers.0.self_attn.q_proj.weight"][4:])
    assert torch.equal(shard["model.layers.0.self_attn.o_proj.weight"], source["model.layers.0.self_attn.o_proj.weight"][:, 4:])
    assert torch.equal(shard["model.layers.0.mlp.gate_proj.weight"], source["model.layers.0.mlp.gate_proj.weight"][8:])
    assert torch.equal(shard["model.layers.0.mlp.down_proj.weight"], source["model.layers.0.mlp.down_proj.weight"][:, 8:])
    assert "model.layers.0.self_attn.o_proj.bias" not in shard


def test_tp_checkpoint_only_rank_zero_keeps_frontend_and_row_bias():
    source = _checkpoint()
    shard = _minimax_h3_tp_state_dict(source, world_size=2, rank=0, keep_frontend=True)

    assert shard["model.embed_tokens.weight"] is source["model.embed_tokens.weight"]
    assert shard["visual.patch_embed.weight"] is source["visual.patch_embed.weight"]
    assert shard["model.layers.0.self_attn.o_proj.bias"] is source["model.layers.0.self_attn.o_proj.bias"]
    assert shard["model.layers.0.input_layernorm.weight"] is source["model.layers.0.input_layernorm.weight"]


def test_tp_checkpoint_rejects_non_divisible_projection():
    source = {"model.layers.0.self_attn.q_proj.weight": torch.empty(7, 4)}

    with pytest.raises(ValueError, match="Cannot tensor-parallel shard"):
        _minimax_h3_tp_state_dict(source, world_size=2, rank=0, keep_frontend=True)


def test_tp_checkpoint_shards_int8_scales_without_slicing_quant_metadata():
    source = {
        "model.layers.0.self_attn.q_proj.weight": torch.empty(8, 4, dtype=torch.int8),
        "model.layers.0.self_attn.q_proj.weight_scale": torch.arange(8).reshape(8, 1),
        "model.layers.0.self_attn.q_proj.comfy_quant": torch.arange(72, dtype=torch.uint8),
        "model.layers.0.self_attn.o_proj.weight": torch.empty(4, 8, dtype=torch.int8),
        "model.layers.0.self_attn.o_proj.weight_scale": torch.arange(4).reshape(4, 1),
        "model.layers.0.self_attn.o_proj.comfy_quant": torch.arange(72, dtype=torch.uint8),
    }

    shard = _minimax_h3_tp_state_dict(source, world_size=2, rank=1, keep_frontend=False)

    assert shard["model.layers.0.self_attn.q_proj.weight"].shape == (4, 4)
    assert torch.equal(
        shard["model.layers.0.self_attn.q_proj.weight_scale"],
        source["model.layers.0.self_attn.q_proj.weight_scale"][4:],
    )
    assert shard["model.layers.0.self_attn.q_proj.comfy_quant"] is source["model.layers.0.self_attn.q_proj.comfy_quant"]
    assert shard["model.layers.0.self_attn.o_proj.weight"].shape == (4, 4)
    assert shard["model.layers.0.self_attn.o_proj.weight_scale"] is source["model.layers.0.self_attn.o_proj.weight_scale"]
    assert shard["model.layers.0.self_attn.o_proj.comfy_quant"] is source["model.layers.0.self_attn.o_proj.comfy_quant"]
