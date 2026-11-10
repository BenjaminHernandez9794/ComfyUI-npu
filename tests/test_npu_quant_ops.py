import json

import torch

import comfy.ops


def test_npu_int8_fallback_dequantizes_before_linear(monkeypatch):
    class FakeQuantizedWeight:
        def __init__(self, value):
            self.value = value

        def dequantize(self):
            return self.value

    input = torch.randn(3, 4)
    weight = torch.randn(2, 4)
    bias = torch.randn(2)

    monkeypatch.setattr(comfy.ops, "_is_npu_int8_linear", lambda *_: True)
    monkeypatch.setattr(comfy.ops, "_can_use_npu_int8_linear", lambda *_args, **_kwargs: False)

    output = comfy.ops._npu_int8_linear_or_dequant(
        input, FakeQuantizedWeight(weight), bias
    )

    expected = torch.nn.functional.linear(input, weight, bias)
    assert torch.allclose(output, expected)


def test_mixed_precision_linear_forward_uses_npu_dispatch(monkeypatch):
    expected = torch.randn(2, 3)
    monkeypatch.setattr(
        comfy.ops,
        "_npu_int8_linear_or_dequant",
        lambda *_args, **_kwargs: expected,
    )
    linear_cls = comfy.ops.mixed_precision_ops().Linear

    output = linear_cls._forward(object(), torch.randn(2, 4), object(), None)

    assert output is expected


def test_quantized_linear_assign_materializes_meta_weight_and_scale():
    linear_cls = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear
    linear = linear_cls(4, 8, bias=False, device=torch.device("meta"))
    quant_config = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
        dtype=torch.uint8,
    )
    state_dict = {
        "weight": torch.zeros((8, 4), dtype=torch.int8),
        "weight_scale": torch.ones((8, 1), dtype=torch.bfloat16),
        "comfy_quant": quant_config,
    }

    missing, unexpected = linear.load_state_dict(state_dict, assign=True)

    assert missing == []
    assert unexpected == []
    assert linear.weight.device.type == "cpu"
    assert linear.weight._params.scale.device.type == "cpu"
    assert not linear.weight.is_meta
    assert not linear.weight._params.scale.is_meta


def test_quantized_apply_normalizes_parameters_inside_inference_mode():
    linear_cls = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear
    linear = linear_cls(4, 8, bias=False, device=torch.device("cpu"))
    quant_config = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
        dtype=torch.uint8,
    )
    state_dict = {
        "weight": torch.zeros((8, 4), dtype=torch.int8),
        "weight_scale": torch.ones((8, 1), dtype=torch.bfloat16),
        "comfy_quant": quant_config,
    }
    linear.load_state_dict(state_dict, assign=True)

    with torch.inference_mode():
        linear._apply(lambda value: value.to(device="cpu", copy=True))

    assert not linear.weight.is_inference()
    assert not linear.weight._params.scale.is_inference()
    assert linear.weight._version >= 0


def test_quantized_apply_cycles_stay_normal_inside_inference_mode():
    linear_cls = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear
    linear = linear_cls(16, 16, bias=False, device=torch.device("cpu"))
    quant_config = torch.tensor(
        list(json.dumps({"format": "int8_tensorwise"}).encode("utf-8")),
        dtype=torch.uint8,
    )
    state_dict = {
        "weight": torch.zeros((16, 16), dtype=torch.int8),
        "weight_scale": torch.ones((16, 1), dtype=torch.bfloat16),
        "comfy_quant": quant_config,
    }
    linear.load_state_dict(state_dict, assign=True)

    with torch.inference_mode():
        for _ in range(3):
            linear._apply(lambda value: value.to(device="cpu", copy=True))
            assert not linear.weight.is_inference()
            assert not linear.weight._params.scale.is_inference()
            assert linear.weight.untyped_storage().nbytes() > 0
            assert linear.weight._params.scale.untyped_storage().nbytes() > 0
