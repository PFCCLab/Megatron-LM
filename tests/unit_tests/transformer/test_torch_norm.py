# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_config import TransformerConfig


def _config(**overrides):
    values = {
        "num_layers": 1,
        "hidden_size": 64,
        "num_attention_heads": 4,
        "normalization": "RMSNorm",
    }
    values.update(overrides)
    config = TransformerConfig(**values)
    config.experimental_attention_variant = "dsa"
    return config


def test_rmsnorm_uses_native_torch_implementation():
    config = _config(use_accuracy_compatible=True, params_dtype=torch.bfloat16)
    norm = WrappedTorchNorm(config=config, hidden_size=64, eps=1e-5)

    assert isinstance(norm, torch.nn.RMSNorm)
    assert norm.weight.dtype == torch.bfloat16


def test_sequence_parallel_is_opt_in_for_native_norm():
    with pytest.raises(AssertionError, match="sequence parallel"):
        WrappedTorchNorm(
            config=_config(sequence_parallel=True, tensor_model_parallel_size=2),
            hidden_size=64,
            eps=1e-5,
        )
    config = _config(
        sequence_parallel=True, tensor_model_parallel_size=2, use_accuracy_compatible=True
    )
    norm = WrappedTorchNorm(config=config, hidden_size=64, eps=1e-5)
    assert all(parameter.sequence_parallel for parameter in norm.parameters())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("variant", [None, "dsa"])
def test_reference_mode_follows_one_switch_and_architecture(enabled, variant):
    config = _config(use_accuracy_compatible=enabled)
    config.experimental_attention_variant = variant
    assert config.uses_dsa_reference is (enabled and variant == "dsa")
    with pytest.raises(AttributeError):
        config.uses_dsa_reference = True
