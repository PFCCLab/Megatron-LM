# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
from __future__ import annotations

import gc
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional, Protocol, cast

import numpy as np
import torch
import torch.nn.functional as F

from megatron.core.dist_checkpointing import ShardedTensor
from megatron.core.dist_checkpointing.mapping import (
    ReplicaId,
    ShardedStateDict,
    ShardedTensorFactory,
)
from megatron.core.fusions.fused_bias_geglu import (
    bias_geglu_impl,
    quick_gelu,
    weighted_bias_quick_geglu_impl,
)
from megatron.core.fusions.fused_bias_gelu import bias_gelu_impl
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl, weighted_bias_swiglu_impl
from megatron.core.transformer.module import MegatronModule, _use_accuracy_compatible
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module, not_none
from megatron.core.utils import (
    get_tensor_model_parallel_group_if_none,
    nvtx_range_pop,
    nvtx_range_push,
)

try:
    import transformer_engine  # pylint: disable=unused-import

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


logger = logging.getLogger(__name__)


class LinearFc1Interface(Protocol):
    """Interface for linear_fc1 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc1 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc1 module."""
        ...


class LinearFc1Builder(Protocol):
    """Protocol describing how to build a linear_fc1 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        gather_output: bool,
        bias: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
        stride: int = 1,
    ) -> LinearFc1Interface:
        """Builds a linear_fc1 layer for MLP."""
        ...


class TEActivationFunctionInterface(Protocol):
    """Interface for activation_function module in MLP."""

    def forward(self, input_: torch.Tensor, /) -> torch.Tensor:
        """Forward method for activation_function module."""
        ...


class TEActivationFunctionBuilder(Protocol):
    """Protocol for activation_function module in MLP."""

    def __call__(self, *, config: TransformerConfig) -> TEActivationFunctionInterface:
        """Builds an activation function module for MLP."""
        ...


class LinearFc2Interface(Protocol):
    """Interface for linear_fc2 module in MLP."""

    def forward(self, hidden_states: torch.Tensor, /) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward method for linear_fc2 module."""
        ...

    def backward_dw(self) -> None:
        """Backward method for linear_fc2 module."""
        ...


class LinearFc2Builder(Protocol):
    """Protocol describing how to build a linear_fc2 layer in MLP."""

    def __call__(
        self,
        input_size: int,
        output_size: int,
        /,
        *,
        config: TransformerConfig,
        init_method: Callable[[torch.Tensor], None],
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        is_expert: bool,
        tp_comm_buffer_name: str | None,
        tp_group: torch.distributed.ProcessGroup | None,
    ) -> LinearFc2Interface:
        """Builds a linear_fc2 layer for MLP."""
        ...


class _WeightedScaleFp64ProbsGrad(torch.autograd.Function):
    """weighted-scale ``x * probs``（x 为 fp32 SwiGLU 输出，probs 为 [tokens,1]）。

    前向数值与普通乘法一致，``dL/dx`` 仍按原 fp32 逐元素（保证 expert/fc1 wgrad 不变）。
    反向的 ``dL/dprobs`` 用 **fp64 从 fc1 输出 o1 重算 silu(gate)·val** 再 Σ_ffn(·grad)：
    - fp64 silu 消除 paddle/torch fp32 silu 的末位差（实测 Ssilu 差 ~5.7e-7）；
    - fp64 reduction 消除 ffn 维累加顺序差；
    与 PF 侧 fp64 probs_grad 完全一致（gate/router wgrad 分叉根因，known-diffs 5.6 思路）。"""

    @staticmethod
    def forward(ctx, x, probs, o1, glu_offset, clamp_val):
        ctx.save_for_backward(x, probs, o1)
        ctx.glu_offset = float(glu_offset)
        ctx.clamp_val = clamp_val
        return x * probs

    @staticmethod
    def backward(ctx, grad_out):
        x, probs, o1 = ctx.saved_tensors
        grad_x = grad_out * probs
        xf = o1.double()
        x_glu, x_linear = torch.chunk(xf, 2, dim=-1)
        if ctx.clamp_val is not None:
            x_glu = x_glu.clamp(min=None, max=ctx.clamp_val)
            x_linear = x_linear.clamp(min=-ctx.clamp_val, max=ctx.clamp_val)
        sv64 = F.silu(x_glu) * (x_linear + ctx.glu_offset)
        gf = grad_out.double()
        grad_probs = (sv64 * gf).sum(dim=-1, keepdim=True).to(probs.dtype)
        return grad_x, grad_probs, None, None, None


@dataclass
class MLPSubmodules:
    """
    The dataclass for ModuleSpecs of MLP submodules
    including  linear fc1, activation function, linear fc2.
    """

    linear_fc1: LinearFc1Builder

    linear_fc2: LinearFc2Builder

    activation_func: TEActivationFunctionBuilder | None = None
    """
    Builder for an activation function module; only used if config.use_te_activation_func is True.
    """


class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.


    Returns an output and a bias to be added to the output.
    If config.add_bias_linear is False, the bias returned is None.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config

        self.input_size = input_size if input_size != None else self.config.hidden_size

        self.tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        if ffn_hidden_size is None:
            if is_expert:
                raise ValueError("MoE MLP requires `ffn_hidden_size`, but it was not provided.")
            warnings.warn(
                "MLP requires ffn_hidden_size, but it was not provided. Using \
                    config.ffn_hidden_size by default.",
                DeprecationWarning,
                stacklevel=2,
            )
            ffn_hidden_size = not_none(self.config.ffn_hidden_size)

        # If this is a gated linear unit we double the output width
        # see https://arxiv.org/pdf/2002.05202.pdf
        # For GLU/SwiGLU, use stride=2 because each TP rank stores interleaved [gate, up] portions.
        # This is critical for correct weight resharding across different TP sizes.
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2
            fc1_stride = 2
            if self.config.use_kitchen:
                # Kitchen Linear doesn't support stride != 1.
                # Weight resharding across TP sizes will have aforementioned problems.
                fc1_stride = 1
        else:
            fc1_stride = 1

        # Use moe_latent_size only for routed experts. 'is_expert' is false for
        # shared_experts.
        use_latent_size = (self.config.moe_latent_size is not None) and is_expert

        self.linear_fc1 = submodules.linear_fc1(
            self.input_size if not use_latent_size else not_none(self.config.moe_latent_size),
            ffn_hidden_size,
            config=self.config,
            init_method=not_none(self.config.init_method),
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
            stride=fc1_stride,
        )

        if self.config.use_te_activation_func and not (submodules.activation_func is None):
            self.activation_func = apply_module(submodules.activation_func(config=self.config))
        else:
            self.activation_func = self.config.activation_func

        self.linear_fc2 = submodules.linear_fc2(
            not_none(self.config.ffn_hidden_size),
            not_none(
                self.config.hidden_size if not use_latent_size else self.config.moe_latent_size
            ),
            config=self.config,
            init_method=not_none(self.config.output_layer_init_method),
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )

    def forward(
        self, hidden_states: torch.Tensor, per_token_scale: torch.Tensor | None = None, **kwargs
    ):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p]
        nvtx_range_push(suffix="linear_fc1")
        intermediate_parallel, bias_parallel = apply_module(self.linear_fc1)(hidden_states)
        nvtx_range_pop(suffix="linear_fc1")

        nvtx_range_push(suffix="activation")
        if self.config.use_te_activation_func:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)
            if per_token_scale is not None:
                original_dtype = intermediate_parallel.dtype
                intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                intermediate_parallel = intermediate_parallel.to(original_dtype)
        elif self.config.bias_activation_fusion:
            if per_token_scale is not None:
                if self.activation_func == F.silu and self.config.gated_linear_unit:
                    # dtype is handled inside the fused kernel
                    intermediate_parallel = weighted_bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                    )
                elif self.activation_func == quick_gelu and self.config.gated_linear_unit:
                    intermediate_parallel = weighted_bias_quick_geglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        per_token_scale.unsqueeze(-1),
                        self.config.activation_func_fp8_input_store,
                        self.config.glu_linear_offset,
                        self.config.activation_func_clamp_value,
                    )
                else:
                    raise ValueError(
                        "Only support fusion of swiglu and quick_gelu with per_token_scale in MLP."
                    )
            else:
                if self.activation_func == F.gelu:
                    if self.config.gated_linear_unit:
                        intermediate_parallel = bias_geglu_impl(
                            intermediate_parallel, bias_parallel
                        )
                    else:
                        assert self.config.add_bias_linear is True
                        intermediate_parallel = bias_gelu_impl(intermediate_parallel, bias_parallel)
                elif self.activation_func == F.silu and self.config.gated_linear_unit:
                    intermediate_parallel = bias_swiglu_impl(
                        intermediate_parallel,
                        bias_parallel,
                        self.config.activation_func_fp8_input_store,
                        self.config.cpu_offloading
                        and self.config.cpu_offloading_activations
                        and HAVE_TE,
                    )
                else:
                    raise ValueError("Only support fusion of gelu and swiglu")
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            # 【修复的问题描述】：MoE expert 内 SwiGLU 与 router prob 相乘的计算精度对齐。
            # PaddleFleet 的 `fused_swiglu_scale` CUDA kernel 在 fp32 下完成 SwiGLU
            # 激活并乘上 per_token_scale（router prob），最后一次性 round 回 bf16；
            # 而 Megatron 原实现是 bf16 SwiGLU + bf16 乘 prob，存在两次 bf16 round，
            # 与 PF 末位有 diff。这里在存在 per_token_scale 时把 GLU 提升到 fp32 计算，
            # 配合下方 fp32 乘 prob 后再 cast 回 bf16，对齐 PF 单次 round 的语义。
            # 由 use_accuracy_compatible 控制：关闭时保留原始 bf16 计算路径。
            _glu_fp32 = per_token_scale is not None and _use_accuracy_compatible()
            if self.config.gated_linear_unit:

                def glu(x):
                    if _glu_fp32:
                        x = x.to(torch.float32)
                    x_glu, x_linear = torch.chunk(x, 2, dim=-1)
                    if (val := self.config.activation_func_clamp_value) is not None:
                        x_glu = x_glu.clamp(min=None, max=val)
                        x_linear = x_linear.clamp(min=-val, max=val)
                    return self.config.activation_func(x_glu) * (
                        x_linear + self.config.glu_linear_offset
                    )

                # 保存 fc1 输出 o1（bf16），供下方 fp64 probs_grad 从 o1 重算 silu·val。
                _o1_ref = intermediate_parallel
                intermediate_parallel = glu(intermediate_parallel)
            else:
                _o1_ref = None
                # 【修复的问题描述】：MoE expert 内 SwiGLU 与 router prob 相乘的计算精度对齐。
                # 非 GLU 分支同样在存在 per_token_scale 时把激活提升到 fp32 计算，
                # 保持与 PaddleFleet fused_swiglu_scale 一致的单次 round 路径。
                if _glu_fp32:
                    intermediate_parallel = self.activation_func(
                        intermediate_parallel.to(torch.float32)
                    )
                else:
                    intermediate_parallel = self.activation_func(intermediate_parallel)

            if per_token_scale is not None:
                if _glu_fp32:
                    # 【修复的问题描述】：MoE expert 内 SwiGLU 与 router prob 相乘的计算精度对齐。
                    # GLU 已在 fp32 下计算，这里把 per_token_scale 也 cast 到 fp32 相乘，
                    # 最后一次性 cast 回原始 bf16 dtype，对齐 PaddleFleet fused_swiglu_scale
                    # 「fp32 激活 × fp32 prob → 单次 bf16 round」的数值路径。
                    original_dtype = hidden_states.dtype
                    # dL/d(per_token_scale) 用 fp64 从 fc1 输出 o1 重算 silu·val 再 Σ_ffn(·grad)，
                    # 对齐 PF fp64 probs_grad（gate wgrad 分叉根因）。前向数值不变、dL/d(act)
                    # 仍 fp32 → expert/fc1 wgrad 不变。
                    if _o1_ref is not None:
                        intermediate_parallel = _WeightedScaleFp64ProbsGrad.apply(
                            intermediate_parallel,
                            per_token_scale.unsqueeze(-1).to(intermediate_parallel.dtype),
                            _o1_ref.detach(),
                            self.config.glu_linear_offset,
                            self.config.activation_func_clamp_value,
                        )
                    else:
                        intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(
                            -1
                        ).to(intermediate_parallel.dtype)
                    intermediate_parallel = intermediate_parallel.to(original_dtype)
                else:
                    # 原始路径：bf16 SwiGLU 输出直接乘 bf16 prob。
                    original_dtype = intermediate_parallel.dtype
                    intermediate_parallel = intermediate_parallel * per_token_scale.unsqueeze(-1)
                    intermediate_parallel = intermediate_parallel.to(original_dtype)

        nvtx_range_pop(suffix="activation")

        # [s, b, h]
        nvtx_range_push(suffix="linear_fc2")

        output, output_bias = apply_module(self.linear_fc2)(
            cast(torch.Tensor, intermediate_parallel)
        )
        nvtx_range_pop(suffix="linear_fc2")

        if per_token_scale is not None and output_bias is not None:
            # if this MLP is an expert, and bias is required, we add the bias to output directly
            # without doing bda later.
            output += output_bias.unsqueeze(0) * per_token_scale.unsqueeze(-1)
            output_bias = None

        return output, output_bias

    # pylint: disable=missing-function-docstring
    def sharded_state_dict(
        self, prefix: str = "", sharded_offsets: tuple = (), metadata: Optional[dict] = None
    ) -> ShardedStateDict:
        """Return the sharded state dictionary of the module."""
        sharded_state_dict = {}
        singleton_local_shards = (metadata or {}).get('singleton_local_shards', False)
        for name, module in self._modules.items():
            sub_sd = module.sharded_state_dict(f"{prefix}{name}.", sharded_offsets, metadata)
            if self.config.gated_linear_unit and name == "linear_fc1":
                for k, v in sub_sd.items():
                    if k in (f"{prefix}{name}.weight", f"{prefix}{name}.bias"):
                        sub_sd[k] = apply_swiglu_sharded_factory(
                            v, sharded_offsets, singleton_local_shards
                        )
            sharded_state_dict.update(sub_sd)
        return sharded_state_dict

    def backward_dw(self):
        self.linear_fc2.backward_dw()
        self.linear_fc1.backward_dw()


# pylint: disable=missing-function-docstring
def apply_swiglu_sharded_factory(
    original_sh_ten, sharded_offsets, singleton_local_shards: bool = False
):
    # We must split the tensor into 2 parts, each sharded separately.
    # This requires a ShardedTensorFactory which `chunk`s during saving
    # and `cat`s during loading

    swiglu_shard_axis = 0
    prepend_axis_num = len(sharded_offsets)
    original_shape = original_sh_ten.local_shape
    original_numel = int(np.prod(original_shape))
    local_axis_size = original_shape[swiglu_shard_axis]
    assert (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] % local_axis_size == 0
    )
    rank_offset = (
        original_sh_ten.global_offset[swiglu_shard_axis + prepend_axis_num] // local_axis_size
    )
    axis_frag = original_sh_ten.axis_fragmentations[swiglu_shard_axis + prepend_axis_num]

    @torch.no_grad()
    def sh_ten_build_fn(
        key: str, t: torch.Tensor, replica_id: ReplicaId, flattened_range: Optional[slice]
    ):
        if singleton_local_shards:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            offset_v = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag)
            w_key = f'{key}_w'
            v_key = f'{key}_v'
        else:
            offset_w = (swiglu_shard_axis + prepend_axis_num, rank_offset, axis_frag * 2)
            offset_v = (
                swiglu_shard_axis + prepend_axis_num,
                rank_offset + axis_frag,
                axis_frag * 2,
            )
            w_key = key
            v_key = key

        tensor_w, tensor_v = torch.chunk(t, 2, dim=swiglu_shard_axis)
        return [
            ShardedTensor.from_rank_offsets(
                w_key,
                tensor_w,
                *sharded_offsets,
                offset_w,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
            ShardedTensor.from_rank_offsets(
                v_key,
                tensor_v,
                *sharded_offsets,
                offset_v,
                replica_id=replica_id,
                prepend_axis_num=prepend_axis_num,
            ),
        ]

    def sh_ten_merge_fn(sub_state_dict):
        with torch.no_grad():
            try:
                return torch.cat(sub_state_dict)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                logger.warning(
                    f"CUDA OutOfMemoryError encountered during tensors merging."
                    f" Switching to CPU merge. (Error: {e})"
                )
                merged_sub_state_dict = torch.cat([t.cpu() for t in sub_state_dict])
                gc.collect()
                torch.cuda.empty_cache()
                return merged_sub_state_dict

    return ShardedTensorFactory(
        original_sh_ten.key,
        original_sh_ten.data,
        sh_ten_build_fn,
        sh_ten_merge_fn,
        original_sh_ten.replica_id,
        flattened_range=original_sh_ten.flattened_range,
    )
