# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from vllm_ascend.utils import enable_custom_op

DEFAULT_LOCAL_MEM_SIZE = 4 * 1024 * 1024 * 1024

DEFAULT_COMM_ALG = "fullmesh_v1"


@dataclass
class LowLatencyShmemTensors:
    combine_x: torch.Tensor
    expand_x_out: torch.Tensor
    dynamic_scales_out: Optional[torch.Tensor]


def _ensure_custom_op_loaded() -> None:
    if not enable_custom_op():
        raise RuntimeError("vllm_ascend_C custom ops are not available; cannot use zero-buffer SHMEM runtime")


def _ensure_zb_op_available(op_name: str) -> None:
    _ensure_custom_op_loaded()
    if not hasattr(torch.ops._C_ascend, op_name):
        raise RuntimeError(
            f"torch.ops._C_ascend.{op_name} is not registered. Rebuild vllm_ascend_C with "
            "VLLM_ASCEND_ENABLE_ZB_OPS=1 to enable zero-buffer SHMEM MoE distribute ops.")


@dataclass
class ShmemMoERuntime:
    rank: int
    world_size: int
    server_ip_port: str | None = None
    local_mem_size: int = DEFAULT_LOCAL_MEM_SIZE
    ext_info: int = 0

    def __post_init__(self) -> None:
        if self.server_ip_port is None:
            self.server_ip_port = os.getenv("VLLM_ASCEND_ZB_SHMEM_URI", "")

    def init(self) -> int:
        _ensure_custom_op_loaded()
        actual_rank = torch.ops._C_ascend.zb_shmem_init(
            self.rank,
            self.world_size,
            self.local_mem_size,
            self.server_ip_port,
        )
        self.rank = int(actual_rank)
        return self.rank

    def alloc(self, element_count: int, element_size: int = 1) -> int:
        _ensure_custom_op_loaded()
        self.ext_info = int(torch.ops._C_ascend.zb_shmem_alloc(element_count, element_size))
        return self.ext_info

    def alloc_tensor(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Allocate a SHMEM-backed NPU tensor.

        Tensor data buffers are independent from ``ext_info``. The latter is
        the metadata/control buffer allocated by :meth:`alloc` and passed to
        zero-buffer kernels.
        """
        _ensure_custom_op_loaded()
        return torch.ops._C_ascend.zb_shmem_alloc_tensor(list(shape), dtype, str(device))

    def alias_tensor(self, base: torch.Tensor, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """Create a tensor view with a different dtype over a SHMEM tensor buffer."""
        _ensure_custom_op_loaded()
        return torch.ops._C_ascend.zb_shmem_alias_tensor(base, list(shape), dtype)

    def allocate_low_latency_tensors(
        self,
        max_recv_tokens: int,
        hidden_size: int,
        device: torch.device | str,
        *,
        use_quant: bool = False,
    ) -> LowLatencyShmemTensors:
        """Allocate the SHMEM tensors required by the zero-buffer low-latency path.

        This mirrors deepep_standalone's ``preallocate_lowlatency_shmem_tensors``:
        ``combine_x`` is always BF16; ``expand_x_out`` aliases it as INT8 when
        quantization is enabled, otherwise it owns a separate BF16 SHMEM buffer.
        ``dynamic_scales_out`` exists only for the quantized path.
        """
        combine_x = self.alloc_tensor([max_recv_tokens, hidden_size], torch.bfloat16, device)
        if use_quant:
            expand_x_out = self.alias_tensor(combine_x, [max_recv_tokens, hidden_size], torch.int8)
            dynamic_scales_out = self.alloc_tensor([max_recv_tokens], torch.float32, device)
        else:
            expand_x_out = self.alloc_tensor([max_recv_tokens, hidden_size], torch.bfloat16, device)
            dynamic_scales_out = None
        return LowLatencyShmemTensors(
            combine_x=combine_x,
            expand_x_out=expand_x_out,
            dynamic_scales_out=dynamic_scales_out,
        )

    def get_ext_info(self) -> int:
        _ensure_custom_op_loaded()
        self.ext_info = int(torch.ops._C_ascend.zb_shmem_get_ext_info())
        return self.ext_info

    def free(self, ptr: int | None = None) -> None:
        _ensure_custom_op_loaded()
        raw_ptr = self.ext_info if ptr is None else ptr
        if raw_ptr:
            torch.ops._C_ascend.zb_shmem_free(raw_ptr)
        if raw_ptr == self.ext_info:
            self.ext_info = 0

    def finalize(self) -> None:
        _ensure_custom_op_loaded()
        torch.ops._C_ascend.zb_shmem_finalize()
        self.ext_info = 0

    def is_initialized(self) -> bool:
        _ensure_custom_op_loaded()
        return bool(torch.ops._C_ascend.zb_shmem_is_initialized())

    def __enter__(self) -> "ShmemMoERuntime":
        self.init()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finalize()


def shmem_moe_distribute_dispatch_zero_buffer(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expand_x_out: torch.Tensor,
    dynamic_scales_out: torch.Tensor,
    assist_info_for_combine_out: torch.Tensor,
    expert_token_nums_out: torch.Tensor,
    ep_recv_count_out: torch.Tensor,
    tp_recv_count_out: torch.Tensor,
    *,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    ext_info: int,
    scales: Optional[torch.Tensor] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    elastic_info: Optional[torch.Tensor] = None,
    tp_world_size: int = 1,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 0,
    shared_expert_rank_num: int = 0,
    quant_mode: int = 0,
    global_bs: int = 0,
    expert_token_nums_type: int = 1,
    comm_alg: str = DEFAULT_COMM_ALG,
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Thin Python wrapper around torch.ops._C_ascend.shmem_moe_distribute_dispatch_zero_buffer.

    All output tensors must be pre-allocated by the caller. ``ext_info`` is the
    SHMEM global virtual address returned by :py:meth:`ShmemMoERuntime.alloc` /
    :py:meth:`ShmemMoERuntime.get_ext_info`.
    """
    _ensure_zb_op_available("shmem_moe_distribute_dispatch_zero_buffer")
    return torch.ops._C_ascend.shmem_moe_distribute_dispatch_zero_buffer(
        x,
        expert_ids,
        scales,
        x_active_mask,
        elastic_info,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        quant_mode,
        global_bs,
        expert_token_nums_type,
        ext_info,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
        expand_x_out,
        dynamic_scales_out,
        assist_info_for_combine_out,
        expert_token_nums_out,
        ep_recv_count_out,
        tp_recv_count_out,
    )


def shmem_moe_distribute_combine_zero_buffer(
    expand_x: torch.Tensor,
    expert_ids: torch.Tensor,
    assist_info_for_combine: torch.Tensor,
    ep_send_count: torch.Tensor,
    expert_scales: torch.Tensor,
    combined_x: torch.Tensor,
    *,
    ep_world_size: int,
    ep_rank_id: int,
    moe_expert_num: int,
    ext_info: int,
    tp_send_count: Optional[torch.Tensor] = None,
    x_active_mask: Optional[torch.Tensor] = None,
    activation_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    group_list: Optional[torch.Tensor] = None,
    expand_scales: Optional[torch.Tensor] = None,
    shared_expert_x: Optional[torch.Tensor] = None,
    elastic_info: Optional[torch.Tensor] = None,
    ori_x: Optional[torch.Tensor] = None,
    const_expert_alpha1: Optional[torch.Tensor] = None,
    const_expert_alpha2: Optional[torch.Tensor] = None,
    const_expert_v: Optional[torch.Tensor] = None,
    tp_world_size: int = 1,
    tp_rank_id: int = 0,
    expert_shard_type: int = 0,
    shared_expert_num: int = 0,
    shared_expert_rank_num: int = 0,
    global_bs: int = 0,
    out_dtype: int = 0,
    comm_quant_mode: int = 0,
    group_list_type: int = 0,
    comm_alg: str = DEFAULT_COMM_ALG,
    zero_expert_num: int = 0,
    copy_expert_num: int = 0,
    const_expert_num: int = 0,
) -> torch.Tensor:
    """Thin Python wrapper around torch.ops._C_ascend.shmem_moe_distribute_combine_zero_buffer.

    ``combined_x`` must be pre-allocated by the caller. ``ext_info`` is the
    SHMEM global virtual address used as the combine source buffer pointer
    (typically the same value used by ``shmem_moe_distribute_dispatch_zero_buffer``).
    """
    _ensure_zb_op_available("shmem_moe_distribute_combine_zero_buffer")
    if tp_send_count is None:
        # The combine tiling marks tp_send_count optional, but still validates
        # its shape and dtype. deepep_standalone always passes an int32 tensor
        # with one entry per TP rank even when tp_world_size == 1.
        tp_send_count = torch.empty((tp_world_size,), dtype=torch.int32, device=expand_x.device)
    return torch.ops._C_ascend.shmem_moe_distribute_combine_zero_buffer(
        expand_x,
        expert_ids,
        assist_info_for_combine,
        ep_send_count,
        expert_scales,
        tp_send_count,
        x_active_mask,
        activation_scale,
        weight_scale,
        group_list,
        expand_scales,
        shared_expert_x,
        elastic_info,
        ori_x,
        const_expert_alpha1,
        const_expert_alpha2,
        const_expert_v,
        ep_world_size,
        ep_rank_id,
        moe_expert_num,
        tp_world_size,
        tp_rank_id,
        expert_shard_type,
        shared_expert_num,
        shared_expert_rank_num,
        global_bs,
        out_dtype,
        comm_quant_mode,
        ext_info,
        group_list_type,
        comm_alg,
        zero_expert_num,
        copy_expert_num,
        const_expert_num,
        combined_x,
    )
