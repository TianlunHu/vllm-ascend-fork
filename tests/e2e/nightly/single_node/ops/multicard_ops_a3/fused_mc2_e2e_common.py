# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Weight/input helpers for fused MC2 single-op e2e tests."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch_npu

ACL_FORMAT_FRACTAL_NZ = 29


def mc2_intermediate_size(hidden: int) -> int:
    import os

    override = os.environ.get("VLLM_ASCEND_MOE_MC2_TEST_INTERMEDIATE")
    if override:
        return int(override)
    zb = os.environ.get("VLLM_ASCEND_ZB_TEST_MOE_INTERMEDIATE")
    if zb:
        return int(zb)
    return max(hidden // 2, 512)


def build_w8a8_nz_expert_weights(
    num_local_experts: int,
    hidden: int,
    moe_intermediate: int,
    device: torch.device | str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Build NZ int8 expert weights + int64 scales for dispatch_ffn_combine."""
    gmm1_out = moe_intermediate * 2
    torch_npu.npu.config.allow_internal_format = True

    w1_list: List[torch.Tensor] = []
    w2_list: List[torch.Tensor] = []
    scale1_list: List[torch.Tensor] = []
    scale2_list: List[torch.Tensor] = []

    for _ in range(num_local_experts):
        w1 = torch.randint(-16, 16, (hidden, gmm1_out), dtype=torch.int8, device=device)
        w2 = torch.randint(-16, 16, (moe_intermediate, hidden), dtype=torch.int8, device=device)
        w1_list.append(torch_npu.npu_format_cast(w1, ACL_FORMAT_FRACTAL_NZ))
        w2_list.append(torch_npu.npu_format_cast(w2, ACL_FORMAT_FRACTAL_NZ))
        scale1_list.append(
            torch.randint(0, 2, (gmm1_out, ), dtype=torch.int64, device=device))
        scale2_list.append(
            torch.randint(0, 2, (hidden, ), dtype=torch.int64, device=device))

    return w1_list, w2_list, scale1_list, scale2_list


def build_w8a8_nz_stacked_expert_weights(
    num_local_experts: int,
    hidden: int,
    moe_intermediate: int,
    device: torch.device | str,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Build list-wrapped stacked weights for dispatch_gmm_combine_decode."""
    gmm1_out = moe_intermediate * 2
    torch_npu.npu.config.allow_internal_format = True

    gmm1 = torch.randint(
        -16,
        16,
        (num_local_experts, hidden, gmm1_out),
        dtype=torch.int8,
        device=device,
    )
    gmm2 = torch.randint(
        -16,
        16,
        (num_local_experts, moe_intermediate, hidden),
        dtype=torch.int8,
        device=device,
    )
    gmm1 = torch_npu.npu_format_cast(gmm1, ACL_FORMAT_FRACTAL_NZ)
    gmm2 = torch_npu.npu_format_cast(gmm2, ACL_FORMAT_FRACTAL_NZ)
    gmm1_scale = torch.randint(
        0, 2, (num_local_experts, gmm1_out), dtype=torch.int64, device=device)
    gmm2_scale = torch.randint(
        0, 2, (num_local_experts, hidden), dtype=torch.int64, device=device)

    return (
        [gmm1],
        [gmm1_scale.float()],
        [gmm2],
        [gmm2_scale.float()],
    )


def build_random_topk(
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    device: torch.device | str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    expert_idx = torch.randint(
        0,
        num_experts,
        (num_tokens, num_topk),
        dtype=torch.int32,
        device=device,
    )
    probs = torch.randn((num_tokens, num_topk), dtype=torch.float32, device=device)
    probs = probs.abs()
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return expert_idx, probs


def verify_fused_output(out: torch.Tensor, expected_shape: torch.Size, rank: int) -> None:
    assert out.shape == expected_shape, (
        f"rank {rank}: output shape {tuple(out.shape)} != expected {tuple(expected_shape)}")
    assert torch.isfinite(out).all(), f"rank {rank}: fused output contains non-finite values"
