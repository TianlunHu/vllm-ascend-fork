# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared helpers for MC2 MoE dispatch/combine e2e tests (ZB + PTA baseline)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch_npu


def mc2_test_mode(env_prefix: str, default: str = "correctness") -> str:
    return os.environ.get(f"{env_prefix}_MODE", default).strip().lower()


def mc2_hccl_port(env_prefix: str = "VLLM_ASCEND_MOE_MC2_TEST") -> int:
    zb_port = os.environ.get("VLLM_ASCEND_ZB_TEST_HCCL_PORT")
    if zb_port:
        return int(zb_port)
    return int(os.environ.get(f"{env_prefix}_HCCL_PORT", "29500"))


def mc2_int_env(name: str, zb_fallback: str, default: str) -> int:
    raw = os.environ.get(name)
    if raw is not None:
        return int(raw)
    zb_raw = os.environ.get(zb_fallback)
    if zb_raw is not None:
        return int(zb_raw)
    return int(default)


def mc2_world_size() -> int:
    return mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE",
        "VLLM_ASCEND_ZB_TEST_WORLD_SIZE",
        "8",
    )


def mc2_bench_iters() -> tuple[int, int]:
    warmups = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_WARMUPS",
        "VLLM_ASCEND_ZB_TEST_NUM_WARMUPS",
        "50",
    )
    tests = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS",
        "VLLM_ASCEND_ZB_TEST_NUM_TESTS",
        "100",
    )
    return warmups, tests


def mc2_profile_iters() -> int:
    tests_default = str(
        mc2_int_env(
            "VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS",
            "VLLM_ASCEND_ZB_TEST_NUM_TESTS",
            "100",
        )
    )
    return mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_PROFILE_TESTS",
        "VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS",
        tests_default,
    )


def mc2_trace_dir(default: str) -> str:
    return os.environ.get(
        "VLLM_ASCEND_MOE_MC2_TEST_TRACE_DIR",
        os.environ.get("VLLM_ASCEND_ZB_TEST_TRACE_DIR", default),
    )


def mc2_bool_env(primary: str, zb_fallback: str) -> bool:
    for key in (primary, zb_fallback):
        raw = os.environ.get(key, "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
    return False


def mc2_w8a8_enabled() -> bool:
    return mc2_bool_env("VLLM_ASCEND_MOE_MC2_TEST_W8A8", "VLLM_ASCEND_ZB_TEST_W8A8")


def mc2_full_moe_enabled() -> bool:
    return mc2_bool_env("VLLM_ASCEND_MOE_MC2_TEST_FULL_MOE", "VLLM_ASCEND_ZB_TEST_FULL_MOE")


def mc2_intermediate_size() -> int:
    return mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_INTERMEDIATE",
        "VLLM_ASCEND_ZB_TEST_INTERMEDIATE",
        "512",
    )


def mc2_max_tokens_per_rank() -> int | None:
    """Optional cudagraph capture cap per TP rank (mirrors serving ``_num_tokens_per_tp_rank``).

    Example: ``max_num_seqs=128``, ``tp=4`` → ``(128 + 3) // 4 = 32``.
    When unset, ZB SHMEM pools are sized from ``num_tokens`` only.
    """
    raw = os.environ.get("VLLM_ASCEND_MOE_MC2_TEST_MAX_TOKENS_PER_RANK")
    if raw is None or raw == "":
        raw = os.environ.get("VLLM_ASCEND_ZB_TEST_MAX_TOKENS_PER_RANK")
    if raw is None or raw == "":
        return None
    return int(raw)


def mc2_shape_config(world_size: int) -> dict:
    """Read tensor shapes; falls back to VLLM_ASCEND_ZB_TEST_* for cross-test parity."""
    num_tokens = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS",
        "VLLM_ASCEND_ZB_TEST_NUM_TOKENS",
        "32",
    )
    hidden = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_HIDDEN",
        "VLLM_ASCEND_ZB_TEST_HIDDEN",
        "2048",
    )
    num_topk = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK",
        "VLLM_ASCEND_ZB_TEST_NUM_TOPK",
        "8",
    )
    num_experts = mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS",
        "VLLM_ASCEND_ZB_TEST_NUM_EXPERTS",
        str(max(world_size * 2, 16)),
    )
    assert num_experts % world_size == 0, "num_experts must be divisible by world_size"
    assert num_topk <= num_experts, (
        f"num_topk={num_topk} must be <= num_experts={num_experts} "
        "(build_fixed_inputs uses random.sample without replacement)"
    )
    num_local_experts = num_experts // world_size
    global_bs = num_tokens * world_size
    return {
        "num_tokens": num_tokens,
        "hidden": hidden,
        "num_topk": num_topk,
        "num_experts": num_experts,
        "num_local_experts": num_local_experts,
        "global_bs": global_bs,
    }


def get_group_ep(rank: int) -> str:
    group = dist.group.WORLD
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def normalize_topk_weights(topk_weights: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
    valid_mask = (topk_idx >= 0).to(topk_weights.dtype)
    masked_weights = topk_weights * valid_mask
    denom = masked_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return masked_weights / denom


def build_fixed_inputs(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device="npu") * (rank + 1)

    topk_idx_cpu = torch.empty((num_tokens, num_topk), dtype=torch.int32)
    expert_range = range(num_experts)
    for token_id in range(num_tokens):
        topk_idx_cpu[token_id] = torch.tensor(random.sample(expert_range, num_topk), dtype=torch.int32)
    topk_idx = topk_idx_cpu.to(device="npu")

    denom = float(num_topk * (num_topk + 1)) / 2.0
    topk_weights = (torch.arange(1, num_topk + 1, dtype=torch.float32, device="npu") / denom).repeat(num_tokens, 1)
    return x, topk_idx, topk_weights


def verify_combine_local(
    combined_x: torch.Tensor,
    original_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    rank: int,
    atol: float = 5e-5,
    rtol: float = 5e-5,
) -> None:
    normalized_weights = normalize_topk_weights(topk_weights.float(), topk_idx)
    weight_sum = normalized_weights.sum(dim=1).view(-1, 1)
    expected_x = original_x.float() * weight_sum

    actual_np = combined_x.float().cpu().numpy()
    expected_np = expected_x.cpu().numpy()
    passed = np.allclose(actual_np, expected_np, atol=atol, rtol=rtol)

    abs_diff = float(np.max(np.abs(actual_np - expected_np)))
    rel_diff = float(np.max(np.abs(actual_np - expected_np) / (np.abs(expected_np) + 1e-12)))
    assert passed, (
        f"rank {rank}: combine mismatch max_abs={abs_diff:.3e} max_rel={rel_diff:.3e} (atol={atol}, rtol={rtol})"
    )


def seed_worker(rank: int) -> None:
    random.seed(rank + 42)
    np.random.seed(rank + 42)
    torch.manual_seed(rank + 42)


def build_w8a8_expert_weights(
    num_local_experts: int,
    hidden: int,
    intermediate: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Local expert weights for w8a8 dynamic MoE (FRACTAL_NZ int8 + bf16 scales)."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    gmm1_weight = torch.randint(
        -16,
        16,
        (num_local_experts, hidden, intermediate * 2),
        dtype=torch.int8,
        generator=gen,
    )
    gmm2_weight = torch.randint(
        -16,
        16,
        (num_local_experts, intermediate, hidden),
        dtype=torch.int8,
        generator=gen,
    )
    gmm1_weight_scale = (torch.rand((num_local_experts, intermediate * 2), generator=gen) * 0.003 + 0.0015).bfloat16()
    gmm2_weight_scale = (torch.rand((num_local_experts, hidden), generator=gen) * 0.003 + 0.0015).bfloat16()

    gmm1_weight = torch_npu.npu_format_cast(gmm1_weight.npu(device=device), torch_npu.Format.FRACTAL_NZ)
    gmm2_weight = torch_npu.npu_format_cast(gmm2_weight.npu(device=device), torch_npu.Format.FRACTAL_NZ)
    gmm1_weight_scale = gmm1_weight_scale.to(device=device)
    gmm2_weight_scale = gmm2_weight_scale.to(device=device)
    return gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale


def w8a8_gmm1_swiglu(
    expand_x: torch.Tensor,
    gmm1_weight: torch.Tensor,
    gmm1_weight_scale: torch.Tensor,
    dynamic_scales: torch.Tensor,
    expert_token_nums: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_list = expert_token_nums.to(torch.int32)
    y1_int32 = torch_npu.npu_grouped_matmul(
        x=[expand_x],
        weight=[gmm1_weight],
        split_item=3,
        group_list_type=1,
        group_type=0,
        group_list=group_list,
        output_dtype=torch.int32,
    )[0]
    return torch_npu.npu_dequant_swiglu_quant(
        x=y1_int32,
        weight_scale=gmm1_weight_scale.to(torch.float32),
        activation_scale=dynamic_scales,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=group_list,
        activate_left=True,
        quant_mode=1,
    )


def w8a8_gmm2(
    y1: torch.Tensor,
    y1_scale: torch.Tensor,
    gmm2_weight: torch.Tensor,
    gmm2_weight_scale: torch.Tensor,
    expert_token_nums: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    group_list = expert_token_nums.to(torch.int32)
    gmm2_scale = gmm2_weight_scale.to(torch.float32)
    if out is None:
        return torch_npu.npu_grouped_matmul(
            x=[y1],
            weight=[gmm2_weight],
            scale=[gmm2_scale],
            per_token_scale=[y1_scale],
            split_item=2,
            group_list_type=1,
            group_type=0,
            group_list=group_list,
            output_dtype=output_dtype,
        )[0]
    from vllm_ascend.ops.fused_moe.zb_runtime import zb_moe_grouped_matmul_gmm2_out

    num_rows = y1.size(0)
    if num_rows > out.size(0):
        raise ValueError(
            f"gmm2 output rows {num_rows} exceed SHMEM combine_x capacity {out.size(0)}"
        )
    zb_moe_grouped_matmul_gmm2_out(
        y1,
        [gmm2_weight],
        group_list,
        out,
        scale=[gmm2_scale],
        per_token_scale=[y1_scale],
        split_item=2,
        group_type=0,
        group_list_type=1,
    )
    return out


def verify_tensors_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rank: int,
    *,
    label: str,
    atol: float = 5e-2,
    rtol: float = 5e-2,
) -> None:
    actual_np = actual.float().cpu().numpy()
    expected_np = expected.float().cpu().numpy()
    passed = np.allclose(actual_np, expected_np, atol=atol, rtol=rtol)
    abs_diff = float(np.max(np.abs(actual_np - expected_np)))
    rel_diff = float(np.max(np.abs(actual_np - expected_np) / (np.abs(expected_np) + 1e-12)))
    assert passed, (
        f"rank {rank}: {label} mismatch max_abs={abs_diff:.3e} max_rel={rel_diff:.3e} "
        f"(atol={atol}, rtol={rtol})"
    )
