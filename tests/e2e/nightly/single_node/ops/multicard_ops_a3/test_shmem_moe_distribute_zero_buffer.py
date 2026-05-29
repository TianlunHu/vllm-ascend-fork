#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Correctness test for the SHMEM zero-buffer MoE distribute ops.

End-to-end ``dispatch -> combine`` round-trip on real NPUs, mirroring
``deepep_standalone``'s ``test_fixed_correctness_low_latency`` but exercising
the operators that were upstreamed into ``vllm-ascend-fork``:

  - ``torch.ops._C_ascend.shmem_moe_distribute_dispatch_zero_buffer``
  - ``torch.ops._C_ascend.shmem_moe_distribute_combine_zero_buffer``

The local check is the same the standalone test uses: with ``x[i] = rank + 1``
(constant per row) the combined output must equal ``x * sum_normalized_topk_weights``
for every valid token, since the ops only move tokens around and apply the
softmax-style weighted sum on combine.

Requires:
  - the package built with ``VLLM_ASCEND_ENABLE_ZB_OPS=1`` so both the runtime
    bindings and the dispatch/combine ops are registered.
  - a reachable SHMEM control endpoint (``VLLM_ASCEND_ZB_SHMEM_URI``).
  - an A3 box with at least ``VLLM_ASCEND_ZB_TEST_WORLD_SIZE`` NPUs (default 8).
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from vllm_ascend.ops.fused_moe.shmem_runtime import (
    ShmemMoERuntime,
    shmem_moe_distribute_combine_zero_buffer,
    shmem_moe_distribute_dispatch_zero_buffer,
)
from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _shmem_server_ipport() -> str:
    raw = os.environ.get("VLLM_ASCEND_ZB_SHMEM_URI", "tcp://127.0.0.1:29555")
    if raw.startswith("tcp://"):
        return raw[len("tcp://"):]
    return raw


def _hccl_master_port() -> int:
    return int(os.environ.get("VLLM_ASCEND_ZB_TEST_HCCL_PORT", "29500"))


def _normalize_topk_weights(topk_weights: torch.Tensor,
                            topk_idx: torch.Tensor) -> torch.Tensor:
    valid_mask = (topk_idx >= 0).to(topk_weights.dtype)
    masked_weights = topk_weights * valid_mask
    denom = masked_weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return masked_weights / denom


def _build_fixed_inputs(
    num_tokens: int,
    hidden: int,
    num_topk: int,
    num_experts: int,
    rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reproduces deepep_standalone's ``build_fixed_inputs`` semantics."""
    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16,
                   device="npu") * (rank + 1)

    topk_idx_cpu = torch.empty((num_tokens, num_topk), dtype=torch.int32)
    expert_range = range(num_experts)
    for token_id in range(num_tokens):
        topk_idx_cpu[token_id] = torch.tensor(
            random.sample(expert_range, num_topk), dtype=torch.int32)
    topk_idx = topk_idx_cpu.to(device="npu")

    denom = float(num_topk * (num_topk + 1)) / 2.0
    topk_weights = (torch.arange(
        1, num_topk + 1, dtype=torch.float32, device="npu") / denom).repeat(
            num_tokens, 1)
    return x, topk_idx, topk_weights


def _allocate_aux_tensors(
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    num_local_experts: int,
    num_max_tokens: int,
    device: str,
) -> dict:
    """Mirrors deepep_standalone's auxiliary buffer layout for the ZB path.

    See ``Buffer::low_latency_dispatch`` in deep_ep_standalone/csrc/deepep/deep_ep.cpp:
      - assist_info_for_combine (``expandIdx``): max(num_tokens*num_topk, num_max_tokens*16)
      - expert_token_nums      (``packed_recv_count``): [num_local_experts] int64
      - ep_recv_count                                : [num_experts * num_ranks] int32
      - tp_recv_count                                : [1] int32
    """
    max_size = max(num_tokens * num_topk, num_max_tokens * 16)
    return {
        "assist_info_for_combine":
        torch.empty((max_size, ), dtype=torch.int32, device=device),
        "expert_token_nums":
        torch.empty((num_local_experts, ), dtype=torch.int64, device=device),
        "ep_recv_count":
        torch.empty((num_experts * num_ranks, ),
                    dtype=torch.int32,
                    device=device),
        "tp_recv_count":
        torch.empty((1, ), dtype=torch.int32, device=device),
    }


def _verify_combine_local(
    combined_x: torch.Tensor,
    original_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    rank: int,
    atol: float = 5e-5,
    rtol: float = 5e-5,
) -> None:
    """Identical to deepep_standalone's ``verify_combine_local``."""
    normalized_weights = _normalize_topk_weights(topk_weights.float(),
                                                 topk_idx)
    weight_sum = normalized_weights.sum(dim=1).view(-1, 1)
    expected_x = original_x.float() * weight_sum

    actual_np = combined_x.float().cpu().numpy()
    expected_np = expected_x.cpu().numpy()
    passed = np.allclose(actual_np, expected_np, atol=atol, rtol=rtol)

    abs_diff = float(np.max(np.abs(actual_np - expected_np)))
    rel_diff = float(
        np.max(np.abs(actual_np - expected_np) / (np.abs(expected_np) + 1e-12)))
    assert passed, (f"rank {rank}: combine mismatch max_abs={abs_diff:.3e} "
                    f"max_rel={rel_diff:.3e} (atol={atol}, rtol={rtol})")


def _worker(rank: int, world_size: int, port: int,
            results: mp.SimpleQueue) -> None:
    try:
        torch_npu.npu.set_device(rank)

        random.seed(rank + 42)
        np.random.seed(rank + 42)
        torch.manual_seed(rank + 42)

        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
        )

        num_tokens = int(os.environ.get("VLLM_ASCEND_ZB_TEST_NUM_TOKENS",
                                        "32"))
        hidden = int(os.environ.get("VLLM_ASCEND_ZB_TEST_HIDDEN", "2048"))
        num_topk = int(os.environ.get("VLLM_ASCEND_ZB_TEST_NUM_TOPK", "8"))
        num_experts = int(
            os.environ.get("VLLM_ASCEND_ZB_TEST_NUM_EXPERTS",
                           str(max(world_size * 2, 16))))
        assert num_experts % world_size == 0, (
            "num_experts must be divisible by world_size")
        num_local_experts = num_experts // world_size

        global_bs = num_tokens * world_size
        num_max_tokens = global_bs * num_local_experts

        device = f"npu:{rank}"

        runtime = ShmemMoERuntime(
            rank=rank,
            world_size=world_size,
            server_ip_port=_shmem_server_ipport(),
        )
        runtime.init()
        runtime.alloc(element_count=2 * 1024 * 1024, element_size=4)

        bundle = runtime.allocate_low_latency_tensors(
            max_recv_tokens=num_max_tokens,
            hidden_size=hidden,
            device=device,
            use_quant=False,
        )

        aux = _allocate_aux_tensors(num_tokens, num_topk, num_experts,
                                    world_size, num_local_experts,
                                    num_max_tokens, device)

        x, topk_idx, topk_weights = _build_fixed_inputs(
            num_tokens, hidden, num_topk, num_experts, rank)

        dist.barrier()

        shmem_moe_distribute_dispatch_zero_buffer(
            x=x,
            expert_ids=topk_idx,
            expand_x_out=bundle.expand_x_out,
            dynamic_scales_out=bundle.expand_x_out.new_empty(
                num_max_tokens, dtype=torch.float32),
            assist_info_for_combine_out=aux["assist_info_for_combine"],
            expert_token_nums_out=aux["expert_token_nums"],
            ep_recv_count_out=aux["ep_recv_count"],
            tp_recv_count_out=aux["tp_recv_count"],
            ep_world_size=world_size,
            ep_rank_id=rank,
            moe_expert_num=num_experts,
            ext_info=runtime.ext_info,
            global_bs=global_bs,
        )

        torch.npu.synchronize()
        dist.barrier()

        combined_x = torch.empty((num_tokens, hidden),
                                 dtype=torch.bfloat16,
                                 device=device)

        # Per deepep_standalone (`Buffer::low_latency_combine`):
        #   - kernel `expandX` = SHMEM ``combine_x`` (staging area peers put into)
        #   - kernel `oriX`   = this rank's dispatch output (``expand_x_out``)
        #   - kernel `XOut`   = a fresh BF16 [num_tokens, hidden] tensor
        shmem_moe_distribute_combine_zero_buffer(
            expand_x=bundle.combine_x,
            expert_ids=topk_idx,
            assist_info_for_combine=aux["assist_info_for_combine"],
            ep_send_count=aux["ep_recv_count"],
            expert_scales=topk_weights,
            combined_x=combined_x,
            ori_x=bundle.expand_x_out,
            ep_world_size=world_size,
            ep_rank_id=rank,
            moe_expert_num=num_experts,
            ext_info=runtime.ext_info,
            global_bs=global_bs,
        )

        torch.npu.synchronize()
        dist.barrier()

        _verify_combine_local(combined_x, x, topk_weights, topk_idx, rank)

        runtime.finalize()
        dist.destroy_process_group()

        results.put((rank, True, None))
    except Exception as exc:  # pragma: no cover - reported via queue
        results.put((rank, False, repr(exc)))


@torch.inference_mode()
def test_shmem_moe_distribute_zero_buffer_roundtrip() -> None:
    if not hasattr(torch.ops._C_ascend,
                   "shmem_moe_distribute_dispatch_zero_buffer"):
        raise AssertionError(
            "shmem_moe_distribute_dispatch_zero_buffer not registered; rebuild "
            "vllm_ascend_C with VLLM_ASCEND_ENABLE_ZB_OPS=1")

    world_size = int(os.environ.get("VLLM_ASCEND_ZB_TEST_WORLD_SIZE", "8"))
    port = _hccl_master_port() + random.randint(0, 10000)
    mp.set_start_method("fork", force=True)

    results: mp.SimpleQueue = mp.SimpleQueue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_worker,
                       args=(rank, world_size, port, results))
        p.start()
        processes.append(p)

    statuses = [results.get() for _ in range(world_size)]
    for p in processes:
        p.join()

    failures = [(r, msg) for r, ok, msg in statuses if not ok]
    assert not failures, f"ZB dispatch/combine failures: {failures}"
