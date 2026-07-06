#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Correctness and optional perf/prof tests for SHMEM zero-buffer MoE on real NPUs.

Paths:
  - ``dispatch -> combine`` (identity round-trip, no expert MLP)
  - ``dispatch -> gmm1 -> gmm2 -> combine`` (full w8a8 MoE layer, ``--full-moe``)

Operators under test (ZB):
  - ``torch.ops._C_ascend.zb_moe_distribute_dispatch`` / ``zb_moe_distribute_combine``
  - ``torch.ops._C_ascend.zb_moe_grouped_matmul_gmm2_out`` (full layer only)

PTA baseline (MC2 V2 + grouped matmul):
  - ``torch_npu.npu_moe_distribute_dispatch_v2`` / ``combine_v2``

Modes (``VLLM_ASCEND_ZB_TEST_MODE``):
  - ``correctness`` (default)
  - ``bench``: wall clock + kineto for ZB vs PTA
  - ``profile``: msprof trace

Examples:
  # w8a8 dispatch+combine only
  python .../test_zb_moe_distribute.py --mode bench --w8a8 --world-size 4 --num-tokens 8

  # full w8a8 MoE layer (dispatch + gmm1 + gmm2 + combine)
  python .../test_zb_moe_distribute.py --mode bench --full-moe \\
    --world-size 4 --num-tokens 8 --hidden 2048 --num-experts 256 --num-topk 8 \\
    --intermediate 512

  VLLM_ASCEND_ZB_TEST_MODE=profile \\
    VLLM_ASCEND_ZB_TEST_TRACE_DIR=./traces/zb_moe \\
    python tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_zb_moe_distribute.py

Requires:
  - package built with ``Ascend SHMEM installed at /usr/local/Ascend/shmem/latest``
  - ``VLLM_ASCEND_ZB_SHMEM_URI`` reachable across EP ranks
  - A3 with at least ``VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE`` NPUs (default 8)

Shape/bench env (shared with PTA/Fused baseline tests):
  ``VLLM_ASCEND_MOE_MC2_TEST_*`` (preferred), legacy ``VLLM_ASCEND_ZB_TEST_*`` fallback.

Serve-like ZB pool sizing (``max_num_seqs=128``, ``tp=4`` → per-rank cap 32)::

  VLLM_ASCEND_MOE_MC2_TEST_MAX_TOKENS_PER_RANK=32 \\
    python .../test_zb_moe_distribute.py --mode bench --world-size 4 --num-tokens 8 ...
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
from moe_mc2_e2e_common import (  # type: ignore[import-not-found,import-untyped]
    build_w8a8_expert_weights,
    mc2_bench_iters,
    mc2_full_moe_enabled,
    mc2_hccl_port,
    mc2_int_env,
    mc2_intermediate_size,
    mc2_max_tokens_per_rank,
    mc2_profile_iters,
    mc2_shape_config,
    mc2_trace_dir,
    mc2_w8a8_enabled,
    mc2_world_size,
    verify_tensors_close,
    w8a8_gmm1_swiglu,
    w8a8_gmm2,
)
from zb_moe_prof_utils import (  # type: ignore[import-not-found,import-untyped]
    V2_MOE_KERNELS,
    ZB_MOE_KERNELS,
    bench,
    bench_kineto,
    bench_moe_combine,
    bench_moe_dispatch,
    msprof_kernel_summary,
    print_kernel_table,
    print_msprof_trace_info,
    print_wallclock_table,
    profile_msprof,
)

from vllm_ascend.ops.fused_moe.zb_runtime import (
    LowLatencyZbTensors,
    ZbMoERuntime,
    compute_low_latency_max_recv_tokens,
    estimate_local_mem_size,
    zb_moe_distribute_combine,
    zb_moe_distribute_dispatch,
)
from vllm_ascend.utils import enable_custom_op

enable_custom_op()
torch_npu.npu.config.allow_internal_format = True


def _test_mode() -> str:
    return os.environ.get("VLLM_ASCEND_ZB_TEST_MODE", "correctness").strip().lower()


def _shmem_server_ipport() -> str:
    raw = os.environ.get("VLLM_ASCEND_ZB_SHMEM_URI", "tcp://127.0.0.1:29555")
    return raw if "://" in raw else f"tcp://{raw}"


def _hccl_master_port() -> int:
    return mc2_hccl_port()


def _get_group_ep(rank: int) -> str:
    group = dist.group.WORLD
    backend = group._get_backend(torch.device("npu"))
    return backend.get_hccl_comm_name(rank)


def _normalize_topk_weights(topk_weights: torch.Tensor, topk_idx: torch.Tensor) -> torch.Tensor:
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


def _allocate_aux_tensors(
    num_tokens: int,
    num_topk: int,
    num_experts: int,
    num_ranks: int,
    num_local_experts: int,
    num_max_tokens: int,
    device: str,
) -> dict:
    max_size = max(num_tokens * num_topk, num_max_tokens * 16)
    return {
        "assist_info_for_combine": torch.empty((max_size,), dtype=torch.int32, device=device),
        "expert_token_nums": torch.empty((num_local_experts,), dtype=torch.int64, device=device),
        "ep_recv_count": torch.empty((num_experts * num_ranks,), dtype=torch.int32, device=device),
        "tp_recv_count": torch.empty((1,), dtype=torch.int32, device=device),
        "dynamic_scales": torch.empty((num_max_tokens,), dtype=torch.float32, device=device),
    }


@dataclass
class ZbMoeOpContext:
    rank: int
    world_size: int
    num_tokens: int
    hidden: int
    intermediate: int
    num_topk: int
    num_experts: int
    num_local_experts: int
    global_bs: int
    num_max_tokens: int
    use_w8a8: bool
    full_moe: bool
    device: str
    group_ep: str
    runtime: ZbMoERuntime
    bundle: LowLatencyZbTensors
    aux: dict
    x: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    combined_x: torch.Tensor
    pta_combined_x: torch.Tensor
    gmm1_weight: torch.Tensor | None = None
    gmm1_weight_scale: torch.Tensor | None = None
    gmm2_weight: torch.Tensor | None = None
    gmm2_weight_scale: torch.Tensor | None = None
    pta_expand_x: torch.Tensor | None = None
    pta_dynamic_scales: torch.Tensor | None = None
    pta_expert_token_nums: torch.Tensor | None = None
    pta_assist_info: torch.Tensor | None = None
    pta_ep_send_counts: torch.Tensor | None = None
    pta_tp_send_counts: torch.Tensor | None = None
    pta_expand_scales: torch.Tensor | None = None
    _pta_gmm2_out: torch.Tensor | None = None

    @property
    def quant_mode(self) -> int:
        return 2 if self.use_w8a8 else 0

    @property
    def combine_comm_quant_mode(self) -> int:
        # After gmm2, combine input is bf16 expert output (see test_dispatch_gmm_combine_decode).
        return 0 if self.full_moe else self.quant_mode

    def _zb_combine_ori_x(self) -> torch.Tensor | None:
        if self.full_moe or self.use_w8a8:
            return None
        return self.bundle.expand_x_out

    def run_zb_dispatch(self) -> None:
        zb_moe_distribute_dispatch(
            x=self.x,
            expert_ids=self.topk_idx,
            expand_x_out=self.bundle.expand_x_out,
            dynamic_scales_out=self.aux["dynamic_scales"],
            assist_info_for_combine_out=self.aux["assist_info_for_combine"],
            expert_token_nums_out=self.aux["expert_token_nums"],
            ep_recv_count_out=self.aux["ep_recv_count"],
            tp_recv_count_out=self.aux["tp_recv_count"],
            ep_world_size=self.world_size,
            ep_rank_id=self.rank,
            moe_expert_num=self.num_experts,
            ext_info=self.runtime.ext_info,
            quant_mode=self.quant_mode,
            global_bs=self.global_bs,
            shared_expert_num=1,
        )

    def run_zb_gmm(self) -> None:
        assert self.full_moe and self.use_w8a8
        assert self.gmm1_weight is not None and self.gmm1_weight_scale is not None
        assert self.gmm2_weight is not None and self.gmm2_weight_scale is not None
        expand_x = self.bundle.expand_x_out.detach()
        y1, y1_scale = w8a8_gmm1_swiglu(
            expand_x,
            self.gmm1_weight,
            self.gmm1_weight_scale,
            self.aux["dynamic_scales"],
            self.aux["expert_token_nums"],
        )
        w8a8_gmm2(
            y1,
            y1_scale,
            self.gmm2_weight,
            self.gmm2_weight_scale,
            self.aux["expert_token_nums"],
            out=self.bundle.combine_x,
        )

    def run_zb_combine(self) -> None:
        expand_scales = self.aux["dynamic_scales"] if self.combine_comm_quant_mode == 2 else None
        zb_moe_distribute_combine(
            expand_x=self.bundle.combine_x,
            expert_ids=self.topk_idx,
            assist_info_for_combine=self.aux["assist_info_for_combine"],
            ep_send_count=self.aux["ep_recv_count"],
            expert_scales=self.topk_weights,
            combined_x=self.combined_x,
            tp_send_count=self.aux["tp_recv_count"],
            ori_x=self._zb_combine_ori_x(),
            expand_scales=expand_scales,
            ep_world_size=self.world_size,
            ep_rank_id=self.rank,
            moe_expert_num=self.num_experts,
            ext_info=self.runtime.ext_info,
            global_bs=self.global_bs,
            comm_quant_mode=self.combine_comm_quant_mode,
        )

    def run_zb_dispatch_combine(self) -> None:
        self.run_zb_dispatch()
        if self.full_moe:
            self.run_zb_gmm()
        self.run_zb_combine()

    def run_pta_dispatch(self) -> None:
        if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            raise RuntimeError("npu_moe_distribute_dispatch_v2 unavailable on this CANN build")
        outputs = torch_npu.npu_moe_distribute_dispatch_v2(
            x=self.x,
            expert_ids=self.topk_idx,
            expert_scales=self.topk_weights,
            group_ep=self.group_ep,
            ep_world_size=self.world_size,
            ep_rank_id=self.rank,
            moe_expert_num=self.num_experts,
            group_tp=self.group_ep,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=0,
            quant_mode=self.quant_mode,
            global_bs=self.global_bs,
            expert_token_nums_type=1,
        )
        (
            self.pta_expand_x,
            self.pta_dynamic_scales,
            self.pta_assist_info,
            self.pta_expert_token_nums,
            self.pta_ep_send_counts,
            self.pta_tp_send_counts,
            self.pta_expand_scales,
        ) = outputs[0:7]

    def run_pta_gmm(self) -> None:
        assert self.full_moe and self.use_w8a8
        assert self.pta_expand_x is not None and self.pta_dynamic_scales is not None
        assert self.pta_expert_token_nums is not None
        assert self.gmm1_weight is not None and self.gmm1_weight_scale is not None
        assert self.gmm2_weight is not None and self.gmm2_weight_scale is not None
        y1, y1_scale = w8a8_gmm1_swiglu(
            self.pta_expand_x,
            self.gmm1_weight,
            self.gmm1_weight_scale,
            self.pta_dynamic_scales,
            self.pta_expert_token_nums,
        )
        self._pta_gmm2_out = w8a8_gmm2(
            y1,
            y1_scale,
            self.gmm2_weight,
            self.gmm2_weight_scale,
            self.pta_expert_token_nums,
        )

    def run_pta_combine(self, *, expand_x: torch.Tensor | None = None) -> None:
        if self.pta_expand_x is None:
            self.run_pta_dispatch()
        assert self.pta_assist_info is not None
        assert self.pta_ep_send_counts is not None
        assert self.pta_tp_send_counts is not None
        combine_expand_x = expand_x
        if combine_expand_x is None:
            if self.full_moe:
                combine_expand_x = self._pta_gmm2_out
            else:
                combine_expand_x = self.pta_expand_x
        assert combine_expand_x is not None
        self.pta_combined_x = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=combine_expand_x,
            expert_ids=self.topk_idx,
            assist_info_for_combine=self.pta_assist_info,
            ep_send_counts=self.pta_ep_send_counts,
            expert_scales=self.topk_weights,
            tp_send_counts=self.pta_tp_send_counts,
            expand_scales=self.pta_expand_scales,
            group_ep=self.group_ep,
            ep_world_size=self.world_size,
            ep_rank_id=self.rank,
            moe_expert_num=self.num_experts,
            group_tp=self.group_ep,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=1,
            shared_expert_rank_num=0,
            global_bs=self.global_bs,
            comm_quant_mode=self.combine_comm_quant_mode,
        )

    def run_pta_dispatch_combine(self) -> None:
        self.run_pta_dispatch()
        if self.full_moe:
            self.run_pta_gmm()
        self.run_pta_combine()


def _build_context(rank: int, world_size: int) -> ZbMoeOpContext:
    cfg = mc2_shape_config(world_size)
    use_w8a8 = mc2_w8a8_enabled()
    full_moe = mc2_full_moe_enabled()
    if full_moe and not use_w8a8:
        raise ValueError("--full-moe requires w8a8 comm quant (pass --w8a8 or set VLLM_ASCEND_MOE_MC2_TEST_W8A8=1)")

    num_tokens = cfg["num_tokens"]
    hidden = cfg["hidden"]
    intermediate = mc2_intermediate_size()
    num_topk = cfg["num_topk"]
    num_experts = cfg["num_experts"]
    num_local_experts = cfg["num_local_experts"]
    global_bs = cfg["global_bs"]
    # Match TokenDispatcherWithZB / serving: global_bs * min(local_experts, topk), floored at 1024.
    num_max_tokens = compute_low_latency_max_recv_tokens(
        num_tokens_per_rank=num_tokens,
        ep_world_size=world_size,
        num_local_experts=num_local_experts,
        experts_per_token=num_topk,
        max_tokens_per_rank=mc2_max_tokens_per_rank(),
    )
    device = f"npu:{rank}"

    local_mem_size = estimate_local_mem_size(
        num_max_tokens,
        hidden,
        use_quant=use_w8a8,
        moe_expert_num=num_experts,
        ep_world_size=world_size,
    )
    runtime = ZbMoERuntime(
        rank=rank,
        world_size=world_size,
        server_ip_port=_shmem_server_ipport(),
        local_mem_size=local_mem_size,
    )
    runtime.init()
    runtime.alloc_ext_info()
    bundle = runtime.allocate_low_latency_tensors(
        max_recv_tokens=num_max_tokens,
        hidden_size=hidden,
        device=device,
        use_quant=use_w8a8,
    )
    aux = _allocate_aux_tensors(
        num_tokens, num_topk, num_experts, world_size, num_local_experts, num_max_tokens, device
    )
    x, topk_idx, topk_weights = _build_fixed_inputs(num_tokens, hidden, num_topk, num_experts, rank)
    combined_x = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    pta_combined_x = torch.empty((num_tokens, hidden), dtype=torch.bfloat16, device=device)

    gmm1_weight = gmm1_weight_scale = gmm2_weight = gmm2_weight_scale = None
    if full_moe:
        gmm1_weight, gmm1_weight_scale, gmm2_weight, gmm2_weight_scale = build_w8a8_expert_weights(
            num_local_experts,
            hidden,
            intermediate,
            device,
            seed=rank + 1000,
        )

    return ZbMoeOpContext(
        rank=rank,
        world_size=world_size,
        num_tokens=num_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_topk=num_topk,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        global_bs=global_bs,
        num_max_tokens=num_max_tokens,
        use_w8a8=use_w8a8,
        full_moe=full_moe,
        device=device,
        group_ep=_get_group_ep(rank),
        runtime=runtime,
        bundle=bundle,
        aux=aux,
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        combined_x=combined_x,
        pta_combined_x=pta_combined_x,
        gmm1_weight=gmm1_weight,
        gmm1_weight_scale=gmm1_weight_scale,
        gmm2_weight=gmm2_weight,
        gmm2_weight_scale=gmm2_weight_scale,
    )


def _verify_combine_local(
    combined_x: torch.Tensor,
    original_x: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_idx: torch.Tensor,
    rank: int,
    atol: float = 5e-5,
    rtol: float = 5e-5,
) -> None:
    normalized_weights = _normalize_topk_weights(topk_weights.float(), topk_idx)
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


def _worker(rank: int, world_size: int, port: int, results: mp.SimpleQueue) -> None:
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

        mode = _test_mode()
        ctx = _build_context(rank, world_size)
        dist.barrier()

        if mode == "correctness":
            _run_correctness(ctx)
        elif mode == "bench":
            _run_bench(ctx)
        elif mode == "profile":
            _run_profile(ctx)
        else:
            raise ValueError(f"Unknown VLLM_ASCEND_ZB_TEST_MODE={mode!r}")

        ctx.runtime.finalize()
        dist.destroy_process_group()
        results.put((rank, True, None))
    except Exception as exc:  # pragma: no cover - reported via queue
        results.put((rank, False, repr(exc)))


def _run_correctness(ctx: ZbMoeOpContext) -> None:
    def _sync(step: str) -> None:
        torch.npu.synchronize()
        dist.barrier()
        if ctx.rank == 0:
            print(f"  correctness: {step} OK", flush=True)

    ctx.run_pta_dispatch()
    _sync("PTA dispatch")
    if ctx.full_moe:
        ctx.run_pta_gmm()
        _sync("PTA gmm1+gmm2")
    ctx.run_pta_combine()
    _sync("PTA combine")

    ctx.run_zb_dispatch()
    _sync("ZB dispatch")
    if ctx.full_moe:
        ctx.run_zb_gmm()
        _sync("ZB gmm1+gmm2")
    ctx.run_zb_combine()
    _sync("ZB combine")

    if ctx.use_w8a8 or ctx.full_moe:
        verify_tensors_close(
            ctx.combined_x,
            ctx.pta_combined_x,
            ctx.rank,
            label="ZB vs PTA MC2",
            atol=8e-2 if ctx.full_moe else 5e-2,
            rtol=8e-2 if ctx.full_moe else 5e-2,
        )
    else:
        _verify_combine_local(ctx.combined_x, ctx.x, ctx.topk_weights, ctx.topk_idx, ctx.rank)


def _run_bench(ctx: ZbMoeOpContext) -> None:
    num_warmups, num_tests = mc2_bench_iters()

    if ctx.full_moe:
        zb_roundtrip = bench(partial(ctx.run_zb_dispatch_combine), num_warmups, num_tests)
        pta_roundtrip = bench(partial(ctx.run_pta_dispatch_combine), num_warmups, num_tests)
        if ctx.rank == 0:
            print(
                f"\n  Full MoE w8a8 round-trip (rank {ctx.rank}): "
                f"ZB={zb_roundtrip[0]*1e3:.4f} ms PTA={pta_roundtrip[0]*1e3:.4f} ms "
                f"(warmups={num_warmups} iters={num_tests})\n",
                flush=True,
            )
        dist.barrier()
        return

    zb_dispatch = bench_moe_dispatch(
        ctx.run_zb_dispatch,
        ctx.run_zb_combine,
        num_warmups,
        num_tests,
    )
    zb_combine = bench_moe_combine(
        ctx.run_zb_dispatch,
        ctx.run_zb_combine,
        num_warmups,
        num_tests,
    )
    pta_dispatch = bench_moe_dispatch(
        ctx.run_pta_dispatch,
        ctx.run_pta_combine,
        num_warmups,
        num_tests,
    )
    pta_combine = bench_moe_combine(
        ctx.run_pta_dispatch,
        ctx.run_pta_combine,
        num_warmups,
        num_tests,
    )
    zb_roundtrip = bench(
        partial(ctx.run_zb_dispatch_combine),
        num_warmups,
        num_tests,
    )
    pta_roundtrip = bench(
        partial(ctx.run_pta_dispatch_combine),
        num_warmups,
        num_tests,
    )

    print_wallclock_table(
        rank=ctx.rank,
        num_tokens=ctx.num_tokens,
        hidden=ctx.hidden,
        num_topk=ctx.num_topk,
        num_experts=ctx.num_experts,
        num_ranks=ctx.world_size,
        zb_dispatch_avg=zb_dispatch[0],
        zb_combine_avg=zb_combine[0],
        pta_dispatch_avg=pta_dispatch[0],
        pta_combine_avg=pta_combine[0],
        zb_roundtrip_avg=zb_roundtrip[0],
        pta_roundtrip_avg=pta_roundtrip[0],
        num_warmups=num_warmups,
        num_tests=num_tests,
    )

    zb_kernels = bench_kineto(
        partial(ctx.run_zb_dispatch_combine),
        kernel_names=ZB_MOE_KERNELS,
        num_warmups=num_warmups,
        num_tests=num_tests,
        suppress_kineto_output=True,
    )
    pta_kernels = bench_kineto(
        partial(ctx.run_pta_dispatch_combine),
        kernel_names=V2_MOE_KERNELS,
        num_warmups=num_warmups,
        num_tests=num_tests,
        suppress_kineto_output=True,
    )
    print_kernel_table(
        rank=ctx.rank,
        label="ZB SHMEM kernels (same round-trip session as wall-clock)",
        kernel_names=ZB_MOE_KERNELS,
        dispatch_t=zb_kernels[0],
        combine_t=zb_kernels[1],
        num_warmups=num_warmups,
        num_tests=num_tests,
    )
    print_kernel_table(
        rank=ctx.rank,
        label="PTA MC2 V2 kernels (same round-trip session as wall-clock)",
        kernel_names=V2_MOE_KERNELS,
        dispatch_t=pta_kernels[0],
        combine_t=pta_kernels[1],
        num_warmups=num_warmups,
        num_tests=num_tests,
    )
    if ctx.rank == 0:
        path_label = "full MoE (D+GMM1+GMM2+C)" if ctx.full_moe else "dispatch+combine only"
        quant_label = "w8a8 (quant_mode=2)" if ctx.use_w8a8 else "bf16 (quant_mode=0)"
        print(f"\n  Path: {path_label} | Quantization: {quant_label}\n", flush=True)
        zb_kineto_total = (zb_kernels[0] + zb_kernels[1]) * 1e3
        pta_kineto_total = (pta_kernels[0] + pta_kernels[1]) * 1e3
        zb_rt_ms = zb_roundtrip[0] * 1e3
        pta_rt_ms = pta_roundtrip[0] * 1e3
        print(
            f"\n  Cross-check (rank 0): round-trip wall-clock ZB={zb_rt_ms:.4f} ms "
            f"PTA={pta_rt_ms:.4f} ms | kineto kernel-sum ZB={zb_kineto_total:.4f} ms "
            f"PTA={pta_kineto_total:.4f} ms\n"
            "  (wall-clock includes sync/Python; kineto is device kernel dur only)\n",
            flush=True,
        )
    dist.barrier()


def _run_profile(ctx: ZbMoeOpContext) -> None:
    num_warmups, num_tests = mc2_bench_iters()
    profile_iters = mc2_profile_iters()
    trace_dir = mc2_trace_dir("./traces/zb_moe")
    os.makedirs(trace_dir, exist_ok=True)

    zb_root = os.path.join(trace_dir, "zb")
    pta_root = os.path.join(trace_dir, "pta_v2")
    zb_worker = f"rank{ctx.rank}_zb"
    pta_worker = f"rank{ctx.rank}_pta_v2"

    profile_msprof(
        partial(ctx.run_zb_dispatch_combine),
        trace_root=zb_root,
        worker_name=zb_worker,
        num_warmups=num_warmups,
        num_tests=profile_iters,
        suppress_output=(ctx.rank != 0),
    )
    dist.barrier()
    profile_msprof(
        partial(ctx.run_pta_dispatch_combine),
        trace_root=pta_root,
        worker_name=pta_worker,
        num_warmups=num_warmups,
        num_tests=profile_iters,
        suppress_output=(ctx.rank != 0),
    )
    dist.barrier()

    zb_summary = msprof_kernel_summary(zb_root, ZB_MOE_KERNELS)
    pta_summary = msprof_kernel_summary(pta_root, V2_MOE_KERNELS)

    print_msprof_trace_info(
        rank=ctx.rank,
        label="ZB SHMEM",
        trace_root=zb_root,
        num_warmups=num_warmups,
        num_tests=profile_iters,
        kernel_names=ZB_MOE_KERNELS,
        kernel_durations=zb_summary,
    )
    print_msprof_trace_info(
        rank=ctx.rank,
        label="PTA MC2 V2",
        trace_root=pta_root,
        num_warmups=num_warmups,
        num_tests=profile_iters,
        kernel_names=V2_MOE_KERNELS,
        kernel_durations=pta_summary,
    )
    if ctx.rank == 0:
        print(
            f"\n  Full msprof traces saved under: {trace_dir}\n"
            "  Layout: zb/ and pta_v2/ per rank (*_ascend_pt bundles)\n"
            "  Inspect ASCEND_PROFILER_OUTPUT/trace_view.json in MindStudio Insight.\n",
            flush=True,
        )


def _launch_multiprocess(world_size: int | None = None) -> None:
    if not hasattr(torch.ops._C_ascend, "zb_moe_distribute_dispatch"):
        raise AssertionError(
            "zb_moe_distribute_dispatch not registered; rebuild "
            "vllm_ascend_C with Ascend SHMEM installed at /usr/local/Ascend/shmem/latest"
        )

    world_size = world_size or mc2_world_size()
    port = _hccl_master_port() + random.randint(0, 10000)
    mp.set_start_method("fork", force=True)

    results: mp.SimpleQueue = mp.SimpleQueue()
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=_worker, args=(rank, world_size, port, results))
        p.start()
        processes.append(p)

    statuses = [results.get() for _ in range(world_size)]
    for p in processes:
        p.join()

    failures = [(r, msg) for r, ok, msg in statuses if not ok]
    assert not failures, f"ZB test failures (mode={_test_mode()}): {failures}"


@torch.inference_mode()
def test_zb_moe_distribute_roundtrip() -> None:
    if _test_mode() != "correctness":
        pytest.skip(f"skip correctness test when VLLM_ASCEND_ZB_TEST_MODE={_test_mode()}")
    _launch_multiprocess()


@torch.inference_mode()
@pytest.mark.skipif(
    _test_mode() != "bench",
    reason="set VLLM_ASCEND_ZB_TEST_MODE=bench to run wall-clock comparison",
)
def test_zb_moe_distribute_bench() -> None:
    _launch_multiprocess()


@torch.inference_mode()
@pytest.mark.skipif(
    _test_mode() != "profile",
    reason="set VLLM_ASCEND_ZB_TEST_MODE=profile to export kineto traces",
)
def test_zb_moe_distribute_profile() -> None:
    _launch_multiprocess()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SHMEM zero-buffer MoE dispatch/combine correctness and perf tests")
    parser.add_argument(
        "--mode",
        type=str,
        default=os.environ.get("VLLM_ASCEND_ZB_TEST_MODE", "correctness"),
        choices=["correctness", "bench", "profile"],
        help="correctness | bench (wall clock + kineto summary) | profile (full msprof trace)",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=mc2_world_size(),
    )
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=mc2_int_env(
            "VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS",
            "VLLM_ASCEND_ZB_TEST_NUM_TOKENS",
            "32",
        ),
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=mc2_int_env(
            "VLLM_ASCEND_MOE_MC2_TEST_HIDDEN",
            "VLLM_ASCEND_ZB_TEST_HIDDEN",
            "2048",
        ),
    )
    parser.add_argument(
        "--num-topk",
        type=int,
        default=mc2_int_env(
            "VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK",
            "VLLM_ASCEND_ZB_TEST_NUM_TOPK",
            "8",
        ),
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=None,
        help="defaults to max(world_size * 2, 16)",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=mc2_bench_iters()[0],
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=mc2_bench_iters()[1],
    )
    parser.add_argument(
        "--num-profile-tests",
        type=int,
        default=mc2_profile_iters(),
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=mc2_trace_dir("./traces/zb_moe"),
    )
    parser.add_argument(
        "--w8a8",
        action="store_true",
        help="enable w8a8 comm quant (quant_mode=2) for both ZB and PTA MC2 paths",
    )
    parser.add_argument(
        "--full-moe",
        action="store_true",
        help="run dispatch + gmm1 + gmm2 + combine (implies --w8a8)",
    )
    parser.add_argument(
        "--intermediate",
        type=int,
        default=mc2_intermediate_size(),
        help="MoE intermediate size for --full-moe expert weights",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.environ["VLLM_ASCEND_ZB_TEST_MODE"] = args.mode
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE"] = str(args.world_size)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS"] = str(args.num_tokens)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_HIDDEN"] = str(args.hidden)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK"] = str(args.num_topk)
    if args.num_experts is not None:
        os.environ["VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS"] = str(args.num_experts)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_NUM_WARMUPS"] = str(args.num_warmups)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS"] = str(args.num_tests)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_NUM_PROFILE_TESTS"] = str(args.num_profile_tests)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_TRACE_DIR"] = args.trace_dir
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_INTERMEDIATE"] = str(args.intermediate)
    if args.w8a8 or args.full_moe:
        os.environ["VLLM_ASCEND_MOE_MC2_TEST_W8A8"] = "1"
        os.environ["VLLM_ASCEND_ZB_TEST_W8A8"] = "1"
    if args.full_moe:
        os.environ["VLLM_ASCEND_MOE_MC2_TEST_FULL_MOE"] = "1"
        os.environ["VLLM_ASCEND_ZB_TEST_FULL_MOE"] = "1"
    # Mirror resolved values for legacy ZB_TEST_* readers.
    os.environ["VLLM_ASCEND_ZB_TEST_WORLD_SIZE"] = str(args.world_size)
    os.environ["VLLM_ASCEND_ZB_TEST_NUM_TOKENS"] = str(args.num_tokens)
    os.environ["VLLM_ASCEND_ZB_TEST_HIDDEN"] = str(args.hidden)
    os.environ["VLLM_ASCEND_ZB_TEST_NUM_TOPK"] = str(args.num_topk)
    if args.num_experts is not None:
        os.environ["VLLM_ASCEND_ZB_TEST_NUM_EXPERTS"] = str(args.num_experts)
    os.environ["VLLM_ASCEND_ZB_TEST_NUM_WARMUPS"] = str(args.num_warmups)
    os.environ["VLLM_ASCEND_ZB_TEST_NUM_TESTS"] = str(args.num_tests)
    os.environ["VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS"] = str(args.num_profile_tests)
    os.environ["VLLM_ASCEND_ZB_TEST_TRACE_DIR"] = args.trace_dir
    os.environ["VLLM_ASCEND_ZB_TEST_INTERMEDIATE"] = str(args.intermediate)
    _launch_multiprocess(args.world_size)
