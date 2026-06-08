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
"""PTA MC2 baseline: ``npu_moe_distribute_dispatch_v2`` / ``combine_v2`` e2e tests.

Mirrors ``test_shmem_moe_distribute_zero_buffer.py`` but exercises only the CANN PTA
path used in vLLM serving baseline (no SHMEM / no ZB ops).

Use the same shape env vars as the ZB test for apples-to-apples comparison:
  VLLM_ASCEND_MOE_MC2_TEST_*  (or legacy VLLM_ASCEND_ZB_TEST_NUM_TOKENS / HIDDEN / ...)

Modes (``VLLM_ASCEND_PTA_MC2_TEST_MODE``):
  - ``correctness`` (default): dispatch -> combine round-trip verify
  - ``bench``: NPU-event wall clock + Kineto kernel summary
  - ``profile``: export chrome trace (rank<N>_pta_v2.json)

Examples:
  ./run_moe_distribute_v2_baseline_test.sh
  ./run_moe_distribute_v2_baseline_test.sh bench
  ./run_moe_distribute_v2_baseline_test.sh profile
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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from moe_mc2_e2e_common import (
    build_fixed_inputs,
    get_group_ep,
    mc2_hccl_port,
    mc2_int_env,
    mc2_shape_config,
    mc2_test_mode,
    seed_worker,
    verify_combine_local,
)
from zb_moe_prof_utils import (
    V2_MOE_KERNELS,
    bench,
    bench_kineto,
    print_kernel_table,
    print_pta_baseline_wallclock_table,
)

ENV_PREFIX = "VLLM_ASCEND_PTA_MC2_TEST"


def _test_mode() -> str:
    return mc2_test_mode(ENV_PREFIX)


def _world_size() -> int:
    return mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE",
        "VLLM_ASCEND_ZB_TEST_WORLD_SIZE",
        "8",
    )


@dataclass
class PtaMoeOpContext:
    rank: int
    world_size: int
    num_tokens: int
    hidden: int
    num_topk: int
    num_experts: int
    global_bs: int
    device: str
    group_ep: str
    x: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    expand_x: torch.Tensor | None = None
    assist_info: torch.Tensor | None = None
    ep_send_counts: torch.Tensor | None = None
    tp_send_counts: torch.Tensor | None = None
    expand_scales: torch.Tensor | None = None
    combined_x: torch.Tensor | None = None

    def run_dispatch(self) -> None:
        if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
            raise RuntimeError(
                "npu_moe_distribute_dispatch_v2 unavailable; upgrade CANN/PTA")
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
            shared_expert_rank_num=0,
            quant_mode=0,
            global_bs=self.global_bs,
            expert_token_nums_type=1,
        )
        (
            self.expand_x,
            _dynamic_scales,
            self.assist_info,
            _expert_token_nums,
            self.ep_send_counts,
            self.tp_send_counts,
            self.expand_scales,
        ) = outputs[0:7]

    def run_combine(self) -> None:
        if self.expand_x is None:
            self.run_dispatch()
        assert self.assist_info is not None
        assert self.ep_send_counts is not None
        assert self.tp_send_counts is not None
        self.combined_x = torch_npu.npu_moe_distribute_combine_v2(
            expand_x=self.expand_x,
            expert_ids=self.topk_idx,
            assist_info_for_combine=self.assist_info,
            ep_send_counts=self.ep_send_counts,
            expert_scales=self.topk_weights,
            tp_send_counts=self.tp_send_counts,
            expand_scales=self.expand_scales,
            group_ep=self.group_ep,
            ep_world_size=self.world_size,
            ep_rank_id=self.rank,
            moe_expert_num=self.num_experts,
            group_tp=self.group_ep,
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_rank_num=0,
            global_bs=self.global_bs,
            comm_quant_mode=0,
        )

    def run_dispatch_combine(self) -> None:
        self.run_dispatch()
        self.run_combine()


def _build_context(rank: int, world_size: int) -> PtaMoeOpContext:
    shape = mc2_shape_config(world_size)
    device = f"npu:{rank}"
    x, topk_idx, topk_weights = build_fixed_inputs(
        shape["num_tokens"],
        shape["hidden"],
        shape["num_topk"],
        shape["num_experts"],
        rank,
    )
    return PtaMoeOpContext(
        rank=rank,
        world_size=world_size,
        num_tokens=shape["num_tokens"],
        hidden=shape["hidden"],
        num_topk=shape["num_topk"],
        num_experts=shape["num_experts"],
        global_bs=shape["global_bs"],
        device=device,
        group_ep=get_group_ep(rank),
        x=x,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
    )


def _worker(rank: int, world_size: int, port: int,
            results: mp.SimpleQueue) -> None:
    try:
        torch_npu.npu.set_device(rank)
        seed_worker(rank)

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
            raise ValueError(f"Unknown {ENV_PREFIX}_MODE={mode!r}")

        dist.destroy_process_group()
        results.put((rank, True, None))
    except Exception as exc:  # pragma: no cover
        results.put((rank, False, repr(exc)))


def _run_correctness(ctx: PtaMoeOpContext) -> None:
    ctx.run_dispatch()
    torch.npu.synchronize()
    dist.barrier()

    ctx.run_combine()
    torch.npu.synchronize()
    dist.barrier()

    assert ctx.combined_x is not None
    verify_combine_local(
        ctx.combined_x,
        ctx.x,
        ctx.topk_weights,
        ctx.topk_idx,
        ctx.rank,
    )


def _run_bench(ctx: PtaMoeOpContext) -> None:
    num_warmups = mc2_int_env(
        f"{ENV_PREFIX}_NUM_WARMUPS",
        "VLLM_ASCEND_ZB_TEST_NUM_WARMUPS",
        "10",
    )
    num_tests = mc2_int_env(
        f"{ENV_PREFIX}_NUM_TESTS",
        "VLLM_ASCEND_ZB_TEST_NUM_TESTS",
        "100",
    )

    ctx.run_dispatch()
    torch.npu.synchronize()
    dist.barrier()

    dispatch_stats = bench(partial(ctx.run_dispatch), num_warmups, num_tests)
    combine_stats = bench(partial(ctx.run_combine), num_warmups, num_tests)

    print_pta_baseline_wallclock_table(
        rank=ctx.rank,
        num_tokens=ctx.num_tokens,
        hidden=ctx.hidden,
        num_topk=ctx.num_topk,
        num_experts=ctx.num_experts,
        num_ranks=ctx.world_size,
        dispatch_avg=dispatch_stats[0],
        combine_avg=combine_stats[0],
        num_warmups=num_warmups,
        num_tests=num_tests,
    )

    kernel_iters = min(30, num_tests)
    kernel_stats = bench_kineto(
        partial(ctx.run_dispatch_combine),
        kernel_names=V2_MOE_KERNELS,
        num_tests=kernel_iters,
        suppress_kineto_output=True,
    )
    print_kernel_table(
        rank=ctx.rank,
        label="PTA MC2 V2 kernels (baseline)",
        kernel_names=V2_MOE_KERNELS,
        dispatch_t=kernel_stats[0],
        combine_t=kernel_stats[1],
        num_tests=kernel_iters,
    )
    dist.barrier()


def _run_profile(ctx: PtaMoeOpContext) -> None:
    num_warmups = mc2_int_env(
        f"{ENV_PREFIX}_NUM_WARMUPS",
        "VLLM_ASCEND_ZB_TEST_NUM_WARMUPS",
        "10",
    )
    num_tests = mc2_int_env(
        f"{ENV_PREFIX}_NUM_PROFILE_TESTS",
        "VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS",
        "30",
    )
    trace_dir = os.environ.get(
        f"{ENV_PREFIX}_TRACE_DIR",
        os.environ.get("VLLM_ASCEND_ZB_TEST_TRACE_DIR", "./traces/pta_mc2_baseline"),
    )
    os.makedirs(trace_dir, exist_ok=True)

    for _ in range(num_warmups):
        ctx.run_dispatch_combine()
    torch.npu.synchronize()
    dist.barrier()

    trace_path = os.path.join(trace_dir, f"rank{ctx.rank}_pta_v2.json")
    kernel_stats = bench_kineto(
        partial(ctx.run_dispatch_combine),
        kernel_names=V2_MOE_KERNELS,
        num_tests=num_tests,
        trace_path=trace_path,
        suppress_kineto_output=(ctx.rank != 0),
    )
    dist.barrier()

    print_kernel_table(
        rank=ctx.rank,
        label="PTA MC2 V2 kernels (baseline)",
        kernel_names=V2_MOE_KERNELS,
        dispatch_t=kernel_stats[0],
        combine_t=kernel_stats[1],
        num_tests=num_tests,
        trace_path=trace_path,
    )
    if ctx.rank == 0:
        print(
            f"\n  Chrome traces saved under: {trace_dir}\n"
            "  Files: rank<N>_pta_v2.json\n"
            "  Open with chrome://tracing or Perfetto UI.\n",
            flush=True,
        )


def _launch_multiprocess(world_size: int | None = None) -> None:
    if not hasattr(torch_npu, "npu_moe_distribute_dispatch_v2"):
        raise AssertionError(
            "npu_moe_distribute_dispatch_v2 not available on this CANN build")

    world_size = world_size or _world_size()
    port = mc2_hccl_port() + random.randint(0, 10000)
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
    assert not failures, f"PTA MC2 baseline failures (mode={_test_mode()}): {failures}"


@torch.inference_mode()
def test_moe_distribute_v2_baseline_roundtrip() -> None:
    if _test_mode() != "correctness":
        pytest.skip(f"skip when {ENV_PREFIX}_MODE={_test_mode()}")
    _launch_multiprocess()


@torch.inference_mode()
@pytest.mark.skipif(
    _test_mode() != "bench",
    reason=f"set {ENV_PREFIX}_MODE=bench",
)
def test_moe_distribute_v2_baseline_bench() -> None:
    _launch_multiprocess()


@torch.inference_mode()
@pytest.mark.skipif(
    _test_mode() != "profile",
    reason=f"set {ENV_PREFIX}_MODE=profile",
)
def test_moe_distribute_v2_baseline_profile() -> None:
    _launch_multiprocess()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PTA MC2 MoeDistribute*V2 baseline e2e tests")
    parser.add_argument(
        "--mode",
        type=str,
        default=os.environ.get(f"{ENV_PREFIX}_MODE", "correctness"),
        choices=["correctness", "bench", "profile"],
    )
    parser.add_argument("--world-size", type=int, default=_world_size())
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
        "--num-warmups",
        type=int,
        default=mc2_int_env(f"{ENV_PREFIX}_NUM_WARMUPS",
                           "VLLM_ASCEND_ZB_TEST_NUM_WARMUPS", "10"),
    )
    parser.add_argument(
        "--num-tests",
        type=int,
        default=mc2_int_env(f"{ENV_PREFIX}_NUM_TESTS",
                           "VLLM_ASCEND_ZB_TEST_NUM_TESTS", "100"),
    )
    parser.add_argument(
        "--num-profile-tests",
        type=int,
        default=mc2_int_env(f"{ENV_PREFIX}_NUM_PROFILE_TESTS",
                           "VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS", "30"),
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=os.environ.get(f"{ENV_PREFIX}_TRACE_DIR",
                               "./traces/pta_mc2_baseline"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.environ[f"{ENV_PREFIX}_MODE"] = args.mode
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE"] = str(args.world_size)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS"] = str(args.num_tokens)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_HIDDEN"] = str(args.hidden)
    os.environ[f"{ENV_PREFIX}_NUM_WARMUPS"] = str(args.num_warmups)
    os.environ[f"{ENV_PREFIX}_NUM_TESTS"] = str(args.num_tests)
    os.environ[f"{ENV_PREFIX}_NUM_PROFILE_TESTS"] = str(args.num_profile_tests)
    os.environ[f"{ENV_PREFIX}_TRACE_DIR"] = args.trace_dir
    _launch_multiprocess(args.world_size)
