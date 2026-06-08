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
"""Fused MC2 serving baseline single-op tests (dispatch+GMM+combine in one kernel).

Mirrors vLLM serving when ``VLLM_ASCEND_ENABLE_FUSED_MC2`` is set:

  - ``1`` → ``torch.ops._C_ascend.dispatch_ffn_combine``  (W8A8, EP<=32)
  - ``2`` → ``torch.ops._C_ascend.dispatch_gmm_combine_decode`` (decode W8A8)

Modes (``VLLM_ASCEND_FUSED_MC2_TEST_MODE``):
  - ``correctness`` (default)
  - ``bench``
  - ``profile``

Examples:
  ./run_fused_mc2_baseline_test.sh
  VLLM_ASCEND_FUSED_MC2_TEST_VARIANT=2 ./run_fused_mc2_baseline_test.sh bench
  ./run_fused_mc2_baseline_test.sh profile
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

from fused_mc2_e2e_common import (
    build_random_topk,
    build_w8a8_nz_expert_weights,
    build_w8a8_nz_stacked_expert_weights,
    mc2_intermediate_size,
    verify_fused_output,
)
from moe_mc2_e2e_common import (
    get_group_ep,
    mc2_bench_iters,
    mc2_hccl_port,
    mc2_int_env,
    mc2_profile_iters,
    mc2_shape_config,
    mc2_test_mode,
    seed_worker,
)
from vllm_ascend.utils import enable_custom_op
from zb_moe_prof_utils import (
    FUSED_MC2_FFNC_KERNEL,
    FUSED_MC2_GMMCD_KERNEL,
    bench,
    bench_kineto,
    msprof_kernel_summary,
    print_fused_mc2_wallclock_table,
    print_msprof_trace_info,
    print_single_kernel_table,
    profile_msprof,
)

enable_custom_op()

ENV_PREFIX = "VLLM_ASCEND_FUSED_MC2_TEST"


def _test_mode() -> str:
    return mc2_test_mode(ENV_PREFIX)


def _world_size() -> int:
    return mc2_int_env(
        "VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE",
        "VLLM_ASCEND_ZB_TEST_WORLD_SIZE",
        "8",
    )


def _fused_variant() -> int:
    raw = os.environ.get(
        f"{ENV_PREFIX}_VARIANT",
        os.environ.get("VLLM_ASCEND_ENABLE_FUSED_MC2", "1"),
    )
    variant = int(raw)
    if variant not in (1, 2):
        raise ValueError(f"Fused MC2 variant must be 1 or 2, got {variant}")
    return variant


@dataclass
class FusedMc2OpContext:
    rank: int
    world_size: int
    variant: int
    num_tokens: int
    hidden: int
    moe_intermediate: int
    num_topk: int
    num_experts: int
    num_local_experts: int
    global_bs: int
    device: str
    group_ep: str
    x: torch.Tensor
    expert_idx: torch.Tensor
    probs: torch.Tensor
    weight1: list
    weight2: list
    scale1: list
    scale2: list
    out: torch.Tensor
    expert_token_nums: torch.Tensor
    x_active_mask: torch.Tensor | None = None

    @property
    def kernel_name(self) -> str:
        return (FUSED_MC2_FFNC_KERNEL if self.variant == 1 else FUSED_MC2_GMMCD_KERNEL)[0]

    @property
    def kernel_label(self) -> str:
        return ("dispatch_ffn_combine" if self.variant == 1 else "dispatch_gmm_combine_decode")

    def run_fused(self) -> None:
        if self.variant == 1:
            self._run_dispatch_ffn_combine()
        else:
            self._run_dispatch_gmm_combine_decode()

    def _run_dispatch_ffn_combine(self) -> None:
        kwargs = dict(
            x=self.x,
            weight1=self.weight1,
            weight2=self.weight2,
            expert_idx=self.expert_idx,
            bias1=torch.tensor([]),
            bias2=torch.tensor([]),
            scale1=self.scale1,
            scale2=self.scale2,
            probs=self.probs,
            group=self.group_ep,
            max_output_size=int(
                os.environ.get(f"{ENV_PREFIX}_MAX_OUTPUT_SIZE", "65536")),
            swiglu_limit=int(os.environ.get(f"{ENV_PREFIX}_SWIGLU_LIMIT", "0")),
            out=self.out,
            expert_token_nums=self.expert_token_nums,
        )
        if self.x_active_mask is not None:
            kwargs["x_active_mask"] = self.x_active_mask
        torch.ops._C_ascend.dispatch_ffn_combine(**kwargs)

    def _run_dispatch_gmm_combine_decode(self) -> None:
        smooth_scales = torch.zeros(128, dtype=torch.float32, device=self.device)
        out, _expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(
            x=self.x,
            expert_ids=self.expert_idx,
            gmm1_permuted_weight=self.weight1,
            gmm1_permuted_weight_scale=self.scale1,
            gmm2_weight=self.weight2,
            gmm2_weight_scale=self.scale2,
            expert_scales=self.probs,
            expert_smooth_scales=smooth_scales,
            x_active_mask=self.x_active_mask,
            group_ep=self.group_ep,
            ep_rank_size=self.world_size,
            ep_rank_id=self.rank,
            moe_expert_num=self.num_experts,
            global_bs=self.global_bs,
        )
        self.out = out


def _build_context(rank: int, world_size: int) -> FusedMc2OpContext:
    variant = _fused_variant()
    if variant == 1 and world_size > 32:
        raise ValueError("dispatch_ffn_combine requires EP world_size <= 32")

    shape = mc2_shape_config(world_size)
    hidden = shape["hidden"]
    moe_intermediate = mc2_intermediate_size(hidden)
    device = f"npu:{rank}"

    x = torch.randn((shape["num_tokens"], hidden), dtype=torch.bfloat16, device=device)
    expert_idx, probs = build_random_topk(
        shape["num_tokens"],
        shape["num_topk"],
        shape["num_experts"],
        device,
    )
    out = torch.empty_like(x)
    expert_token_nums = torch.zeros(
        (shape["num_local_experts"], ),
        dtype=torch.int32,
        device=device,
    )

    use_mc2_mask = os.environ.get(f"{ENV_PREFIX}_USE_MC2_MASK", "0") not in ("", "0", "false", "False")
    x_active_mask = None
    if use_mc2_mask:
        x_active_mask = torch.ones((shape["num_tokens"], ), dtype=torch.bool, device=device)

    if variant == 1:
        w1, w2, s1, s2 = build_w8a8_nz_expert_weights(
            shape["num_local_experts"],
            hidden,
            moe_intermediate,
            device,
        )
    else:
        w1, s1, w2, s2 = build_w8a8_nz_stacked_expert_weights(
            shape["num_local_experts"],
            hidden,
            moe_intermediate,
            device,
        )

    return FusedMc2OpContext(
        rank=rank,
        world_size=world_size,
        variant=variant,
        num_tokens=shape["num_tokens"],
        hidden=hidden,
        moe_intermediate=moe_intermediate,
        num_topk=shape["num_topk"],
        num_experts=shape["num_experts"],
        num_local_experts=shape["num_local_experts"],
        global_bs=shape["global_bs"],
        device=device,
        group_ep=get_group_ep(rank),
        x=x,
        expert_idx=expert_idx,
        probs=probs,
        weight1=w1,
        weight2=w2,
        scale1=s1,
        scale2=s2,
        out=out,
        expert_token_nums=expert_token_nums,
        x_active_mask=x_active_mask,
    )


def _worker(rank: int, world_size: int, port: int,
            results: mp.SimpleQueue) -> None:
    try:
        if not hasattr(torch.ops._C_ascend, "dispatch_ffn_combine"):
            raise RuntimeError("dispatch_ffn_combine not registered in vllm_ascend_C")

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


def _run_correctness(ctx: FusedMc2OpContext) -> None:
    ctx.run_fused()
    torch.npu.synchronize()
    dist.barrier()
    verify_fused_output(ctx.out, ctx.x.shape, ctx.rank)


def _run_bench(ctx: FusedMc2OpContext) -> None:
    num_warmups, num_tests = mc2_bench_iters()

    fused_stats = bench(partial(ctx.run_fused), num_warmups, num_tests)
    print_fused_mc2_wallclock_table(
        rank=ctx.rank,
        variant=ctx.variant,
        kernel_label=ctx.kernel_label,
        num_tokens=ctx.num_tokens,
        hidden=ctx.hidden,
        num_topk=ctx.num_topk,
        num_experts=ctx.num_experts,
        num_ranks=ctx.world_size,
        moe_intermediate=ctx.moe_intermediate,
        fused_avg=fused_stats[0],
        num_warmups=num_warmups,
        num_tests=num_tests,
    )

    kernel_name = ctx.kernel_name
    kernel_t = bench_kineto(
        partial(ctx.run_fused),
        kernel_names=kernel_name,
        num_warmups=num_warmups,
        num_tests=num_tests,
        suppress_kineto_output=True,
    )
    print_single_kernel_table(
        rank=ctx.rank,
        label=f"Fused MC2 variant={ctx.variant}",
        kernel_name=kernel_name,
        duration_t=float(kernel_t),
        num_warmups=num_warmups,
        num_tests=num_tests,
    )
    dist.barrier()


def _run_profile(ctx: FusedMc2OpContext) -> None:
    num_warmups, _ = mc2_bench_iters()
    profile_iters = mc2_profile_iters()
    trace_dir = os.environ.get(
        f"{ENV_PREFIX}_TRACE_DIR",
        "./traces/fused_mc2_baseline",
    )
    os.makedirs(trace_dir, exist_ok=True)

    suffix = "ffn_combine" if ctx.variant == 1 else "gmm_combine_decode"
    trace_root = os.path.join(trace_dir, f"fused_mc2_{suffix}")
    worker_name = f"rank{ctx.rank}_fused_mc2_{suffix}"
    profile_msprof(
        partial(ctx.run_fused),
        trace_root=trace_root,
        worker_name=worker_name,
        num_warmups=num_warmups,
        num_tests=profile_iters,
        suppress_output=(ctx.rank != 0),
    )
    dist.barrier()

    summary = msprof_kernel_summary(trace_root, ctx.kernel_name)
    kernel_duration = summary[0] if summary else None
    print_msprof_trace_info(
        rank=ctx.rank,
        label=f"Fused MC2 variant={ctx.variant}",
        trace_root=trace_root,
        num_warmups=num_warmups,
        num_tests=profile_iters,
        kernel_names=(ctx.kernel_name,),
        kernel_durations=(kernel_duration,) if kernel_duration is not None else None,
    )
    if ctx.rank == 0:
        print(
            f"\n  Full msprof traces saved under: {trace_dir}/fused_mc2_{suffix}/\n"
            "  Inspect ASCEND_PROFILER_OUTPUT/trace_view.json in MindStudio Insight.\n",
            flush=True,
        )


def _launch_multiprocess(world_size: int | None = None) -> None:
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
    assert not failures, (
        f"Fused MC2 baseline failures (mode={_test_mode()}, variant={_fused_variant()}): "
        f"{failures}")


@torch.inference_mode()
def test_fused_mc2_baseline_roundtrip() -> None:
    if _test_mode() != "correctness":
        pytest.skip(f"skip when {ENV_PREFIX}_MODE={_test_mode()}")
    _launch_multiprocess()


@torch.inference_mode()
@pytest.mark.skipif(_test_mode() != "bench", reason=f"set {ENV_PREFIX}_MODE=bench")
def test_fused_mc2_baseline_bench() -> None:
    _launch_multiprocess()


@torch.inference_mode()
@pytest.mark.skipif(_test_mode() != "profile", reason=f"set {ENV_PREFIX}_MODE=profile")
def test_fused_mc2_baseline_profile() -> None:
    _launch_multiprocess()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fused MC2 single-op baseline e2e tests")
    parser.add_argument(
        "--mode",
        type=str,
        default=os.environ.get(f"{ENV_PREFIX}_MODE", "correctness"),
        choices=["correctness", "bench", "profile"],
    )
    parser.add_argument(
        "--variant",
        type=int,
        default=_fused_variant(),
        choices=[1, 2],
        help="1=dispatch_ffn_combine, 2=dispatch_gmm_combine_decode",
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
        "--moe-intermediate",
        type=int,
        default=None,
        help="defaults to hidden//2 (min 512)",
    )
    parser.add_argument(
        "--num-warmups",
        type=int,
        default=mc2_int_env(f"{ENV_PREFIX}_NUM_WARMUPS",
                           "VLLM_ASCEND_ZB_TEST_NUM_WARMUPS", "50"),
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
        default=mc2_profile_iters(),
    )
    parser.add_argument(
        "--trace-dir",
        type=str,
        default=os.environ.get(f"{ENV_PREFIX}_TRACE_DIR", "./traces/fused_mc2_baseline"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    os.environ[f"{ENV_PREFIX}_MODE"] = args.mode
    os.environ[f"{ENV_PREFIX}_VARIANT"] = str(args.variant)
    os.environ["VLLM_ASCEND_ENABLE_FUSED_MC2"] = str(args.variant)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE"] = str(args.world_size)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS"] = str(args.num_tokens)
    os.environ["VLLM_ASCEND_MOE_MC2_TEST_HIDDEN"] = str(args.hidden)
    if args.moe_intermediate is not None:
        os.environ["VLLM_ASCEND_MOE_MC2_TEST_INTERMEDIATE"] = str(args.moe_intermediate)
    os.environ[f"{ENV_PREFIX}_NUM_WARMUPS"] = str(args.num_warmups)
    os.environ[f"{ENV_PREFIX}_NUM_TESTS"] = str(args.num_tests)
    os.environ[f"{ENV_PREFIX}_NUM_PROFILE_TESTS"] = str(args.num_profile_tests)
    os.environ[f"{ENV_PREFIX}_TRACE_DIR"] = args.trace_dir
    _launch_multiprocess(args.world_size)
