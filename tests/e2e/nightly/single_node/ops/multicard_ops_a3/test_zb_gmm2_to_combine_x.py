#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Verify whether ``npu_grouped_matmul`` can write gmm2 output directly into the
# SHMEM ``combine_x`` buffer so ZB combine can use ``ori_x=None`` (no
# ``CopyValidExpandXToShmem``).
#
# Run on A3 (built with VLLM_ASCEND_ENABLE_ZB_OPS=1):
#
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   export VLLM_ASCEND_ZB_SHMEM_URI=tcp://127.0.0.1:29556
#   ./run_zb_gmm2_to_combine_x_test.sh
#
# Or directly (default world_size=2 for TP=2-like setups):
#
#   VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE=2 \\
#   VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS=32 \\
#   VLLM_ASCEND_MOE_MC2_TEST_HIDDEN=2048 \\
#   python tests/e2e/nightly/single_node/ops/multicard_ops_a3/test_zb_gmm2_to_combine_x.py
#
# Note: CANN grouped matmul requires hidden in [1024, 8192] on A3; do not use 128.
# Pass criteria (rank 0 prints summary):
#   1. ``zb_moe_grouped_matmul_gmm2_out`` writes into ``combine_x`` slice.
#   2. ``combine(ori_x=gmm_out)`` matches ``combine(ori_x=None)`` after (1).

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu

from vllm_ascend.ops.fused_moe.shmem_runtime import (
    shmem_moe_distribute_combine_zero_buffer,
    shmem_moe_distribute_dispatch_zero_buffer,
    zb_moe_grouped_matmul_gmm2_out,
)
from vllm_ascend.utils import enable_custom_op

from moe_mc2_e2e_common import mc2_hccl_port, mc2_shape_config, mc2_world_size
from test_shmem_moe_distribute_zero_buffer import (
    ZbMoeOpContext,
    _build_context as _build_zb_context_base,
)

enable_custom_op()

_GMM_HIDDEN_MIN = 1024
_GMM_HIDDEN_MAX = 8192


def _validate_shape_config(world_size: int) -> dict:
    cfg = mc2_shape_config(world_size)
    hidden = cfg["hidden"]
    if hidden < _GMM_HIDDEN_MIN or hidden > _GMM_HIDDEN_MAX:
        raise ValueError(
            f"hidden={hidden} is outside CANN npu_grouped_matmul supported range "
            f"[{_GMM_HIDDEN_MIN}, {_GMM_HIDDEN_MAX}]; set "
            "VLLM_ASCEND_MOE_MC2_TEST_HIDDEN=2048 (or another value in range).")
    if cfg["num_topk"] > cfg["num_experts"]:
        raise ValueError(
            f"num_topk={cfg['num_topk']} exceeds num_experts={cfg['num_experts']}; "
            "set VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS >= num_topk "
            "(default for world_size=2 is 16).")
    return cfg


def _gmm_rows(ctx: ZbMoeOpContext) -> int:
    counts = ctx.aux["expert_token_nums"]
    rows = int(counts.sum().item())
    if rows <= 0:
        raise RuntimeError(
            f"rank {ctx.rank}: expert_token_nums sum is 0 after dispatch; "
            "cannot infer gmm2 row count.")
    cap = ctx.bundle.combine_x.size(0)
    if rows > cap:
        raise RuntimeError(
            f"rank {ctx.rank}: gmm rows={rows} exceed combine_x capacity={cap}")
    return rows


def _run_minimal_gmm2(
    ctx: ZbMoeOpContext,
    rows: int,
    *,
    out_buf: torch.Tensor | None = None,
) -> tuple[torch.Tensor, str | None]:
    """Identity-ish grouped matmul stand-in for gmm2 (single group, BF16).

    Uses dispatch ``expand_x_out`` as input so shapes match the real serving path.
    Returns ``(output_tensor, output_kw_used)``.
    """
    hidden = ctx.hidden
    device = ctx.device
    x = ctx.bundle.expand_x_out[:rows].detach()

    w = torch.eye(hidden, dtype=torch.bfloat16, device=device).unsqueeze(0)
    group_list = torch.tensor([rows], dtype=torch.int64, device=device)

    if out_buf is not None:
        zb_moe_grouped_matmul_gmm2_out(
            x,
            [w],
            group_list,
            out_buf,
            split_item=2,
            group_type=0,
            group_list_type=0,
        )
        return out_buf, "zb_moe_grouped_matmul_gmm2_out"

    out = torch.empty((rows, hidden), dtype=torch.bfloat16, device=device)
    zb_moe_grouped_matmul_gmm2_out(
        x,
        [w],
        group_list,
        out,
        split_item=2,
        group_type=0,
        group_list_type=0,
    )
    return out, None


def _run_zb_dispatch(ctx: ZbMoeOpContext) -> None:
    shmem_moe_distribute_dispatch_zero_buffer(
        x=ctx.x,
        expert_ids=ctx.topk_idx,
        expand_x_out=ctx.bundle.expand_x_out,
        dynamic_scales_out=ctx.aux["dynamic_scales"],
        assist_info_for_combine_out=ctx.aux["assist_info_for_combine"],
        expert_token_nums_out=ctx.aux["expert_token_nums"],
        ep_recv_count_out=ctx.aux["ep_recv_count"],
        tp_recv_count_out=ctx.aux["tp_recv_count"],
        ep_world_size=ctx.world_size,
        ep_rank_id=ctx.rank,
        moe_expert_num=ctx.num_experts,
        ext_info=ctx.runtime.ext_info,
        global_bs=ctx.global_bs,
    )


def _run_zb_combine(
    ctx: ZbMoeOpContext,
    combined_x: torch.Tensor,
    *,
    ori_x: torch.Tensor | None,
) -> None:
    shmem_moe_distribute_combine_zero_buffer(
        expand_x=ctx.bundle.combine_x,
        expert_ids=ctx.topk_idx,
        assist_info_for_combine=ctx.aux["assist_info_for_combine"],
        ep_send_count=ctx.aux["ep_recv_count"],
        expert_scales=ctx.topk_weights,
        combined_x=combined_x,
        tp_send_count=ctx.aux["tp_recv_count"],
        ori_x=ori_x,
        ep_world_size=ctx.world_size,
        ep_rank_id=ctx.rank,
        moe_expert_num=ctx.num_experts,
        ext_info=ctx.runtime.ext_info,
        global_bs=ctx.global_bs,
    )


@dataclass
class Gmm2CombineXResult:
    rank: int
    gmm_y_kw: str | None
    gmm_into_shmem_ok: bool
    combine_match: bool
    max_abs_diff: float
    message: str


def _verify_once(ctx: ZbMoeOpContext) -> Gmm2CombineXResult:
    _run_zb_dispatch(ctx)
    torch.npu.synchronize()
    rows = _gmm_rows(ctx)

    # --- Step 1: auto-alloc gmm2 (baseline tensor) ---
    try:
        gmm_auto, _ = _run_minimal_gmm2(ctx, rows, out_buf=None)
    except Exception as exc:
        return Gmm2CombineXResult(
            rank=ctx.rank,
            gmm_y_kw=None,
            gmm_into_shmem_ok=False,
            combine_match=False,
            max_abs_diff=-1.0,
            message=f"gmm2 auto-alloc failed: {exc!r}",
        )

    # --- Step 2: try writing gmm2 directly into SHMEM combine_x ---
    shmem_out = ctx.bundle.combine_x[:rows]
    shmem_out.zero_()
    try:
        _, y_kw = _run_minimal_gmm2(ctx, rows, out_buf=shmem_out)
    except Exception as exc:
        return Gmm2CombineXResult(
            rank=ctx.rank,
            gmm_y_kw=None,
            gmm_into_shmem_ok=False,
            combine_match=False,
            max_abs_diff=-1.0,
            message=f"gmm2 into combine_x failed: {exc!r}",
        )
    torch.npu.synchronize()

    gmm_shmem_ok = torch.allclose(
        gmm_auto.float(),
        shmem_out.float(),
        rtol=1e-2,
        atol=1e-2,
    )
    if not gmm_shmem_ok:
        max_diff = (gmm_auto.float() - shmem_out.float()).abs().max().item()
        return Gmm2CombineXResult(
            rank=ctx.rank,
            gmm_y_kw=y_kw,
            gmm_into_shmem_ok=False,
            combine_match=False,
            max_abs_diff=float(max_diff),
            message="gmm auto vs combine_x slice mismatch",
        )

    # --- Step 3: combine with ori_x=gmm_auto (kernel copy path) ---
    _run_zb_dispatch(ctx)
    torch.npu.synchronize()
    rows = _gmm_rows(ctx)
    gmm_auto2, _ = _run_minimal_gmm2(ctx, rows, out_buf=None)
    combined_ref = torch.empty_like(ctx.combined_x)
    _run_zb_combine(ctx, combined_ref, ori_x=gmm_auto2)
    torch.npu.synchronize()

    # --- Step 4: combine with ori_x=None (gmm already in combine_x) ---
    _run_zb_dispatch(ctx)
    torch.npu.synchronize()
    rows = _gmm_rows(ctx)
    shmem_out2 = ctx.bundle.combine_x[:rows]
    shmem_out2.zero_()
    _run_minimal_gmm2(ctx, rows, out_buf=shmem_out2)
    torch.npu.synchronize()
    combined_nocopy = torch.empty_like(ctx.combined_x)
    _run_zb_combine(ctx, combined_nocopy, ori_x=None)
    torch.npu.synchronize()

    combine_ok = torch.allclose(
        combined_ref.float(),
        combined_nocopy.float(),
        rtol=1e-2,
        atol=1e-2,
    )
    max_diff = (combined_ref.float() - combined_nocopy.float()).abs().max().item()
    msg = "PASS" if combine_ok else "combine(ori_x=*) != combine(ori_x=None)"
    return Gmm2CombineXResult(
        rank=ctx.rank,
        gmm_y_kw=y_kw,
        gmm_into_shmem_ok=True,
        combine_match=combine_ok,
        max_abs_diff=float(max_diff),
        message=msg,
    )


def _worker(rank: int, world_size: int, port: int, results: mp.SimpleQueue) -> None:
    try:
        torch_npu.npu.set_device(rank)
        random.seed(rank + 7)
        np.random.seed(rank + 7)
        torch.manual_seed(rank + 7)

        dist.init_process_group(
            backend="hccl",
            rank=rank,
            world_size=world_size,
            init_method=f"tcp://127.0.0.1:{port}",
        )

        ctx = _build_zb_context_base(rank, world_size)
        dist.barrier()

        if rank == 0:
            cfg = _validate_shape_config(world_size)
            print(
                f"[rank0] shape: tokens={cfg['num_tokens']} hidden={cfg['hidden']} "
                f"topk={cfg['num_topk']} experts={cfg['num_experts']} "
                f"world_size={world_size}",
                flush=True,
            )
            print("[rank0] using torch.ops._C_ascend.zb_moe_grouped_matmul_gmm2_out", flush=True)
        else:
            _validate_shape_config(world_size)

        result = _verify_once(ctx)
        dist.barrier()

        ctx.runtime.finalize()
        dist.destroy_process_group()
        results.put(result)
    except Exception as exc:
        results.put(
            Gmm2CombineXResult(
                rank=rank,
                gmm_y_kw=None,
                gmm_into_shmem_ok=False,
                combine_match=False,
                max_abs_diff=-1.0,
                message=f"ERROR: {exc!r}",
            ))


def _launch(world_size: int) -> list[Gmm2CombineXResult]:
    if not hasattr(torch.ops._C_ascend, "shmem_moe_distribute_dispatch_zero_buffer"):
        raise RuntimeError(
            "ZB ops not registered; rebuild vllm_ascend_C with VLLM_ASCEND_ENABLE_ZB_OPS=1")
    if not hasattr(torch.ops._C_ascend, "zb_moe_grouped_matmul_gmm2_out"):
        raise RuntimeError(
            "zb_moe_grouped_matmul_gmm2_out not registered; rebuild vllm_ascend_C with "
            "VLLM_ASCEND_ENABLE_ZB_OPS=1")

    port = mc2_hccl_port() + random.randint(0, 10000)
    mp.set_start_method("fork", force=True)
    results_q: mp.SimpleQueue = mp.SimpleQueue()
    procs = []
    for rank in range(world_size):
        p = mp.Process(target=_worker, args=(rank, world_size, port, results_q))
        p.start()
        procs.append(p)

    out = [results_q.get() for _ in range(world_size)]
    for p in procs:
        p.join()
    return out


def _print_summary(results: list[Gmm2CombineXResult]) -> None:
    print("\n=== ZB gmm2 -> combine_x verification ===", flush=True)
    for r in sorted(results, key=lambda x: x.rank):
        print(
            f"  rank {r.rank}: gmm_y_kw={r.gmm_y_kw!r} "
            f"gmm_into_shmem={r.gmm_into_shmem_ok} "
            f"combine_match={r.combine_match} "
            f"max_abs_diff={r.max_abs_diff:.3e} "
            f"msg={r.message}",
            flush=True,
        )

    failures = [r for r in results if r.message.startswith("ERROR") or not r.combine_match]
    if failures:
        print("\nRESULT: FAIL — see rank lines above.", flush=True)
        raise SystemExit(1)

    print("\nRESULT: PASS — gmm2 can target SHMEM combine_x; combine(ori_x=None) matches.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify gmm2 output can be written into SHMEM combine_x (Plan A1)")
    parser.add_argument("--world-size", type=int, default=mc2_world_size())
    args = parser.parse_args()
    _validate_shape_config(args.world_size)
    results = _launch(args.world_size)
    _print_summary(results)


if __name__ == "__main__":
    main()
