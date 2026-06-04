# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Torch-free helpers for ZB SHMEM MC2 device visibility.

CANN locks ``ASCEND_RT_VISIBLE_DEVICES`` at process init. For DP>1 cross-DP HyBM,
worker subprocesses expand to the full MC2 list before ``import torch_npu``, then
select the card via a **global EP device index** (same rule as vLLM ``gpu_worker``).

General model (any DP, TP, PP, PCP on a single node)::

    ep_world_size = DP * TP * PP * PCP
    global_ep_rank = data_parallel_rank * (TP * PP * PCP) + partition_local_rank
    device_rank = global_ep_rank   # after expand to logical 0 .. ep_world_size-1

Each EngineCore still owns only ``local_world_size`` workers (partition_local_rank
in ``0 .. TP*PP*PCP-1``); ZB aclshmem uses ``rank=global_ep_rank`` with
``world_size=ep_world_size`` across all DP partitions.

For clusters where physical ids are not ``0..N-1``, set
``VLLM_ASCEND_ZB_SHMEM_MC2_VISIBLE_DEVICES`` to the full physical list (length
``ep_world_size``). Multi-node DP (``nnodes_within_dp > 1``) is not enabled yet.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


def _env_enabled(name: str) -> bool:
    raw = os.getenv(name, "")
    return raw not in ("", "0", "false", "False")


def parse_visible_devices() -> list[int]:
    visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if not visible:
        return []
    return [int(part.strip()) for part in visible.split(",") if part.strip()]


def get_zb_mc2_visible_devices(ep_rank: int, ep_world_size: int, physical_device_id: int) -> str:
    override = os.getenv("VLLM_ASCEND_ZB_SHMEM_MC2_VISIBLE_DEVICES", "").strip()
    if override:
        return override
    device_base = physical_device_id - ep_rank
    if device_base < 0:
        device_base = 0
    return ",".join(str(device_base + i) for i in range(ep_world_size))


def compute_ep_world_size(vllm_config: VllmConfig) -> int:
    parallel_config = vllm_config.parallel_config
    return (
        parallel_config.data_parallel_size
        * parallel_config.prefill_context_parallel_size
        * parallel_config.tensor_parallel_size
        * parallel_config.pipeline_parallel_size
    )


def should_use_zb_mc2_full_visible(vllm_config: VllmConfig) -> bool:
    if not _env_enabled("VLLM_ASCEND_ENABLE_ZB_SHMEM"):
        return False
    parallel_config = vllm_config.parallel_config
    if parallel_config.data_parallel_size <= 1:
        return False
    if not parallel_config.enable_expert_parallel:
        return False
    if parallel_config.distributed_executor_backend in ("ray", "external_launcher"):
        return False
    if parallel_config.data_parallel_backend == "ray":
        return False
    if parallel_config.nnodes_within_dp != 1:
        return False
    return True


def resolve_dp_device_offset(parallel_config) -> int:
    """Global DP index used to map per-engine TP local rank -> NPU id."""
    if parallel_config.data_parallel_size <= 1:
        return 0

    dp_rank = int(getattr(parallel_config, "data_parallel_rank", 0) or 0)
    dp_local = parallel_config.data_parallel_rank_local
    dp_local_int = int(dp_local) if dp_local is not None else None
    dp_index = int(getattr(parallel_config, "data_parallel_index", 0) or 0)

    # Prefer an explicitly non-zero global rank. ``data_parallel_rank`` defaults to
    # 0 in ParallelConfig and is often left stale in worker subprocess configs.
    if dp_rank > 0:
        return dp_rank
    if dp_local_int is not None:
        # ``data_parallel_rank_local`` is set by EngineCore for the owning DP
        # partition (0 included). Do not override with partition env hints.
        return dp_local_int
    if dp_index > 0:
        return dp_index

    partition_base = os.getenv("VLLM_ASCEND_ZB_SHMEM_PARTITION_DEVICE_BASE", "").strip()
    if partition_base:
        tp_pp = tp_pp_world_size(parallel_config)
        if tp_pp > 0 and int(partition_base) > 0:
            return int(partition_base) // tp_pp

    return dp_rank


def tp_pp_world_size(parallel_config) -> int:
    return (
        parallel_config.pipeline_parallel_size
        * parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )


def resolve_mc2_visible_devices(vllm_config: VllmConfig) -> str:
    """Return the full MC2 physical device list all EP ranks must share."""
    override = os.getenv("VLLM_ASCEND_ZB_SHMEM_MC2_VISIBLE_DEVICES", "").strip()
    if override:
        return override

    ep_world_size = compute_ep_world_size(vllm_config)
    visible = parse_visible_devices()
    if len(visible) >= ep_world_size:
        return ",".join(str(device_id) for device_id in visible[:ep_world_size])

    parallel_config = vllm_config.parallel_config
    dp_offset = resolve_dp_device_offset(parallel_config)
    tp_pp = tp_pp_world_size(parallel_config)

    if visible:
        device_base = visible[0] - dp_offset * tp_pp
    else:
        device_base = 0
    if device_base < 0:
        device_base = 0
    return get_zb_mc2_visible_devices(0, ep_world_size, device_base)


def is_mc2_full_visible_env(vllm_config: VllmConfig) -> bool:
    """True when the process env already exposes the full MC2 device list."""
    if not should_use_zb_mc2_full_visible(vllm_config):
        return False
    ep_world_size = compute_ep_world_size(vllm_config)
    visible = parse_visible_devices()
    if len(visible) < ep_world_size:
        return False
    expected = resolve_mc2_visible_devices(vllm_config)
    actual = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "").strip()
    if actual == expected:
        return True
    # Identity-mapped MC2 list after expand: 0..ep_world_size-1
    return visible[:ep_world_size] == list(range(ep_world_size))


def should_adjust_mc2_device_rank(vllm_config: VllmConfig) -> bool:
    if not should_use_zb_mc2_full_visible(vllm_config):
        return False
    return len(parse_visible_devices()) >= compute_ep_world_size(vllm_config)


def apply_zb_mc2_worker_visible_env(
    vllm_config: VllmConfig,
    rank: int,
    local_rank: int,
) -> bool:
    """Expand visible devices before CANN init in a worker subprocess."""
    if not should_use_zb_mc2_full_visible(vllm_config):
        return False

    mc2_visible = resolve_mc2_visible_devices(vllm_config)
    previous = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
    visible_before = parse_visible_devices()
    if visible_before:
        os.environ["VLLM_ASCEND_ZB_SHMEM_PARTITION_DEVICE_BASE"] = str(visible_before[0])
    if previous == mc2_visible:
        return True

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = mc2_visible
    logger.warning(
        "[ZB-SHMEM] worker pre-CANN visible expand rank=%d local_rank=%d "
        "ASCEND_RT_VISIBLE_DEVICES: %r -> %r",
        rank,
        local_rank,
        previous,
        mc2_visible,
    )
    return True


def compute_mc2_device_rank(vllm_config: VllmConfig, partition_local_rank: int) -> int:
    """Map per-EngineCore TP-local rank to global logical NPU id (== global EP rank)."""
    if not should_adjust_mc2_device_rank(vllm_config):
        return partition_local_rank
    parallel_config = vllm_config.parallel_config
    dp_offset = resolve_dp_device_offset(parallel_config)
    return partition_local_rank + dp_offset * tp_pp_world_size(parallel_config)


def validate_mc2_device_rank(vllm_config: VllmConfig, device_rank: int) -> None:
    """Raise if device_rank is outside the MC2 / EP world."""
    if not should_use_zb_mc2_full_visible(vllm_config):
        return
    ep_world_size = compute_ep_world_size(vllm_config)
    visible_count = len(parse_visible_devices())
    if device_rank < 0 or device_rank >= ep_world_size:
        raise RuntimeError(
            f"[ZB-SHMEM] device_rank={device_rank} out of range for ep_world_size="
            f"{ep_world_size} (DP={vllm_config.parallel_config.data_parallel_size}, "
            f"TP={vllm_config.parallel_config.tensor_parallel_size})"
        )
    if visible_count < ep_world_size:
        raise RuntimeError(
            f"[ZB-SHMEM] ASCEND_RT_VISIBLE_DEVICES exposes {visible_count} devices but "
            f"ep_world_size={ep_world_size}; expand or set "
            "VLLM_ASCEND_ZB_SHMEM_MC2_VISIBLE_DEVICES"
        )


def adjust_local_rank_for_zb_mc2(vllm_config: VllmConfig, local_rank: int) -> int:
    """Alias for :func:`compute_mc2_device_rank` (partition-local rank in, device id out)."""
    return compute_mc2_device_rank(vllm_config, local_rank)


def describe_mc2_device_bind(vllm_config: VllmConfig, partition_local_rank: int) -> dict[str, object]:
    parallel_config = vllm_config.parallel_config
    ep_world_size = compute_ep_world_size(vllm_config)
    device_rank = compute_mc2_device_rank(vllm_config, partition_local_rank)
    return {
        "partition_local_rank": partition_local_rank,
        "device_rank": device_rank,
        "global_ep_rank": device_rank,
        "ep_world_size": ep_world_size,
        "dp_offset": resolve_dp_device_offset(parallel_config),
        "data_parallel_rank": getattr(parallel_config, "data_parallel_rank", None),
        "data_parallel_rank_local": parallel_config.data_parallel_rank_local,
        "data_parallel_index": parallel_config.data_parallel_index,
        "visible_devices": parse_visible_devices(),
        "ascend_rt_visible_devices": os.getenv("ASCEND_RT_VISIBLE_DEVICES", ""),
    }
