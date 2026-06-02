# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Torch-free helpers for ZB SHMEM MC2 device visibility.

CANN locks ``ASCEND_RT_VISIBLE_DEVICES`` at process init. For DP>1 cross-DP HyBM,
worker subprocesses expand to the full MC2 list before ``import torch_npu``, then
select the card via DP-adjusted ``local_rank`` (global ep rank).
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
    dp_local_rank = parallel_config.data_parallel_rank_local
    if dp_local_rank is None:
        dp_local_rank = parallel_config.data_parallel_index or 0
    tp_pp_world_size = (
        parallel_config.pipeline_parallel_size
        * parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )

    if visible:
        device_base = visible[0] - dp_local_rank * tp_pp_world_size
    else:
        device_base = 0
    if device_base < 0:
        device_base = 0
    return get_zb_mc2_visible_devices(0, ep_world_size, device_base)


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
    if previous == mc2_visible:
        return True

    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = mc2_visible
    if _env_enabled("VLLM_ASCEND_ZB_SHMEM_DEBUG"):
        logger.warning(
            "[ZB-SHMEM] worker pre-CANN visible expand rank=%d local_rank=%d "
            "ASCEND_RT_VISIBLE_DEVICES: %r -> %r",
            rank,
            local_rank,
            previous,
            mc2_visible,
        )
    return True


def adjust_local_rank_for_zb_mc2(vllm_config: VllmConfig, local_rank: int) -> int:
    if not should_use_zb_mc2_full_visible(vllm_config):
        return local_rank
    parallel_config = vllm_config.parallel_config
    dp_local_rank = parallel_config.data_parallel_rank_local
    if dp_local_rank is None:
        dp_local_rank = parallel_config.data_parallel_index or 0
    tp_pp_world_size = (
        parallel_config.pipeline_parallel_size
        * parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    return local_rank + dp_local_rank * tp_pp_world_size
