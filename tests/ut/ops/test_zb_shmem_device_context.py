#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm_ascend.ops.fused_moe import shmem_runtime
from vllm_ascend.ops.fused_moe import zb_shmem_device_env as device_env


def _make_parallel_config(**overrides):
    defaults = dict(
        data_parallel_size=2,
        data_parallel_rank_local=1,
        data_parallel_index=1,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        local_world_size=2,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_vllm_config(**parallel_overrides):
    return SimpleNamespace(parallel_config=_make_parallel_config(**parallel_overrides))


@pytest.mark.parametrize(
    ("visible", "logical", "expected_physical", "needs_remap"),
    [
        (None, 2, 2, False),
        ("0,1,2,3", 1, 1, False),
        ("4,5,6,7", 0, 4, True),
        ("4,5,6,7", 3, 7, True),
    ],
)
def test_get_zb_physical_device_id_mapping(
    monkeypatch: pytest.MonkeyPatch,
    visible: str | None,
    logical: int,
    expected_physical: int,
    needs_remap: bool,
) -> None:
    if visible is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)

    with patch.object(shmem_runtime.torch.npu, "current_device", return_value=logical):
        physical = shmem_runtime.get_zb_physical_device_id()
        ctx = shmem_runtime.describe_zb_device_context()

    assert physical == expected_physical
    assert ctx["physical_device_id"] == expected_physical
    assert ctx["logical_device_id"] == logical
    assert ctx["needs_device_remap"] is needs_remap
    assert ctx["dp1_passthrough"] is not needs_remap


@pytest.mark.parametrize(
    ("ep_rank", "ep_world_size", "physical", "expected_mc2_visible"),
    [
        (0, 4, 0, "0,1,2,3"),
        (2, 4, 2, "0,1,2,3"),
        (3, 4, 3, "0,1,2,3"),
    ],
)
def test_get_zb_mc2_visible_devices(
    ep_rank: int,
    ep_world_size: int,
    physical: int,
    expected_mc2_visible: str,
) -> None:
    assert (
        device_env.get_zb_mc2_visible_devices(ep_rank, ep_world_size, physical)
        == expected_mc2_visible
    )


def test_apply_zb_mc2_worker_visible_env_expands_before_cann(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ZB_SHMEM", "1")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")

    vllm_config = _make_vllm_config()
    applied = device_env.apply_zb_mc2_worker_visible_env(vllm_config, rank=0, local_rank=0)

    assert applied is True
    assert os.getenv("ASCEND_RT_VISIBLE_DEVICES") == "0,1,2,3"


def test_resolve_mc2_visible_devices_from_dp_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")
    vllm_config = _make_vllm_config(data_parallel_rank_local=1)
    assert device_env.resolve_mc2_visible_devices(vllm_config) == "0,1,2,3"


def test_adjust_local_rank_for_zb_mc2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ZB_SHMEM", "1")
    vllm_config = _make_vllm_config(data_parallel_rank_local=1)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")
    assert device_env.adjust_local_rank_for_zb_mc2(vllm_config, local_rank=0) == 0

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1,2,3")
    assert device_env.adjust_local_rank_for_zb_mc2(vllm_config, local_rank=0) == 2
    assert device_env.adjust_local_rank_for_zb_mc2(vllm_config, local_rank=1) == 3


def test_should_not_expand_for_dp1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_ZB_SHMEM", "1")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0,1")
    vllm_config = _make_vllm_config(data_parallel_size=1, data_parallel_rank_local=0)

    assert device_env.should_use_zb_mc2_full_visible(vllm_config) is False
    assert device_env.apply_zb_mc2_worker_visible_env(vllm_config, rank=0, local_rank=0) is False
