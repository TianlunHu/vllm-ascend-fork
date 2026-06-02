#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#
from unittest.mock import patch

import pytest

from vllm_ascend.ops.fused_moe import shmem_runtime


@pytest.mark.parametrize(
    ("visible", "logical", "expected_physical", "needs_remap"),
    [
        (None, 2, 2, False),  # DP=1, env unset
        ("0,1,2,3", 1, 1, False),  # DP=1, identity mapping
        ("4,5,6,7", 0, 4, True),  # DP>1 worker, logical 0 -> physical 4
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
        shmem_runtime.get_zb_mc2_visible_devices(ep_rank, ep_world_size, physical)
        == expected_mc2_visible
    )
