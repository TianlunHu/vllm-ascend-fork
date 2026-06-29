# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import pytest

from vllm_ascend.ops.fused_moe import zb_runtime


class TestResolveZbShmemUri:
    def setup_method(self) -> None:
        zb_runtime.set_zb_distributed_init_method(None)

    def teardown_method(self) -> None:
        zb_runtime.set_zb_distributed_init_method(None)

    def test_derives_dedicated_port_from_hccl_rendezvous(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_ASCEND_ZB_SHMEM_URI", raising=False)
        monkeypatch.delenv("VLLM_ASCEND_ZB_URI", raising=False)
        monkeypatch.setenv("VLLM_ASCEND_ZB_SHMEM_PORT_OFFSET", "10000")
        zb_runtime.set_zb_distributed_init_method("tcp://127.0.0.1:29500")

        assert zb_runtime.resolve_zb_shmem_uri() == "tcp://127.0.0.1:39500"

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_ASCEND_ZB_SHMEM_URI", "tcp://10.0.0.1:29999")
        zb_runtime.set_zb_distributed_init_method("tcp://127.0.0.1:29500")

        assert zb_runtime.resolve_zb_shmem_uri() == "tcp://10.0.0.1:29999"

    def test_master_addr_fallback_uses_offset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_ASCEND_ZB_SHMEM_URI", raising=False)
        monkeypatch.delenv("VLLM_ASCEND_ZB_URI", raising=False)
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29600")

        assert zb_runtime.resolve_zb_shmem_uri() == "tcp://127.0.0.1:39600"
