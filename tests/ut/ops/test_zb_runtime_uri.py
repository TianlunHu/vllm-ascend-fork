# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import os

import pytest

from vllm_ascend.ops.fused_moe import zb_runtime


class TestResolveZbShmemUri:
    def setup_method(self) -> None:
        zb_runtime.set_zb_shmem_conf_store_uri(None)

    def teardown_method(self) -> None:
        zb_runtime.set_zb_shmem_conf_store_uri(None)

    def test_uses_reserved_conf_store_uri(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_ASCEND_ZB_SHMEM_URI", raising=False)
        monkeypatch.delenv("VLLM_ASCEND_ZB_URI", raising=False)
        zb_runtime.set_zb_shmem_conf_store_uri("tcp://127.0.0.1:45289")

        assert zb_runtime.resolve_zb_shmem_uri() == "tcp://127.0.0.1:45289"

    def test_env_override_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VLLM_ASCEND_ZB_SHMEM_URI", "tcp://10.0.0.1:29999")
        zb_runtime.set_zb_shmem_conf_store_uri("tcp://127.0.0.1:45289")

        assert zb_runtime.resolve_zb_shmem_uri() == "tcp://10.0.0.1:29999"

    def test_missing_reservation_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VLLM_ASCEND_ZB_SHMEM_URI", raising=False)
        monkeypatch.delenv("VLLM_ASCEND_ZB_URI", raising=False)

        with pytest.raises(RuntimeError, match="conf-store URI is unavailable"):
            zb_runtime.resolve_zb_shmem_uri()


class TestParseTcpHostPort:
    def test_ipv4(self) -> None:
        assert zb_runtime._parse_tcp_host_port("tcp://127.0.0.1:29500") == ("127.0.0.1", 29500)

    def test_ipv6(self) -> None:
        assert zb_runtime._parse_tcp_host_port("tcp://[::1]:29500") == ("::1", 29500)


class TestZbWorkerDeviceIndex:
    def test_resolve_keeps_local_rank_without_zb(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeConfig:
            enable_mc2_zb = False

        monkeypatch.setattr(zb_runtime, "get_ascend_config", lambda: FakeConfig())

        class ParallelConfig:
            data_parallel_size = 2
            data_parallel_index = 1
            tensor_parallel_size = 2
            pipeline_parallel_size = 1
            prefill_context_parallel_size = 1

        assert zb_runtime.resolve_zb_worker_device_index(ParallelConfig(), local_rank=0) == 0

    def test_prepare_expands_visible_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")

        class FakeConfig:
            enable_mc2_zb = True

        monkeypatch.setattr(zb_runtime, "get_ascend_config", lambda: FakeConfig())

        class ParallelConfig:
            data_parallel_size = 2
            data_parallel_index = 1
            tensor_parallel_size = 2
            pipeline_parallel_size = 1
            prefill_context_parallel_size = 1
            nnodes_within_dp = 1

        zb_runtime.prepare_zb_visible_devices_before_set_device(ParallelConfig())
        assert os.getenv("ASCEND_RT_VISIBLE_DEVICES") == "0,1,2,3"

    def test_resolve_uses_physical_index_for_dp_gt1(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeConfig:
            enable_mc2_zb = True

        monkeypatch.setattr(zb_runtime, "get_ascend_config", lambda: FakeConfig())

        class ParallelConfig:
            data_parallel_size = 2
            data_parallel_index = 1
            tensor_parallel_size = 2
            pipeline_parallel_size = 1
            prefill_context_parallel_size = 1
            nnodes_within_dp = 1

        assert zb_runtime.resolve_zb_worker_device_index(ParallelConfig(), local_rank=0) == 2
        assert zb_runtime.resolve_zb_worker_device_index(ParallelConfig(), local_rank=1) == 3

