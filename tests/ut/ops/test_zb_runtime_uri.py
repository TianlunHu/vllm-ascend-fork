# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

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


class TestZbPhysicalDeviceMapping:
    def test_resolve_without_visible_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)

        class FakeNpu:
            @staticmethod
            def current_device() -> int:
                return 1

        monkeypatch.setitem(__import__("sys").modules, "torch_npu", type("torch_npu", (), {"npu": FakeNpu})())

        assert zb_runtime._resolve_zb_physical_device_id() == 1

    def test_resolve_with_visible_devices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,3")

        class FakeNpu:
            @staticmethod
            def current_device() -> int:
                return 0

        monkeypatch.setitem(__import__("sys").modules, "torch_npu", type("torch_npu", (), {"npu": FakeNpu})())

        assert zb_runtime._resolve_zb_physical_device_id() == 2

    def test_align_skips_single_rank(self) -> None:
        zb_runtime._align_zb_shmem_device_visibility(1)

