# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""ZB SHMEM: hook worker spawn to expand MC2 visible devices before CANN init.

Must be imported from ``patch/platform/__init__.py`` so EngineCore/worker
subprocesses (spawn) re-apply the patch via ``adapt_patch(is_global_patch=True)``.
"""

from __future__ import annotations

from multiprocessing.synchronize import Lock as LockType

from vllm.config import VllmConfig
from vllm.utils import numa_utils
from vllm.utils.system_utils import get_mp_context
from vllm.v1.executor.multiproc_executor import UnreadyWorkerProcHandle, WorkerProc

_ZB_SHMEM_PATCHED = False


def worker_process_target(vllm_config: VllmConfig):
    from vllm_ascend.ops.fused_moe.zb_shmem_device_env import should_use_zb_mc2_full_visible

    if should_use_zb_mc2_full_visible(vllm_config):
        from vllm_ascend.worker.zb_shmem_worker_entry import ascend_worker_main

        return ascend_worker_main
    return WorkerProc.worker_main


def _make_worker_process_zb_aware(*, daemon: bool):
    def make_worker_process(
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        input_shm_handle,
        shared_worker_lock: LockType,
        is_driver_worker: bool = False,
        inherited_fds: list[int] | None = None,
    ) -> UnreadyWorkerProcHandle:
        context = get_mp_context()
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        if inherited_fds is not None:
            inherited_fds = inherited_fds.copy()
            inherited_fds.extend((ready_reader.fileno(), death_writer.fileno()))
        process_kwargs = {
            "vllm_config": vllm_config,
            "local_rank": local_rank,
            "rank": rank,
            "distributed_init_method": distributed_init_method,
            "input_shm_handle": input_shm_handle,
            "ready_pipe": ready_writer,
            "death_pipe": death_reader,
            "shared_worker_lock": shared_worker_lock,
            "is_driver_worker": is_driver_worker,
            "inherited_fds": inherited_fds if inherited_fds is not None else [],
        }
        proc = context.Process(
            target=worker_process_target(vllm_config),
            kwargs=process_kwargs,
            name=f"VllmWorker-{rank}",
            daemon=daemon,
        )
        with numa_utils.configure_subprocess(vllm_config, local_rank, process_kind="worker"):
            proc.start()
        ready_writer.close()
        death_reader.close()
        return UnreadyWorkerProcHandle(proc, rank, ready_reader, death_writer)

    return staticmethod(make_worker_process)


def apply_zb_shmem_worker_patch() -> None:
    global _ZB_SHMEM_PATCHED
    if _ZB_SHMEM_PATCHED:
        return

    WorkerProc.make_worker_process = _make_worker_process_zb_aware(daemon=True)

    try:
        from vllm_ascend.patch.platform.patch_multiproc_executor import AscendWorkerProc

        AscendWorkerProc.make_worker_process = _make_worker_process_zb_aware(daemon=False)
    except ImportError:
        pass

    _ZB_SHMEM_PATCHED = True


def _maybe_apply_on_import() -> None:
    raw = __import__("os").getenv("VLLM_ASCEND_ENABLE_ZB_SHMEM", "")
    if raw not in ("", "0", "false", "False"):
        apply_zb_shmem_worker_patch()


_maybe_apply_on_import()
