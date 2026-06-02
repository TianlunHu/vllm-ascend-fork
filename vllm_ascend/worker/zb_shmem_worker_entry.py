# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Worker subprocess entry: expand MC2 visible devices before CANN init.

Must not import ``torch`` or ``torch_npu`` in this module.
"""


def ascend_worker_main(*args, **kwargs):
    vllm_config = kwargs.get("vllm_config")
    if vllm_config is not None:
        from vllm_ascend.ops.fused_moe.zb_shmem_device_env import apply_zb_mc2_worker_visible_env

        apply_zb_mc2_worker_visible_env(
            vllm_config,
            kwargs.get("rank", 0),
            kwargs.get("local_rank", 0),
        )

    from vllm.v1.executor.multiproc_executor import WorkerProc

    return WorkerProc.worker_main(*args, **kwargs)
