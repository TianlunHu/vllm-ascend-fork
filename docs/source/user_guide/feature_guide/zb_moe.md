# ZB MoE for Expert Parallel

**ZB** stands for **zero buffer**: SHMEM-backed MoE dispatch/combine that avoids
extra staging buffers on the hot path. Throughout this project, paths, operators,
and config keys use the `zb_` prefix only (for example `zb_moe_distribute_dispatch`).

## Overview

Zero buffer MoE is an optional expert-parallel (EP) communication path on Ascend A3/A5.
When enabled, MoE **dispatch** and **combine** use custom operators backed by Ascend SHMEM
instead of the default PTA `npu_moe_distribute_dispatch_v2` / `combine_v2` path. Expert MLP
**gmm2** can write directly into the SHMEM `combine_x` buffer, so combine can run with
`ori_x=None` and skip the inner copy on the hot path.

The feature is **opt-in**. When ZB is disabled, vLLM Ascend keeps the existing PTA MC2 MoE
behavior.

## Requirements

### Build time

Install [Ascend SHMEM](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/developmentguide/opdevg/opdevg_00001.html)
at the default path `/usr/local/Ascend/shmem/latest`, then build vLLM Ascend from source:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -e . --no-build-isolation
```

If SHMEM is detected, ZB custom ops are compiled automatically. No extra build environment
variable is required.

Verify registration after install:

```bash
python3 -c "
import torch
for op in (
    'zb_moe_distribute_dispatch',
    'zb_moe_distribute_combine',
    'zb_moe_grouped_matmul_gmm2_out',
):
    print(op, hasattr(torch.ops._C_ascend, op))
"
```

All three checks should print `True`.

### Runtime

Enable ZB via `--additional-config`:

| Key | Required | Description |
| --- | -------- | ----------- |
| `enable_mc2_zb` | Yes (to enable) | Set to `true` to use `TokenDispatcherWithZB` for SHMEM dispatch/combine. Default: `false`. |

aclshmem conf-store URI is taken from the same HCCL / torch.distributed rendezvous as worker
startup (`MASTER_ADDR` + `MASTER_PORT`, i.e. the `distributed_init_method` passed to
`init_distributed_environment`). No separate URI is required in `additional_config`.

Legacy environment variable `VLLM_ASCEND_ENABLE_ZB` still works as a fallback when
`enable_mc2_zb` is unset.

Optional tuning (defaults are usually sufficient):

| Environment variable | Description |
| -------------------- | ----------- |
| `VLLM_ASCEND_ZB_LOCAL_MEM_SIZE` | Override local SHMEM pool size (bytes). |
| `VLLM_ASCEND_ZB_EXT_INFO_BYTES` | Size of the per-process `ext_info` SHMEM allocation. |
| `VLLM_ASCEND_ZB_POOL_SLACK_BYTES` | Extra slack added when estimating local memory. |
| `VLLM_ASCEND_ZB_SHMEM_URI` | Override conf-store URI (standalone e2e tests only; serving reuses HCCL rendezvous). |

See also [Environment Variables](../configuration/env_vars.md) for the full list pulled from
`vllm_ascend/envs.py`.

## How to Use

```bash
vllm serve <moe-model> \
  --enable-expert-parallel \
  --additional-config '{"enable_mc2_zb": true}' \
  ...
```

## Limitations (current)

- **Hardware:** A3/A5 only (requires MC2 extra-args path).
- **DP:** DP=1 only for the ZB serving path.
- **Incompatible with** `enable_mc2_hierarchy_comm` in Ascend config.
- **MXFP** gmm2 direct-to-`combine_x` is not supported yet.

## Testing

Multicard A3 e2e scripts live under
`tests/e2e/nightly/single_node/ops/multicard_ops_a3/`:

```bash
cd tests/e2e/nightly/single_node/ops/multicard_ops_a3
export VLLM_ASCEND_ZB_SHMEM_URI=tcp://127.0.0.1:29556

./run_zb_moe_distribute_test.sh              # dispatch/combine correctness
./run_zb_gmm2_to_combine_x_test.sh         # gmm2 -> combine_x full path
./run_zb_moe_distribute_test.sh bench      # optional benchmark vs PTA v2
```

Standalone e2e tests set `MASTER_ADDR` / `MASTER_PORT` or `VLLM_ASCEND_ZB_SHMEM_URI` explicitly
because they do not go through the full vLLM worker bootstrap.
