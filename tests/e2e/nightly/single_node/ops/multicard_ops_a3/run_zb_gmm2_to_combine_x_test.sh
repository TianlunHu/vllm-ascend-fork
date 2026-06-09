#!/usr/bin/env bash
# Verify gmm2 can write into SHMEM combine_x (eliminate combine kernel copy).
#
# Usage:
#   ./run_zb_gmm2_to_combine_x_test.sh
#   VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE=2 ./run_zb_gmm2_to_combine_x_test.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
TEST_PY="${SCRIPT_DIR}/test_zb_gmm2_to_combine_x.py"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export VLLM_ASCEND_ZB_SHMEM_URI="${VLLM_ASCEND_ZB_SHMEM_URI:-tcp://127.0.0.1:29556}"

# CANN grouped matmul on A3 requires hidden in [1024, 8192]; match other ZB e2e tests.
export VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE="${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE:-2}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS:-32}"
export VLLM_ASCEND_MOE_MC2_TEST_HIDDEN="${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN:-2048}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK:-8}"
# Must be >= num_topk and divisible by world_size; 16 matches mc2_shape_config default.
export VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS:-16}"

echo "=== ZB gmm2 -> combine_x test ==="
echo "  repo:       ${REPO_ROOT}"
echo "  world_size: ${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE}"
echo "  num_tokens: ${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS}"
echo "  hidden:     ${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN}"
echo "  num_topk:   ${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK}"
echo "  num_experts:${VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS}"
echo "  shmem_uri:  ${VLLM_ASCEND_ZB_SHMEM_URI}"
echo ""

cd "${REPO_ROOT}"
python "${TEST_PY}" --world-size "${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE}"
