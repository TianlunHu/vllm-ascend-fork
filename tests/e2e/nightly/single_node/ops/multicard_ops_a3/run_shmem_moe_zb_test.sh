#!/usr/bin/env bash
# Run SHMEM zero-buffer MoE dispatch/combine e2e tests (correctness / bench / profile).
#
# Profile mode writes Chrome/Kineto JSON traces. Default output directory:
#   <repo-root>/traces/zb_moe_<timestamp>/
# Files per rank (world_size=R):
#   rank0_zb_shmem.json, rank0_pta_v2.json, ... rank{R-1}_*.json
#
# Usage:
#   ./run_shmem_moe_zb_test.sh                  # correctness
#   ./run_shmem_moe_zb_test.sh bench            # wall-clock + kineto summary (no trace files)
#   ./run_shmem_moe_zb_test.sh profile          # export traces + kineto summary
#   Compare with PTA baseline: ./run_moe_distribute_v2_baseline_test.sh profile
#
# Environment (optional overrides):
#   VLLM_ASCEND_ZB_SHMEM_URI       SHMEM control endpoint (default tcp://127.0.0.1:29555)
#   VLLM_ASCEND_ZB_TEST_WORLD_SIZE  EP world size / NPU count (default 8)
#   VLLM_ASCEND_ZB_TEST_NUM_TOKENS  tokens per rank (default 32)
#   VLLM_ASCEND_ZB_TEST_HIDDEN      hidden size (default 2048)
#   VLLM_ASCEND_ZB_TEST_NUM_WARMUPS warmup iters (default 10)
#   VLLM_ASCEND_ZB_TEST_NUM_TESTS   bench iters (default 100)
#   VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS profile iters (default 30)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
TEST_PY="${SCRIPT_DIR}/test_shmem_moe_distribute_zero_buffer.py"

MODE="${1:-correctness}"
TRACE_DIR_ARG="${2:-}"

case "${MODE}" in
  correctness|bench|profile) ;;
  -h|--help|help)
    sed -n '1,24p' "$0"
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE} (use correctness | bench | profile)" >&2
    exit 1
    ;;
esac

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export VLLM_ASCEND_ZB_TEST_MODE="${MODE}"
export VLLM_ASCEND_ZB_SHMEM_URI="${VLLM_ASCEND_ZB_SHMEM_URI:-tcp://127.0.0.1:29555}"
export VLLM_ASCEND_ZB_TEST_WORLD_SIZE="${VLLM_ASCEND_ZB_TEST_WORLD_SIZE:-8}"
export VLLM_ASCEND_ZB_TEST_NUM_TOKENS="${VLLM_ASCEND_ZB_TEST_NUM_TOKENS:-32}"
export VLLM_ASCEND_ZB_TEST_HIDDEN="${VLLM_ASCEND_ZB_TEST_HIDDEN:-2048}"
export VLLM_ASCEND_ZB_TEST_NUM_TOPK="${VLLM_ASCEND_ZB_TEST_NUM_TOPK:-8}"
export VLLM_ASCEND_ZB_TEST_NUM_WARMUPS="${VLLM_ASCEND_ZB_TEST_NUM_WARMUPS:-10}"
export VLLM_ASCEND_ZB_TEST_NUM_TESTS="${VLLM_ASCEND_ZB_TEST_NUM_TESTS:-100}"
export VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS="${VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS:-30}"

if [[ -n "${TRACE_DIR_ARG}" ]]; then
  TRACE_DIR="$(cd "${TRACE_DIR_ARG}" 2>/dev/null && pwd || echo "${TRACE_DIR_ARG}")"
  if [[ "${TRACE_DIR}" != /* ]]; then
    TRACE_DIR="${REPO_ROOT}/${TRACE_DIR}"
  fi
else
  TRACE_DIR="${REPO_ROOT}/traces/zb_moe_$(date +%Y%m%d_%H%M%S)"
fi
export VLLM_ASCEND_ZB_TEST_TRACE_DIR="${TRACE_DIR}"

echo "=== SHMEM MoE ZB e2e test ==="
echo "  repo:        ${REPO_ROOT}"
echo "  mode:        ${MODE}"
echo "  world_size:  ${VLLM_ASCEND_ZB_TEST_WORLD_SIZE}"
echo "  num_tokens:  ${VLLM_ASCEND_ZB_TEST_NUM_TOKENS}"
echo "  hidden:      ${VLLM_ASCEND_ZB_TEST_HIDDEN}"
echo "  shmem_uri:   ${VLLM_ASCEND_ZB_SHMEM_URI}"
if [[ "${MODE}" == "profile" ]]; then
  echo "  trace_dir:   ${TRACE_DIR}"
  mkdir -p "${TRACE_DIR}"
fi
echo ""

cd "${REPO_ROOT}"
python "${TEST_PY}" \
  --mode "${MODE}" \
  --world-size "${VLLM_ASCEND_ZB_TEST_WORLD_SIZE}" \
  --num-tokens "${VLLM_ASCEND_ZB_TEST_NUM_TOKENS}" \
  --hidden "${VLLM_ASCEND_ZB_TEST_HIDDEN}" \
  --num-warmups "${VLLM_ASCEND_ZB_TEST_NUM_WARMUPS}" \
  --num-tests "${VLLM_ASCEND_ZB_TEST_NUM_TESTS}" \
  --num-profile-tests "${VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS}" \
  --trace-dir "${TRACE_DIR}"

if [[ "${MODE}" == "profile" ]]; then
  echo ""
  echo "=== Profile traces ==="
  echo "  directory: ${TRACE_DIR}"
  ls -lh "${TRACE_DIR}"/*.json 2>/dev/null || echo "  (no .json files found — check test logs above)"
  echo ""
  echo "  Open with chrome://tracing or https://ui.perfetto.dev/"
  echo "  ZB path:  rank<N>_zb_shmem.json"
  echo "  PTA path: rank<N>_pta_v2.json"
elif [[ "${MODE}" == "bench" ]]; then
  echo ""
  echo "=== bench mode ==="
  echo "  Printed wall-clock + kineto kernel tables on rank 0 stdout."
  echo "  No trace JSON files (use: $0 profile)."
fi
