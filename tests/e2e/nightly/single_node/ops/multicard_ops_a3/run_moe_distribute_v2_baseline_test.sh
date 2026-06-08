#!/usr/bin/env bash
# PTA MC2 baseline: npu_moe_distribute_dispatch_v2 / combine_v2 e2e (no SHMEM).
#
# Profile traces (mode=profile): full msprof under pta_v2/rank<N>_pta_v2.*_ascend_pt/
#
# For fair comparison with ZB test, use the same shape parameters on both scripts
# (MOE_MC2_TEST_* or ZB_TEST_NUM_TOKENS / HIDDEN are both read).
#
# Usage:
#   ./run_moe_distribute_v2_baseline_test.sh
#   ./run_moe_distribute_v2_baseline_test.sh bench
#   ./run_moe_distribute_v2_baseline_test.sh profile
#   ./run_moe_distribute_v2_baseline_test.sh profile /path/to/traces

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
TEST_PY="${SCRIPT_DIR}/test_moe_distribute_v2_baseline.py"

MODE="${1:-correctness}"
TRACE_DIR_ARG="${2:-}"

case "${MODE}" in
  correctness|bench|profile) ;;
  -h|--help|help)
    sed -n '1,18p' "$0"
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

export VLLM_ASCEND_PTA_MC2_TEST_MODE="${MODE}"
export VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE="${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE:-${VLLM_ASCEND_ZB_TEST_WORLD_SIZE:-8}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS:-${VLLM_ASCEND_ZB_TEST_NUM_TOKENS:-32}}"
export VLLM_ASCEND_MOE_MC2_TEST_HIDDEN="${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN:-${VLLM_ASCEND_ZB_TEST_HIDDEN:-2048}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK:-${VLLM_ASCEND_ZB_TEST_NUM_TOPK:-8}}"
export VLLM_ASCEND_PTA_MC2_TEST_NUM_WARMUPS="${VLLM_ASCEND_PTA_MC2_TEST_NUM_WARMUPS:-${VLLM_ASCEND_ZB_TEST_NUM_WARMUPS:-10}}"
export VLLM_ASCEND_PTA_MC2_TEST_NUM_TESTS="${VLLM_ASCEND_PTA_MC2_TEST_NUM_TESTS:-${VLLM_ASCEND_ZB_TEST_NUM_TESTS:-100}}"
export VLLM_ASCEND_PTA_MC2_TEST_NUM_PROFILE_TESTS="${VLLM_ASCEND_PTA_MC2_TEST_NUM_PROFILE_TESTS:-${VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS:-30}}"

if [[ -n "${TRACE_DIR_ARG}" ]]; then
  TRACE_DIR="$(cd "${TRACE_DIR_ARG}" 2>/dev/null && pwd || echo "${TRACE_DIR_ARG}")"
  if [[ "${TRACE_DIR}" != /* ]]; then
    TRACE_DIR="${REPO_ROOT}/${TRACE_DIR}"
  fi
else
  TRACE_DIR="${REPO_ROOT}/traces/pta_mc2_baseline_$(date +%Y%m%d_%H%M%S)"
fi
export VLLM_ASCEND_PTA_MC2_TEST_TRACE_DIR="${TRACE_DIR}"

echo "=== PTA MC2 V2 baseline e2e test ==="
echo "  repo:        ${REPO_ROOT}"
echo "  mode:        ${MODE}"
echo "  world_size:  ${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE}"
echo "  num_tokens:  ${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS}"
echo "  hidden:      ${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN}"
if [[ "${MODE}" == "profile" ]]; then
  echo "  trace_dir:   ${TRACE_DIR}"
  mkdir -p "${TRACE_DIR}"
fi
echo ""

cd "${REPO_ROOT}"
python "${TEST_PY}" \
  --mode "${MODE}" \
  --world-size "${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE}" \
  --num-tokens "${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS}" \
  --hidden "${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN}" \
  --num-warmups "${VLLM_ASCEND_PTA_MC2_TEST_NUM_WARMUPS}" \
  --num-tests "${VLLM_ASCEND_PTA_MC2_TEST_NUM_TESTS}" \
  --num-profile-tests "${VLLM_ASCEND_PTA_MC2_TEST_NUM_PROFILE_TESTS}" \
  --trace-dir "${TRACE_DIR}"

if [[ "${MODE}" == "profile" ]]; then
  echo ""
  echo "=== Profile traces (PTA baseline, msprof) ==="
  echo "  directory: ${TRACE_DIR}/pta_v2/"
  find "${TRACE_DIR}" -name '*_ascend_pt' -type d 2>/dev/null | head -20 || true
  echo ""
  echo "  Compare with ZB: run run_shmem_moe_zb_test.sh profile with same shapes."
  echo "  Inspect ASCEND_PROFILER_OUTPUT/trace_view.json in MindStudio Insight."
elif [[ "${MODE}" == "bench" ]]; then
  echo ""
  echo "=== bench mode (PTA baseline only) ==="
  echo "  Wall-clock + kineto tables printed on rank 0 stdout."
fi
