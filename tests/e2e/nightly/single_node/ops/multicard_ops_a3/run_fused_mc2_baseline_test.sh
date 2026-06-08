#!/usr/bin/env bash
# Fused MC2 serving baseline single-op test (dispatch+GMM+combine fused kernel).
#
# Maps to vLLM serving env VLLM_ASCEND_ENABLE_FUSED_MC2:
#   variant 1 (default) → dispatch_ffn_combine      (W8A8, EP<=32)
#   variant 2           → dispatch_gmm_combine_decode (decode W8A8)
#
# Profile traces: fused_mc2_<variant>/rank<N>_*.ascend_pt/ (full msprof)
#
# Usage:
#   ./run_fused_mc2_baseline_test.sh
#   ./run_fused_mc2_baseline_test.sh bench
#   ./run_fused_mc2_baseline_test.sh profile
#   VLLM_ASCEND_FUSED_MC2_TEST_VARIANT=2 ./run_fused_mc2_baseline_test.sh bench
#   ./run_fused_mc2_baseline_test.sh profile /path/to/traces

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
TEST_PY="${SCRIPT_DIR}/test_fused_mc2_baseline.py"

MODE="${1:-correctness}"
TRACE_DIR_ARG="${2:-}"
VARIANT="${VLLM_ASCEND_FUSED_MC2_TEST_VARIANT:-${VLLM_ASCEND_ENABLE_FUSED_MC2:-1}}"

case "${MODE}" in
  correctness|bench|profile) ;;
  -h|--help|help)
    sed -n '1,22p' "$0"
    exit 0
    ;;
  *)
    echo "Unknown mode: ${MODE} (use correctness | bench | profile)" >&2
    exit 1
    ;;
esac

if [[ "${VARIANT}" != "1" && "${VARIANT}" != "2" ]]; then
  echo "Fused MC2 variant must be 1 or 2, got ${VARIANT}" >&2
  exit 1
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

export VLLM_ASCEND_FUSED_MC2_TEST_MODE="${MODE}"
export VLLM_ASCEND_FUSED_MC2_TEST_VARIANT="${VARIANT}"
export VLLM_ASCEND_ENABLE_FUSED_MC2="${VARIANT}"
export VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE="${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE:-${VLLM_ASCEND_ZB_TEST_WORLD_SIZE:-8}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS:-${VLLM_ASCEND_ZB_TEST_NUM_TOKENS:-32}}"
export VLLM_ASCEND_MOE_MC2_TEST_HIDDEN="${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN:-${VLLM_ASCEND_ZB_TEST_HIDDEN:-2048}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK:-${VLLM_ASCEND_ZB_TEST_NUM_TOPK:-8}}"
export VLLM_ASCEND_FUSED_MC2_TEST_NUM_WARMUPS="${VLLM_ASCEND_FUSED_MC2_TEST_NUM_WARMUPS:-${VLLM_ASCEND_ZB_TEST_NUM_WARMUPS:-10}}"
export VLLM_ASCEND_FUSED_MC2_TEST_NUM_TESTS="${VLLM_ASCEND_FUSED_MC2_TEST_NUM_TESTS:-${VLLM_ASCEND_ZB_TEST_NUM_TESTS:-100}}"
export VLLM_ASCEND_FUSED_MC2_TEST_NUM_PROFILE_TESTS="${VLLM_ASCEND_FUSED_MC2_TEST_NUM_PROFILE_TESTS:-${VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS:-30}}"

if [[ -n "${TRACE_DIR_ARG}" ]]; then
  TRACE_DIR="$(cd "${TRACE_DIR_ARG}" 2>/dev/null && pwd || echo "${TRACE_DIR_ARG}")"
  if [[ "${TRACE_DIR}" != /* ]]; then
    TRACE_DIR="${REPO_ROOT}/${TRACE_DIR}"
  fi
else
  TRACE_DIR="${REPO_ROOT}/traces/fused_mc2_baseline_$(date +%Y%m%d_%H%M%S)"
fi
export VLLM_ASCEND_FUSED_MC2_TEST_TRACE_DIR="${TRACE_DIR}"

if [[ "${VARIANT}" == "1" && "${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE}" -gt 32 ]]; then
  echo "WARN: dispatch_ffn_combine requires EP world_size <= 32 (got ${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE})" >&2
fi

echo "=== Fused MC2 baseline e2e test ==="
echo "  repo:        ${REPO_ROOT}"
echo "  mode:        ${MODE}"
echo "  variant:     ${VARIANT} (VLLM_ASCEND_ENABLE_FUSED_MC2)"
if [[ "${VARIANT}" == "1" ]]; then
  echo "  kernel:      dispatch_ffn_combine"
else
  echo "  kernel:      dispatch_gmm_combine_decode"
fi
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
  --variant "${VARIANT}" \
  --world-size "${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE}" \
  --num-tokens "${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS}" \
  --hidden "${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN}" \
  --num-warmups "${VLLM_ASCEND_FUSED_MC2_TEST_NUM_WARMUPS}" \
  --num-tests "${VLLM_ASCEND_FUSED_MC2_TEST_NUM_TESTS}" \
  --num-profile-tests "${VLLM_ASCEND_FUSED_MC2_TEST_NUM_PROFILE_TESTS}" \
  --trace-dir "${TRACE_DIR}"

if [[ "${MODE}" == "profile" ]]; then
  echo ""
  echo "=== Profile traces (Fused MC2, msprof) ==="
  echo "  directory: ${TRACE_DIR}"
  find "${TRACE_DIR}" -name '*_ascend_pt' -type d 2>/dev/null | head -20 || true
  echo ""
  echo "  Compare three serving paths with same shapes:"
  echo "    ./run_fused_mc2_baseline_test.sh profile  # this test"
  echo "    ./run_moe_distribute_v2_baseline_test.sh profile"
  echo "    ./run_shmem_moe_zb_test.sh profile"
  echo "  Inspect ASCEND_PROFILER_OUTPUT/trace_view.json in MindStudio Insight."
elif [[ "${MODE}" == "bench" ]]; then
  echo ""
  echo "=== bench mode (Fused MC2) ==="
  echo "  Single fused op timing on rank 0 stdout."
fi
