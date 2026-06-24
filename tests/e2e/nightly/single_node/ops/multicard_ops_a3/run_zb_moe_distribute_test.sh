#!/usr/bin/env bash
# Run SHMEM zero-buffer MoE dispatch/combine e2e tests (correctness / bench / profile).
#
# Profile mode writes full msprof traces (CPU+NPU, same config as vLLM serving profiler).
# Default output directory:
#   <repo-root>/traces/zb_moe_<timestamp>/
# Per rank:
#   zb/rank<N>_zb.*_ascend_pt/ASCEND_PROFILER_OUTPUT/
#   pta_v2/rank<N>_pta_v2.*_ascend_pt/ASCEND_PROFILER_OUTPUT/
#
# Usage:
#   ./run_zb_moe_distribute_test.sh                  # correctness
#   ./run_zb_moe_distribute_test.sh bench            # wall-clock + kineto summary (no trace files)
#   ./run_zb_moe_distribute_test.sh profile          # full msprof trace + optional kernel summary
#   Compare with PTA baseline: ./run_moe_distribute_v2_baseline_test.sh profile
#
# Environment (shape/bench — shared across ZB / PTA / Fused tests):
#   VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE, _NUM_TOKENS, _HIDDEN, _NUM_TOPK, ...
#   (legacy VLLM_ASCEND_ZB_TEST_* still accepted as fallback)
# ZB-only:
#   VLLM_ASCEND_ZB_SHMEM_URI, VLLM_ASCEND_ZB_TEST_MODE

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"
TEST_PY="${SCRIPT_DIR}/test_zb_moe_distribute_zero_buffer.py"

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

# Shared shape/bench env: MOE_MC2_TEST_* takes precedence over ZB_TEST_*.
export VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE="${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE:-${VLLM_ASCEND_ZB_TEST_WORLD_SIZE:-8}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS:-${VLLM_ASCEND_ZB_TEST_NUM_TOKENS:-32}}"
export VLLM_ASCEND_MOE_MC2_TEST_HIDDEN="${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN:-${VLLM_ASCEND_ZB_TEST_HIDDEN:-2048}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK:-${VLLM_ASCEND_ZB_TEST_NUM_TOPK:-8}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_WARMUPS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_WARMUPS:-${VLLM_ASCEND_ZB_TEST_NUM_WARMUPS:-50}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS:-${VLLM_ASCEND_ZB_TEST_NUM_TESTS:-100}}"
export VLLM_ASCEND_MOE_MC2_TEST_NUM_PROFILE_TESTS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_PROFILE_TESTS:-${VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS:-${VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS}}}"
if [[ -n "${VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS:-${VLLM_ASCEND_ZB_TEST_NUM_EXPERTS:-}}" ]]; then
  export VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS:-${VLLM_ASCEND_ZB_TEST_NUM_EXPERTS}}"
fi

# Keep ZB_TEST_* in sync so older code paths still see resolved values.
export VLLM_ASCEND_ZB_TEST_WORLD_SIZE="${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE}"
export VLLM_ASCEND_ZB_TEST_NUM_TOKENS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS}"
export VLLM_ASCEND_ZB_TEST_HIDDEN="${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN}"
export VLLM_ASCEND_ZB_TEST_NUM_TOPK="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK}"
export VLLM_ASCEND_ZB_TEST_NUM_WARMUPS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_WARMUPS}"
export VLLM_ASCEND_ZB_TEST_NUM_TESTS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS}"
export VLLM_ASCEND_ZB_TEST_NUM_PROFILE_TESTS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_PROFILE_TESTS}"
if [[ -n "${VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS:-}" ]]; then
  export VLLM_ASCEND_ZB_TEST_NUM_EXPERTS="${VLLM_ASCEND_MOE_MC2_TEST_NUM_EXPERTS}"
fi

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
echo "  world_size:  ${VLLM_ASCEND_MOE_MC2_TEST_WORLD_SIZE}"
echo "  num_tokens:  ${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOKENS}"
echo "  hidden:      ${VLLM_ASCEND_MOE_MC2_TEST_HIDDEN}"
echo "  num_topk:    ${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK}"
echo "  shmem_uri:   ${VLLM_ASCEND_ZB_SHMEM_URI}"
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
  --num-topk "${VLLM_ASCEND_MOE_MC2_TEST_NUM_TOPK}" \
  --num-warmups "${VLLM_ASCEND_MOE_MC2_TEST_NUM_WARMUPS}" \
  --num-tests "${VLLM_ASCEND_MOE_MC2_TEST_NUM_TESTS}" \
  --num-profile-tests "${VLLM_ASCEND_MOE_MC2_TEST_NUM_PROFILE_TESTS}" \
  --trace-dir "${TRACE_DIR}"

if [[ "${MODE}" == "profile" ]]; then
  echo ""
  echo "=== Profile traces (msprof) ==="
  echo "  directory: ${TRACE_DIR}"
  find "${TRACE_DIR}" -name '*_ascend_pt' -type d 2>/dev/null | head -20 || true
  echo ""
  echo "  Inspect ASCEND_PROFILER_OUTPUT/trace_view.json in MindStudio Insight"
  echo "  or run: python -c \"from torch_npu.profiler.profiler import analyse; analyse('<path/to/*_ascend_pt>')\""
  echo "  ZB path:  ${TRACE_DIR}/zb/"
  echo "  PTA path: ${TRACE_DIR}/pta_v2/"
elif [[ "${MODE}" == "bench" ]]; then
  echo ""
  echo "=== bench mode ==="
  echo "  Printed wall-clock + kineto kernel tables on rank 0 stdout."
  echo "  No trace JSON files (use: $0 profile)."
fi
