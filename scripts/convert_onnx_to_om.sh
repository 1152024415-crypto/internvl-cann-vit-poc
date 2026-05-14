#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:-${PROJECT_ROOT}/artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx}"
OUTPUT_PREFIX="${2:-${PROJECT_ROOT}/artifacts/om/internvl3_5_vit_fp32_opset18_staticpos}"
LOG_PATH="${OMG_LOG_PATH:-${OUTPUT_PREFIX}.omg.log}"
REQUIRE_NPU_PARTITION="${REQUIRE_NPU_PARTITION:-1}"

find_omg() {
  if [[ -n "${OMG_BIN:-}" && -x "${OMG_BIN}" ]]; then
    printf '%s\n' "${OMG_BIN}"
    return 0
  fi

  if command -v omg >/dev/null 2>&1; then
    command -v omg
    return 0
  fi

  local candidates=(
    "${CANN_KIT_HOME:-}/tools/tools_omg/omg"
    "${CANN_HOME:-}/tools/tools_omg/omg"
    "${ASCEND_TOOLKIT_HOME:-}/tools/tools_omg/omg"
    "${HOME}/cann-kit/tools/tools_omg/omg"
    "${HOME}/CANNKit/tools/tools_omg/omg"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  local search_roots=(
    "${CANN_KIT_HOME:-}"
    "${CANN_HOME:-}"
    "${ASCEND_TOOLKIT_HOME:-}"
    "${HOME}/cann-kit"
    "${HOME}/CANNKit"
  )

  for root in "${search_roots[@]}"; do
    if [[ -n "${root}" && -d "${root}" ]]; then
      local found
      found="$(find "${root}" -path '*/tools_omg/omg' -type f -perm /111 2>/dev/null | head -n 1 || true)"
      if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return 0
      fi
    fi
  done

  return 1
}

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ONNX model not found: ${MODEL_PATH}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_PREFIX}")"

if ! OMG="$(find_omg)"; then
  cat >&2 <<'EOF'
OMG tool not found.

Set OMG_BIN to the full omg path, or set CANN_KIT_HOME to the CANN Kit root.
Expected tool path shape:
  $CANN_KIT_HOME/tools/tools_omg/omg

Example:
  export CANN_KIT_HOME=$HOME/cann-kit
  ./scripts/convert_onnx_to_om.sh
EOF
  exit 3
fi

echo "Using OMG: ${OMG}"
echo "Input ONNX: ${MODEL_PATH}"
echo "Output prefix: ${OUTPUT_PREFIX}"
echo "OMG log: ${LOG_PATH}"

mkdir -p "$(dirname "${LOG_PATH}")"

"${OMG}" \
  --model "${MODEL_PATH}" \
  --framework 5 \
  --input_shape "pixel_values:1,3,448,448" \
  --output "${OUTPUT_PREFIX}" 2>&1 | tee "${LOG_PATH}"

if [[ "${REQUIRE_NPU_PARTITION}" == "1" ]]; then
  LAST_PARTITION_LINE="$(grep "partition type NPU:" "${LOG_PATH}" | tail -n 1 || true)"

  if [[ -z "${LAST_PARTITION_LINE}" ]]; then
    cat >&2 <<EOF
Could not prove the OM has an NPU partition.

Expected the OMG log to contain a partition summary with NPU > 0.
Set REQUIRE_NPU_PARTITION=0 only for explicit CPU fallback experiments.

Log: ${LOG_PATH}
EOF
    exit 6
  fi

  if [[ "${LAST_PARTITION_LINE}" == *"partition type NPU:0"* ]]; then
    cat >&2 <<EOF
CPU-only OM is not acceptable for this project.

OMG generated an OM, but its partition summary says NPU:0. Install the matching
Kirin platform plugin under tools/platform, then rerun conversion.

Last partition summary:
${LAST_PARTITION_LINE}

Log: ${LOG_PATH}
EOF
    exit 5
  fi

  if ! grep -Eq "partition type NPU:[1-9][0-9]*" <<<"${LAST_PARTITION_LINE}"; then
    cat >&2 <<EOF
Could not parse a positive NPU partition count.

Expected the final OMG partition summary to contain NPU > 0.
Set REQUIRE_NPU_PARTITION=0 only for explicit CPU fallback experiments.

Last partition summary:
${LAST_PARTITION_LINE}

Log: ${LOG_PATH}
EOF
    exit 6
  fi
fi

echo "Expected OM output: ${OUTPUT_PREFIX}.om"
