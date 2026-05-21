#!/usr/bin/env sh
set -eu

CASE_ID="${1:-}"
if [ -z "$CASE_ID" ]; then
  echo "Usage: $0 CASE_ID" >&2
  exit 2
fi
case "$CASE_ID" in
  *[!A-Za-z0-9_.-]*)
    echo "Unsafe CASE_ID: $CASE_ID" >&2
    exit 2
    ;;
esac

HDC="${HDC:-hdc}"
DEVICE_DIR="${DEVICE_DIR:-/data/local/tmp/internvl_om_compare}"
MODEL_NAME="${MODEL_NAME:-split_internvit_v6_exclude_conv_int8_8.omc}"
RUNNER_NAME="${RUNNER_NAME:-model_run_tool_internal}"

case "$DEVICE_DIR" in
  ""|*[!A-Za-z0-9_./-]*)
    echo "Unsafe DEVICE_DIR: $DEVICE_DIR" >&2
    exit 2
    ;;
esac
case "$MODEL_NAME" in
  ""|*[!A-Za-z0-9_.-]*)
    echo "Unsafe MODEL_NAME: $MODEL_NAME" >&2
    exit 2
    ;;
esac
case "$RUNNER_NAME" in
  ""|*[!A-Za-z0-9_.-]*)
    echo "Unsafe RUNNER_NAME: $RUNNER_NAME" >&2
    exit 2
    ;;
esac
case "${HDC_TARGET:-}" in
  *[!A-Za-z0-9_.:-]*)
    echo "Unsafe HDC_TARGET: $HDC_TARGET" >&2
    exit 2
    ;;
esac

target_args=""
if [ "${HDC_TARGET:-}" ]; then
  target_args="-t ${HDC_TARGET}"
fi

hdc_shell_log() {
  log_path=$1
  shell_command=$2
  status_file="${log_path}.status.$$"
  rm -f "$status_file"
  set +e
  # shellcheck disable=SC2086
  { "$HDC" $target_args shell "$shell_command"; printf '%s\n' "$?" > "$status_file"; } 2>&1 | tee "$log_path"
  tee_status=$?
  set -e
  cmd_status=1
  if [ -f "$status_file" ]; then
    cmd_status=$(cat "$status_file")
  fi
  rm -f "$status_file"
  if [ "$cmd_status" -ne 0 ]; then
    return "$cmd_status"
  fi
  if [ "$tee_status" -ne 0 ]; then
    return "$tee_status"
  fi
  return 0
}

DEVICE_INPUT="$DEVICE_DIR/inputs/${CASE_ID}_pixel_values_fp32.bin"
DEVICE_DUMP="$DEVICE_DIR/dump_$CASE_ID"
DEVICE_PROC="$DEVICE_DIR/dump_proc_$CASE_ID"
LOCAL_DIR="quantized_v6/layer_debug/omc/$CASE_ID"
LOCAL_PROC_DIR="quantized_v6/layer_debug/omc/${CASE_ID}_proc"
INVENTORY_JSON="quantized_v6/layer_debug/omc/${CASE_ID}_inventory.json"
INVENTORY_CSV="quantized_v6/layer_debug/omc/${CASE_ID}_inventory.csv"
RUN_LOG="quantized_v6/layer_debug/omc/${CASE_ID}_run.log"
DATA_PROC_LOG="quantized_v6/layer_debug/omc/${CASE_ID}_data_proc.log"

# shellcheck disable=SC2086
if ! "$HDC" $target_args shell "if test -f '$DEVICE_INPUT'; then echo OK_INPUT; else echo MISSING_INPUT; fi" | grep -q OK_INPUT; then
  echo "Missing device input: $DEVICE_INPUT" >&2
  exit 1
fi

# shellcheck disable=SC2086
mkdir -p "$(dirname "$RUN_LOG")"
# shellcheck disable=SC2086
"$HDC" $target_args shell "mkdir -p '$DEVICE_DUMP'"
hdc_shell_log "$RUN_LOG" "cd '$DEVICE_DIR'; chmod +x '$RUNNER_NAME'; export LD_LIBRARY_PATH=.; rm -rf 'dump_$CASE_ID'; mkdir -p '$DEVICE_DUMP'; ./'$RUNNER_NAME' --model='$DEVICE_DIR/$MODEL_NAME' --input='$DEVICE_INPUT' --output_dir='$DEVICE_DUMP' --enable_item=2 --times=1"
# shellcheck disable=SC2086
"$HDC" $target_args shell "mkdir -p '$DEVICE_PROC'"
hdc_shell_log "$DATA_PROC_LOG" "cd '$DEVICE_DIR'; chmod +x data_proc_tool_internal 2>/dev/null || true; export LD_LIBRARY_PATH=.; rm -rf 'dump_proc_$CASE_ID'; mkdir -p '$DEVICE_PROC'; ./data_proc_tool_internal --result_path='$DEVICE_PROC'"

# shellcheck disable=SC2086
if ! "$HDC" $target_args shell "if test -d '$DEVICE_DUMP'; then echo OK_DUMP; else echo MISSING_DUMP; fi" | grep -q OK_DUMP; then
  echo "Missing device dump: $DEVICE_DUMP" >&2
  exit 1
fi

rm -rf "$LOCAL_DIR" "$LOCAL_PROC_DIR"
# shellcheck disable=SC2086
"$HDC" $target_args file recv "$DEVICE_DUMP" "$LOCAL_DIR"
# shellcheck disable=SC2086
"$HDC" $target_args file recv "$DEVICE_PROC" "$LOCAL_PROC_DIR" || true

uv run --no-sync python scripts/compare_omc_layers.py inventory --dump-dir "$LOCAL_DIR" --output-json "$INVENTORY_JSON" --output-csv "$INVENTORY_CSV"
