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
DEVICE_PROFILE="$DEVICE_DIR/profile_$CASE_ID"
DEVICE_PROC="$DEVICE_DIR/profile_proc_$CASE_ID"
LOCAL_DIR="quantized_v6/layer_debug/profile/$CASE_ID"
LOCAL_PROC_DIR="quantized_v6/layer_debug/profile/${CASE_ID}_proc"
RUN_LOG="quantized_v6/layer_debug/profile/${CASE_ID}_run.log"
DATA_PROC_LOG="quantized_v6/layer_debug/profile/${CASE_ID}_data_proc.log"

mkdir -p "$(dirname "$RUN_LOG")"
# shellcheck disable=SC2086
"$HDC" $target_args shell "mkdir -p '$DEVICE_PROFILE' '$DEVICE_PROC'"
hdc_shell_log "$RUN_LOG" "cd '$DEVICE_DIR'; chmod +x '$RUNNER_NAME'; export LD_LIBRARY_PATH=.; rm -rf 'profile_$CASE_ID'; mkdir -p '$DEVICE_PROFILE'; ./'$RUNNER_NAME' --model='$DEVICE_DIR/$MODEL_NAME' --input='$DEVICE_INPUT' --output_dir='$DEVICE_PROFILE' --enable_item=1 --times=1"
hdc_shell_log "$DATA_PROC_LOG" "cd '$DEVICE_DIR'; chmod +x data_proc_tool_internal 2>/dev/null || true; export LD_LIBRARY_PATH=.; rm -rf 'profile_proc_$CASE_ID'; mkdir -p '$DEVICE_PROC'; ./data_proc_tool_internal --result_path='$DEVICE_PROC'"

rm -rf "$LOCAL_DIR" "$LOCAL_PROC_DIR"
# shellcheck disable=SC2086
"$HDC" $target_args file recv "$DEVICE_PROFILE" "$LOCAL_DIR"
# shellcheck disable=SC2086
"$HDC" $target_args file recv "$DEVICE_PROC" "$LOCAL_PROC_DIR"
