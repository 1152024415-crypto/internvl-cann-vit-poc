#!/usr/bin/env sh
set -eu

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

"$HDC" list targets
# shellcheck disable=SC2086
"$HDC" $target_args shell "if test -d '$DEVICE_DIR'; then echo OK_DIR; else echo MISSING_DIR; fi"
# shellcheck disable=SC2086
"$HDC" $target_args shell "if test -f '$DEVICE_DIR/$RUNNER_NAME'; then echo OK_RUNNER; else echo MISSING_RUNNER; fi"
# shellcheck disable=SC2086
"$HDC" $target_args shell "if test -f '$DEVICE_DIR/$MODEL_NAME'; then echo OK_MODEL; else echo MISSING_MODEL; fi"
