# Yellow-Zone ONNX to OM DDK Runbook

This document is for the yellow-zone agent that has access to the correct
Huawei CANN Kit / DDK package for the target device. The blue-zone OM is not
trusted for device validation. Regenerate the OM in yellow zone with the
yellow-zone DDK, then validate it before packaging it into the HarmonyOS demo.

## Goal

Convert the InternVL3.5 ViT + projector ONNX model into a CANN `.om` offline
model with the yellow-zone DDK, with enough logs to diagnose DDK, platform,
shape, and NPU placement problems. The current target phone is Kirin 9030, but
the conversion must follow the yellow-zone DDK's official requirements rather
than assumptions from the blue-zone environment.

The intended model contract is fixed:

```text
input_name  = pixel_values
input_shape = [1, 3, 448, 448]
input_dtype = fp32

output      = visual_tokens
output_shape = [1, 256, 1024]
```

Do not generate or accept dynamic-shape, multi-patch, or 5D ONNX/OM artifacts
for this first proof.

## Official References

Use these Huawei documents as the source of truth:

- CANN Kit model conversion example:
  https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-model-conversion-example
- CANN Kit model conversion preparation:
  https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparing-for-model-conversion
- CANN Kit OMG parameters:
  https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-overall-parameter
- CANN Kit Tools download / platform plugin download:
  https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-preparations#tools%E4%B8%8B%E8%BD%BD

Important points from the official docs:

- OMG converts Caffe, TensorFlow, ONNX, and MindSpore models into OM offline
  models.
- ONNX conversion uses `--framework 5`.
- Current CANN Kit ONNX support is opset 7 to 18.
- OMG is under `tools/tools_omg` and runs on 64-bit Linux.
- `--mode 3` is precheck-only and can write a check report.
- `--mode 1` can convert an OM to JSON for structure inspection.
- `--input_shape` must be specified when the ONNX input dimension is dynamic.
- `--dynamic_dims` is only for explicit pre-declared dynamic-shape models.
  Do not use it for this first proof.
- `--platform` is a target-platform conversion parameter. Use it when the
  selected OMG mode or yellow-zone DDK workflow requires target-specific model
  generation.

## About `--platform` And Platform Plugins

This project does not write custom operators.

That matters because the decision to install a Kirin 9030 platform package
should not be justified by custom-op development; there is no custom-op
development in this project. Platform targeting and custom operators are
separate topics:

```text
custom operator / AscendC plugin  = needed only when adding unsupported ops
platform targeting / platform data = used by OMG/runtime for target-specific OM handling
```

Use this rule:

```text
If the official yellow-zone DDK command works without --platform, keep that run
and its logs as the baseline attempt.

If using --platform kirin9030, the matching platform package must be installed
and visible to the DDK.

If the yellow-zone DDK or official docs require --platform for the chosen target
mode, follow that requirement.
```

If uncertain, produce two logged attempts:

```text
attempt A: minimal official ONNX conversion without --platform
attempt B: target-specific conversion with --platform kirin9030
```

Do not accept either attempt unless the resulting OM passes the static checks
and then passes HarmonyOS device runtime validation.

## Inputs

Use this ONNX model:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx
```

Expected properties:

```text
opset = 18
input = pixel_values [1, 3, 448, 448]
output = [1, 256, 1024]
no Resize op
no dynamic batch
no 5D input
```

If the ONNX file is missing, pull the latest repo and download the ONNX artifact
from the project release. Do not use the blue-zone generated `.om` as the input.

## Hard Constraints

These are non-negotiable for this pass:

```text
Do not use --dynamic_dims.
Do not use input_shape with -1.
Do not use 5D input such as [1, N, 3, 448, 448].
Do not convert multi-patch or dynamic high-resolution variants.
Do not accept CPU-only OM output.
Do not strip or discard OMG logs.
```

Why: this proof isolates one static image patch. Dynamic high-resolution and
multi-patch handling belong to a later phase. If a 5D or dynamic-shape ONNX
appears, stop and fix the ONNX export, not the OMG command.

## Directory Layout

Use one timestamped run directory so the logs are not overwritten:

```bash
cd /path/to/internvl-cann-vit-poc

export RUN_ID="$(date +%Y%m%d-%H%M%S)"
export RUN_DIR="$PWD/artifacts/yellow-ddk-runs/$RUN_ID"
export ONNX="$PWD/artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx"
export OUT_PREFIX="$RUN_DIR/internvl3_5_vit_projector_fp32_opset18_staticpos_yellow_ddk"
export LOG_DIR="$RUN_DIR/logs"

mkdir -p "$LOG_DIR"
```

The run should produce:

```text
logs/env.log
logs/onnx_static_check.log
logs/omg_help.log
logs/omg_precheck.log
logs/omg_precheck_report.txt
logs/omg_convert.log
logs/om_to_json.log
logs/om_summary.log
internvl3_5_vit_projector_fp32_opset18_staticpos_yellow_ddk.om
internvl3_5_vit_projector_fp32_opset18_staticpos_yellow_ddk.json
MANIFEST.sha256
```

## Locate Yellow-Zone OMG

Use the yellow-zone DDK. Do not point `OMG_BIN` at a blue-zone package.

```bash
export CANN_KIT_HOME=/absolute/path/to/yellow-zone/cann-kit
export OMG_BIN="$CANN_KIT_HOME/tools/tools_omg/omg"
# Optional. Set only for a target-specific conversion attempt.
# Example: export OMG_PLATFORM=kirin9030
export OMG_PLATFORM="${OMG_PLATFORM:-}"

test -x "$OMG_BIN"
"$OMG_BIN" --help > "$LOG_DIR/omg_help.log" 2>&1
```

If `OMG_PLATFORM` is empty, the commands below run the minimal official ONNX
conversion path without `--platform`. If `OMG_PLATFORM=kirin9030`, the commands
include `--platform kirin9030`; in that case, the matching platform package must
be installed and visible to the DDK.

## Capture Environment Logs

Run this before conversion:

```bash
{
  echo "=== date ==="
  date -Is
  echo "=== uname ==="
  uname -a
  echo "=== pwd ==="
  pwd
  echo "=== omg ==="
  echo "OMG_BIN=$OMG_BIN"
  ls -l "$OMG_BIN"
  echo "=== platform plugins ==="
  find "$CANN_KIT_HOME/tools/platform" -maxdepth 3 -type f -o -type d 2>/dev/null | sort | head -n 200
  echo "=== env ==="
  env | grep -E 'CANN|ASCEND|OMG|LD_LIBRARY_PATH|PATH' | sort
  echo "=== onnx ==="
  ls -lh "$ONNX"
  sha256sum "$ONNX"
} 2>&1 | tee "$LOG_DIR/env.log"
```

The log must show the yellow-zone DDK path and the available platform packages.
If `--platform kirin9030` is used, the log must also show the installed
`kirin9030` platform package. If `--platform` is not used, do not fail only
because the platform package is absent; judge the result by OMG logs and device
runtime validation.

## Static ONNX Gate

Run this check before OMG:

```bash
uv run python - <<'PY' 2>&1 | tee "$LOG_DIR/onnx_static_check.log"
from pathlib import Path
import os
import onnx

onnx_path = Path(os.environ["ONNX"])
model = onnx.load(onnx_path)
onnx.checker.check_model(model)

print(f"onnx={onnx_path}")
print("checker=ok")
print("opsets=" + ",".join(str(op.version) for op in model.opset_import))

inputs = []
for value in model.graph.input:
    dims = []
    tensor_type = value.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    inputs.append((value.name, dims, tensor_type.elem_type))
    print(f"input={value.name} dims={dims} elem_type={tensor_type.elem_type}")

outputs = []
for value in model.graph.output:
    dims = []
    tensor_type = value.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    outputs.append((value.name, dims, tensor_type.elem_type))
    print(f"output={value.name} dims={dims} elem_type={tensor_type.elem_type}")

op_types = {}
for node in model.graph.node:
    op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
print("op_counts=" + ",".join(f"{k}:{op_types[k]}" for k in sorted(op_types)))

expected_input = ("pixel_values", [1, 3, 448, 448])
if not inputs or inputs[0][0] != expected_input[0] or inputs[0][1] != expected_input[1]:
    raise SystemExit(f"bad input contract: expected {expected_input}, got {inputs}")

for name, dims, _ in inputs:
    if len(dims) != 4:
        raise SystemExit(f"5D/non-4D input is forbidden: {name} {dims}")
    if any(not isinstance(dim, int) for dim in dims):
        raise SystemExit(f"dynamic input dim is forbidden: {name} {dims}")

if "Resize" in op_types:
    raise SystemExit("Resize op is forbidden in the static-position CANN ONNX")

print("static_contract=ok")
PY
```

Expected:

```text
checker=ok
input=pixel_values dims=[1, 3, 448, 448]
static_contract=ok
```

If this fails, stop. Do not try to fix it with `--dynamic_dims`.

## OMG Precheck

Run precheck and keep both stdout and the report. Some DDK builds may return a
non-zero process code even when the generated precheck report says all items
passed, so inspect the report content rather than relying only on `$?`.

```bash
OMG_PLATFORM_ARG=()
if [ -n "${OMG_PLATFORM:-}" ]; then
  OMG_PLATFORM_ARG=(--platform "$OMG_PLATFORM")
fi

set +e
"$OMG_BIN" \
  --mode 3 \
  --model "$ONNX" \
  --framework 5 \
  --input_shape "pixel_values:1,3,448,448" \
  "${OMG_PLATFORM_ARG[@]}" \
  --check_report "$LOG_DIR/omg_precheck_report.txt" \
  2>&1 | tee "$LOG_DIR/omg_precheck.log"
export OMG_PRECHECK_EXIT="${PIPESTATUS[0]}"
set -e

echo "OMG_PRECHECK_EXIT=$OMG_PRECHECK_EXIT" | tee -a "$LOG_DIR/omg_precheck.log"
test -s "$LOG_DIR/omg_precheck_report.txt"
```

Review:

```bash
grep -Ei "fail|error|unsupported|not support|warning|pass|success" \
  "$LOG_DIR/omg_precheck.log" "$LOG_DIR/omg_precheck_report.txt" \
  | tee "$LOG_DIR/omg_precheck_summary.log"
```

Gate:

```text
No unsupported operator.
No shape rank error.
No dynamic shape requirement.
No 5D input.
Precheck report must not contain failed checks.
```

If the report is unclear, keep the whole run directory and ask for review.

## OMG Conversion

Run the conversion with static shape:

```bash
set -o pipefail
OMG_PLATFORM_ARG=()
if [ -n "${OMG_PLATFORM:-}" ]; then
  OMG_PLATFORM_ARG=(--platform "$OMG_PLATFORM")
fi

"$OMG_BIN" \
  --model "$ONNX" \
  --framework 5 \
  --input_shape "pixel_values:1,3,448,448" \
  "${OMG_PLATFORM_ARG[@]}" \
  --output "$OUT_PREFIX" \
  2>&1 | tee "$LOG_DIR/omg_convert.log"
```

The expected success line is:

```text
OMG generate offline model success
```

The output should be:

```text
$OUT_PREFIX.om
```

Do not pass:

```text
--dynamic_dims
--input_shape "pixel_values:-1,3,448,448"
--input_shape "pixel_values:1,-1,3,448,448"
--input_shape "pixel_values:1,N,3,448,448"
```

## Required Log Summary

Produce a compact summary:

```bash
{
  echo "=== success markers ==="
  grep -Ei "OMG generate offline model success|success|failed|error|unsupported|not support" "$LOG_DIR/omg_convert.log"

  echo "=== NPU / CPU markers ==="
  grep -Ei "AI_NPUCL|CPUCL|partition type|NPU|GPU|ISP|GetPlatformVersion|platform" "$LOG_DIR/omg_convert.log"

  echo "=== dynamic shape markers ==="
  grep -Ei "dynamic|dynamic_dims|input_shape|-1|shape|rank|dim" "$LOG_DIR/omg_convert.log"
} 2>&1 | tee "$LOG_DIR/om_summary.log"
```

Gate:

```text
Accept:
- OMG success marker exists.
- OM file exists and is non-empty.
- Log shows the yellow-zone DDK path and the exact platform handling used.
- If `--platform kirin9030` is used, log shows the matching platform package.
- Prefer positive NPU evidence: AI_NPUCL lines or partition type NPU > 0.
- No CPU-only evidence.

Reject:
- partition type NPU:0
- CPUCL fallback as the only backend
- unsupported operator
- dynamic shape/dynamic dims request
- 5D input or rank mismatch
- missing platform plugin when `--platform kirin9030` was requested
```

If the log has `GetPlatformVersion` errors or `partition type NPU:0`, do not
ship that OM. Fix the DDK/platform handling and rerun conversion.

## OM to JSON Inspection

Convert the OM to JSON for structural inspection:

```bash
"$OMG_BIN" \
  --mode 1 \
  --om "$OUT_PREFIX.om" \
  --json "$OUT_PREFIX.json" \
  2>&1 | tee "$LOG_DIR/om_to_json.log"
```

Check the JSON exists and record shape evidence:

```bash
test -s "$OUT_PREFIX.json"

grep -Ei "pixel_values|1, 3, 448, 448|1,3,448,448|256|1024|Node_Output|visual" \
  "$OUT_PREFIX.json" \
  | head -n 200 \
  | tee "$LOG_DIR/om_json_shape_snippet.log"
```

The JSON check is static evidence only. It does not prove that the phone NPU can
run the model.

## Manifest

Create a manifest:

```bash
{
  sha256sum "$ONNX"
  sha256sum "$OUT_PREFIX.om"
  test -f "$OUT_PREFIX.json" && sha256sum "$OUT_PREFIX.json"
  find "$LOG_DIR" -type f -maxdepth 1 -print0 | sort -z | xargs -0 sha256sum
} | tee "$RUN_DIR/MANIFEST.sha256"
```

Also record file sizes:

```bash
ls -lh "$ONNX" "$OUT_PREFIX.om" "$OUT_PREFIX.json" "$LOG_DIR"/* | tee "$RUN_DIR/files.txt"
```

## What To Upload Back

Upload these files to GitHub Release or otherwise make them available to blue
zone:

```text
$OUT_PREFIX.om
$OUT_PREFIX.json
$RUN_DIR/MANIFEST.sha256
$RUN_DIR/files.txt
$LOG_DIR/env.log
$LOG_DIR/onnx_static_check.log
$LOG_DIR/omg_help.log
$LOG_DIR/omg_precheck.log
$LOG_DIR/omg_precheck_report.txt
$LOG_DIR/omg_precheck_summary.log
$LOG_DIR/omg_convert.log
$LOG_DIR/om_summary.log
$LOG_DIR/om_to_json.log
$LOG_DIR/om_json_shape_snippet.log
```

The logs are part of the deliverable. A naked `.om` is not enough.

## Final Device Validation

OMG runs on Linux and produces an offline model. It does not prove the phone NPU
can authenticate and run the model. Final proof still requires the HarmonyOS
demo:

```text
Official Smoke succeeds
Official Classify succeeds
InternVL Load Model succeeds
InternVL Run Once succeeds for dog/cat
InternVL output shape = [1, 256, 1024]
cosine >= 0.999
mean_abs_diff <= 0.01
20-run stability succeeds
```

If the official sample succeeds but InternVL fails at `OH_NNCompilation_Build`,
the issue is likely specific to the InternVL OM or conversion path. If the
official sample also fails, investigate yellow-zone SDK, signing, DDK, device
runtime, or CANN Kit installation first.

## Common Mistakes

Do not do this:

```bash
# Wrong: dynamic shape.
--input_shape "pixel_values:-1,3,448,448" --dynamic_dims "1;2;5"

# Wrong: 5D multi-patch input.
--input_shape "pixel_values:1,12,3,448,448"

# Wrong: requesting a target platform without the matching platform package.
--platform kirin9030
# but no tools/platform/kirin9030 plugin installed

# Wrong: assuming the Kirin 9030 package is needed because we write custom ops.
# This project writes no custom ops. Treat --platform as a target-specific
# conversion variable, not a custom-op requirement.

# Wrong: accepting CPU-only logs.
partition type NPU:0, CPU:1
```

Correct first-pass minimal command when the yellow-zone DDK supports conversion
without explicit `--platform`:

```bash
"$OMG_BIN" \
  --model "$ONNX" \
  --framework 5 \
  --input_shape "pixel_values:1,3,448,448" \
  --output "$OUT_PREFIX" \
  2>&1 | tee "$LOG_DIR/omg_convert.log"
```

Correct target-specific command when the yellow-zone DDK workflow requires or
benefits from Kirin 9030 targeting:

```bash
"$OMG_BIN" \
  --model "$ONNX" \
  --framework 5 \
  --input_shape "pixel_values:1,3,448,448" \
  --platform kirin9030 \
  --output "$OUT_PREFIX" \
  2>&1 | tee "$LOG_DIR/omg_convert.log"
```
