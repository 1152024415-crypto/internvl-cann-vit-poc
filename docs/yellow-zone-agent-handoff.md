# Yellow-Zone Agent Handoff

This is the first document to give to a yellow-zone AI or engineer. The goal is
to let the yellow-zone side regenerate the OM with its own DDK, validate it with
logs, then run the HarmonyOS device demo.

## First Read Order

Read these files in order. Do not load every document at once.

```text
AGENTS.md
docs/current-status.md
docs/yellow-zone-onnx-to-om-ddk-runbook.md
docs/yellow-zone-validation-runbook.md
docs/stage-4-vit-projector-chain.md
```

The authoritative current workflow is:

```text
1. Use yellow-zone DDK/OMG to regenerate the OM from ONNX.
2. Keep all OMG logs, precheck report, OM-to-JSON output, and manifest.
3. Put the regenerated OM into the HarmonyOS demo rawfile directory.
4. Run official CANN sample smoke/classification first.
5. Run InternVL Load Model, dog/cat Run Once, then 20-run stability.
```

Do not treat the blue-zone-generated OM as final proof. It may be useful as a
historical artifact, but the yellow-zone DDK is the source of truth for the OM
that will be tested on the phone.

## Model Contract

This project is validating only the InternVL3.5 ViT + projector path:

```text
pixel_values [1, 3, 448, 448]
-> vision_model
-> pixel_shuffle
-> mlp1
-> visual_tokens [1, 256, 1024]
```

It is not full VLM text generation. It does not output class labels or text.

Hard requirements for the first pass:

```text
ONNX input name  = pixel_values
ONNX input shape = [1, 3, 448, 448]
ONNX dtype       = fp32
ONNX opset       = 18
OM output shape  = [1, 256, 1024]
No dynamic input dimension
No --dynamic_dims
No 5D input such as [1, N, 3, 448, 448]
No dynamic high-resolution / multi-patch path
No CPU-only OM
```

## About `--platform` And Kirin 9030

The target phone reported by the user is Kirin 9030.

Do not confuse these two concepts:

```text
custom operator / AscendC plugin  = needed only when writing custom operators
platform targeting / platform data = used by OMG/runtime for target-specific model handling
```

We are not writing custom operators in this project. Therefore the Kirin 9030
platform package is not required because of custom-op development.

The platform question is only about OM generation and validation:

```text
If the yellow-zone DDK supports the official minimal ONNX conversion without
--platform, run it and keep the logs.

If generating a target-specific OM with --platform kirin9030, the matching
platform package must be installed and visible to the yellow-zone DDK.

If the official yellow-zone docs or sample commands require --platform for the
chosen target mode, follow that requirement.
```

If uncertain, produce two logged attempts:

```text
attempt A: official/minimal ONNX -> OM command without --platform
attempt B: target-specific ONNX -> OM command with --platform kirin9030
```

Accept only an OM that passes the static checks and then passes phone-side
runtime validation. A conversion log that produces CPU-only placement is not
acceptable, even if an `.om` file exists.

## Required Yellow-Zone Deliverables

Return or upload the complete run directory, not only the `.om`:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos*.om
internvl3_5_vit_projector_fp32_opset18_staticpos*.json
MANIFEST.sha256
files.txt
logs/env.log
logs/onnx_static_check.log
logs/omg_help.log
logs/omg_precheck.log
logs/omg_precheck_report.txt
logs/omg_precheck_summary.log
logs/omg_convert.log
logs/om_summary.log
logs/om_to_json.log
logs/om_json_shape_snippet.log
```

The logs must show:

```text
yellow-zone DDK/OMG path
ONNX SHA256
ONNX input/output shape
OMG command line
precheck result
conversion success or exact failure
NPU/CPU/platform-related markers from OMG output
OM-to-JSON success or exact failure
```

## Code Map

Only patch these areas unless the compiler points somewhere else:

```text
demo/                                      DevEco project root to open
demo/entry/src/main/ets/pages/Index.ets    ArkTS validation UI
demo/entry/src/main/cpp/napi_init.cpp      NAPI exports
demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp
                                           CANN/NN runtime load, build, RunSync
demo/entry/src/main/cpp/validation/rawfile_loader.cpp
                                           rawfile model/tensor loading
demo/entry/src/main/cpp/validation/tensor_metrics.cpp
                                           cosine/diff checks
demo/entry/src/main/cpp/CMakeLists.txt     native library link settings
demo/entry/src/main/resources/rawfile/     OM and fp32 validation tensors
```

If the official CANN sample fails, debug SDK/device/app integration before
changing InternVL model conversion. If the official sample passes and InternVL
fails, focus on the InternVL OM and OMG logs.

## Device Validation Gates

Run the HarmonyOS app from `demo/` on a physical phone. Do not use an emulator
as NPU proof.

Passing gates:

```text
Official Smoke succeeds
Official Classify succeeds for guitar or cup
InternVL Load Model succeeds
selected device contains HIAI_F / ACCELERATOR
OH_NNCompilation_Build succeeds
OH_NNExecutor_RunSync succeeds
dog output shape = [1, 256, 1024]
cat output shape = [1, 256, 1024]
dog/cat cosine >= 0.999
dog/cat mean_abs_diff <= 0.01
20-run stability succeeds
latency values are recorded
```

If Official Smoke fails, debug the yellow-zone SDK/device/app runtime first. If
Official Smoke succeeds but InternVL fails, focus on the InternVL OM conversion
path.

## Prompt To Give The Yellow-Zone AI

Use this prompt directly:

```text
You are working in the yellow-zone clone of
https://github.com/1152024415-crypto/internvl-cann-vit-poc.

Read these files first:
1. AGENTS.md
2. docs/current-status.md
3. docs/yellow-zone-onnx-to-om-ddk-runbook.md
4. docs/yellow-zone-validation-runbook.md
5. docs/stage-4-vit-projector-chain.md

Your task:
- Regenerate the InternVL ViT + projector OM from the CANN-specific ONNX using
  the yellow-zone DDK/OMG.
- Keep full logs and manifest exactly as requested by the runbook.
- Do not use dynamic shape, --dynamic_dims, -1 dimensions, or 5D input.
- Do not accept CPU-only OM output.
- Do not assume the Kirin 9030 platform package is needed because of custom
  operators; this project writes no custom ops. Treat --platform kirin9030 as a
  target-specific conversion variable. If the official yellow-zone DDK command
  works without --platform, log that attempt; if --platform is used, verify the
  matching platform package is installed.
- After OM generation, put the accepted OM into
  demo/entry/src/main/resources/rawfile/
  as internvl3_5_vit_projector_fp32_opset18_staticpos.om.
- Open demo/ in DevEco Studio, build for arm64-v8a, install on a physical
  HarmonyOS phone, then run Official Smoke, Official Classify, InternVL Load
  Model, dog/cat Run Once, and Run 20 Stability.
- Record all results in docs/harmonyos-om-runtime-validation.md.

Return the exact OMG command, generated file SHA256, key log lines, phone model,
HarmonyOS version, selected runtime device, and validation metrics.
```
