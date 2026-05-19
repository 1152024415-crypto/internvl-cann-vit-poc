# Current Status

Last updated: 2026-05-15

## Mainline

The active deployment target is the InternVL3.5 **ViT + projector** path:

```text
image -> preprocess -> vision_model -> pixel_shuffle -> mlp1 -> visual_tokens
```

This is not full VLM text generation. It produces visual embeddings for the LLM:

```text
visual_tokens [1, 256, 1024]
```

Target phone reported by the user:

```text
Kirin 9030
```

Important: do not require the Kirin 9030 platform package on the grounds of
custom-operator development. This project writes no custom operators. Treat
`--platform kirin9030` as a target-specific OMG conversion variable: if it is
used, the matching platform package must be installed; if the official
yellow-zone DDK workflow works without `--platform`, keep that run and judge it
by logs plus phone-side validation.

## Completed

Model split and PyTorch baselines:

```text
artifacts/baseline-projector-dog/baseline_output.pt
artifacts/baseline-projector-cat/baseline_output.pt
```

ONNX export and validation:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
input  = pixel_values [1, 3, 448, 448]
output = visual_tokens [1, 256, 1024]
Resize op count = 0
dog cosine vs PyTorch = 1.0000001192092896
cat cosine vs PyTorch = 1.0000001192092896
```

CANN-specific ONNX surgery:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx
static_class_embedding = true
removed_nodes = 9
removed embedding Equal/Where/Expand = yes
dog cosine vs PyTorch = 1.0000001192092896
cat cosine vs PyTorch = 1.0000001192092896
SHA256 = 215A6248B2C5A259A531472210E31282791F50945917CEF6419A238C19E893C2
```

Historical blue-zone OM artifact:

```text
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.om
size = 1236219952 bytes
SHA256 = 8D081689805763B786BE003B5627061DFB9324EDF3DF7DF0226C8F5A9C093FA7
OMG platform = kirin9030
OMG result = success
AI_NPUCL lines = 21
CPUCL lines = 0
partition type NPU:0 lines = 0
Linux static validation = pass
precheck report = success, pass 1096, fail 0
OM JSON input = pixel_values [1, 3, 448, 448]
OM JSON output = Node_Output [1, 256, 1024]
GitHub Release upload verified = yes at the time of upload
GitHub asset digest = sha256:8d081689805763b786be003b5627061dfb9324edf3df7df0226c8f5a9c093fa7
```

Release assets uploaded:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
internvl3_5_vit_projector_fp32_opset18_staticpos.om
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
```

Important: after the blue-zone/yellow-zone DDK mismatch was found, this Release
`.om` is no longer the authoritative final runtime artifact. Use it only as
historical context unless it is regenerated or explicitly confirmed by the
yellow-zone DDK run. The next trusted OM should be produced in yellow zone using
`docs/yellow-zone-onnx-to-om-ddk-runbook.md`, then validated on device.

Small tracked artifact metadata:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
artifacts/release/artifacts-manifest.sha256.txt
```

HarmonyOS demo workspace:

```text
demo/
```

It now contains a Native C++ validation path for the ViT + projector OM.
The app exposes:

```text
listTestCases()
loadModel(resourceManager)
unloadModel()
runOfficialSmoke(resourceManager)
runOfficialClassification(resourceManager, imageName)
runOnce(resourceManager, caseName)
runStability(resourceManager, caseName, repeatCount)
```

The native side follows the CANN Kit endpoint deployment lifecycle:

```text
read .om rawfile
select HIAI_F
HMS_HiAICompatibility_CheckFromBuffer
OH_NNCompilation_ConstructWithOfflineModelBuffer
OH_NNCompilation_SetDevice
HMS_HiAIOptions_SetBandMode(HIAI_BANDMODE_NORMAL)
HMS_HiAIOptions_SetModelDeviceOrder(HIAI_EXECUTE_DEVICE_NPU)
OH_NNCompilation_Build
OH_NNExecutor_Construct
destroy compilation
reuse executor for RunSync
```

The NAPI entry points return Promises so model loading/building and `RunSync`
do not execute on the ArkUI event thread.

## INT8 Quantization (Stage 5)

FP32 ONNX → OMG produces CPU-only OM (588 ops not supported by NPUCL).
Solution: INT8 quantization via `dopt_onnx_py3` changes op patterns so NPU
can execute them natively.

Verified on Linux server (100.102.192.199, cann_lm_310 conda env):

```text
dopt_onnx_py3/dopt_so.py \
  --framework 5 -m 0 \
  --model split_internvit_projector_fixed_b1_no_resize.onnx \
  --cal_conf int8_8_config_v6.prototxt \
  --output split_internvit_v6_exclude_conv_int8_8.onnx \
  --input_shape 'pixel_values:1,3,448,448' \
  --out_nodes 'visual_features' \
  --compress_conf split_internvit_v6_exclude_conv_int8_8_param \
  --device_idx 0

Result: Quantize model success
```

Key config (`int8_8_config_v6.prototxt`):

```text
strategy: Quant_INT8-8
device: USE_CPU
exclude_op: /vision_model/embeddings/patch_embedding/Conv
preprocess_parameter: { input_type: IMAGE, ImageNet normalize, ./calib_images }
```

Next: OMG conversion with `--compress_conf` → device validation.

Reference: `docs/stage-5-int8-quantization-runbook.md`

## Not Completed

Device-side work still pending:

```text
OMG conversion of INT8 ONNX with --compress_conf
compile/install on a yellow-zone physical HarmonyOS device
confirm HIAI_F / NPU placement with the INT8 OM
confirm OH_NNExecutor_RunSync succeeds
compare output against baseline on device
measure latency, memory, cold start, 20-run stability
```

Preprocessing pending on device:

```text
decode image
resize to 448 x 448
RGB
normalize
HWC -> BCHW
float32 tensor
```

The raw fp32 validation tensors are tracked in git, so yellow-zone validation
only needs the checked-out code plus the yellow-zone-regenerated `.om` file. The
official SqueezeNet CANN sample OM is also tracked in git as
`official_squeezenet_hiai.om` for a small known-good runtime smoke test. The
official `cup` and `guitar` images, labels, and generated 227 x 227 BGR inputs
are tracked as a second known-good classification check.
