# Current Status

Last updated: 2026-05-14

## Mainline

The active deployment target is the InternVL3.5 **ViT + projector** path:

```text
image -> preprocess -> vision_model -> pixel_shuffle -> mlp1 -> visual_tokens
```

This is not full VLM text generation. It produces visual embeddings for the LLM:

```text
visual_tokens [1, 256, 1024]
```

Target phone platform for NPU OM generation:

```text
Kirin 9030
platform plugin package = kirin9030-plugin-6.0.1.0
expected SHA256 = 3B32EFFC5AF9804628CB9287E88CC28ED381877ADB15DD85BF8D66E3BE805251
```

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

Release OM artifact:

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
GitHub Release upload verified = yes
GitHub asset digest = sha256:8d081689805763b786be003b5627061dfb9324edf3df7df0226c8f5a9c093fa7
```

Release assets uploaded:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
internvl3_5_vit_projector_fp32_opset18_staticpos.om
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
```

Important: the uploaded Release `.om` now matches the Linux-validated Kirin
9030 replacement OM. It is ready for yellow-zone HarmonyOS device validation.

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

## Not Completed

Device-side work still pending:

```text
compile/install on a yellow-zone physical HarmonyOS device
confirm HIAI_F / NPU placement with the replacement OM
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
only needs the checked-out code plus the Release `.om` file. The official
SqueezeNet CANN sample OM is also tracked in git as
`official_squeezenet_hiai.om` for a small known-good runtime smoke test. The
official `cup` and `guitar` images, labels, and generated 227 x 227 BGR inputs
are tracked as a second known-good classification check.
