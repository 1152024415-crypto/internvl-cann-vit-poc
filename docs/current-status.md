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

Local replacement OM conversion:

```text
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.om
size = 1236219952 bytes
SHA256 = 33CA510F80C02C5C990C7050E23F434A6863C94D0D074603E2A29E69D81ADE7B
OMG platform = kirin9030
OMG result = success
AI_NPUCL lines = 21
CPUCL lines = 0
partition type NPU:0 lines = 0
```

Release assets uploaded:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
internvl3_5_vit_projector_fp32_opset18_staticpos.om
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
```

Important: the currently uploaded Release `.om` is the old CPU-only asset and
is not a valid NPU validation asset. A local replacement OM has been generated
from the CANN-specific ONNX with Kirin 9030 platform evidence, but it has not
been uploaded to GitHub Release yet.

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
runOnce(resourceManager, caseName)
runStability(resourceManager, caseName, repeatCount)
```

The native side follows the CANN Kit endpoint deployment lifecycle:

```text
read .om rawfile
select HIAI_F
OH_NNCompilation_ConstructWithOfflineModelBuffer
OH_NNCompilation_SetDevice
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
upload replacement OM
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
only needs the checked-out code plus the Release `.om` file.
