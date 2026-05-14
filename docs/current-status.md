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

OM conversion:

```text
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.om
OMG result = success, but current artifact is CPU-only
latest partition summary = NPU:0, CPU:1, GPU:0, ISP:0
```

Release assets uploaded:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
internvl3_5_vit_projector_fp32_opset18_staticpos.om
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
```

Important: the currently uploaded `.om` is not a valid NPU validation asset.
Yellow-zone logs confirmed that the file loads intact and `HIAI_F` is found,
but `OH_NNCompilation_Build` fails with model authentication because the OM was
generated without an NPU partition. Install the matching Kirin platform plugin
into WSL, reconvert, and upload a replacement OM before retrying device NPU
runtime validation.

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
install matching Kirin platform plugin into WSL
reconvert projector OM and verify OMG partition summary has NPU > 0
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
