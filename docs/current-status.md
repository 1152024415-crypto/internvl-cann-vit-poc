# Current Status

Last updated: 2026-05-13

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
OMG result = success
```

Release assets uploaded:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
internvl3_5_vit_projector_fp32_opset18_staticpos.om
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
```

Small tracked artifact metadata:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
artifacts/release/artifacts-manifest.sha256.txt
```

HarmonyOS demo workspace:

```text
demo/
```

It is currently a blank DevEco Native C++ template with a sample NAPI `add`
function. CANN/NN runtime inference code has not been added yet.

## Not Completed

Device-side work still pending:

```text
load .om from resources/rawfile
select HIAI_F / NPU device
build OH_NN executor
run OH_NNExecutor_RunSync
compare output against baseline
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

Recommended next blue-zone task:

```text
export raw fp32 validation tensors:
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
```

These `.bin` files allow the yellow-zone C++ demo to validate OM runtime first,
before implementing image preprocessing.
