# Stage 4: ViT + Projector Chain

## Current Status

Status: ViT + projector chain is available in PyTorch baseline, ONNX, and OM
formats.

This is the chain we care about before handing image features to the LLM:

```text
image
-> resize/RGB/normalize/BCHW
-> vision_model
-> pixel_shuffle
-> mlp1
-> visual_tokens
```

The output is not a class label or text answer. It is a tensor of visual
embeddings:

```text
visual_tokens = [1, 256, 1024]
```

## PyTorch Baseline Artifacts

Dog:

```text
data/test-images/dog.jpg
artifacts/baseline-projector-dog/baseline_output.pt
artifacts/baseline-projector-dog/baseline_metadata.json
```

Cat:

```text
data/test-images/cat.jpg
artifacts/baseline-projector-cat/baseline_output.pt
artifacts/baseline-projector-cat/baseline_metadata.json
```

Both projector baselines have:

```text
input_shape = [1, 3, 448, 448]
output_shape = [1, 256, 1024]
output_kind = projected_visual_tokens
include_projector = true
```

## ONNX Artifact

The ViT + projector ONNX artifact is:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
```

Metadata:

```text
target = projector
input_name = pixel_values
input_shape = [1, 3, 448, 448]
output_name = visual_tokens
output_shape = [1, 256, 1024]
opset = 18
dtype = fp32
static_position_embedding = true
```

File size:

```text
1237299338 bytes
```

SHA256:

```text
C8997ADB449A6DB2ADEC7FA79D8DAEFC00B920EEB220C3B66C57439750BD23D1
```

ONNX structure check:

```text
onnx.checker.check_model = ok
Resize op count = 0
input = pixel_values [1, 3, 448, 448]
output = visual_tokens [1, 256, 1024]
```

## ONNX Runtime Verification

Reusable command:

```powershell
uv run internvl-onnx-verify `
  --onnx artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos.onnx `
  --image data\test-images\dog.jpg `
  --output-name visual_tokens `
  --save-output artifacts\onnx-projector-dog-output.pt `
  --compare artifacts\baseline-projector-dog\baseline_output.pt
```

Saved ONNX Runtime outputs:

```text
artifacts/onnx-projector-dog-output.pt    1050407 bytes
artifacts/onnx-projector-cat-output.pt    1050407 bytes
```

Dog image:

```text
ONNX shape = [1, 256, 1024]
PyTorch baseline shape = [1, 256, 1024]
max_abs_diff = 0.00026726722717285156
mean_abs_diff = 0.0000025306071620434523
cosine = 1.0000001192092896
```

Cat image:

```text
ONNX shape = [1, 256, 1024]
PyTorch baseline shape = [1, 256, 1024]
max_abs_diff = 0.0015463829040527344
mean_abs_diff = 0.000002828864126058761
cosine = 1.0000001192092896
```

This verifies that the ONNX projector chain matches the PyTorch projector
baseline for more than one image.

## OM Artifact

The ViT + projector OM artifact is:

```text
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

File size:

```text
1236220257 bytes
```

SHA256:

```text
6E26AF05E176C18D91B513C89289FD384E91C4DB220C1F5393ACBD75E16E764C
```

Conversion log:

```text
C:\Users\11520\wsl-onnx-to-om.log
```

OMG result:

```text
OMG generate offline model success
```

## Verification Limits

The OM file has been generated, but we have not run OM inference yet. The local
tooling has OMG for conversion, not a full HarmonyOS/CANN runtime path for
executing OM with image tensors.

The OMG log still contains platform warnings:

```text
GetPlatformVersion: Read platform version error
partition type NPU:0, CPU:1, GPU:0, ISP:0
```

So the current verified chain is:

```text
PyTorch projector baseline: verified
ONNX projector inference: verified against PyTorch
ONNX projector -> OM conversion: verified
OM runtime inference: not verified yet
```

The next non-HarmonyOS task is to keep improving validation around ONNX and
model artifacts. Device runtime validation must wait until we have a workable
HarmonyOS/CANN runtime environment.
