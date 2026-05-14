# Stage 4: ViT + Projector Chain

## Current Status

Status: ViT + projector chain is verified in PyTorch baseline and ONNX. The
Kirin 9030 replacement OM has been generated from a CANN-specific ONNX, passed
Linux static validation, and been uploaded to GitHub Release. Device runtime
validation is still pending.

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

## CANN-Specific ONNX

The first projector ONNX still contained a fixed-batch class-token
`Expand/Cast` helper chain. With `--platform kirin9030`, OMG loaded NPUCL but
rejected that helper chain through `Equal`, `Select`, and `BroadcastTo`.

The CANN-specific ONNX artifact is:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx
```

Graph surgery:

```text
static_class_embedding = true
removed_nodes = 9
Equal op count = 0
Where op count = 0
Expand op count = 0
```

File size:

```text
1237297904 bytes
```

SHA256:

```text
215A6248B2C5A259A531472210E31282791F50945917CEF6419A238C19E893C2
```

CANN-specific ONNX Runtime verification:

```text
dog cosine vs PyTorch = 1.0000001192092896
cat cosine vs PyTorch = 1.0000001192092896
```

## OM Artifact

The ViT + projector OM artifact is:

```text
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

File size:

```text
1236219952 bytes
```

SHA256:

```text
8D081689805763B786BE003B5627061DFB9324EDF3DF7DF0226C8F5A9C093FA7
```

Conversion log:

```text
C:\Users\11520\wsl-onnx-to-om.log
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.omg.log
```

OMG result:

```text
OMG generate offline model success
OMG platform = kirin9030
AI_NPUCL lines = 21
CPUCL lines = 0
partition type NPU:0 lines = 0
```

Linux static validation:

```text
OMG --mode 3 pre-check report = success
pre-check pass = 1096
pre-check fail = 0
OMG --mode 1 OM to JSON = success
OM JSON op count = 1167
OM JSON input = pixel_values [1, 3, 448, 448]
OM JSON output = Node_Output [1, 256, 1024]
GitHub Release asset verified = yes
```

`omg --mode 3` currently returns process exit code 1 even when the generated
pre-check report says `result=success` and `fail=0`. We use the report contents
for the ONNX/operator support check, then require OM-to-JSON to exit
successfully for the generated OM file check.

## Verification Limits

The old Release OM was CPU-only. The 2026-05-14 yellow-zone device run
confirmed this mattered. The old OM file was loaded intact:

```text
om_bytes=1236220257
selected_device=name=HIAI_F type=ACCELERATOR
OH_NNCompilation_Build failed code=1
Authentication failed, input model cannot run by npu
```

The old root cause was:

```text
partition type NPU:0, CPU:1, GPU:0, ISP:0
```

The replacement OM was generated after:

```text
Kirin 9030 platform plugin installed
OMG called with --platform kirin9030
CANN-specific ONNX removed static class-token Equal/Where/Expand
```

The latest OMG log does not print a partition summary. The host-side evidence is
therefore weaker than a phone run, but it is materially different from the old
CPU-only conversion: it shows `AI_NPUCL`, does not show `CPUCL`, and does not
show `partition type NPU:0`.

So the current verified chain is:

```text
PyTorch projector baseline: verified
ONNX projector inference: verified against PyTorch
CANN-specific ONNX inference: verified against PyTorch
ONNX projector -> OM conversion: host-side Kirin 9030 NPU-targeted conversion generated
OM runtime inference: not verified yet
```

The next step is to copy the Release `.om` into the yellow-zone HarmonyOS app
rawfile directory and rerun device validation.
