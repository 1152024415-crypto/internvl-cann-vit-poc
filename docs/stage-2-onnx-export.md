# Stage 2: ONNX Export

## Current Status

Status: pure ViT and ViT + projector ONNX exports complete.

The static pure ViT path and the static ViT + projector path have been exported
to ONNX and verified with both ONNX checker and ONNX Runtime numerical
comparison against saved PyTorch baselines.

Completed:

- Added `internvl-export-onnx` CLI.
- Added `onnx` and `onnxruntime` dependencies.
- Added lightweight unit tests for ONNX wrappers and metadata.
- Exported pure ViT target with static `1 x 3 x 448 x 448` input.
- Used ONNX opset 18.
- Kept Flash Attention disabled.
- Verified graph input/output shape.
- Ran `onnx.checker.check_model`.
- Ran ONNX Runtime CPU inference and compared against PyTorch baselines.

Not completed:

- Full VLM text-generation ONNX export has not been attempted.

## Export Command

```powershell
uv run internvl-export-onnx `
  --image data\sample.jpg `
  --output artifacts\onnx\internvl3_5_vit_fp32_opset18.onnx `
  --target vision `
  --device cpu `
  --dtype fp32 `
  --opset 18 `
  --local-files-only
```

Generated files:

```text
artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx               1216395796 bytes
artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx.metadata.json        574 bytes
```

Export metadata:

```text
target = vision
dtype = fp32
device = cpu
opset = 18
input_name = pixel_values
output_name = last_hidden_state
input_shape = [1, 3, 448, 448]
output_shape = [1, 1025, 1024]
dynamic_axes = false
```

## Structure Check

Command:

```powershell
uv run python -B -c "import onnx; p='artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx'; m=onnx.load(p); print('ir_version', m.ir_version); print('opset', [(o.domain, o.version) for o in m.opset_import]); print('inputs', [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]); print('outputs', [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]); print('nodes', len(m.graph.node)); onnx.checker.check_model(m); print('checker ok')"
```

Observed:

```text
ir_version = 8
opset = [('', 18)]
inputs = [('pixel_values', [1, 3, 448, 448])]
outputs = [('last_hidden_state', [1, 1025, 1024])]
nodes = 1732
checker ok
```

Top ONNX op counts:

```text
Constant: 488
Add: 169
MatMul: 144
Unsqueeze: 144
Mul: 121
Cast: 99
Transpose: 74
Shape: 73
Gather: 72
Squeeze: 72
Concat: 51
Reshape: 50
LayerNormalization: 48
Div: 48
Split: 24
Softmax: 24
Erf: 24
Conv: 1
Resize: 1
```

The presence of `MatMul`, `Softmax`, `LayerNormalization`, and `Erf` is expected for a normal transformer attention and GELU path. `Resize` comes from the ViT position embedding interpolation path, even though the input is fixed to `448 x 448`.

## Numerical Check

Command:

```powershell
uv run python -B -c "import numpy as np, onnxruntime as ort, torch; from internvl_vit_poc.preprocess import load_static_pixel_values; x=load_static_pixel_values('data/sample.jpg').numpy(); sess=ort.InferenceSession('artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx', providers=['CPUExecutionProvider']); y=sess.run(['last_hidden_state'], {'pixel_values': x})[0]; ref=torch.load('artifacts/baseline-vit/baseline_output.pt', map_location='cpu').numpy(); diff=np.abs(y-ref); cos=float(np.dot(y.reshape(-1), ref.reshape(-1))/(np.linalg.norm(y.reshape(-1))*np.linalg.norm(ref.reshape(-1)))); print('onnx_shape', y.shape); print('ref_shape', ref.shape); print('max_abs_diff', float(diff.max())); print('mean_abs_diff', float(diff.mean())); print('cosine', cos)"
```

Observed:

```text
onnx_shape = (1, 1025, 1024)
ref_shape = (1, 1025, 1024)
max_abs_diff = 0.0034332275390625
mean_abs_diff = 5.648157184623415e-06
cosine = 0.9999999403953552
```

This is a good export result for FP32. The ONNX graph is numerically aligned with the PyTorch baseline.

## Why These Choices

Static input:

```text
1 x 3 x 448 x 448
```

This avoids dynamic image patch counts and dynamic axes. Static shape is simpler for ONNX export and for the next CANN OMG conversion.

Opset 18:

```text
opset = 18
```

CANN Kit documentation states ONNX conversion supports opset 7-18, so opset 18 is the newest safe target for this path.

Flash Attention disabled:

```text
use_flash_attn = false
use_fa3 = false
```

This keeps attention as ordinary graph operations such as `MatMul` and `Softmax`. That is easier to inspect and more likely to be convertible by CANN than a fused GPU-specific attention kernel.

FP32 first:

```text
dtype = fp32
```

The original model may use bfloat16, but FP32 is better for the first reference export because it gives a cleaner numerical baseline. FP16 can be attempted after the conversion path is proven.

## CANN-Compatible Static Position Export

The first fp32 ONNX export converted correctly in ONNX Runtime, but OMG rejected
its single `Resize` node. The node came from InternVL's bicubic position
embedding interpolation.

For the fixed `1 x 3 x 448 x 448` PoC, the ViT patch grid is already `32 x 32`,
so position embedding interpolation is unnecessary. The CANN-targeted export is:

```text
artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx
```

Validation:

```text
onnx.checker.check_model = ok
Resize op count = 0
ONNX Runtime vs PyTorch cosine = 0.9999999403953552
ONNX Runtime vs PyTorch mean_abs_diff = 5.648157184623415e-06
```

## Next Step

Use the CANN-compatible projector ONNX for OM conversion:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
```

The projector ONNX details and dog/cat verification numbers are tracked in:

```text
docs/stage-4-vit-projector-chain.md
```
