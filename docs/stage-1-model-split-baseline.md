# Stage 1: Model Split and PyTorch Baseline

## Current Status

Status: pure ViT and projector baselines complete.

The project scaffolding, split utilities, fixed-size image preprocessing, baseline CLI, uv environment, and unit tests are in place. A real pure ViT baseline and two projector baselines have been generated from `OpenGVLab/InternVL3_5-1B-Instruct`.

Completed:

- Created the uv-managed project at `D:\proj\internvl-cann-vit-poc`.
- Pinned Python to 3.12 through `.python-version`.
- Added dependencies in `pyproject.toml` and generated `uv.lock`.
- Implemented `split_state_dict()` for extracting `vision_model.*` weights.
- Added optional projector inclusion for `vision_model + mlp1`.
- Implemented fixed `1 x 3 x 448 x 448` image preprocessing.
- Implemented `internvl-vit-baseline` CLI.
- Pinned the default Hugging Face revision to `5648fa26ff23acaba53588936d9f1dfaf305f522`.
- Added `--local-files-only` for cache-only runs after files are downloaded.
- Added unit tests for split behavior and metadata shape recording.
- Added `einops` and `timm`, which are required by InternVL remote modeling code.
- Loaded fixed-revision InternVL weights with `trust_remote_code=True` after explicit authorization.
- Generated pure ViT baseline artifacts under `artifacts/baseline-vit`.
- Downloaded two test images under `data/test-images`.
- Generated projector baseline artifacts for `dog.jpg` and `cat.jpg`.
- Verified tests with `uv run python -B -m unittest discover -s tests`.

Not completed:

- ONNX export and CANN conversion are intentionally out of scope for this stage.

## Generated Baseline Artifacts

Pure ViT baseline command:

```powershell
uv run internvl-vit-baseline `
  --image data\sample.jpg `
  --output-dir artifacts\baseline-vit `
  --device cpu `
  --dtype fp32
```

Artifacts:

```text
artifacts/baseline-vit/vision_state.pt        1216155097 bytes
artifacts/baseline-vit/vision_config.json            835 bytes
artifacts/baseline-vit/baseline_output.pt        4200033 bytes
artifacts/baseline-vit/baseline_metadata.json        619 bytes
```

Observed metadata:

```text
input_shape = [1, 3, 448, 448]
output_shape = [1, 1025, 1024]
output_kind = vision_last_hidden_state
dtype = fp32
device = cpu
include_projector = false
```

Loaded baseline tensor check:

```text
shape = (1, 1025, 1024)
dtype = torch.float32
mean = 0.00448997737839818
std = 1.7406953573226929
```

## Generated Projector Baselines

Downloaded test images:

```text
data/test-images/dog.jpg    661378 bytes
data/test-images/cat.jpg    279603 bytes
```

Image sources:

```text
dog.jpg = https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg
cat.jpg = https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg
```

Dog projector command:

```powershell
uv run internvl-vit-baseline `
  --image data\test-images\dog.jpg `
  --output-dir artifacts\baseline-projector-dog `
  --device cpu `
  --dtype fp32 `
  --include-projector `
  --local-files-only
```

Dog observed metadata:

```text
input_shape = [1, 3, 448, 448]
output_shape = [1, 256, 1024]
output_kind = projected_visual_tokens
dtype = fp32
device = cpu
include_projector = true
```

Dog tensor check:

```text
shape = (1, 256, 1024)
dtype = torch.float32
mean = -0.0018174485303461552
std = 0.41595539450645447
```

Cat projector command:

```powershell
uv run internvl-vit-baseline `
  --image data\test-images\cat.jpg `
  --output-dir artifacts\baseline-projector-cat `
  --device cpu `
  --dtype fp32 `
  --include-projector `
  --local-files-only
```

Cat observed metadata:

```text
input_shape = [1, 3, 448, 448]
output_shape = [1, 256, 1024]
output_kind = projected_visual_tokens
dtype = fp32
device = cpu
include_projector = true
```

Cat tensor check:

```text
shape = (1, 256, 1024)
dtype = torch.float32
mean = 0.005221450701355934
std = 0.4335198700428009
```

Dog vs cat tensor difference:

```text
mean_abs_diff = 0.31287381052970886
max_abs_diff = 3.1300559043884277
```

The projector output shape confirms that `pixel_shuffle + mlp1` reduces the visual sequence from pure ViT's `1025` tokens to `256` projected visual tokens.

## What Model Splitting Means

InternVL3.5 is a VLM, not a pure vision model. Its high-level path is:

```text
image
-> vision_model
-> pixel_shuffle
-> mlp1
-> visual_tokens
-> language_model
-> text answer
```

PyTorch stores weights in a `state_dict`, which is a dictionary from module-path names to tensors. In the full VLM, ViT weights live under the parent module name `vision_model`:

```text
vision_model.encoder.layers.0.attn.qkv.weight
vision_model.embeddings.position_embedding
language_model.model.layers.0.self_attn.q_proj.weight
mlp1.0.weight
```

For a standalone `InternVisionModel`, the same ViT weights are expected without the parent prefix:

```text
encoder.layers.0.attn.qkv.weight
embeddings.position_embedding
```

So default splitting does this:

```text
vision_model.xxx -> xxx
language_model.xxx -> dropped
mlp1.xxx -> dropped
```

The key rename is required because PyTorch `load_state_dict()` matches weights by key names. The tensor values are not mathematically changed; only their dictionary keys are rewritten to match the target module structure.

## Projector Mode

The optional `--include-projector` mode keeps:

```text
vision_model.*
mlp1.*
```

This is not a pure ViT split. It represents the visual path before the LLM consumes image features.

Without projector:

```text
image -> vision_model -> last_hidden_state
```

Expected fixed-shape output for one `448 x 448` image is approximately:

```text
1 x 1025 x 1024
```

The `1025` dimension is `32 x 32` patch tokens plus one CLS token.

With projector:

```text
image -> vision_model -> pixel_shuffle -> mlp1 -> visual_tokens
```

`pixel_shuffle` is token grid rearrangement/downsampling, not classification. It packs nearby spatial tokens into fewer tokens with a larger channel dimension. `mlp1` is a vision-to-language projection layer, not a class prediction head. Its output is continuous visual embeddings for the LLM.

## Fixed Baseline Input

The first PoC uses:

```text
1 x 3 x 448 x 448
```

Meaning:

- `1`: one image / one patch.
- `3`: RGB channels.
- `448 x 448`: fixed image size.
- Layout: NCHW.

This intentionally avoids dynamic high-resolution mode, multi-patch routing, and dynamic shape handling. Those features are useful later, but they make ONNX export, CANN conversion, and output comparison harder to debug.

## Flash Attention Decision

Flash Attention should stay disabled for stage 1.

Reason:

- The first target is debuggable export and conversion, not maximum PyTorch GPU speed.
- A normal attention graph is easier to export as `MatMul -> Softmax -> MatMul`.
- Flash Attention is usually implemented as a fused GPU kernel, which can export poorly or appear as a custom/fused op.
- CANN Kit's current ONNX conversion path is based on ONNX models supported by OMG, so relying on runtime-specific attention fusion is risky at the first step.

This does not mean Flash Attention is bad. It is useful for GPU inference and training. It is just not the safest first dependency for a PyTorch -> ONNX -> CANN OM path.

## Current Commands

Set up or refresh the environment:

```powershell
uv sync
```

Run tests:

```powershell
uv run python -B -m unittest discover -s tests
```

Show baseline CLI:

```powershell
uv run internvl-vit-baseline --help
```

The baseline CLI pins model files and remote code by default:

```text
revision = 5648fa26ff23acaba53588936d9f1dfaf305f522
code_revision = 5648fa26ff23acaba53588936d9f1dfaf305f522
```

This reduces the risk of accidentally executing a changed `main` branch. It does not remove the inherent risk of `trust_remote_code=True`; the downloaded model code should still be reviewed before execution.

Run the intended baseline after model loading is confirmed:

```powershell
uv run internvl-vit-baseline `
  --image path\to\image.jpg `
  --output-dir artifacts\baseline `
  --device cpu `
  --dtype fp32
```

## Completion Criteria for Stage 1

The pure ViT portion of Stage 1 is complete because these files exist and have been inspected:

```text
artifacts/baseline/vision_state.pt
artifacts/baseline/vision_config.json
artifacts/baseline/baseline_output.pt
artifacts/baseline/baseline_metadata.json
```

The projector portion is complete because these files exist and have been inspected:

```text
artifacts/baseline-projector-dog/baseline_output.pt
artifacts/baseline-projector-dog/baseline_metadata.json
artifacts/baseline-projector-cat/baseline_output.pt
artifacts/baseline-projector-cat/baseline_metadata.json
```

The metadata should record:

- model id;
- input image path;
- input shape;
- output shape;
- dtype;
- device;
- whether projector mode was used.
