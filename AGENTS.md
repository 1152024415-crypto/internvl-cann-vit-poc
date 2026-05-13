# Agent Map

This file is the short map for agent work in this repository. Keep it concise;
put detailed reasoning in `docs/`.

## Project Goal

Deploy the InternVL3.5 visual path on Huawei CANN/HarmonyOS:

```text
image -> vision_model -> pixel_shuffle -> mlp1 -> visual_tokens
```

The current target is **ViT + projector**, not the full language model.

## Current Verified Chain

Read `docs/stage-4-vit-projector-chain.md` first for the current mainline.

Verified:

```text
PyTorch projector baseline
ONNX projector inference vs PyTorch baseline
ONNX projector -> OM conversion with CANN Kit OMG
```

Not verified yet:

```text
OM runtime inference on HarmonyOS device
NPU/HIAI_F placement
latency, memory, cold start, 20-run stability
```

## Important Artifacts

Large artifacts are not committed to git. They live in GitHub Releases.
See `docs/release-artifacts.md`.

Small metadata files are tracked:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
artifacts/release/artifacts-manifest.sha256.txt
```

Do not commit:

```text
*.onnx
*.om
*.pt
DDK-tools*.zip
artifacts/ release payloads other than tracked metadata
```

## Documentation Map

Use these files instead of loading every document at once:

```text
docs/index.md                         high-level doc index
docs/current-status.md                current state snapshot
docs/search-guide.md                  keywords and search commands
docs/stage-1-model-split-baseline.md  model split and PyTorch baselines
docs/stage-2-onnx-export.md           ONNX export and ONNX Runtime checks
docs/stage-3-onnx-to-om.md            OMG conversion and OM notes
docs/stage-4-vit-projector-chain.md   current ViT + projector chain
docs/release-artifacts.md             GitHub Release asset workflow
docs/next-steps.md                    active next steps
```

## Development Rules

Use `uv` for Python commands:

```powershell
uv run python -B -m unittest discover -s tests
```

Keep the first PoC static:

```text
input = pixel_values [1, 3, 448, 448]
output = visual_tokens [1, 256, 1024]
dtype = fp32
opset = 18
```

Do not re-enable FlashAttention/FA3 for ONNX/CANN export.

Do not reintroduce bicubic position-embedding `Resize` into the CANN-targeted
ONNX. The `staticpos` export bypasses it for fixed `448 x 448` input.

## HarmonyOS Demo

`demo/` is the HarmonyOS Native C++ workspace. Generated DevEco/Hvigor folders
must stay out of git.

Current demo state: blank Native C++ template with sample NAPI `add` function.
Real CANN/NN runtime inference code is not implemented yet.

The first device validation should use raw fp32 tensor files:

```text
dog_pixel_values_fp32.bin -> OM -> dog_visual_tokens_fp32.bin comparison
```

This separates OM runtime validation from image preprocessing validation.

## Verification Expectations

Every material change should state what was verified. Prefer:

```text
unit tests
ONNX checker
ONNX Runtime vs PyTorch cosine/max error
OMG conversion log
device-side output comparison once HarmonyOS SDK is available
```
