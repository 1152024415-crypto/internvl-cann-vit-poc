# InternVL CANN ViT PoC

This project is a PoC for deploying the ViT + projector side of
`OpenGVLab/InternVL3_5-1B-Instruct` on-device.

Current scope:

- load the Hugging Face model on a desktop machine;
- disable flash-attention-only paths for export-friendly behavior;
- split and save the vision tower state;
- run a fixed `1 x 3 x 448 x 448` PyTorch baseline;
- save tensor output plus metadata for later ONNX and CANN comparison;
- export static fp32 ONNX models with opset 18;
- convert pure ViT and ViT + projector ONNX artifacts to OM with CANN Kit OMG.

The first baseline intentionally avoids dynamic high-resolution preprocessing,
multi-patch routing, dynamic shapes, and quantization.

## Python Environment

Use `uv` for the Python environment. The project pins Python 3.12 through
`.python-version`; avoid running the system `python` directly because it may
point to Python 3.14, which is not a safe target for PyTorch/Transformers.

```powershell
uv sync
```

## Run Unit Tests

```powershell
uv run python -B -m unittest discover -s tests
```

## Run Baseline

```powershell
uv run internvl-vit-baseline `
  --image path\to\image.jpg `
  --output-dir artifacts\baseline `
  --device cpu `
  --dtype fp32
```

For a GPU machine, use `--device cuda --dtype fp16` or `--dtype bf16` if the
GPU supports it.

## Run ONNX Export

```powershell
uv run internvl-export-onnx `
  --image data\test-images\dog.jpg `
  --output artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos.onnx `
  --target projector `
  --device cpu `
  --dtype fp32 `
  --opset 18 `
  --local-files-only `
  --static-position-embedding
```

## Verify Projector ONNX With An Image

```powershell
uv run internvl-onnx-verify `
  --onnx artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos.onnx `
  --image data\test-images\dog.jpg `
  --output-name visual_tokens `
  --save-output artifacts\onnx-projector-dog-output.pt `
  --compare artifacts\baseline-projector-dog\baseline_output.pt
```

## Stage Docs

- `docs/stage-1-model-split-baseline.md`
- `docs/stage-2-onnx-export.md`
- `docs/stage-3-onnx-to-om.md`
- `docs/stage-4-vit-projector-chain.md`
