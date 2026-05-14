# Release Artifacts

Large model artifacts are not committed to git. They are published as GitHub
Release assets so the source repository stays lightweight.

Current release:

```text
https://github.com/1152024415-crypto/internvl-cann-vit-poc/releases/tag/v0.1.0-artifacts
```

## Current OM Replacement Status

As of 2026-05-14, the Release asset named below is stale:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

The uploaded file was generated before the Kirin 9030 platform plugin and
CANN-specific ONNX surgery were added. The yellow-zone device run showed that
old file could be read and `HIAI_F` could be selected, but
`OH_NNCompilation_Build` rejected it because the host OMG log showed:

```text
partition type NPU:0, CPU:1, GPU:0, ISP:0
```

Replace the Release asset manually with this local file:

```text
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.om
size = 1236219952 bytes
SHA256 = 33CA510F80C02C5C990C7050E23F434A6863C94D0D074603E2A29E69D81ADE7B
```

Keep the asset name unchanged so the yellow-zone runbook and app packaging
steps continue to work:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

The replacement OM was generated from:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx
SHA256 = 215A6248B2C5A259A531472210E31282791F50945917CEF6419A238C19E893C2
```

Host-side conversion evidence:

```text
OMG platform = kirin9030
OMG generate offline model success = 1
AI_NPUCL lines = 21
CPUCL lines = 0
partition type NPU:0 lines = 0
```

This is still not final device proof. Final proof requires a yellow-zone
HarmonyOS phone run where `OH_NNCompilation_Build` and
`OH_NNExecutor_RunSync` both succeed.

Included artifacts:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
internvl3_5_vit_projector_fp32_opset18_staticpos.om
onnx-projector-dog-output.pt
onnx-projector-cat-output.pt
pytorch-projector-dog-baseline_output.pt
pytorch-projector-cat-baseline_output.pt
artifacts-manifest.json
artifacts-manifest.sha256.txt
```

Not included:

```text
DDK-tools-next-6.0.1.0.zip
```

The DDK/CANN tools package should be downloaded from Huawei's official site and
verified with the SHA256 recorded in the Stage 3 notes.

## Publish Command

```powershell
powershell -ExecutionPolicy Bypass -File scripts\publish_github_release_artifacts.ps1
```

The script uses the GitHub token available through Git Credential Manager and
does not print the token.

## Validation Tensor Assets

The HarmonyOS runtime validation also needs raw fp32 tensor files. The four
current validation `.bin` files are tracked in git under:

```text
demo/entry/src/main/resources/rawfile/
```

They are small enough to keep in source control:

```text
dog_pixel_values_fp32.bin     2408448 bytes
dog_visual_tokens_fp32.bin    1048576 bytes
cat_pixel_values_fp32.bin     2408448 bytes
cat_visual_tokens_fp32.bin    1048576 bytes
```

Generate them with:

```powershell
uv run internvl-export-validation-tensors `
  --output-dir artifacts\validation-tensors `
  --case dog=data\test-images\dog.jpg,artifacts\baseline-projector-dog\baseline_output.pt `
  --case cat=data\test-images\cat.jpg,artifacts\baseline-projector-cat\baseline_output.pt
```

After regenerating, copy these files into the demo rawfile directory and commit
them with the matching metadata:

```text
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
dog.metadata.json
cat.metadata.json
```

The small `dog.metadata.json` and `cat.metadata.json` files are tracked under
`demo/entry/src/main/resources/rawfile/` for the current tensor set. Update them
only when regenerating the validation tensors.
