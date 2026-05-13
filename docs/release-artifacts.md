# Release Artifacts

Large model artifacts are not committed to git. They are published as GitHub
Release assets so the source repository stays lightweight.

Current release:

```text
https://github.com/1152024415-crypto/internvl-cann-vit-poc/releases/tag/v0.1.0-artifacts
```

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

The HarmonyOS runtime validation also needs raw fp32 tensor files. They are not
tracked by git because they are binary release artifacts.

Generate them with:

```powershell
uv run internvl-export-validation-tensors `
  --output-dir artifacts\validation-tensors `
  --case dog=data\test-images\dog.jpg,artifacts\baseline-projector-dog\baseline_output.pt `
  --case cat=data\test-images\cat.jpg,artifacts\baseline-projector-cat\baseline_output.pt
```

Upload these files to the same GitHub Release as the OM:

```text
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
dog.metadata.json
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
cat.metadata.json
```
