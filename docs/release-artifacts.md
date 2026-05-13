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
