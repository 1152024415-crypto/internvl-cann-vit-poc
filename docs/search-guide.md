# Search Guide

Use this file as a keyword map for fast context lookup.

## Main Keywords

`projector`

Current mainline path. Search:

```powershell
rg -n "projector|visual_tokens|mlp1|pixel_shuffle" docs src tests
```

`staticpos`

CANN-targeted export with static position embedding. Search:

```powershell
rg -n "staticpos|static_position_embedding|position embedding|Resize" docs src tests
```

`OMG`

ONNX to OM conversion and CANN Kit tool notes. Search:

```powershell
rg -n "OMG|ONNX to OM|tools_omg|CANN Kit|ResizeConvertFunc" docs scripts
```

`HarmonyOS`

Device-side plan and demo workspace. Search:

```powershell
rg -n "HarmonyOS|NAPI|HIAI_F|OH_NN|RunSync|rawfile" docs demo
```

`release`

GitHub Release artifact workflow. Search:

```powershell
rg -n "Release|artifact|sha256|manifest" docs scripts artifacts/release
```

## Important Files

Current state:

```text
docs/current-status.md
docs/stage-4-vit-projector-chain.md
```

Model/export code:

```text
src/internvl_vit_poc/baseline.py
src/internvl_vit_poc/onnx_export.py
src/internvl_vit_poc/onnx_verify.py
src/internvl_vit_poc/preprocess.py
src/internvl_vit_poc/model_split.py
```

Conversion/publishing scripts:

```text
scripts/convert_onnx_to_om.sh
scripts/convert_wsl_onnx_to_om.ps1
scripts/publish_github_release_artifacts.ps1
```

HarmonyOS template:

```text
demo/entry/src/main/cpp/CMakeLists.txt
demo/entry/src/main/cpp/napi_init.cpp
demo/entry/src/main/ets/pages/Index.ets
```

## Artifact Names

Projector ONNX:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
```

Projector OM:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

Projector metadata:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
```

SHA256 list:

```text
artifacts-manifest.sha256.txt
```

## Verification Commands

Python tests:

```powershell
uv run python -B -m unittest discover -s tests
```

Projector ONNX verification:

```powershell
uv run internvl-onnx-verify `
  --onnx artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos.onnx `
  --image data\test-images\dog.jpg `
  --output-name visual_tokens `
  --compare artifacts\baseline-projector-dog\baseline_output.pt
```

Release asset check:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command `
  "Invoke-RestMethod https://api.github.com/repos/1152024415-crypto/internvl-cann-vit-poc/releases/tags/v0.1.0-artifacts"
```
