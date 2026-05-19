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

Read `docs/yellow-zone-agent-handoff.md` first when working in or handing off
to the yellow-zone environment. Read `docs/stage-4-vit-projector-chain.md` for
the model chain details.

Verified:

```text
PyTorch projector baseline
ONNX projector inference vs PyTorch baseline
blue-zone ONNX projector -> OM conversion with CANN Kit OMG, host-side only
FP32 ONNX -> yellow-zone DDK OMG -> CPU-only OM (588 ops unsupported by NPUCL)
FP32 ONNX -> dopt_onnx_py3 INT8 quantization -> INT8 ONNX (Quantize model success)
```

Not verified yet:

```text
INT8 ONNX + --compress_conf -> OMG -> OM (NPU placement check)
OM runtime inference on HarmonyOS device
NPU/HIAI_F placement
latency, memory, cold start, 20-run stability
```

Key finding: FP32 ONNX cannot go NPU on Kirin 9030 with current DDK. Must use
INT8 quantization path (`dopt_onnx_py3`) which changes op patterns to ones NPU
supports natively. See `docs/stage-5-int8-quantization-runbook.md`.

## Important Artifacts

Large artifacts are not committed to git. They live in GitHub Releases.
See `docs/release-artifacts.md`.

Small metadata files are tracked:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx.metadata.json
artifacts/release/artifacts-manifest.sha256.txt
demo/entry/src/main/resources/rawfile/dog.metadata.json
demo/entry/src/main/resources/rawfile/cat.metadata.json
```

Validation `.bin` tensors are also tracked because they are small enough for git
and make yellow-zone validation simpler:

```text
demo/entry/src/main/resources/rawfile/dog_pixel_values_fp32.bin
demo/entry/src/main/resources/rawfile/dog_visual_tokens_fp32.bin
demo/entry/src/main/resources/rawfile/cat_pixel_values_fp32.bin
demo/entry/src/main/resources/rawfile/cat_visual_tokens_fp32.bin
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
docs/yellow-zone-agent-handoff.md     first document for yellow-zone AI handoff
docs/stage-1-model-split-baseline.md  model split and PyTorch baselines
docs/stage-2-onnx-export.md           ONNX export and ONNX Runtime checks
docs/stage-3-onnx-to-om.md            OMG conversion and OM notes
docs/stage-4-vit-projector-chain.md   current ViT + projector chain
docs/stage-5-int8-quantization-runbook.md INT8 quantization + OMG with compress_conf
docs/release-artifacts.md             GitHub Release asset workflow
docs/yellow-zone-onnx-to-om-ddk-runbook.md yellow-zone DDK OM regeneration guide
docs/yellow-zone-validation-runbook.md yellow-zone device validation runbook
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

Do not require the Kirin 9030 platform package on the grounds of custom-operator
development. This project writes no custom operators. Treat `--platform kirin9030`
as one target-specific OMG conversion choice: if used, the matching
platform package must be installed; if omitted by the official yellow-zone DDK
flow, judge the result by logs and phone-side validation.

## HarmonyOS Demo

`demo/` is the HarmonyOS Native C++ workspace. Generated DevEco/Hvigor folders
must stay out of git.

Current demo state: Native C++ validation demo with NAPI methods:

```text
listTestCases()
loadModel(resourceManager)
unloadModel()
runOfficialSmoke(resourceManager)
runOfficialClassification(resourceManager, imageName)
runOnce(resourceManager, caseName)
runStability(resourceManager, caseName, repeatCount)
```

`loadModel` follows the CANN Kit endpoint deployment flow: read `.om` from
rawfile, select `HIAI_F`, call `OH_NNCompilation_ConstructWithOfflineModelBuffer`,
`OH_NNCompilation_Build`, then create and cache one `OH_NNExecutor`. `runOnce`
and `runStability` reuse that executor instead of recompiling the OM.

The native target is `arm64-v8a` only. This PoC validates physical-phone NN
runtime / NPU behavior, not x86_64 emulator behavior.

The demo links `libneural_network_core.so` for `OH_NN*` APIs. Treat
`libhiai_foundation.so` as optional if the SDK exposes it for the active ABI.

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
