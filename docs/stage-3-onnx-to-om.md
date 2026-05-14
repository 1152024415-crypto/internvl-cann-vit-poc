# Stage 3: ONNX to OM

## Current Status

Status: the old ViT + projector OM was CPU-only. A replacement projector OM has
now been generated for Kirin 9030 from a CANN-specific ONNX. Device runtime
validation is still pending.

WSL status from the installation logs:

```text
WSL version: 2.7.3.0
Distro: InternVL-Ubuntu-22.04
Distro WSL version: 2
Ubuntu: 22.04.1 LTS
Default Linux user: root
```

The Codex sandbox runs as a different Windows user, so it cannot directly see the per-user WSL distribution. WSL commands for this stage are run through Windows scripts under the interactive user context and logged to files under `C:\Users\11520`.

Latest environment check:

```text
Log file: C:\Users\11520\wsl-omg-check.log
WSL user: root
Project in WSL: /root/internvl-cann-vit-poc
ONNX copied: yes
Conversion script copied: yes
OMG path: /root/cann-kit/tools/tools_omg/omg
convert script result: success, exit=0
```

Important 2026-05-14 finding: the old `success, exit=0` result only meant OMG
wrote an `.om` file. It did not mean the file could run on the phone NPU. The
old log contained:

```text
dlopen so failed: libai_npucore_itf.so: cannot open shared object file
GetPlatformVersion: Read platform version error
partition type NPU:0, CPU:1, GPU:0, ISP:0
```

That meant the generated OM was CPU-only.

The replacement flow fixed two issues:

```text
1. Install kirin9030 platform plugin under /root/cann-kit/tools/platform/kirin9030.
2. Pass --platform kirin9030 to OMG.
3. Remove the fixed-batch class-token Equal/Where/Expand helper chain from ONNX.
```

## What OM Means

`OM` is Huawei CANN's offline model format.

For this project:

```text
PyTorch model -> ONNX model -> OM model -> CANN Kit runtime on device
```

The CANN conversion ONNX file is:

```text
artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx
```

The pure ViT OM artifact is:

```text
artifacts/om/internvl3_5_vit_fp32_opset18_staticpos.om
```

The replacement ViT + projector OM artifact is:

```text
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.om
size = 1236219952 bytes
SHA256 = 33CA510F80C02C5C990C7050E23F434A6863C94D0D074603E2A29E69D81ADE7B
```

Projector OM details are tracked in:

```text
docs/stage-4-vit-projector-chain.md
```

## Input ONNX

The current ONNX file for CANN conversion is the pure ViT export with fixed
position embedding:

```text
artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx
```

It replaces the first ONNX export:

```text
artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx
```

because the first export contained one `Resize` node from bicubic position
embedding interpolation. OMG rejected that node:

```text
onnx_util.cpp ResizeConvertFunc: mode == "nearest" || mode == "linear" false
```

For the fixed `1 x 3 x 448 x 448` PoC, the patch grid is already `32 x 32`,
which matches the learned position embedding grid. The interpolation is
therefore unnecessary and is bypassed during export.

Current CANN ONNX contract:

```text
input_name = pixel_values
input_shape = [1, 3, 448, 448]
output_name = last_hidden_state
output_shape = [1, 1025, 1024]
opset = 18
dtype = fp32
```

It has already passed:

```text
onnx.checker.check_model = ok
ONNX Runtime vs PyTorch cosine = 0.9999999403953552
ONNX Runtime vs PyTorch mean_abs_diff = 5.648157184623415e-06
Resize op count = 0
```

For the ViT + projector deployment path, use the CANN-specific projector ONNX:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx
```

It is derived from:

```text
artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos.onnx
```

by replacing the fixed batch=1 class-token `Expand/Cast` helper chain with the
existing class embedding initializer and pruning 9 dead nodes. ONNX Runtime
verification remains aligned with the PyTorch projector baseline:

```text
dog cosine = 1.0000001192092896
cat cosine = 1.0000001192092896
```

## Conversion Script

The project has a Linux conversion script:

```text
scripts/convert_onnx_to_om.sh
```

Default command in WSL:

```bash
cd /root/internvl-cann-vit-poc
./scripts/convert_onnx_to_om.sh
```

Equivalent explicit command:

```bash
./scripts/convert_onnx_to_om.sh \
  ./artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx \
  ./artifacts/om/internvl3_5_vit_fp32_opset18_staticpos
```

Internally it runs OMG like this:

```bash
omg \
  --model ./artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx \
  --framework 5 \
  --input_shape "pixel_values:1,3,448,448" \
  --output ./artifacts/om/internvl3_5_vit_fp32_opset18_staticpos \
  --platform kirin9030
```

## OMG Tool Location

The script looks for `omg` in this order:

```text
$OMG_BIN
PATH
$CANN_KIT_HOME/tools/tools_omg/omg
$CANN_HOME/tools/tools_omg/omg
$ASCEND_TOOLKIT_HOME/tools/tools_omg/omg
$HOME/cann-kit/tools/tools_omg/omg
$HOME/CANNKit/tools/tools_omg/omg
recursive search under the roots above
```

If CANN Kit is installed somewhere else, set:

```bash
export OMG_BIN=/absolute/path/to/omg
```

or:

```bash
export CANN_KIT_HOME=/absolute/path/to/cann-kit
```

## Preparation Script

The Windows helper script:

```text
scripts/prepare_wsl_stage3.ps1
```

copies these files into WSL:

```text
/root/internvl-cann-vit-poc/artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx
/root/internvl-cann-vit-poc/artifacts/onnx/internvl3_5_vit_fp32_opset18.onnx.metadata.json
/root/internvl-cann-vit-poc/artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx
/root/internvl-cann-vit-poc/artifacts/onnx/internvl3_5_vit_fp32_opset18_staticpos.onnx.metadata.json
/root/internvl-cann-vit-poc/scripts/convert_onnx_to_om.sh
```

It also checks whether an OMG executable is already available.

The Windows check script:

```text
scripts/check_wsl_omg.ps1
```

runs the Linux diagnostic script:

```text
scripts/check_wsl_omg.sh
```

and writes the result to:

```text
C:\Users\11520\wsl-omg-check.log
```

## Installing CANN Kit Tools Into WSL

Huawei's CANN Kit Tools package is not bundled in this repo. Download the
official CANN Kit / DDK Tools package from Huawei, then import it into WSL with:

```powershell
Start-Process -FilePath powershell.exe -ArgumentList @(
  '-NoProfile',
  '-ExecutionPolicy',
  'Bypass',
  '-File',
  'D:\proj\internvl-cann-vit-poc\scripts\install_cann_tools_to_wsl.ps1',
  '-PackagePath',
  'C:\Users\11520\Downloads\YOUR_CANN_TOOLS_PACKAGE.zip'
) -Verb RunAs -Wait
```

The helper accepts `.zip`, `.tar.gz`, `.tgz`, and `.tar` packages. It imports
the package to:

```text
/root/cann-kit
```

and checks for a file matching:

```text
*/tools_omg/omg
```

After import, rerun:

```powershell
Start-Process -FilePath powershell.exe -ArgumentList @(
  '-NoProfile',
  '-ExecutionPolicy',
  'Bypass',
  '-File',
  'D:\proj\internvl-cann-vit-poc\scripts\check_wsl_omg.ps1'
) -Verb RunAs -Wait
```

The imported package used for this conversion was:

```text
DDK-tools-next-6.0.1.0.zip
SHA256 = 1B2822FB9E5FE7443782915C6F34B4A2CE5C028207E7782514BD93970FF8E48A
```

This package provides the common tools, including OMG. It is not enough by
itself for NPU-targeted OM generation. The target phone platform is Kirin 9030,
so import `kirin9030-plugin-6.0.1.0.zip`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\install_cann_platform_plugin_to_wsl.ps1 `
  -PluginPackagePath C:\Users\11520\Downloads\kirin9030-plugin-6.0.1.0.zip
```

Expected plugin SHA256:

```text
3B32EFFC5AF9804628CB9287E88CC28ED381877ADB15DD85BF8D66E3BE805251
```

The plugin is installed into:

```text
/root/cann-kit/tools/platform/<kirin...>
```

## Copying OM Back To Windows

The Windows helper script:

```text
scripts/copy_wsl_om_to_windows.ps1
```

copies:

```text
/root/internvl-cann-vit-poc/artifacts/om/internvl3_5_vit_fp32_opset18_staticpos.om
```

to:

```text
D:\proj\internvl-cann-vit-poc\artifacts\om\internvl3_5_vit_fp32_opset18_staticpos.om
```

and writes the log to:

```text
C:\Users\11520\wsl-om-copy.log
```

## Conversion Notes

The first OMG run failed on the original ONNX because CANN Kit rejected bicubic
`Resize`. The static-position ONNX removed that op and converted successfully.

The old successful OMG log still contained platform warnings:

```text
GetPlatformVersion: Read platform version error
partition type NPU:0, CPU:1, GPU:0, ISP:0
```

The replacement OMG log does not print a partition summary, but it does show:

```text
OMG generate offline model success = 1
AI_NPUCL lines = 21
CPUCL lines = 0
partition type NPU:0 lines = 0
```

This is host-side evidence that OMG used the Kirin 9030 NPU path instead of the
old CPUCL path. It is not final proof of device runtime support; the final proof
is still `OH_NNCompilation_Build` and `OH_NNExecutor_RunSync` on the HarmonyOS
device.

## Expected Failure Modes After OM Generation

If the generated OM fails on device, inspect support for these ONNX-derived ops first:

```text
LayerNormalization
Softmax
MatMul
Conv
```

`Resize` was the first actual blocker and has been removed from the static
position embedding ONNX export. The second blocker was the fixed-batch
class-token `Equal`/`Where`/`Expand` helper chain; it is removed by
`internvl-simplify-onnx-for-cann`.
