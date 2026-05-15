# Yellow-Zone HarmonyOS Validation Runbook

This is the direct runbook for validating the InternVL3.5 ViT + projector OM in
the yellow-zone HarmonyOS environment.

The validation target is:

```text
pixel_values [1, 3, 448, 448]
-> internvl3_5_vit_projector_fp32_opset18_staticpos.om
-> visual_tokens [1, 256, 1024]
```

This does not validate image decoding, resize, normalization, AIPP, or full VLM
text generation.

Official CANN Kit references used for this demo:

```text
Deployment full flow:
https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-whole-deployment-process

Model inference API flow:
https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-model-inference

App integration model flow:
https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-integration-model
```

## Current Asset Status

The app needs this runtime asset in `resources/rawfile`:

```text
demo/entry/src/main/resources/rawfile/internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

Use the yellow-zone-regenerated OM from
`docs/yellow-zone-onnx-to-om-ddk-runbook.md`. Do not treat a blue-zone-generated
Release OM as final proof after the blue-zone/yellow-zone DDK mismatch was
found.

Historical note: the original 2026-05-13 uploaded `.om` was a CPU-only OM from
the OMG log:

```text
partition type NPU:0, CPU:1, GPU:0, ISP:0
```

That old OM must not be used as proof of NPU runtime validation. Yellow-zone
runtime already confirmed it could be read and that `HIAI_F` could be selected,
but `OH_NNCompilation_Build` rejected it during authentication.

Another 2026-05-14 blue-zone-generated replacement OM had stronger host-side
evidence:

```text
release asset = internvl3_5_vit_projector_fp32_opset18_staticpos.om
size = 1236219952 bytes
SHA256 = 8D081689805763B786BE003B5627061DFB9324EDF3DF7DF0226C8F5A9C093FA7
GitHub digest = sha256:8d081689805763b786be003b5627061dfb9324edf3df7df0226c8f5a9c093fa7
source ONNX = artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx
OMG platform = kirin9030
AI_NPUCL lines = 21
CPUCL lines = 0
partition type NPU:0 lines = 0
Linux static validation = pass
pre-check report = success, pass 1096, fail 0
OM JSON input = pixel_values [1, 3, 448, 448]
OM JSON output = Node_Output [1, 256, 1024]
```

That host-side result is useful context, but it is not final because the user
later found that the blue-zone DDK is not the DDK that should produce the phone
validation OM. The current required flow is:

```text
yellow-zone ONNX -> yellow-zone DDK/OMG -> yellow-zone OM -> HarmonyOS device validation
```

The yellow-zone runtime demo also uses these raw fp32 validation tensors:

```text
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
```

These `.bin` files are tracked in git under:

```text
demo/entry/src/main/resources/rawfile/
```

The small metadata files are tracked in the same directory:

```text
demo/entry/src/main/resources/rawfile/dog.metadata.json
demo/entry/src/main/resources/rawfile/cat.metadata.json
```

## Step 0: Check Tracked Validation Tensors

After `git pull`, these files should already exist:

```text
demo/entry/src/main/resources/rawfile/dog_pixel_values_fp32.bin
demo/entry/src/main/resources/rawfile/dog_visual_tokens_fp32.bin
demo/entry/src/main/resources/rawfile/cat_pixel_values_fp32.bin
demo/entry/src/main/resources/rawfile/cat_visual_tokens_fp32.bin
demo/entry/src/main/resources/rawfile/dog.metadata.json
demo/entry/src/main/resources/rawfile/cat.metadata.json
```

Expected binary sizes:

```text
dog_pixel_values_fp32.bin    2408448 bytes
cat_pixel_values_fp32.bin    2408448 bytes
dog_visual_tokens_fp32.bin   1048576 bytes
cat_visual_tokens_fp32.bin   1048576 bytes
```

The tracked metadata contains SHA256 values for the current tensor set. If the
metadata and `.bin` files do not match, the app reports `metadata_mismatch`.

## Step 1: Clone The Repo In Yellow Zone

On the yellow-zone workstation:

```powershell
git clone https://github.com/1152024415-crypto/internvl-cann-vit-poc.git
cd internvl-cann-vit-poc
```

If the repo already exists:

```powershell
git pull origin main
```

## Step 2: Add The OM Runtime Asset

Create the rawfile directory:

```powershell
New-Item -ItemType Directory -Force demo\entry\src\main\resources\rawfile
```

Copy the yellow-zone-regenerated OM into the rawfile directory and name it:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

Put it here:

```text
demo/entry/src/main/resources/rawfile/
```

Do not put the ONNX file in the HarmonyOS app. The device demo loads the `.om`
file only. The `.bin` and `.metadata.json` files should already be present from
git.

If the yellow-zone OM has already been uploaded to GitHub Release, this optional
PowerShell helper can download it. Do not use this helper for an old blue-zone
OM:

```powershell
$repo = "1152024415-crypto/internvl-cann-vit-poc"
$tag = "v0.1.0-artifacts"
$dest = "demo\entry\src\main\resources\rawfile"
New-Item -ItemType Directory -Force $dest | Out-Null

$required = @(
  "internvl3_5_vit_projector_fp32_opset18_staticpos.om"
)

$release = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/tags/$tag"
foreach ($name in $required) {
  $asset = $release.assets | Where-Object { $_.name -eq $name }
  if (-not $asset) {
    Write-Error "Missing Release asset: $name"
    continue
  }
  Invoke-WebRequest -Uri $asset.browser_download_url -OutFile (Join-Path $dest $name)
}
```

After copying the accepted OM, the rawfile directory should contain:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
dog.metadata.json
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
cat.metadata.json
```

Expected binary sizes:

```text
dog_pixel_values_fp32.bin    2408448 bytes
cat_pixel_values_fp32.bin    2408448 bytes
dog_visual_tokens_fp32.bin   1048576 bytes
cat_visual_tokens_fp32.bin   1048576 bytes
```

## Step 3: Open The HarmonyOS Project

Open DevEco Studio.

Open this directory as the project:

```text
internvl-cann-vit-poc/demo
```

Do not open the repository root as the HarmonyOS project.

Let DevEco sync the project. If it asks for SDK configuration, use the yellow-zone
HarmonyOS SDK/CANN-capable setup.

## Step 4: Build And Install

Use a physical HarmonyOS device. Do not use an emulator for this validation.

The demo native build is intentionally limited to:

```text
arm64-v8a
```

This validation targets phone-side NN runtime / NPU execution. The x86_64
emulator path is not a valid NPU validation target and may not ship the same
HiAI/CANN runtime libraries.

Build and run the `entry` module from DevEco Studio.

The app packages these rawfile assets into the HAP:

```text
.om
dog/cat input .bin
dog/cat expected output .bin
dog/cat metadata .json
```

If packaging fails because the `.om` is too large, record the exact error. That
means the next task is packaging strategy, not model accuracy.

Native logs use tag:

```text
InternVLNative
```

Useful load stages:

```text
load_model_read_om_start
load_model_read_om_done
load_model_select_hiai_f
load_model_construct_compilation
hiai_compatibility_check
load_model_set_hiai_build_options
hiai_build_options_done
load_model_build_start
load_model_create_executor
load_model_done
```

`load_model_read_om_done` prints `om_bytes`. `load_model_select_hiai_f`
prints `selected_device`. Build failures include the numeric
`OH_NN_ReturnCode` in the UI error message.

## Step 5: Run Single-Case Validation

Open the app. The title should be:

```text
InternVL OM Validation
```

The case buttons come from native `listTestCases()`. Expected cases:

```text
dog
cat
```

Before loading the large InternVL model, run the official sample OM smoke test:

```text
Tap Official Smoke
```

This loads `official_squeezenet_hiai.om`, which is copied from Huawei's public
CANN Kit C++ sample and tracked in git because it is only about 2.5 MB. The
smoke test uses the same native CANN build setup as the InternVL path, then
creates tensors, fills the input buffer with zeroes, and calls
`OH_NNExecutor_RunSync`. It does not check image classification accuracy.

Expected passing smoke result:

```text
ok: true
device: name=HIAI_F type=ACCELERATOR
model_bytes: 2496459
input_count: 1
output_count: 1
input_bytes: > 0
output_bytes: > 0
```

If `Official Smoke` fails, investigate the yellow-zone SDK/device/app runtime
before debugging the InternVL OM. If `Official Smoke` passes but `Load Model`
fails, the current failure is specific to the InternVL OM or its conversion
settings.

After smoke passes, run the official image classifier:

```text
Tap Guitar or Cup
Tap Official Classify
```

The UI displays the official sample images from `resources/base/media` and the
native side runs their preprocessed 227 x 227 BGR planar inputs:

```text
official_guitar_bgr_227.bin
official_cup_bgr_227.bin
official_labels_caffe.txt
```

A passing classifier run should show:

```text
ok: true
image: official_guitar or official_cup
device: name=HIAI_F type=ACCELERATOR
input_bytes: 154587
output_elements: > 0
top1: ...
top2: ...
top3: ...
```

Use this result only to prove that the official CANN sample can perform real
inference in our app. It is not part of the InternVL accuracy baseline.

First load and compile the OM:

```text
Tap Load Model
Wait until status shows Model loaded
```

This is intentionally separate from `Run Once`. It mirrors the CANN Kit
deployment flow: load OM, select `HIAI_F`, build the compilation, create the
executor, then reuse that executor for inference.

For dog:

```text
Select dog
Tap Run Once
```

For cat:

```text
Select cat
Tap Run Once
```

A passing single run should show:

```text
ok: true
output_elements: 262144
output_shape: [1,256,1024]
output_shape_ok: true
finite: true
cosine: >= 0.999
mean_abs_diff: <= 0.01
```

The `device` line should be recorded exactly. A good result should indicate an
accelerator/NPU device named `HIAI_F`, for example:

```text
name=HIAI_F type=ACCELERATOR
```

If `Load Model` reports `device_selection_failed` and the available device list
does not contain `HIAI_F`, do not count the run as NPU validation. Record it as
a runtime-placement/device-support issue.

## Step 6: Run 20-Run Stability

For dog:

```text
Select dog
Tap Run 20 Stability
```

For cat:

```text
Select cat
Tap Run 20 Stability
```

A passing stability run should show:

```text
ok: true
repeat: 20
success: 20
avg latency recorded
min latency recorded
max latency recorded
```

The current implementation compiles the OM once in `Load Model` and reuses the
same `OH_NNExecutor` for every `Run Once` and `Run 20 Stability` call. The
20-run number is therefore a warm executor stability check, not a repeated model
compile benchmark.

## Step 7: Record Results

Record the results in:

```text
docs/harmonyos-om-runtime-validation.md
```

Fill the `Device Results` section:

```text
Device model:
HarmonyOS version:
CANN Kit / SDK version:
date:

Dog single run:
ok:
device:
latency_ms:
output_shape:
cosine:
mean_abs_diff:
max_abs_diff:

Cat single run:
ok:
device:
latency_ms:
output_shape:
cosine:
mean_abs_diff:
max_abs_diff:

20-run stability:
dog_success_count:
dog_avg_latency_ms:
cat_success_count:
cat_avg_latency_ms:
```

Then commit and push from yellow zone:

```powershell
git add docs\harmonyos-om-runtime-validation.md
git commit -m "Record HarmonyOS OM runtime validation results"
git push origin main
```

## Error Checklist

`missing_artifact`

```text
The app cannot find a rawfile. Check file names and location:
demo/entry/src/main/resources/rawfile/
```

`unable to find library -lhiai_foundation`

```text
The native linker could not find libhiai_foundation.so for the current SDK/ABI.
This is a build/link configuration issue, not an OM conversion issue.

The demo links libneural_network_core.so directly and treats
libhiai_foundation.so as optional because the checked-in native code only calls
OH_NN* APIs from neural_network_core. Keep the build target on arm64-v8a and
run on a physical device.

If DevEco still tries to build x86_64 after git pull, clean the project or
delete stale native build output under demo/entry.cxx and demo/entry/build.
```

`metadata_mismatch`

```text
The tracked metadata does not match the current .bin files. Regenerate tensors,
update metadata, and use matching .bin files.
```

`om_compilation_failed`

```text
The NN runtime could not compile/load the OM. Check that the device and CANN
runtime support this OM. Also confirm the OM file is not corrupted.

If logs contain:

Authentication failed, fail to authenticate model
Authentication failed, input model cannot run by npu

then the OM reached NNRt/CANN, but the NPU backend rejected it during model
authentication. Record the exact `OH_NNCompilation_Build failed code=...`, the
`om_bytes` value, the selected device, and the full CANN/NNRt/DLSA log section.
This is not an ArkUI or NAPI-threading failure.

The native demo now follows the public CANN Kit C++ sample's build setup before
`OH_NNCompilation_Build`:

```text
HMS_HiAICompatibility_CheckFromBuffer
HMS_HiAIOptions_SetBandMode(HIAI_BANDMODE_NORMAL)
HMS_HiAIOptions_SetModelDeviceOrder(HIAI_EXECUTE_DEVICE_NPU)
```

In a fresh run, look for these additional stages:

```text
hiai_compatibility_check compatibility=...
load_model_set_hiai_build_options
hiai_build_options_done band_mode=normal execute_device_order=npu
```

If `hiai_build_options_done` says `skipped=...`, the active SDK/ABI did not
provide `libhiai_foundation.so` plus the CANN Kit headers. If the options are
applied and the same authentication failure remains, compare against the
official SqueezeNet CANN sample on the same phone. If the official sample also
fails, treat it as yellow-zone SDK/device/signing environment. If the official
sample passes, treat our ViT projector OM as the failing component.

For the old 2026-05-14 failure, the matching OMG log shows:

partition type NPU:0, CPU:1, GPU:0, ISP:0

That means the old OM is CPU-only. The current fix is not "blindly reuse the
blue-zone replacement OM"; it is to regenerate the OM in the yellow zone with
the correct DDK, keep the OMG logs, and then rerun this device validation.
```

`device_selection_failed`

```text
The native validator could not find `HIAI_F`, or SetDevice failed after finding
it. Record available device information from the UI/logs.
```

`executor_create_failed`

```text
Compilation succeeded but executor/tensor setup failed. Check SDK/runtime
compatibility first.
```

`run_sync_failed`

```text
OH_NNExecutor_RunSync failed. Keep the logs and record whether the selected
device was accelerator or CPU.
```

`output_size_mismatch`

```text
The runtime output shape was not [1,256,1024]. Record the reported output_shape.
```

`compare_failed`

```text
The runtime returned output, but cosine/mean_abs_diff failed thresholds. Record
cosine, mean_abs_diff, max_abs_diff, and device placement.
```

## Files To Patch If Yellow-Zone SDK Differs

If the yellow-zone SDK has different NN runtime signatures, patch only:

```text
demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp
```

Do not mix SDK compatibility patches with model artifact changes or UI changes.

If rawfile API signatures differ, patch only:

```text
demo/entry/src/main/cpp/validation/rawfile_loader.cpp
```

Keep all other files stable unless the compiler points to them directly.
