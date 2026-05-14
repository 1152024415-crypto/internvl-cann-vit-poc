# HarmonyOS OM Runtime Validation

This stage validates the runtime path for the InternVL3.5 ViT + projector OM:

```text
pixel_values [1, 3, 448, 448]
-> OM runtime
-> visual_tokens [1, 256, 1024]
```

It does not validate image preprocessing or full VLM text generation.

## Blue-Zone Scope

Blue-zone execution did not compile the HarmonyOS app and did not run validation
on a physical HarmonyOS device. This document is the yellow-zone manual
checklist to run after the release assets are available in an environment with
DevEco Studio, the HarmonyOS SDK/CANN Kit, and a compatible device.

## Yellow-Zone Manual Checklist

1. Pull the repository on the yellow-zone workstation:

```text
git clone <repo-url>
cd internvl-cann-vit-poc
```

If the repository already exists, update it to the intended commit or release
branch before opening DevEco Studio.

2. Download the `.om` asset listed below from the project release page or the
approved artifact location.

3. Place every downloaded asset in:

```text
demo/entry/src/main/resources/rawfile/
```

The file names must match the names in this document because the demo loads
them as rawfile resources.

4. Open the HarmonyOS project in DevEco Studio by selecting the `demo/`
directory, not the repository root.

5. Build and install the demo on a physical HarmonyOS device with the required
SDK and CANN Kit/runtime support. Do not record simulator or emulator results
for this validation.

6. Tap `Load Model` and wait for `Model loaded`. This performs the CANN Kit
runtime setup once: load the OM, select `HIAI_F`, build the compilation, and
create a reusable executor.

7. Run the Dog single-run validation and record the fields in the Device Results
template.

8. Run the Cat single-run validation and record the fields in the Device Results
template.

9. Run the 20-run Dog/Cat stability validation and record the success counts and
average latencies in the Device Results template.

During blue-zone review, the NN runtime adapter was checked against the local
DevEco header:

```text
neural_network_runtime/neural_network_core.h
```

The checked-in adapter uses the current `OH_NNCompilation_*`, `OH_NNTensor_*`,
and `OH_NNExecutor_RunSync` signatures from that header. If the yellow-zone SDK
still differs, isolate the compatibility patch to:

```text
demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp
```

Do not mix NN API compatibility edits with release asset changes or unrelated
demo changes.

## Required Runtime Assets

Download this file in the yellow-zone environment:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

Put it in:

```text
demo/entry/src/main/resources/rawfile/
```

The validation `.bin` files and small metadata files are tracked in git under
the same rawfile directory:

```text
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
dog.metadata.json
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
cat.metadata.json
```

They are safe to replace with freshly generated tensors and metadata if the
validation set is regenerated, but the `.bin` and `.metadata.json` files must
always match each other.

The native validator reads these metadata files at runtime and checks that they
match the selected case, expected model file, tensor file names, byte counts, and
initial numeric thresholds. The validator also queries the actual NN runtime
output shape after `RunSync` and reports it in the UI.

## Build Notes

The native build is limited to `arm64-v8a` because this validation is meant for
a physical HarmonyOS phone with NN runtime / NPU support. Do not use x86_64
emulator results as OM runtime validation results.

The official CANN Kit C++ sample links `libhiai_foundation.so` and sets HiAI
build options before `OH_NNCompilation_Build`. This demo follows that pattern
when the SDK exposes the library and headers for the active ABI:

```text
HMS_HiAICompatibility_CheckFromBuffer
HMS_HiAIOptions_SetBandMode(HIAI_BANDMODE_NORMAL)
HMS_HiAIOptions_SetModelDeviceOrder(HIAI_EXECUTE_DEVICE_NPU)
```

If `libhiai_foundation.so` or the CANN Kit headers are unavailable for an ABI,
the native code logs that the HiAI option step was skipped. Do not use emulator
or x86_64 results as NPU validation.

## Expected Shapes

```text
input:  pixel_values  [1, 3, 448, 448]   fp32   2408448 bytes
output: visual_tokens [1, 256, 1024]     fp32   1048576 bytes
```

## Pass Criteria

The first run passes when:

```text
Load Model succeeds
device is HIAI_F
compilation and executor creation succeed once
RunSync returns success
output shape is [1, 256, 1024]
all output values are finite
cosine >= 0.999
mean_abs_diff <= 0.01
```

The 20-run stability test passes when all 20 runs reuse the loaded executor,
return success, and meet the same numeric thresholds.

## Device Results

Device:

```text
model:
HarmonyOS version:
CANN Kit / SDK version:
date:
```

Dog single run:

```text
ok:
device:
latency_ms:
cosine:
mean_abs_diff:
max_abs_diff:
```

Cat single run:

```text
ok:
device:
latency_ms:
cosine:
mean_abs_diff:
max_abs_diff:
```

20-run stability:

```text
dog_success_count:
dog_avg_latency_ms:
cat_success_count:
cat_avg_latency_ms:
```
