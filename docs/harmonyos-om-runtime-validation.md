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

2. Download the release assets listed below from the project release page or the
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

6. Run the Dog single-run validation and record the fields in the Device Results
template.

7. Run the Cat single-run validation and record the fields in the Device Results
template.

8. Run the 20-run Dog/Cat stability validation and record the success counts and
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

## Required Release Assets

Download these files in the yellow-zone environment:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
```

Put them in:

```text
demo/entry/src/main/resources/rawfile/
```

The small `dog.metadata.json` and `cat.metadata.json` files are tracked in the
rawfile directory. They are also safe to replace with freshly generated release
metadata if the validation tensors are regenerated.

The native validator reads these metadata files at runtime and checks that they
match the selected case, expected model file, tensor file names, byte counts, and
initial numeric thresholds. The validator also queries the actual NN runtime
output shape after `RunSync` and reports it in the UI.

## Expected Shapes

```text
input:  pixel_values  [1, 3, 448, 448]   fp32   2408448 bytes
output: visual_tokens [1, 256, 1024]     fp32   1048576 bytes
```

## Pass Criteria

The first run passes when:

```text
OM loads successfully
compilation and executor creation succeed
RunSync returns success
output shape is [1, 256, 1024]
all output values are finite
cosine >= 0.999
mean_abs_diff <= 0.01
```

The 20-run stability test passes when all 20 runs return success and meet the
same numeric thresholds.

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
