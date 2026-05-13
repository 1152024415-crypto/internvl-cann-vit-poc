# HarmonyOS OM Runtime Validation Design

Date: 2026-05-13

## Purpose

Validate that the InternVL3.5 ViT + projector OM can run on a HarmonyOS device
in the yellow-zone environment.

The validated model chain is:

```text
pixel_values [1, 3, 448, 448]
-> vision_model
-> pixel_shuffle
-> mlp1
-> visual_tokens [1, 256, 1024]
```

This is not full VLM text generation. It only validates the visual embedding
path that will later feed the language model.

## Current Baseline

Already verified in the blue-zone environment:

```text
PyTorch projector baseline
ONNX projector inference vs PyTorch baseline
ONNX projector -> OM conversion by CANN Kit OMG
GitHub Release upload for ONNX/OM
HarmonyOS Native C++ demo scaffold
```

Not verified yet:

```text
HarmonyOS device loads the OM
Runtime selects HIAI_F / NPU
OH_NNExecutor_RunSync succeeds
OM output matches PyTorch/ONNX baseline
20-run stability and latency are acceptable
```

## Non-Goals

The first device validation will not include:

```text
JPG/PNG decoding on device
resize/RGB/normalize preprocessing on device
AIPP preprocessing offload
dynamic image sizes
multi-patch dynamic shape
full LLM text generation
int8/fp16 quantization
```

These are deferred because they add extra failure causes. The first test should
prove the OM runtime path itself before mixing in image preprocessing.

## Options Considered

### Option A: OM Smoke Test With Dummy Input

Use all-zero or random `pixel_values` generated in C++.

Pros:

- Smallest demo code.
- No extra artifact generation.
- Quickly proves that OM loading and `RunSync` do not crash.

Cons:

- Does not prove the result is numerically correct.
- If output is wrong but shape is right, the test may still look successful.

### Option B: Raw Tensor Numeric Validation

Generate raw fp32 input and expected output files in Python:

```text
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
```

The HarmonyOS demo reads the `.om` and `.bin` files from `resources/rawfile`,
runs the model, and compares the output tensor against the expected tensor.

Pros:

- Separates model runtime validation from image preprocessing validation.
- Gives a clear pass/fail signal using shape, max error, mean error, and cosine.
- Reuses the existing PyTorch/ONNX verified baseline.
- Easy to reproduce in blue zone with a Python script.

Cons:

- Requires extra `.bin` artifacts.
- The `.bin` files are test tensors, not model/compiler outputs.

### Option C: Full Image-To-Output Validation

The HarmonyOS demo reads an image, preprocesses it to `pixel_values`, runs OM,
and compares visual tokens.

Pros:

- Closest to the final product path.
- Exercises image decode and preprocessing code.

Cons:

- Harder to debug because preprocessing and OM runtime can fail together.
- More C++/ArkTS code before we know the OM itself runs on device.
- More sensitive to small resize or normalization differences.

## Selected Approach

Use Option B first.

The first milestone is:

```text
Python-generated fp32 tensor input
-> HarmonyOS Native C++ reads tensor
-> OM runtime executes ViT + projector
-> C++ compares output against Python-generated reference
```

This gives the strongest early signal with the fewest moving parts. Once this
passes, image preprocessing can be added as a separate milestone.

## Blue-Zone Responsibilities

The blue-zone repository provides:

```text
1. Python script to export raw fp32 validation tensors.
2. Metadata JSON describing tensor names, shapes, dtype, byte count, SHA256.
3. HarmonyOS Native C++ code for loading OM and raw tensors.
4. ArkTS UI to trigger one run and a 20-run stability test.
5. Documentation for yellow-zone setup and expected results.
```

Large binary artifacts stay out of git:

```text
*.om
*.onnx
*.pt
*.bin
```

They can be attached to GitHub Releases for yellow-zone download.

Small metadata files can be tracked in git:

```text
*.metadata.json
*.sha256.txt
```

## Yellow-Zone Responsibilities

In the yellow-zone environment, the user should:

```text
1. Pull the GitHub repo.
2. Download the OM and validation `.bin` artifacts from GitHub Release.
3. Put OM and `.bin` files under the demo rawfile resource directory.
4. Open the demo in DevEco Studio.
5. Build and install on a compatible HarmonyOS device.
6. Run the validation screen and record results.
```

The expected runtime path is:

```text
read rawfile OM bytes
-> OH_NNCompilation_ConstructWithOfflineModelBuffer
-> select HIAI_F / NPU device when available
-> build compilation
-> create executor
-> bind input tensor pixel_values
-> bind output tensor visual_tokens
-> OH_NNExecutor_RunSync
-> compare output
```

## Demo Behavior

The first demo screen should be utilitarian, not a product UI.

It should show:

```text
model file present/missing
test case selected: dog or cat
runtime device selection result
single-run status
output shape
max_abs_diff
mean_abs_diff
cosine similarity
single-run latency
20-run pass/fail and average latency
last error message
```

The existing sample NAPI `add` function should be replaced by a small validation
API surface:

```text
listTestCases()
runOnce(testCaseName)
runStability(testCaseName, repeatCount)
```

The NAPI return value should be structured data, not only log text, so ArkTS can
display the result clearly.

## Validation Data Format

The raw tensor `.bin` files are little-endian contiguous fp32 arrays.

Input:

```text
name = pixel_values
shape = [1, 3, 448, 448]
dtype = fp32
layout = BCHW
elements = 602112
bytes = 2408448
```

Output:

```text
name = visual_tokens
shape = [1, 256, 1024]
dtype = fp32
elements = 262144
bytes = 1048576
```

Each test case should have a metadata JSON file that records:

```text
source_image
model_artifact
input_bin
expected_output_bin
input_shape
output_shape
dtype
sha256
baseline_source
```

## Pass Criteria

A single test case passes if:

```text
OM file is loaded successfully
compilation and executor creation succeed
RunSync returns success
output shape is [1, 256, 1024]
all output values are finite
cosine similarity is close to 1.0
max_abs_diff is within an agreed tolerance
mean_abs_diff is small
```

Initial numeric tolerance should be permissive until real device behavior is
observed:

```text
cosine >= 0.999
mean_abs_diff <= 0.01
max_abs_diff recorded, not hard-failed at first
```

After one real device run, tighten the thresholds based on observed fp32
runtime differences.

The stability test passes if 20 consecutive runs:

```text
return success
produce finite output
stay within numeric thresholds
do not leak obvious memory
record average/min/max latency
```

## Error Handling

The C++ layer should return explicit error stages:

```text
missing_artifact
read_rawfile_failed
om_compilation_failed
device_selection_failed
executor_create_failed
input_size_mismatch
output_size_mismatch
run_sync_failed
compare_failed
```

Device selection failure should not hide other information. If the API allows a
CPU fallback, the demo should report the actual device path instead of claiming
NPU success.

## Risks

The main risks are:

```text
The converted OM may contain ops unsupported by the phone runtime.
The runtime may place part or all of the graph on CPU.
The 1.2 GB OM may exceed memory or packaging limits.
Huawei SDK headers or libraries may differ from public examples.
Resource packaging may not accept very large rawfiles without project setting changes.
```

Mitigations:

```text
Keep first input static: [1, 3, 448, 448].
Start with fp32 and no quantization.
Report actual device placement when possible.
Keep tensor validation separate from preprocessing.
Track large artifacts in GitHub Release, not git.
```

## Future Milestones

After raw tensor validation passes:

```text
1. Add device-side image preprocessing and compare against Python pixel_values.
2. Add full image-to-visual_tokens validation.
3. Measure cold start and memory more carefully.
4. Try fp16 export/conversion if fp32 is too large or slow.
5. Consider int8 quantization only after fp32/fp16 correctness is established.
6. Revisit dynamic patch count after the static single-patch path is stable.
```
