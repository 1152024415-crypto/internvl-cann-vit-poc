# HarmonyOS OM Runtime Validation Implementation Plan

Status note, 2026-05-14: this plan is historical. The checked-in implementation
has since been adjusted to match the CANN Kit deployment lifecycle more closely:
`loadModel(resourceManager)` loads/builds the OM once, selects `HIAI_F` by device
name/id, creates a cached `OH_NNExecutor`, and `runOnce` / `runStability` reuse
that executor through Promise-based NAPI calls. Use
`docs/yellow-zone-validation-runbook.md` and `docs/current-status.md` as the
current source of truth.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible yellow-zone validation path where HarmonyOS Native C++ loads the InternVL3.5 ViT + projector OM, runs raw fp32 tensor input, and compares the output against a Python-generated reference.

**Architecture:** Blue-zone Python exports little-endian fp32 `.bin` test tensors and metadata from already verified image preprocessing and baseline outputs. Yellow-zone HarmonyOS demo reads `.om` and `.bin` files from `resources/rawfile`, executes the OM through the NN runtime, computes output metrics in C++, and returns structured results to ArkTS.

**Tech Stack:** Python 3.12 managed by `uv`, NumPy, PyTorch tensor loading, HarmonyOS ArkTS, Native C++ NAPI, HarmonyOS rawfile API, HarmonyOS NN runtime / CANN Kit offline model APIs.

---

## Post-Implementation Note

During centralized review, the Task 6 NN runtime adapter was corrected against
the local DevEco `neural_network_runtime/neural_network_core.h` header. The
checked-in implementation uses pointer-returning offline compilation and
executor construction APIs, device IDs from `OH_NNDevice_GetAllDevicesID`, and
`NN_Tensor*` arrays passed directly to `OH_NNExecutor_RunSync`. The source code
is authoritative if it differs from the earlier Task 6 draft snippet below.

## Scope Check

This plan implements one coherent milestone: raw tensor validation for the existing static fp32 ViT + projector OM.

The plan does not implement image decode, resize, RGB conversion, normalization, AIPP, dynamic patch count, fp16 conversion, int8 quantization, or full VLM text generation.

## File Structure

Create or modify these files:

```text
src/internvl_vit_poc/validation_tensors.py
tests/test_validation_tensors.py
pyproject.toml
.gitignore
docs/harmonyos-om-runtime-validation.md
docs/release-artifacts.md
demo/entry/src/main/resources/rawfile/README.md
demo/entry/src/main/cpp/validation/validation_constants.h
demo/entry/src/main/cpp/validation/tensor_metrics.h
demo/entry/src/main/cpp/validation/tensor_metrics.cpp
demo/entry/src/main/cpp/validation/rawfile_loader.h
demo/entry/src/main/cpp/validation/rawfile_loader.cpp
demo/entry/src/main/cpp/validation/nn_runtime_validator.h
demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp
demo/entry/src/main/cpp/validation/napi_result.h
demo/entry/src/main/cpp/validation/napi_result.cpp
demo/entry/src/main/cpp/napi_init.cpp
demo/entry/src/main/cpp/CMakeLists.txt
demo/entry/src/main/cpp/types/libentry/Index.d.ts
demo/entry/src/main/ets/pages/Index.ets
```

Responsibilities:

```text
validation_tensors.py
  Export raw fp32 input/output tensors and metadata for one or more test cases.

tensor_metrics.*
  Compare fp32 output buffers with shape checks, finite checks, max_abs_diff,
  mean_abs_diff, and cosine similarity.

rawfile_loader.*
  Read `.om`, `.bin`, and `.metadata.json` files from HarmonyOS rawfile resources.

nn_runtime_validator.*
  Own the OM execution path: load model bytes, compile offline model, bind input
  and output buffers, call RunSync, and return C++ validation data.

napi_result.*
  Convert C++ validation structs to NAPI JavaScript objects.

napi_init.cpp
  Expose listTestCases(), runOnce(), and runStability() to ArkTS.

Index.ets
  Display artifact presence, runtime status, output metrics, latency, and errors.
```

## Task 1: Python Validation Tensor Exporter

**Files:**

- Create: `src/internvl_vit_poc/validation_tensors.py`
- Create: `tests/test_validation_tensors.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing tests for raw fp32 export**

Create `tests/test_validation_tensors.py`:

```python
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class ValidationTensorTests(unittest.TestCase):
    def test_write_fp32_bin_writes_little_endian_float32(self):
        from internvl_vit_poc.validation_tensors import write_fp32_bin

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tensor.bin"
            tensor = np.array([[1.0, 2.5, -3.0]], dtype=np.float64)

            info = write_fp32_bin(path, tensor)

            self.assertEqual(info["path"], str(path))
            self.assertEqual(info["shape"], [1, 3])
            self.assertEqual(info["dtype"], "float32")
            self.assertEqual(info["byte_count"], 12)
            loaded = np.fromfile(path, dtype="<f4")
            np.testing.assert_allclose(loaded, np.array([1.0, 2.5, -3.0], dtype=np.float32))

    def test_build_case_metadata_records_files_and_sha256(self):
        from internvl_vit_poc.validation_tensors import build_case_metadata, write_fp32_bin

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_info = write_fp32_bin(root / "dog_pixel_values_fp32.bin", np.zeros((1, 3), dtype=np.float32))
            output_info = write_fp32_bin(root / "dog_visual_tokens_fp32.bin", np.ones((1, 2), dtype=np.float32))

            metadata = build_case_metadata(
                case_name="dog",
                source_image="data/test-images/dog.jpg",
                model_artifact="internvl3_5_vit_projector_fp32_opset18_staticpos.om",
                baseline_source="artifacts/baseline-projector-dog/baseline_output.pt",
                input_info=input_info,
                output_info=output_info,
            )

            self.assertEqual(metadata["case_name"], "dog")
            self.assertEqual(metadata["input"]["shape"], [1, 3])
            self.assertEqual(metadata["expected_output"]["shape"], [1, 2])
            self.assertEqual(len(metadata["input"]["sha256"]), 64)
            self.assertEqual(len(metadata["expected_output"]["sha256"]), 64)

    def test_save_case_metadata_writes_json(self):
        from internvl_vit_poc.validation_tensors import save_case_metadata

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dog.metadata.json"
            metadata = {"case_name": "dog", "input": {"shape": [1, 3]}}

            save_case_metadata(path, metadata)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), metadata)
```

- [ ] **Step 2: Run the new test file and confirm it fails**

Run:

```powershell
uv run python -B -m unittest tests.test_validation_tensors
```

Expected:

```text
ModuleNotFoundError: No module named 'internvl_vit_poc.validation_tensors'
```

- [ ] **Step 3: Implement validation tensor utilities and CLI**

Create `src/internvl_vit_poc/validation_tensors.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values

INPUT_SHAPE = [1, 3, 448, 448]
OUTPUT_SHAPE = [1, 256, 1024]
MODEL_ARTIFACT = "internvl3_5_vit_projector_fp32_opset18_staticpos.om"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_fp32_bin(path: str | Path, tensor: Any) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(tensor, dtype="<f4")
    array.tofile(path)
    return {
        "path": str(path),
        "shape": [int(dim) for dim in array.shape],
        "dtype": "float32",
        "byte_count": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def load_reference_tensor(path: str | Path) -> np.ndarray:
    import torch

    loaded = torch.load(path, map_location="cpu")
    if hasattr(loaded, "detach"):
        loaded = loaded.detach().cpu().numpy()
    return np.asarray(loaded, dtype=np.float32)


def build_case_metadata(
    *,
    case_name: str,
    source_image: str,
    model_artifact: str,
    baseline_source: str,
    input_info: dict[str, Any],
    output_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "case_name": case_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_image": source_image,
        "model_artifact": model_artifact,
        "baseline_source": baseline_source,
        "input": input_info,
        "expected_output": output_info,
        "thresholds": {
            "cosine_min": 0.999,
            "mean_abs_diff_max": 0.01,
        },
    }


def save_case_metadata(path: str | Path, metadata: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def export_case(
    *,
    case_name: str,
    image_path: str | Path,
    expected_output_path: str | Path,
    output_dir: str | Path,
    model_artifact: str = MODEL_ARTIFACT,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    pixel_values = load_static_pixel_values(image_path, STATIC_IMAGE_SIZE).numpy().astype(np.float32)
    expected_output = load_reference_tensor(expected_output_path)

    if list(pixel_values.shape) != INPUT_SHAPE:
        raise ValueError(f"input shape mismatch: got {list(pixel_values.shape)}, expected {INPUT_SHAPE}")
    if list(expected_output.shape) != OUTPUT_SHAPE:
        raise ValueError(f"output shape mismatch: got {list(expected_output.shape)}, expected {OUTPUT_SHAPE}")

    input_info = write_fp32_bin(output_dir / f"{case_name}_pixel_values_fp32.bin", pixel_values)
    output_info = write_fp32_bin(output_dir / f"{case_name}_visual_tokens_fp32.bin", expected_output)
    metadata = build_case_metadata(
        case_name=case_name,
        source_image=str(image_path),
        model_artifact=model_artifact,
        baseline_source=str(expected_output_path),
        input_info=input_info,
        output_info=output_info,
    )
    save_case_metadata(output_dir / f"{case_name}.metadata.json", metadata)
    return metadata


def parse_case(value: str) -> tuple[str, str, str]:
    parts = value.split("=", 1)
    if len(parts) != 2 or not parts[0]:
        raise argparse.ArgumentTypeError("case must be name=image_path,baseline_output_path")
    paths = parts[1].split(",", 1)
    if len(paths) != 2 or not paths[0] or not paths[1]:
        raise argparse.ArgumentTypeError("case must be name=image_path,baseline_output_path")
    return parts[0], paths[0], paths[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw fp32 validation tensors for HarmonyOS OM tests.")
    parser.add_argument("--output-dir", required=True, help="Directory for .bin and metadata outputs.")
    parser.add_argument(
        "--case",
        action="append",
        type=parse_case,
        required=True,
        help="Test case in the form name=image_path,baseline_output.pt",
    )
    parser.add_argument("--model-artifact", default=MODEL_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [
        export_case(
            case_name=name,
            image_path=image_path,
            expected_output_path=expected_output_path,
            output_dir=args.output_dir,
            model_artifact=args.model_artifact,
        )
        for name, image_path, expected_output_path in args.case
    ]
    print(json.dumps({"cases": results}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register the CLI and ignore raw binary tensors**

Modify `pyproject.toml` script section:

```toml
[project.scripts]
internvl-vit-baseline = "internvl_vit_poc.baseline:main"
internvl-export-onnx = "internvl_vit_poc.onnx_export:main"
internvl-onnx-verify = "internvl_vit_poc.onnx_verify:main"
internvl-export-validation-tensors = "internvl_vit_poc.validation_tensors:main"
```

Modify `.gitignore`:

```text
*.bin
```

- [ ] **Step 5: Run the tests**

Run:

```powershell
uv run python -B -m unittest tests.test_validation_tensors
```

Expected:

```text
Ran 3 tests
OK
```

- [ ] **Step 6: Run the full Python test suite**

Run:

```powershell
uv run python -B -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit Task 1**

Run:

```powershell
git add .gitignore pyproject.toml src/internvl_vit_poc/validation_tensors.py tests/test_validation_tensors.py
git commit -m "Add validation tensor exporter"
```

## Task 2: Generate Local Validation Artifacts

**Files:**

- Generate ignored files under: `artifacts/validation-tensors/`
- Modify: `docs/harmonyos-om-runtime-validation.md`
- Modify: `docs/release-artifacts.md`

- [ ] **Step 1: Export dog and cat validation tensors**

Run:

```powershell
uv run internvl-export-validation-tensors `
  --output-dir artifacts\validation-tensors `
  --case dog=data\test-images\dog.jpg,artifacts\baseline-projector-dog\baseline_output.pt `
  --case cat=data\test-images\cat.jpg,artifacts\baseline-projector-cat\baseline_output.pt
```

Expected output includes two cases:

```text
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
dog.metadata.json
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
cat.metadata.json
```

- [ ] **Step 2: Verify artifact sizes**

Run:

```powershell
Get-ChildItem artifacts\validation-tensors | Select-Object Name,Length
```

Expected sizes for each case:

```text
*_pixel_values_fp32.bin    2408448
*_visual_tokens_fp32.bin   1048576
*.metadata.json            small JSON file
```

- [ ] **Step 3: Create the yellow-zone validation doc**

Create `docs/harmonyos-om-runtime-validation.md`:

```markdown
# HarmonyOS OM Runtime Validation

This stage validates the runtime path for the InternVL3.5 ViT + projector OM:

```text
pixel_values [1, 3, 448, 448]
-> OM runtime
-> visual_tokens [1, 256, 1024]
```

It does not validate image preprocessing or full VLM text generation.

## Required Release Assets

Download these files in the yellow-zone environment:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
dog.metadata.json
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
cat.metadata.json
```

Put them in:

```text
demo/entry/src/main/resources/rawfile/
```

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
```

- [ ] **Step 4: Update release artifact docs**

Append this section to `docs/release-artifacts.md`:

```markdown
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
```

- [ ] **Step 5: Run docs and artifact checks**

Run:

```powershell
Test-Path artifacts\validation-tensors\dog_pixel_values_fp32.bin
Test-Path artifacts\validation-tensors\dog_visual_tokens_fp32.bin
Test-Path artifacts\validation-tensors\cat_pixel_values_fp32.bin
Test-Path artifacts\validation-tensors\cat_visual_tokens_fp32.bin
```

Expected:

```text
True
True
True
True
```

- [ ] **Step 6: Commit Task 2**

Run:

```powershell
git add docs/harmonyos-om-runtime-validation.md docs/release-artifacts.md
git commit -m "Document HarmonyOS validation tensors"
```

## Task 3: Rawfile Resource Contract

**Files:**

- Create: `demo/entry/src/main/resources/rawfile/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Add the rawfile directory contract**

Create `demo/entry/src/main/resources/rawfile/README.md`:

```markdown
# Runtime Validation Rawfiles

Put release artifacts here before building the HarmonyOS demo in the yellow-zone
environment.

Required files:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
dog_pixel_values_fp32.bin
dog_visual_tokens_fp32.bin
dog.metadata.json
cat_pixel_values_fp32.bin
cat_visual_tokens_fp32.bin
cat.metadata.json
```

The `.om` and `.bin` files are large binary artifacts and must not be committed.
The `.metadata.json` files may be committed only when they are small and useful
for review.
```

- [ ] **Step 2: Keep rawfile release payloads out of git**

Add these lines to `.gitignore`:

```text
demo/entry/src/main/resources/rawfile/*.om
demo/entry/src/main/resources/rawfile/*.bin
```

- [ ] **Step 3: Check ignored artifact behavior**

Run:

```powershell
git check-ignore demo/entry/src/main/resources/rawfile/dog_pixel_values_fp32.bin
git check-ignore demo/entry/src/main/resources/rawfile/internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

Expected:

```text
demo/entry/src/main/resources/rawfile/dog_pixel_values_fp32.bin
demo/entry/src/main/resources/rawfile/internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

- [ ] **Step 4: Commit Task 3**

Run:

```powershell
git add .gitignore demo/entry/src/main/resources/rawfile/README.md
git commit -m "Add HarmonyOS rawfile artifact contract"
```

## Task 4: C++ Tensor Metrics Core

**Files:**

- Create: `demo/entry/src/main/cpp/validation/validation_constants.h`
- Create: `demo/entry/src/main/cpp/validation/tensor_metrics.h`
- Create: `demo/entry/src/main/cpp/validation/tensor_metrics.cpp`
- Modify: `demo/entry/src/main/cpp/CMakeLists.txt`

- [ ] **Step 1: Add validation constants**

Create `demo/entry/src/main/cpp/validation/validation_constants.h`:

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

namespace internvl {

constexpr const char* kModelFile = "internvl3_5_vit_projector_fp32_opset18_staticpos.om";

constexpr std::size_t kInputN = 1;
constexpr std::size_t kInputC = 3;
constexpr std::size_t kInputH = 448;
constexpr std::size_t kInputW = 448;
constexpr std::size_t kInputElementCount = kInputN * kInputC * kInputH * kInputW;
constexpr std::size_t kInputByteCount = kInputElementCount * sizeof(float);

constexpr std::size_t kOutputN = 1;
constexpr std::size_t kOutputTokenCount = 256;
constexpr std::size_t kOutputHiddenSize = 1024;
constexpr std::size_t kOutputElementCount = kOutputN * kOutputTokenCount * kOutputHiddenSize;
constexpr std::size_t kOutputByteCount = kOutputElementCount * sizeof(float);

constexpr double kCosineMin = 0.999;
constexpr double kMeanAbsDiffMax = 0.01;

struct TestCase {
    const char* name;
    const char* inputFile;
    const char* expectedOutputFile;
    const char* metadataFile;
};

constexpr TestCase kTestCases[] = {
    {"dog", "dog_pixel_values_fp32.bin", "dog_visual_tokens_fp32.bin", "dog.metadata.json"},
    {"cat", "cat_pixel_values_fp32.bin", "cat_visual_tokens_fp32.bin", "cat.metadata.json"},
};

} // namespace internvl
```

- [ ] **Step 2: Add tensor metrics declarations**

Create `demo/entry/src/main/cpp/validation/tensor_metrics.h`:

```cpp
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace internvl {

struct MetricsResult {
    bool ok = false;
    bool finite = false;
    std::size_t elementCount = 0;
    double maxAbsDiff = 0.0;
    double meanAbsDiff = 0.0;
    double cosine = 0.0;
    std::string errorStage;
    std::string errorMessage;
};

MetricsResult CompareFp32Tensors(const std::vector<float>& actual, const std::vector<float>& expected);

} // namespace internvl
```

- [ ] **Step 3: Implement tensor metrics**

Create `demo/entry/src/main/cpp/validation/tensor_metrics.cpp`:

```cpp
#include "tensor_metrics.h"

#include "validation_constants.h"

#include <cmath>
#include <limits>

namespace internvl {

MetricsResult CompareFp32Tensors(const std::vector<float>& actual, const std::vector<float>& expected)
{
    MetricsResult result;
    result.elementCount = actual.size();

    if (actual.size() != expected.size()) {
        result.errorStage = "output_size_mismatch";
        result.errorMessage = "Actual and expected output element counts differ";
        return result;
    }

    if (actual.empty()) {
        result.errorStage = "compare_failed";
        result.errorMessage = "Output tensor is empty";
        return result;
    }

    double absDiffSum = 0.0;
    double maxAbsDiff = 0.0;
    double dot = 0.0;
    double actualNorm = 0.0;
    double expectedNorm = 0.0;
    bool finite = true;

    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double a = static_cast<double>(actual[index]);
        const double e = static_cast<double>(expected[index]);
        if (!std::isfinite(a) || !std::isfinite(e)) {
            finite = false;
        }
        const double diff = std::abs(a - e);
        absDiffSum += diff;
        if (diff > maxAbsDiff) {
            maxAbsDiff = diff;
        }
        dot += a * e;
        actualNorm += a * a;
        expectedNorm += e * e;
    }

    result.finite = finite;
    result.maxAbsDiff = maxAbsDiff;
    result.meanAbsDiff = absDiffSum / static_cast<double>(actual.size());
    if (actualNorm > 0.0 && expectedNorm > 0.0) {
        result.cosine = dot / (std::sqrt(actualNorm) * std::sqrt(expectedNorm));
    }

    if (!finite) {
        result.errorStage = "compare_failed";
        result.errorMessage = "Actual or expected output contains non-finite values";
        return result;
    }

    if (result.cosine < kCosineMin || result.meanAbsDiff > kMeanAbsDiffMax) {
        result.errorStage = "compare_failed";
        result.errorMessage = "Output metrics are outside validation thresholds";
        return result;
    }

    result.ok = true;
    return result;
}

} // namespace internvl
```

- [ ] **Step 4: Add metric source to CMake**

Modify `demo/entry/src/main/cpp/CMakeLists.txt`:

```cmake
add_library(entry SHARED
    napi_init.cpp
    validation/tensor_metrics.cpp
)
```

Keep the existing NAPI link line:

```cmake
target_link_libraries(entry PUBLIC libace_napi.z.so)
```

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
git add demo/entry/src/main/cpp/CMakeLists.txt demo/entry/src/main/cpp/validation
git commit -m "Add C++ tensor comparison metrics"
```

## Task 5: HarmonyOS Rawfile Loader

**Files:**

- Create: `demo/entry/src/main/cpp/validation/rawfile_loader.h`
- Create: `demo/entry/src/main/cpp/validation/rawfile_loader.cpp`
- Modify: `demo/entry/src/main/cpp/CMakeLists.txt`

- [ ] **Step 1: Add rawfile loader declarations**

Create `demo/entry/src/main/cpp/validation/rawfile_loader.h`:

```cpp
#pragma once

#include "napi/native_api.h"

#include <cstdint>
#include <string>
#include <vector>

namespace internvl {

struct RawfileReadResult {
    bool ok = false;
    std::vector<std::uint8_t> bytes;
    std::string errorStage;
    std::string errorMessage;
};

RawfileReadResult ReadRawfile(napi_env env, napi_value resourceManager, const std::string& fileName);
std::vector<float> BytesToFp32Vector(const std::vector<std::uint8_t>& bytes);

} // namespace internvl
```

- [ ] **Step 2: Implement rawfile reading**

Create `demo/entry/src/main/cpp/validation/rawfile_loader.cpp`:

```cpp
#include "rawfile_loader.h"

#include "rawfile/raw_file_manager.h"

#include <cstring>

namespace internvl {

RawfileReadResult ReadRawfile(napi_env env, napi_value resourceManager, const std::string& fileName)
{
    RawfileReadResult result;

    NativeResourceManager* nativeResourceManager = OH_ResourceManager_InitNativeResourceManager(env, resourceManager);
    if (nativeResourceManager == nullptr) {
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Failed to initialize NativeResourceManager";
        return result;
    }

    RawFile* rawFile = OH_ResourceManager_OpenRawFile(nativeResourceManager, fileName.c_str());
    if (rawFile == nullptr) {
        result.errorStage = "missing_artifact";
        result.errorMessage = "Missing rawfile: " + fileName;
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
        return result;
    }

    const long rawFileSize = OH_ResourceManager_GetRawFileSize(rawFile);
    if (rawFileSize <= 0) {
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Rawfile is empty: " + fileName;
        OH_ResourceManager_CloseRawFile(rawFile);
        OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);
        return result;
    }

    result.bytes.resize(static_cast<std::size_t>(rawFileSize));
    const int readSize = OH_ResourceManager_ReadRawFile(rawFile, result.bytes.data(), rawFileSize);
    OH_ResourceManager_CloseRawFile(rawFile);
    OH_ResourceManager_ReleaseNativeResourceManager(nativeResourceManager);

    if (readSize != rawFileSize) {
        result.bytes.clear();
        result.errorStage = "read_rawfile_failed";
        result.errorMessage = "Rawfile read size mismatch: " + fileName;
        return result;
    }

    result.ok = true;
    return result;
}

std::vector<float> BytesToFp32Vector(const std::vector<std::uint8_t>& bytes)
{
    std::vector<float> values(bytes.size() / sizeof(float));
    if (!values.empty()) {
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(float));
    }
    return values;
}

} // namespace internvl
```

- [ ] **Step 3: Link the rawfile library**

Modify `demo/entry/src/main/cpp/CMakeLists.txt`:

```cmake
add_library(entry SHARED
    napi_init.cpp
    validation/tensor_metrics.cpp
    validation/rawfile_loader.cpp
)

target_link_libraries(entry PUBLIC
    libace_napi.z.so
    librawfile.z.so
)
```

- [ ] **Step 4: Commit Task 5**

Run:

```powershell
git add demo/entry/src/main/cpp/CMakeLists.txt demo/entry/src/main/cpp/validation/rawfile_loader.h demo/entry/src/main/cpp/validation/rawfile_loader.cpp
git commit -m "Add HarmonyOS rawfile loader"
```

## Task 6: NN Runtime Validator And NAPI API

**Files:**

- Create: `demo/entry/src/main/cpp/validation/nn_runtime_validator.h`
- Create: `demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp`
- Create: `demo/entry/src/main/cpp/validation/napi_result.h`
- Create: `demo/entry/src/main/cpp/validation/napi_result.cpp`
- Modify: `demo/entry/src/main/cpp/napi_init.cpp`
- Modify: `demo/entry/src/main/cpp/CMakeLists.txt`
- Modify: `demo/entry/src/main/cpp/types/libentry/Index.d.ts`

- [ ] **Step 1: Add validator result types**

Create `demo/entry/src/main/cpp/validation/nn_runtime_validator.h`:

```cpp
#pragma once

#include "napi/native_api.h"

#include <string>
#include <vector>

namespace internvl {

struct RunResult {
    bool ok = false;
    std::string caseName;
    std::string deviceType;
    std::string errorStage;
    std::string errorMessage;
    double latencyMs = 0.0;
    std::size_t outputElementCount = 0;
    double maxAbsDiff = 0.0;
    double meanAbsDiff = 0.0;
    double cosine = 0.0;
    bool finite = false;
};

struct StabilityResult {
    bool ok = false;
    std::string caseName;
    int repeatCount = 0;
    int successCount = 0;
    double minLatencyMs = 0.0;
    double maxLatencyMs = 0.0;
    double avgLatencyMs = 0.0;
    std::string errorStage;
    std::string errorMessage;
    RunResult lastRun;
};

std::vector<std::string> ListTestCaseNames();
RunResult RunOnce(napi_env env, napi_value resourceManager, const std::string& caseName);
StabilityResult RunStability(napi_env env, napi_value resourceManager, const std::string& caseName, int repeatCount);

} // namespace internvl
```

- [ ] **Step 2: Implement validator control flow**

Create `demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp`:

```cpp
#include "nn_runtime_validator.h"

#include "rawfile_loader.h"
#include "tensor_metrics.h"
#include "validation_constants.h"

#include "neural_network_runtime/neural_network_core.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <limits>

namespace internvl {

namespace {

const TestCase* FindCase(const std::string& caseName)
{
    for (const auto& testCase : kTestCases) {
        if (caseName == testCase.name) {
            return &testCase;
        }
    }
    return nullptr;
}

RunResult Fail(const std::string& caseName, const std::string& stage, const std::string& message)
{
    RunResult result;
    result.caseName = caseName;
    result.errorStage = stage;
    result.errorMessage = message;
    return result;
}

} // namespace

std::vector<std::string> ListTestCaseNames()
{
    std::vector<std::string> names;
    for (const auto& testCase : kTestCases) {
        names.emplace_back(testCase.name);
    }
    return names;
}

RunResult RunOnce(napi_env env, napi_value resourceManager, const std::string& caseName)
{
    const TestCase* testCase = FindCase(caseName);
    if (testCase == nullptr) {
        return Fail(caseName, "missing_artifact", "Unknown validation test case");
    }

    auto modelBytes = ReadRawfile(env, resourceManager, kModelFile);
    if (!modelBytes.ok) {
        return Fail(caseName, modelBytes.errorStage, modelBytes.errorMessage);
    }

    auto inputBytes = ReadRawfile(env, resourceManager, testCase->inputFile);
    if (!inputBytes.ok) {
        return Fail(caseName, inputBytes.errorStage, inputBytes.errorMessage);
    }
    if (inputBytes.bytes.size() != kInputByteCount) {
        return Fail(caseName, "input_size_mismatch", "Input rawfile byte count does not match [1,3,448,448] fp32");
    }

    auto expectedBytes = ReadRawfile(env, resourceManager, testCase->expectedOutputFile);
    if (!expectedBytes.ok) {
        return Fail(caseName, expectedBytes.errorStage, expectedBytes.errorMessage);
    }
    if (expectedBytes.bytes.size() != kOutputByteCount) {
        return Fail(caseName, "output_size_mismatch", "Expected output rawfile byte count does not match [1,256,1024] fp32");
    }

    std::vector<float> output(kOutputElementCount, 0.0f);

    const auto start = std::chrono::steady_clock::now();

    OH_NNCompilation* compilation = nullptr;
    OH_NNExecutor* executor = nullptr;

    OH_NN_ReturnCode code = OH_NNCompilation_ConstructWithOfflineModelBuffer(
        modelBytes.bytes.data(),
        modelBytes.bytes.size(),
        &compilation
    );
    if (code != OH_NN_SUCCESS || compilation == nullptr) {
        return Fail(caseName, "om_compilation_failed", "OH_NNCompilation_ConstructWithOfflineModelBuffer failed");
    }

    code = OH_NNCompilation_SetDevice(compilation, "HIAI_F");
    if (code != OH_NN_SUCCESS) {
        OH_NNCompilation_Destroy(&compilation);
        return Fail(caseName, "device_selection_failed", "OH_NNCompilation_SetDevice(HIAI_F) failed");
    }

    code = OH_NNCompilation_Build(compilation);
    if (code != OH_NN_SUCCESS) {
        OH_NNCompilation_Destroy(&compilation);
        return Fail(caseName, "om_compilation_failed", "OH_NNCompilation_Build failed");
    }

    code = OH_NNExecutor_Construct(compilation, &executor);
    if (code != OH_NN_SUCCESS || executor == nullptr) {
        OH_NNCompilation_Destroy(&compilation);
        return Fail(caseName, "executor_create_failed", "OH_NNExecutor_Construct failed");
    }

    code = OH_NNExecutor_SetInput(executor, 0, inputBytes.bytes.data(), inputBytes.bytes.size());
    if (code != OH_NN_SUCCESS) {
        OH_NNExecutor_Destroy(&executor);
        OH_NNCompilation_Destroy(&compilation);
        return Fail(caseName, "input_size_mismatch", "OH_NNExecutor_SetInput failed");
    }

    code = OH_NNExecutor_SetOutput(executor, 0, output.data(), kOutputByteCount);
    if (code != OH_NN_SUCCESS) {
        OH_NNExecutor_Destroy(&executor);
        OH_NNCompilation_Destroy(&compilation);
        return Fail(caseName, "output_size_mismatch", "OH_NNExecutor_SetOutput failed");
    }

    code = OH_NNExecutor_RunSync(executor);
    OH_NNExecutor_Destroy(&executor);
    OH_NNCompilation_Destroy(&compilation);

    const auto end = std::chrono::steady_clock::now();
    const double latencyMs = std::chrono::duration<double, std::milli>(end - start).count();

    if (code != OH_NN_SUCCESS) {
        return Fail(caseName, "run_sync_failed", "OH_NNExecutor_RunSync failed");
    }

    const std::vector<float> expected = BytesToFp32Vector(expectedBytes.bytes);
    MetricsResult metrics = CompareFp32Tensors(output, expected);

    RunResult result;
    result.ok = metrics.ok;
    result.caseName = caseName;
    result.deviceType = "HIAI_F";
    result.errorStage = metrics.errorStage;
    result.errorMessage = metrics.errorMessage;
    result.latencyMs = latencyMs;
    result.outputElementCount = output.size();
    result.maxAbsDiff = metrics.maxAbsDiff;
    result.meanAbsDiff = metrics.meanAbsDiff;
    result.cosine = metrics.cosine;
    result.finite = metrics.finite;
    return result;
}

StabilityResult RunStability(napi_env env, napi_value resourceManager, const std::string& caseName, int repeatCount)
{
    if (repeatCount <= 0) {
        repeatCount = 20;
    }

    StabilityResult result;
    result.caseName = caseName;
    result.repeatCount = repeatCount;
    result.minLatencyMs = std::numeric_limits<double>::max();

    double latencySum = 0.0;
    for (int index = 0; index < repeatCount; ++index) {
        RunResult run = RunOnce(env, resourceManager, caseName);
        result.lastRun = run;
        if (!run.ok) {
            result.errorStage = run.errorStage;
            result.errorMessage = run.errorMessage;
            break;
        }
        result.successCount += 1;
        result.minLatencyMs = std::min(result.minLatencyMs, run.latencyMs);
        result.maxLatencyMs = std::max(result.maxLatencyMs, run.latencyMs);
        latencySum += run.latencyMs;
    }

    if (result.successCount > 0) {
        result.avgLatencyMs = latencySum / static_cast<double>(result.successCount);
    } else {
        result.minLatencyMs = 0.0;
    }
    result.ok = result.successCount == repeatCount;
    return result;
}

} // namespace internvl
```

- [ ] **Step 3: Verify NN API signatures in yellow-zone headers**

Before building, inspect the installed HarmonyOS SDK header:

```powershell
rg -n "OH_NNCompilation_ConstructWithOfflineModelBuffer|OH_NNCompilation_SetDevice|OH_NNExecutor_SetInput|OH_NNExecutor_SetOutput|OH_NNExecutor_RunSync" "D:\Huawei\DevEco Studio\sdk"
```

If a function name or signature differs, update only `nn_runtime_validator.cpp` and keep the public NAPI functions unchanged:

```text
listTestCases()
runOnce(resourceManager, caseName)
runStability(resourceManager, caseName, repeatCount)
```

- [ ] **Step 4: Add NAPI result conversion declarations**

Create `demo/entry/src/main/cpp/validation/napi_result.h`:

```cpp
#pragma once

#include "napi/native_api.h"
#include "nn_runtime_validator.h"

namespace internvl {

napi_value ToNapiValue(napi_env env, const RunResult& result);
napi_value ToNapiValue(napi_env env, const StabilityResult& result);
napi_value StringArrayToNapiValue(napi_env env, const std::vector<std::string>& values);

} // namespace internvl
```

- [ ] **Step 5: Add NAPI result conversion implementation**

Create `demo/entry/src/main/cpp/validation/napi_result.cpp`:

```cpp
#include "napi_result.h"

namespace internvl {

namespace {

void SetBool(napi_env env, napi_value object, const char* name, bool value)
{
    napi_value napiValue;
    napi_get_boolean(env, value, &napiValue);
    napi_set_named_property(env, object, name, napiValue);
}

void SetNumber(napi_env env, napi_value object, const char* name, double value)
{
    napi_value napiValue;
    napi_create_double(env, value, &napiValue);
    napi_set_named_property(env, object, name, napiValue);
}

void SetString(napi_env env, napi_value object, const char* name, const std::string& value)
{
    napi_value napiValue;
    napi_create_string_utf8(env, value.c_str(), value.size(), &napiValue);
    napi_set_named_property(env, object, name, napiValue);
}

} // namespace

napi_value ToNapiValue(napi_env env, const RunResult& result)
{
    napi_value object;
    napi_create_object(env, &object);
    SetBool(env, object, "ok", result.ok);
    SetString(env, object, "caseName", result.caseName);
    SetString(env, object, "deviceType", result.deviceType);
    SetString(env, object, "errorStage", result.errorStage);
    SetString(env, object, "errorMessage", result.errorMessage);
    SetNumber(env, object, "latencyMs", result.latencyMs);
    SetNumber(env, object, "outputElementCount", static_cast<double>(result.outputElementCount));
    SetNumber(env, object, "maxAbsDiff", result.maxAbsDiff);
    SetNumber(env, object, "meanAbsDiff", result.meanAbsDiff);
    SetNumber(env, object, "cosine", result.cosine);
    SetBool(env, object, "finite", result.finite);
    return object;
}

napi_value ToNapiValue(napi_env env, const StabilityResult& result)
{
    napi_value object;
    napi_create_object(env, &object);
    SetBool(env, object, "ok", result.ok);
    SetString(env, object, "caseName", result.caseName);
    SetNumber(env, object, "repeatCount", result.repeatCount);
    SetNumber(env, object, "successCount", result.successCount);
    SetNumber(env, object, "minLatencyMs", result.minLatencyMs);
    SetNumber(env, object, "maxLatencyMs", result.maxLatencyMs);
    SetNumber(env, object, "avgLatencyMs", result.avgLatencyMs);
    SetString(env, object, "errorStage", result.errorStage);
    SetString(env, object, "errorMessage", result.errorMessage);
    napi_value lastRun = ToNapiValue(env, result.lastRun);
    napi_set_named_property(env, object, "lastRun", lastRun);
    return object;
}

napi_value StringArrayToNapiValue(napi_env env, const std::vector<std::string>& values)
{
    napi_value array;
    napi_create_array_with_length(env, values.size(), &array);
    for (std::size_t index = 0; index < values.size(); ++index) {
        napi_value item;
        napi_create_string_utf8(env, values[index].c_str(), values[index].size(), &item);
        napi_set_element(env, array, index, item);
    }
    return array;
}

} // namespace internvl
```

- [ ] **Step 6: Replace sample NAPI add with validation APIs**

Replace `demo/entry/src/main/cpp/napi_init.cpp` with:

```cpp
#include "napi/native_api.h"

#include "validation/napi_result.h"
#include "validation/nn_runtime_validator.h"

#include <string>

namespace {

std::string ReadStringArg(napi_env env, napi_value value)
{
    std::size_t length = 0;
    napi_get_value_string_utf8(env, value, nullptr, 0, &length);
    std::string result(length, '\0');
    napi_get_value_string_utf8(env, value, result.data(), result.size() + 1, &length);
    return result;
}

napi_value ListTestCases(napi_env env, napi_callback_info info)
{
    return internvl::StringArrayToNapiValue(env, internvl::ListTestCaseNames());
}

napi_value RunOnce(napi_env env, napi_callback_info info)
{
    std::size_t argc = 2;
    napi_value args[2] = {nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 2) {
        return internvl::ToNapiValue(env, internvl::RunResult{
            .errorStage = "napi_arg_error",
            .errorMessage = "runOnce requires resourceManager and caseName",
        });
    }

    const std::string caseName = ReadStringArg(env, args[1]);
    return internvl::ToNapiValue(env, internvl::RunOnce(env, args[0], caseName));
}

napi_value RunStability(napi_env env, napi_callback_info info)
{
    std::size_t argc = 3;
    napi_value args[3] = {nullptr, nullptr, nullptr};
    napi_get_cb_info(env, info, &argc, args, nullptr, nullptr);
    if (argc < 3) {
        internvl::StabilityResult result;
        result.errorStage = "napi_arg_error";
        result.errorMessage = "runStability requires resourceManager, caseName, and repeatCount";
        return internvl::ToNapiValue(env, result);
    }

    const std::string caseName = ReadStringArg(env, args[1]);
    int32_t repeatCount = 20;
    napi_get_value_int32(env, args[2], &repeatCount);
    return internvl::ToNapiValue(env, internvl::RunStability(env, args[0], caseName, repeatCount));
}

} // namespace

EXTERN_C_START
static napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor desc[] = {
        {"listTestCases", nullptr, ListTestCases, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runOnce", nullptr, RunOnce, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"runStability", nullptr, RunStability, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}
EXTERN_C_END

static napi_module demoModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void*)0),
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void)
{
    napi_module_register(&demoModule);
}
```

- [ ] **Step 7: Update CMake sources and libraries**

Modify `demo/entry/src/main/cpp/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.5.0)
project(demo)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})

if(DEFINED PACKAGE_FIND_FILE)
    include(${PACKAGE_FIND_FILE})
endif()

include_directories(${NATIVERENDER_ROOT_PATH}
                    ${NATIVERENDER_ROOT_PATH}/include)

add_library(entry SHARED
    napi_init.cpp
    validation/tensor_metrics.cpp
    validation/rawfile_loader.cpp
    validation/nn_runtime_validator.cpp
    validation/napi_result.cpp
)

target_link_libraries(entry PUBLIC
    libace_napi.z.so
    librawfile.z.so
    libneural_network_core.so
    libhiai_foundation.so
)
```

- [ ] **Step 8: Update TypeScript declarations**

Replace `demo/entry/src/main/cpp/types/libentry/Index.d.ts` with:

```ts
export interface RunResult {
  ok: boolean;
  caseName: string;
  deviceType: string;
  errorStage: string;
  errorMessage: string;
  latencyMs: number;
  outputElementCount: number;
  maxAbsDiff: number;
  meanAbsDiff: number;
  cosine: number;
  finite: boolean;
}

export interface StabilityResult {
  ok: boolean;
  caseName: string;
  repeatCount: number;
  successCount: number;
  minLatencyMs: number;
  maxLatencyMs: number;
  avgLatencyMs: number;
  errorStage: string;
  errorMessage: string;
  lastRun: RunResult;
}

export const listTestCases: () => string[];
export const runOnce: (resourceManager: object, caseName: string) => RunResult;
export const runStability: (resourceManager: object, caseName: string, repeatCount: number) => StabilityResult;
```

- [ ] **Step 9: Commit Task 6**

Run:

```powershell
git add demo/entry/src/main/cpp demo/entry/src/main/cpp/types/libentry/Index.d.ts
git commit -m "Add HarmonyOS OM validation NAPI"
```

## Task 7: ArkTS Validation Screen

**Files:**

- Modify: `demo/entry/src/main/ets/pages/Index.ets`

- [ ] **Step 1: Replace the sample page with validation controls**

Replace `demo/entry/src/main/ets/pages/Index.ets` with:

```ts
import testNapi, { RunResult, StabilityResult } from 'libentry.so';
import { hilog } from '@kit.PerformanceAnalysisKit';

const DOMAIN = 0x0000;

@Entry
@Component
struct Index {
  @State selectedCase: string = 'dog';
  @State status: string = 'Ready';
  @State runText: string = '';
  @State stabilityText: string = '';

  private formatRun(result: RunResult): string {
    return [
      `ok: ${result.ok}`,
      `case: ${result.caseName}`,
      `device: ${result.deviceType}`,
      `latency_ms: ${result.latencyMs.toFixed(3)}`,
      `output_elements: ${result.outputElementCount}`,
      `finite: ${result.finite}`,
      `cosine: ${result.cosine.toFixed(8)}`,
      `mean_abs_diff: ${result.meanAbsDiff.toExponential(4)}`,
      `max_abs_diff: ${result.maxAbsDiff.toExponential(4)}`,
      `error_stage: ${result.errorStage}`,
      `error_message: ${result.errorMessage}`,
    ].join('\n');
  }

  private formatStability(result: StabilityResult): string {
    return [
      `ok: ${result.ok}`,
      `case: ${result.caseName}`,
      `repeat: ${result.repeatCount}`,
      `success: ${result.successCount}`,
      `avg_latency_ms: ${result.avgLatencyMs.toFixed(3)}`,
      `min_latency_ms: ${result.minLatencyMs.toFixed(3)}`,
      `max_latency_ms: ${result.maxLatencyMs.toFixed(3)}`,
      `error_stage: ${result.errorStage}`,
      `error_message: ${result.errorMessage}`,
    ].join('\n');
  }

  build() {
    Column({ space: 12 }) {
      Text('InternVL OM Validation')
        .fontSize(22)
        .fontWeight(FontWeight.Bold)

      Row({ space: 8 }) {
        Button('Dog')
          .onClick(() => {
            this.selectedCase = 'dog';
          })
        Button('Cat')
          .onClick(() => {
            this.selectedCase = 'cat';
          })
      }

      Text(`Selected: ${this.selectedCase}`)
        .fontSize(16)

      Button('Run Once')
        .width('100%')
        .onClick(() => {
          this.status = 'Running once';
          const result = testNapi.runOnce(getContext(this).resourceManager, this.selectedCase);
          this.runText = this.formatRun(result);
          this.status = result.ok ? 'Run passed' : 'Run failed';
          hilog.info(DOMAIN, 'internvl', 'runOnce ok=%{public}s', `${result.ok}`);
        })

      Button('Run 20 Stability')
        .width('100%')
        .onClick(() => {
          this.status = 'Running stability';
          const result = testNapi.runStability(getContext(this).resourceManager, this.selectedCase, 20);
          this.stabilityText = this.formatStability(result);
          this.status = result.ok ? 'Stability passed' : 'Stability failed';
          hilog.info(DOMAIN, 'internvl', 'runStability ok=%{public}s', `${result.ok}`);
        })

      Text(this.status)
        .fontSize(16)
        .fontWeight(FontWeight.Medium)

      Text('Single Run')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
      Text(this.runText)
        .fontSize(13)
        .width('100%')

      Text('Stability')
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
      Text(this.stabilityText)
        .fontSize(13)
        .width('100%')
    }
    .padding(16)
    .width('100%')
    .height('100%')
  }
}
```

- [ ] **Step 2: Commit Task 7**

Run:

```powershell
git add demo/entry/src/main/ets/pages/Index.ets
git commit -m "Add HarmonyOS validation screen"
```

## Task 8: Yellow-Zone Build And Runtime Verification

**Files:**

- Modify: `docs/harmonyos-om-runtime-validation.md`

- [ ] **Step 1: Build in DevEco Studio**

In yellow zone:

```text
Open demo/ in DevEco Studio
Sync project
Build HAP for a physical HarmonyOS device
Install on device
```

Expected:

```text
Native library compiles
HAP packaging includes rawfile artifacts
Application opens the validation screen
```

- [ ] **Step 2: If NN API signatures fail, patch only the adapter**

When build errors point to NN runtime function names or signatures, update this file only:

```text
demo/entry/src/main/cpp/validation/nn_runtime_validator.cpp
```

Keep these files stable unless the compiler points to them:

```text
demo/entry/src/main/cpp/validation/tensor_metrics.cpp
demo/entry/src/main/cpp/validation/rawfile_loader.cpp
demo/entry/src/main/cpp/validation/napi_result.cpp
demo/entry/src/main/cpp/types/libentry/Index.d.ts
demo/entry/src/main/ets/pages/Index.ets
```

- [ ] **Step 3: Run dog once**

On device:

```text
Select Dog
Tap Run Once
```

Expected UI fields:

```text
ok: true
case: dog
device: HIAI_F
output_elements: 262144
finite: true
cosine: >= 0.999
mean_abs_diff: <= 0.01
```

- [ ] **Step 4: Run cat once**

On device:

```text
Select Cat
Tap Run Once
```

Expected UI fields:

```text
ok: true
case: cat
device: HIAI_F
output_elements: 262144
finite: true
cosine: >= 0.999
mean_abs_diff: <= 0.01
```

- [ ] **Step 5: Run 20-run stability**

On device:

```text
Select Dog
Tap Run 20 Stability
Select Cat
Tap Run 20 Stability
```

Expected:

```text
ok: true
repeat: 20
success: 20
avg_latency_ms: recorded
min_latency_ms: recorded
max_latency_ms: recorded
```

- [ ] **Step 6: Record observed results**

Append real device results to `docs/harmonyos-om-runtime-validation.md`:

```markdown
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
```

- [ ] **Step 7: Commit yellow-zone compatibility fixes and results**

Run:

```powershell
git add demo docs/harmonyos-om-runtime-validation.md
git commit -m "Validate HarmonyOS OM runtime on device"
```

## Final Verification Before Push

- [ ] **Step 1: Run Python tests**

Run:

```powershell
uv run python -B -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 2: Check git status**

Run:

```powershell
git status -sb
```

Expected:

```text
## main...origin/main [ahead N]
```

No untracked source files should remain. Large `.om`, `.onnx`, `.pt`, and `.bin`
files should remain ignored.

- [ ] **Step 3: Push after user approval**

Run:

```powershell
git push origin main
```
