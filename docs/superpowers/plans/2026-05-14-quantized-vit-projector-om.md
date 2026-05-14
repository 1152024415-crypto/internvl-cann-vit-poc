# Quantized ViT Projector OM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a separate quantization PoC that turns the current CANN-compatible ViT + projector ONNX into a smaller INT8 OM, then validates size, Linux static checks, and device accuracy against the current floating-point baseline.

**Architecture:** Keep the current floating-point OM as the control artifact. Add a separate DOPT quantization path that generates calibration data, runs CANN Kit `tools_dopt` ONNX no-training `Quant_INT8-8`, converts the quantized ONNX with OMG using `--compress_conf`, and validates the quantized OM through the same Linux and HarmonyOS runtime gates. Do not replace the baseline Release asset until the quantized OM passes device validation.

**Tech Stack:** Python 3.12 with `uv`, NumPy, Pillow preprocessing, ONNX/ONNX Runtime checks, Huawei CANN Kit `tools_dopt`, Huawei CANN Kit `tools_omg`, WSL Ubuntu 22.04, HarmonyOS Native C++ validation demo.

---

## Current Baseline

The current validated floating-point Release OM is:

```text
artifacts/om/internvl3_5_vit_projector_fp32_opset18_staticpos.om
size = 1236219952 bytes
SHA256 = 8D081689805763B786BE003B5627061DFB9324EDF3DF7DF0226C8F5A9C093FA7
```

The quantization branch must keep that artifact as the reference. Quantized results are published under a different name:

```text
internvl3_5_vit_projector_int8_opset18_staticpos.om
```

## Expected Artifact Flow

```text
data/calibration-images/*.jpg
-> InternVL static preprocess
-> DOPT calibration .bin with 510 header
-> DOPT ONNX no-training quantization
-> quantized ONNX + compress_conf
-> OMG --compress_conf
-> quantized OM
-> Linux static validation
-> HarmonyOS device validation
```

## Files

- Create: `src/internvl_vit_poc/dopt_calibration.py`
  - Export DOPT-compatible calibration binaries from image files.
  - Use the same `load_static_pixel_values()` path as baseline/export.
- Create: `tests/test_dopt_calibration.py`
  - Verify DOPT binary headers, shape, dtype, and payload size.
- Create: `scripts/quantize_onnx_with_dopt.sh`
  - Linux-side DOPT invocation.
  - Locate `tools_dopt/dopt_onnx_py3/dopt_so.py`.
  - Produce quantized ONNX and `compress_conf`.
- Create: `scripts/quantize_wsl_onnx_with_dopt.ps1`
  - Windows wrapper that copies ONNX/calibration data into WSL, runs DOPT, and copies quantized outputs back.
- Modify: `scripts/convert_onnx_to_om.sh`
  - Add optional `COMPRESS_CONF` support.
  - Append `--compress_conf "${COMPRESS_CONF}"` only when provided.
- Modify: `scripts/convert_wsl_onnx_to_om.ps1`
  - Add optional `-CompressConf` parameter and copy it into WSL.
- Create: `scripts/validate_om_static_linux.sh`
  - Run `omg --mode 3` report check, `omg --mode 1` OM-to-JSON, shape checks, and log evidence checks.
  - This should also validate the current floating-point OM and the future quantized OM.
- Create: `docs/stage-5-quantization.md`
  - Record commands, artifacts, accuracy thresholds, and open risks.
- Modify: `docs/current-status.md`
  - Add a Stage 5 status section once the first quantized artifact exists.

## Task 1: Export DOPT Calibration Binaries

**Files:**
- Create: `src/internvl_vit_poc/dopt_calibration.py`
- Create: `tests/test_dopt_calibration.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing DOPT header test**

Create `tests/test_dopt_calibration.py`:

```python
from pathlib import Path

import numpy as np

from internvl_vit_poc.dopt_calibration import DOPT_MAGIC_NCHW, build_dopt_4d_blob


def test_build_dopt_4d_blob_writes_510_header_and_fp32_payload() -> None:
    tensor = np.arange(1 * 3 * 2 * 2, dtype=np.float32).reshape(1, 3, 2, 2)

    blob = build_dopt_4d_blob(tensor)
    header = np.frombuffer(blob[:20], dtype="<i4")
    payload = np.frombuffer(blob[20:], dtype="<f4")

    assert header.tolist() == [DOPT_MAGIC_NCHW, 1, 3, 2, 2]
    assert payload.tolist() == tensor.astype("<f4").reshape(-1).tolist()
    assert len(blob) == 20 + tensor.size * 4
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_dopt_calibration
```

Expected result:

```text
ModuleNotFoundError: No module named 'internvl_vit_poc.dopt_calibration'
```

- [ ] **Step 3: Implement DOPT binary export**

Create `src/internvl_vit_poc/dopt_calibration.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .preprocess import STATIC_IMAGE_SIZE, load_static_pixel_values
from .validation_tensors import sha256_file

DOPT_MAGIC_NCHW = 510
INPUT_SHAPE = [1, 3, 448, 448]


def build_dopt_4d_blob(tensor: np.ndarray) -> bytes:
    array = np.asarray(tensor, dtype="<f4")
    if array.ndim != 4:
        raise ValueError(f"DOPT 4D blob expects rank 4, got {array.ndim}")
    header = np.asarray([DOPT_MAGIC_NCHW, *array.shape], dtype="<i4")
    return header.tobytes() + array.reshape(-1).tobytes()


def export_image(path: Path, output_dir: Path) -> dict[str, object]:
    tensor = load_static_pixel_values(path, STATIC_IMAGE_SIZE).numpy().astype(np.float32)
    if list(tensor.shape) != INPUT_SHAPE:
        raise ValueError(f"input shape mismatch for {path}: {list(tensor.shape)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{path.stem}_pixel_values_dopt.bin"
    output_path.write_bytes(build_dopt_4d_blob(tensor))
    return {
        "source_image": str(path),
        "path": str(output_path),
        "shape": INPUT_SHAPE,
        "dtype": "float32",
        "format": "dopt_4d_510_header",
        "byte_count": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DOPT calibration .bin files for InternVL ViT + projector.")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    images = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        for path in image_dir.glob(pattern)
    )
    if not images:
        raise SystemExit(f"No calibration images found under {image_dir}")

    output_dir = Path(args.output_dir)
    records = [export_image(path, output_dir) for path in images]
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(records), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add CLI entry point**

Modify `pyproject.toml`:

```toml
internvl-export-dopt-calibration = "internvl_vit_poc.dopt_calibration:main"
```

- [ ] **Step 5: Run tests**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_dopt_calibration
uv run --no-sync python -B -m unittest discover -s tests
```

Expected result:

```text
OK
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add pyproject.toml src\internvl_vit_poc\dopt_calibration.py tests\test_dopt_calibration.py
git commit -m "Add DOPT calibration export"
```

## Task 2: Add DOPT Quantization Scripts

**Files:**
- Create: `scripts/quantize_onnx_with_dopt.sh`
- Create: `scripts/quantize_wsl_onnx_with_dopt.ps1`
- Create: `tests/test_dopt_quantization_scripts.py`

- [ ] **Step 1: Write script structure tests**

Create `tests/test_dopt_quantization_scripts.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_linux_dopt_script_uses_onnx_notrain_quantization() -> None:
    script = read_text("scripts/quantize_onnx_with_dopt.sh")

    assert "tools_dopt/dopt_onnx_py3/dopt_so.py" in script
    assert "--framework 5" in script
    assert "--mode 0" in script
    assert "--input_shape" in script
    assert "--out_nodes" in script
    assert "--compress_conf" in script
    assert "visual_tokens" in script


def test_windows_dopt_wrapper_copies_quant_outputs_back() -> None:
    script = read_text("scripts/quantize_wsl_onnx_with_dopt.ps1")

    assert "Invoke-WslChecked" in script
    assert "quantized.onnx" in script
    assert "compress_conf" in script
    assert "$LASTEXITCODE" in script
```

- [ ] **Step 2: Run failing tests**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_dopt_quantization_scripts
```

Expected result:

```text
FileNotFoundError
```

- [ ] **Step 3: Create Linux DOPT script**

Create `scripts/quantize_onnx_with_dopt.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_PATH="${1:-${PROJECT_ROOT}/artifacts/onnx/internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx}"
CAL_CONF="${2:-${PROJECT_ROOT}/artifacts/quantization/dopt_config.prototxt}"
OUTPUT_ONNX="${3:-${PROJECT_ROOT}/artifacts/quantization/internvl3_5_vit_projector_int8_opset18_staticpos.onnx}"
COMPRESS_CONF="${4:-${PROJECT_ROOT}/artifacts/quantization/internvl3_5_vit_projector_int8_compress_conf}"
DOPT_ROOT="${DOPT_ROOT:-${HOME}/cann-kit/tools/tools_dopt}"
DOPT_BIN="${DOPT_BIN:-${DOPT_ROOT}/dopt_onnx_py3/dopt_so.py}"

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ONNX model not found: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${CAL_CONF}" ]]; then
  echo "DOPT calibration config not found: ${CAL_CONF}" >&2
  exit 3
fi
if [[ ! -f "${DOPT_BIN}" ]]; then
  echo "DOPT ONNX entry not found: ${DOPT_BIN}" >&2
  exit 4
fi

mkdir -p "$(dirname "${OUTPUT_ONNX}")"
mkdir -p "$(dirname "${COMPRESS_CONF}")"

python3 "${DOPT_BIN}" \
  --mode 0 \
  --framework 5 \
  --model "${MODEL_PATH}" \
  --cal_conf "${CAL_CONF}" \
  --output "${OUTPUT_ONNX}" \
  --input_format NCHW \
  --input_shape "pixel_values:1,3,448,448" \
  --out_nodes "visual_tokens" \
  --compress_conf "${COMPRESS_CONF}"

test -f "${OUTPUT_ONNX}"
test -f "${COMPRESS_CONF}"
echo "Quantized ONNX: ${OUTPUT_ONNX}"
echo "Compress conf: ${COMPRESS_CONF}"
```

- [ ] **Step 4: Create Windows WSL wrapper**

Create `scripts/quantize_wsl_onnx_with_dopt.ps1` using the `Invoke-WslChecked` pattern from existing WSL scripts. It must accept:

```powershell
param(
    [string]$Distro = "InternVL-Ubuntu-22.04",
    [string]$OnnxFile = "D:\proj\internvl-cann-vit-poc\artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx",
    [string]$CalConf = "D:\proj\internvl-cann-vit-poc\artifacts\quantization\dopt_config.prototxt",
    [string]$OutputName = "internvl3_5_vit_projector_int8_opset18_staticpos"
)
```

It must copy back:

```text
artifacts/quantization/internvl3_5_vit_projector_int8_opset18_staticpos.onnx
artifacts/quantization/internvl3_5_vit_projector_int8_compress_conf
```

- [ ] **Step 5: Run tests**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_dopt_quantization_scripts
uv run --no-sync python -B -m unittest discover -s tests
```

Expected result:

```text
OK
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add scripts\quantize_onnx_with_dopt.sh scripts\quantize_wsl_onnx_with_dopt.ps1 tests\test_dopt_quantization_scripts.py
git commit -m "Add DOPT ONNX quantization scripts"
```

## Task 3: Generate Calibration Data And DOPT Config

**Files:**
- Create: `artifacts/quantization/dopt_config.prototxt` locally; do not commit large calibration bins.
- Modify: `docs/stage-5-quantization.md`

- [ ] **Step 1: Prepare calibration image directory**

Create:

```text
data/calibration-images/
```

For the first PoC, put at least these images there:

```text
data/test-images/dog.jpg
data/test-images/cat.jpg
```

For a more useful calibration run, add 20-50 varied natural images. Do not commit copyrighted or private images unless they are explicitly safe to store.

- [ ] **Step 2: Export DOPT calibration binaries**

Run:

```powershell
uv run internvl-export-dopt-calibration `
  --image-dir data\calibration-images `
  --output-dir artifacts\quantization\calibration-bins `
  --manifest artifacts\quantization\calibration-manifest.json
```

Expected output:

```text
count >= 2
manifest = artifacts\quantization\calibration-manifest.json
```

- [ ] **Step 3: Create DOPT config**

Create `artifacts/quantization/dopt_config.prototxt`:

```text
strategy: "Quant_INT8-8"
device: USE_CPU
preprocess_parameter:
{
    input_type: BINARY
    input_file_path: "/root/internvl-cann-vit-poc/artifacts/quantization/calibration-bins"
}
```

Use `BINARY` because project preprocessing already performs resize, RGB conversion, normalization, and BCHW layout.

- [ ] **Step 4: Document the calibration set**

Create `docs/stage-5-quantization.md` with:

````markdown
# Stage 5: Quantization

## Calibration Set

The first quantization PoC uses DOPT BINARY calibration input generated from the
same InternVL preprocessing as the baseline path.

Calibration manifest:

```text
artifacts/quantization/calibration-manifest.json
```

## Validation Gates

The quantized OM is not accepted unless:

```text
quantized ONNX exists
compress_conf exists
OMG conversion succeeds with --compress_conf
OM-to-JSON succeeds
input shape = [1,3,448,448]
output shape = [1,256,1024]
device RunSync succeeds
dog/cat cosine >= 0.995 for first PoC
mean_abs_diff <= 0.05 for first PoC
```
````

- [ ] **Step 5: Commit docs only**

Do not commit generated calibration bins. Run:

```powershell
git add docs\stage-5-quantization.md
git commit -m "Document quantization calibration setup"
```

## Task 4: Run DOPT ONNX Quantization

**Files:**
- Local artifact: `artifacts/quantization/internvl3_5_vit_projector_int8_opset18_staticpos.onnx`
- Local artifact: `artifacts/quantization/internvl3_5_vit_projector_int8_compress_conf`

- [ ] **Step 1: Run WSL DOPT wrapper**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\quantize_wsl_onnx_with_dopt.ps1 `
  -Distro InternVL-Ubuntu-22.04 `
  -OnnxFile D:\proj\internvl-cann-vit-poc\artifacts\onnx\internvl3_5_vit_projector_fp32_opset18_staticpos_cann.onnx `
  -CalConf D:\proj\internvl-cann-vit-poc\artifacts\quantization\dopt_config.prototxt `
  -OutputName internvl3_5_vit_projector_int8_opset18_staticpos
```

- [ ] **Step 2: Verify artifacts exist**

Run:

```powershell
Get-Item artifacts\quantization\internvl3_5_vit_projector_int8_opset18_staticpos.onnx
Get-Item artifacts\quantization\internvl3_5_vit_projector_int8_compress_conf
Get-FileHash artifacts\quantization\internvl3_5_vit_projector_int8_opset18_staticpos.onnx -Algorithm SHA256
```

Expected:

```text
Both files exist.
Record size and SHA256 in docs/stage-5-quantization.md.
```

## Task 5: Convert Quantized ONNX To OM

**Files:**
- Modify: `scripts/convert_onnx_to_om.sh`
- Modify: `scripts/convert_wsl_onnx_to_om.ps1`
- Test: `tests/test_cann_omg_npu_partition.py`

- [ ] **Step 1: Add test for optional compress_conf support**

Modify `tests/test_cann_omg_npu_partition.py`:

```python
def test_conversion_script_supports_optional_compress_conf(self) -> None:
    script = read_text("scripts/convert_onnx_to_om.sh")

    self.assertIn("COMPRESS_CONF", script)
    self.assertIn("--compress_conf", script)
    self.assertIn("if [[ -n", script)
```

- [ ] **Step 2: Run failing test**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_cann_omg_npu_partition
```

Expected:

```text
FAIL: COMPRESS_CONF not found
```

- [ ] **Step 3: Add `COMPRESS_CONF` to Linux conversion script**

Modify `scripts/convert_onnx_to_om.sh`:

```bash
COMPRESS_CONF="${COMPRESS_CONF:-}"
```

After `OMG_ARGS` is initialized, append:

```bash
if [[ -n "${COMPRESS_CONF}" ]]; then
  if [[ ! -f "${COMPRESS_CONF}" ]]; then
    echo "compress_conf not found: ${COMPRESS_CONF}" >&2
    exit 7
  fi
  OMG_ARGS+=(--compress_conf "${COMPRESS_CONF}")
fi
```

- [ ] **Step 4: Add `-CompressConf` to Windows wrapper**

Modify `scripts/convert_wsl_onnx_to_om.ps1` to accept:

```powershell
[string]$CompressConf = ""
```

If provided, copy it into:

```text
/root/internvl-cann-vit-poc/artifacts/quantization/
```

and invoke Linux conversion with:

```bash
export COMPRESS_CONF="/root/internvl-cann-vit-poc/artifacts/quantization/internvl3_5_vit_projector_int8_compress_conf"
```

- [ ] **Step 5: Convert quantized ONNX**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\convert_wsl_onnx_to_om.ps1 `
  -Distro InternVL-Ubuntu-22.04 `
  -OnnxFile D:\proj\internvl-cann-vit-poc\artifacts\quantization\internvl3_5_vit_projector_int8_opset18_staticpos.onnx `
  -OutputName internvl3_5_vit_projector_int8_opset18_staticpos `
  -CompressConf D:\proj\internvl-cann-vit-poc\artifacts\quantization\internvl3_5_vit_projector_int8_compress_conf
```

- [ ] **Step 6: Record size and hash**

Run:

```powershell
Get-Item artifacts\om\internvl3_5_vit_projector_int8_opset18_staticpos.om
Get-FileHash artifacts\om\internvl3_5_vit_projector_int8_opset18_staticpos.om -Algorithm SHA256
```

Record the size reduction ratio:

```text
ratio = int8_om_size / 1236219952
```

- [ ] **Step 7: Run tests and commit**

Run:

```powershell
uv run --no-sync python -B -m unittest discover -s tests
git add scripts\convert_onnx_to_om.sh scripts\convert_wsl_onnx_to_om.ps1 tests\test_cann_omg_npu_partition.py docs\stage-5-quantization.md
git commit -m "Support quantized OM conversion"
```

## Task 6: Linux Static Validation For Quantized OM

**Files:**
- Create: `scripts/validate_om_static_linux.sh`
- Create: `scripts/validate_wsl_om_static.ps1`
- Test: `tests/test_om_static_validation_scripts.py`

- [ ] **Step 1: Add script tests**

Create `tests/test_om_static_validation_scripts.py`:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_text(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_linux_om_static_validation_checks_shapes_and_log_evidence() -> None:
    script = read_text("scripts/validate_om_static_linux.sh")

    assert "--mode 1" in script
    assert "--mode 3" in script
    assert "pixel_values" in script
    assert "1,3,448,448" in script
    assert "1,256,1024" in script
    assert "AI_NPUCL" in script
    assert "partition type NPU:0" in script


def test_windows_om_static_validation_wrapper_uses_wsl() -> None:
    script = read_text("scripts/validate_wsl_om_static.ps1")

    assert "Invoke-WslChecked" in script
    assert "validate_om_static_linux.sh" in script
    assert "$LASTEXITCODE" in script
```

- [ ] **Step 2: Implement Linux static validator**

The validator must accept:

```bash
./scripts/validate_om_static_linux.sh \
  ./artifacts/onnx/internvl3_5_vit_projector_int8_opset18_staticpos.onnx \
  ./artifacts/om/internvl3_5_vit_projector_int8_opset18_staticpos.om \
  ./artifacts/om/internvl3_5_vit_projector_int8_opset18_staticpos.omg.log
```

It must fail if:

```text
precheck report fail != 0
OM-to-JSON fails
input shape is not [1,3,448,448]
output shape is not [1,256,1024]
OMG log has CPUCL
OMG log has partition type NPU:0
OMG log does not have AI_NPUCL
```

- [ ] **Step 3: Validate quantized OM**

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\validate_wsl_om_static.ps1 `
  -Distro InternVL-Ubuntu-22.04 `
  -OnnxFile D:\proj\internvl-cann-vit-poc\artifacts\quantization\internvl3_5_vit_projector_int8_opset18_staticpos.onnx `
  -OmFile D:\proj\internvl-cann-vit-poc\artifacts\om\internvl3_5_vit_projector_int8_opset18_staticpos.om `
  -OmgLog D:\proj\internvl-cann-vit-poc\artifacts\om\internvl3_5_vit_projector_int8_opset18_staticpos.omg.log
```

Expected:

```text
all checks PASS
```

- [ ] **Step 4: Commit**

Run:

```powershell
uv run --no-sync python -B -m unittest discover -s tests
git add scripts\validate_om_static_linux.sh scripts\validate_wsl_om_static.ps1 tests\test_om_static_validation_scripts.py docs\stage-5-quantization.md
git commit -m "Add OM static validation scripts"
```

## Task 7: Device Validation Of Quantized OM

**Files:**
- Modify: `docs/yellow-zone-validation-runbook.md`
- Modify: `docs/harmonyos-om-runtime-validation.md`

- [ ] **Step 1: Do not replace the baseline OM in Release**

Upload the quantized OM as a new Release asset:

```text
internvl3_5_vit_projector_int8_opset18_staticpos.om
```

Keep the existing baseline asset:

```text
internvl3_5_vit_projector_fp32_opset18_staticpos.om
```

- [ ] **Step 2: Copy quantized OM into rawfile for a separate run**

In yellow zone:

```powershell
Copy-Item internvl3_5_vit_projector_int8_opset18_staticpos.om `
  demo\entry\src\main\resources\rawfile\internvl3_5_vit_projector_fp32_opset18_staticpos.om `
  -Force
```

This temporary copy lets the existing app load the quantized OM without code changes. Record in the notes that the file contents are INT8 even though the rawfile name is the baseline name.

- [ ] **Step 3: Run device validation**

Run:

```text
Load Model
dog -> Run Once
cat -> Run Once
dog -> Run 20 Stability
cat -> Run 20 Stability
```

First PoC acceptance thresholds:

```text
Load Model succeeds
RunSync succeeds
output shape = [1,256,1024]
finite = true
cosine >= 0.995
mean_abs_diff <= 0.05
20-run success count = 20
```

If cosine is between `0.99` and `0.995`, keep the artifact but mark it as accuracy-risk and test with more images. If cosine is below `0.99`, reject the first quantization attempt.

- [ ] **Step 4: Record results**

Update `docs/harmonyos-om-runtime-validation.md`:

```markdown
## Quantized OM Device Results

artifact:
size:
sha256:
device:
date:

Dog:
ok:
cosine:
mean_abs_diff:
max_abs_diff:
latency_ms:

Cat:
ok:
cosine:
mean_abs_diff:
max_abs_diff:
latency_ms:

20-run stability:
dog_success_count:
cat_success_count:
```

- [ ] **Step 5: Commit**

Run:

```powershell
git add docs\harmonyos-om-runtime-validation.md docs\yellow-zone-validation-runbook.md
git commit -m "Record quantized OM device validation"
```

## Task 8: Decide Whether To Promote Quantized OM

**Files:**
- Modify: `docs/current-status.md`
- Modify: `docs/release-artifacts.md`
- Modify: `docs/stage-5-quantization.md`

- [ ] **Step 1: Compare baseline and quantized artifacts**

Record:

```text
baseline_size = 1236219952
quantized_size = measured value
size_ratio = quantized_size / baseline_size
dog_cosine = measured value
cat_cosine = measured value
dog_latency_ms = measured value
cat_latency_ms = measured value
```

- [ ] **Step 2: Promotion rule**

Promote the quantized OM only when:

```text
size_ratio <= 0.70
dog_cosine >= 0.995
cat_cosine >= 0.995
dog 20-run stability succeeds
cat 20-run stability succeeds
no new native crash or packaging failure
```

- [ ] **Step 3: If promoted, update docs**

Update docs to say:

```text
baseline OM remains available
quantized OM is preferred for size-sensitive testing
full app still uses ViT + projector only
preprocessing is still fixed [1,3,448,448]
```

- [ ] **Step 4: Commit**

Run:

```powershell
git add docs\current-status.md docs\release-artifacts.md docs\stage-5-quantization.md
git commit -m "Promote quantized OM validation results"
```

## Execution Notes

- Do not block current yellow-zone validation of the floating-point OM.
- Do not delete or replace the floating-point Release asset.
- Do not use the LLM 4bit flow for this ViT + projector subgraph unless ONNX INT8 is rejected and a separate design is written.
- Use ONNX no-training INT8 first because it directly matches the current CANN-compatible ONNX artifact.
- Calibration quality is the main accuracy risk. Two images are enough only to smoke-test the toolchain, not to judge production quality.

## Self-Review

- Spec coverage: the plan covers calibration data, DOPT quantization, OMG conversion with `compress_conf`, Linux static validation, device validation, and promotion rules.
- Plan scan: result-recording sections list concrete fields that must be filled after measurement.
- Type consistency: artifact names consistently use `fp32` for the current baseline and `int8` for the quantized branch.
