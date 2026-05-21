# ONNX vs OMC Layer Dump Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable one-case workflow that dumps ONNX intermediate tensors from `pixel_values_fp32.bin`, inventories device-side OMC dump files, and compares mapped ONNX tensors against OMC raw tensors to find the first divergent block/layer.

**Architecture:** Extend the existing ONNX intermediate dump utility instead of creating a parallel ONNX runner. Add a focused OMC dump alignment module for raw dump inventory and mapped comparison, because OMC dump names/layouts are device-tool-defined and will not reliably match ONNX graph names. Add a runbook with exact Windows/WSL host commands and `hdc shell` commands for the current `/data/local/tmp/internvl_om_compare` device environment.

**Tech Stack:** Python, `uv`, `onnx`, `onnxruntime`, `numpy`, existing `internvl_vit_poc` utilities, HarmonyOS `hdc`, device `model_run_tool_internal`.

---

## Current Environment Facts

These facts were verified before writing the plan:

```text
Blue-zone current environment:
Windows local workspace + optional WSL + hdc-connected device

Yellow-zone environment:
Windows local workspace + Linux server for DDK/DOPT/OMG + hdc-connected device

Boundary for this implementation:
Do not run DDK, DOPT quantization, or OMG OMC compilation here.
Use the existing OMC already on the device.

Local Windows hdc path:
D:\software\deveco\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe

Connected device:
5NC0125514000008

Device working directory:
/data/local/tmp/internvl_om_compare

Device model:
/data/local/tmp/internvl_om_compare/split_internvit_v6_exclude_conv_int8_8.omc
size ~= 319 MB

Device runner:
/data/local/tmp/internvl_om_compare/model_run_tool_internal

Other device-side tools found in the same directory:
/data/local/tmp/internvl_om_compare/data_proc_tool_internal
/data/local/tmp/internvl_om_compare/model_run_bbit
/data/local/tmp/internvl_om_compare/prof_data_parser_internal
/data/local/tmp/internvl_om_compare/profiling_tool_internal
/data/local/tmp/internvl_om_compare/set_log_level

Device runner dump flag:
--enable_item=2

Device runner profiling flag:
--enable_item=1

Existing device inputs:
/data/local/tmp/internvl_om_compare/inputs/*.bin       # 5-case subset
/data/local/tmp/internvl_om_compare/inputs_all/*.bin   # full set

Existing final outputs:
/data/local/tmp/internvl_om_compare/om_outputs/*.bin
/data/local/tmp/internvl_om_compare/om_outputs_all/*.bin
```

`model_run_tool_internal -h` confirms this command shape:

```bash
./model_run_tool_internal \
  --model=<model.om_or_omc> \
  --input=<input.bin> \
  --output_dir=<out_dir> \
  --enable_item=2 \
  --times=1
```

The current ONNX intermediate utility already adds selected ONNX tensors to
graph outputs and runs ONNX Runtime, but it only accepts image input. This plan
adds raw input-bin support so the host ONNX path consumes exactly the same
bytes as the device OMC path.

## File Structure

Modify:

```text
src/internvl_vit_poc/onnx_intermediate_debug.py
```

Responsibility: keep all ONNX graph-output augmentation and ONNX Runtime dump
logic. Add raw input-bin loading here because it is part of ONNX dump execution.

Create:

```text
src/internvl_vit_poc/omc_dump_align.py
```

Responsibility: inventory pulled OMC dump files, parse mapping CSV rows, load raw
OMC tensors, compare mapped OMC tensors against ONNX `.npz` tensors.

Create:

```text
scripts/compare_omc_layers.py
```

Responsibility: thin CLI wrapper around `internvl_vit_poc.omc_dump_align`.

Modify:

```text
pyproject.toml
```

Responsibility: expose `internvl-compare-omc-layers` CLI entrypoint.

Modify:

```text
tests/test_onnx_intermediate_debug.py
```

Responsibility: add tests for raw input-bin parsing/loading and ONNX dump input
selection.

Create:

```text
tests/test_omc_dump_align.py
```

Responsibility: test OMC dump inventory, mapping CSV parsing, shape/dtype raw
tensor loading, mapped ONNX-vs-OMC comparison.

Create:

```text
docs/onnx-omc-layer-dump-runbook.md
```

Responsibility: exact one-case commands for Windows/WSL ONNX dump, device OMC
dump, `hdc file recv`, inventory, mapping, and compare.

Modify:

```text
docs/index.md
AGENTS.md
```

Responsibility: point future agents to the runbook and plan.

---

### Task 1: Add Raw Input-Bin Support To ONNX Intermediate Dump

**Files:**
- Modify: `src/internvl_vit_poc/onnx_intermediate_debug.py`
- Modify: `tests/test_onnx_intermediate_debug.py`

- [ ] **Step 1: Write failing tests for shape parsing and raw bin loading**

Append these tests to `tests/test_onnx_intermediate_debug.py`:

```python
    def test_parse_shape_accepts_comma_and_x_separators(self):
        from internvl_vit_poc.onnx_intermediate_debug import parse_shape

        self.assertEqual(parse_shape("1,3,448,448"), [1, 3, 448, 448])
        self.assertEqual(parse_shape("1x256x1024"), [1, 256, 1024])

    def test_load_input_array_from_bin_uses_shape_and_dtype(self):
        from internvl_vit_poc.onnx_intermediate_debug import load_input_array

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.bin"
            np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tofile(path)

            array = load_input_array(
                image=None,
                input_bin=path,
                input_shape="1,1,2,2",
                input_dtype="float32",
                image_size=448,
            )

            self.assertEqual(array.shape, (1, 1, 2, 2))
            self.assertEqual(array.dtype, np.float32)
            self.assertAlmostEqual(float(array[0, 0, 1, 1]), 4.0)

    def test_load_input_array_rejects_image_and_bin_together(self):
        from internvl_vit_poc.onnx_intermediate_debug import load_input_array

        with self.assertRaises(ValueError):
            load_input_array(
                image="x.jpg",
                input_bin="x.bin",
                input_shape="1,3,448,448",
                input_dtype="float32",
                image_size=448,
            )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_onnx_intermediate_debug
```

Expected:

```text
ImportError or AttributeError for parse_shape/load_input_array
```

- [ ] **Step 3: Implement shape parsing and raw bin loading**

In `src/internvl_vit_poc/onnx_intermediate_debug.py`, add near the imports:

```python
DTYPES = {
    "float32": np.dtype("<f4"),
    "fp32": np.dtype("<f4"),
    "<f4": np.dtype("<f4"),
    "float16": np.dtype("<f2"),
    "fp16": np.dtype("<f2"),
    "<f2": np.dtype("<f2"),
}
```

Add these functions before `dump_onnx_outputs`:

```python
def parse_shape(value: str) -> list[int]:
    parts = [part for part in re.split(r"[xX,]", value.strip()) if part]
    if not parts:
        raise ValueError("input shape cannot be empty")
    shape = [int(part) for part in parts]
    if any(dim <= 0 for dim in shape):
        raise ValueError(f"input shape must contain positive dimensions: {value}")
    return shape


def dtype_from_name(name: str) -> np.dtype:
    try:
        return DTYPES[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}. Supported: {', '.join(sorted(DTYPES))}") from exc


def load_input_array(
    *,
    image: str | Path | None,
    input_bin: str | Path | None,
    input_shape: str | None,
    input_dtype: str,
    image_size: int,
) -> np.ndarray:
    if image and input_bin:
        raise ValueError("Use either --image or --input-bin, not both")
    if not image and not input_bin:
        raise ValueError("One of --image or --input-bin is required")
    if image:
        return load_static_pixel_values(image, image_size).numpy().astype("float32")
    if not input_shape:
        raise ValueError("--input-shape is required with --input-bin")

    shape = parse_shape(input_shape)
    array = np.fromfile(input_bin, dtype=dtype_from_name(input_dtype)).astype(np.float32, copy=False)
    expected = int(np.prod(shape))
    if array.size != expected:
        raise ValueError(f"{input_bin} has {array.size} elements, expected {expected} for shape {shape}")
    return array.reshape(shape)
```

- [ ] **Step 4: Wire raw input into `dump_onnx_outputs`**

Replace:

```python
pixel_values = load_static_pixel_values(args.image, args.image_size).numpy().astype("float32")
```

with:

```python
pixel_values = load_input_array(
    image=args.image,
    input_bin=args.input_bin,
    input_shape=args.input_shape,
    input_dtype=args.input_dtype,
    image_size=args.image_size,
)
```

In the result payload, add:

```python
"input_bin": args.input_bin,
"input_shape": [int(dim) for dim in pixel_values.shape],
```

- [ ] **Step 5: Update CLI args for dump subcommand**

Change:

```python
dump_parser.add_argument("--image", required=True)
```

to:

```python
dump_parser.add_argument("--image")
dump_parser.add_argument("--input-bin")
dump_parser.add_argument("--input-shape", help="Required with --input-bin, e.g. 1,3,448,448")
dump_parser.add_argument("--input-dtype", default="float32", choices=sorted(DTYPES))
```

- [ ] **Step 6: Run tests and verify pass**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_onnx_intermediate_debug
```

Expected:

```text
OK
```

- [ ] **Step 7: Commit**

```powershell
git add src/internvl_vit_poc/onnx_intermediate_debug.py tests/test_onnx_intermediate_debug.py
git commit -m "Add input-bin support to ONNX intermediate dump"
```

---

### Task 2: Add OMC Dump Inventory Utility

**Files:**
- Create: `src/internvl_vit_poc/omc_dump_align.py`
- Create: `scripts/compare_omc_layers.py`
- Create: `tests/test_omc_dump_align.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write failing tests for inventory**

Create `tests/test_omc_dump_align.py`:

```python
import tempfile
import unittest
from pathlib import Path

import numpy as np


class OmcDumpAlignTests(unittest.TestCase):
    def test_inventory_dump_files_records_sizes_and_candidate_elements(self):
        from internvl_vit_poc.omc_dump_align import inventory_dump_files

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "nested").mkdir()
            np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tofile(root / "nested" / "output_0")
            (root / "log.txt").write_text("ignore text", encoding="utf-8")

            result = inventory_dump_files(root)

            records = result["records"]
            self.assertEqual(len(records), 2)
            output = next(record for record in records if record["relative_path"] == "nested/output_0")
            self.assertEqual(output["byte_count"], 16)
            self.assertEqual(output["fp32_element_count"], 4)
            self.assertEqual(output["fp16_element_count"], 8)
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_omc_dump_align
```

Expected:

```text
ModuleNotFoundError: No module named 'internvl_vit_poc.omc_dump_align'
```

- [ ] **Step 3: Implement inventory module**

Create `src/internvl_vit_poc/omc_dump_align.py`:

```python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .om_onnx_eval import dtype_from_name, read_bin_tensor, save_json
from .onnx_intermediate_debug import load_npz_named
from .onnx_verify import compare_tensors


def inventory_dump_files(root: str | Path) -> dict[str, object]:
    root_path = Path(root)
    records = []
    for path in sorted(p for p in root_path.rglob("*") if p.is_file()):
        byte_count = path.stat().st_size
        records.append(
            {
                "relative_path": path.relative_to(root_path).as_posix(),
                "path": str(path),
                "byte_count": byte_count,
                "fp32_element_count": byte_count // 4 if byte_count % 4 == 0 else None,
                "fp16_element_count": byte_count // 2 if byte_count % 2 == 0 else None,
            }
        )
    return {"root": str(root_path), "count": len(records), "records": records}
```

- [ ] **Step 4: Add CLI skeleton**

Append to `src/internvl_vit_poc/omc_dump_align.py`:

```python
def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory and compare OMC layer dump files.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inventory_parser = subparsers.add_parser("inventory", help="List OMC dump files and byte sizes.")
    inventory_parser.add_argument("--dump-dir", required=True)
    inventory_parser.add_argument("--output-json")
    inventory_parser.add_argument("--output-csv")

    return parser.parse_args(argv)


def save_inventory_csv(path: str | Path, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_path", "path", "byte_count", "fp32_element_count", "fp16_element_count"],
        )
        writer.writeheader()
        writer.writerows(records)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.command == "inventory":
        result = inventory_dump_files(args.dump_dir)
        if args.output_json:
            save_json(args.output_json, result)
        if args.output_csv:
            save_inventory_csv(args.output_csv, result["records"])  # type: ignore[arg-type]
        return result
    raise SystemExit(f"Unknown command: {args.command}")


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, ensure_ascii=False))
```

Create `scripts/compare_omc_layers.py`:

```python
from internvl_vit_poc.omc_dump_align import main


if __name__ == "__main__":
    main()
```

In `pyproject.toml`, add:

```toml
internvl-compare-omc-layers = "internvl_vit_poc.omc_dump_align:main"
```

- [ ] **Step 5: Run tests and CLI help**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_omc_dump_align
uv run --no-sync python scripts\compare_omc_layers.py --help
uv run --no-sync python scripts\compare_omc_layers.py inventory --help
```

Expected:

```text
OK
usage: compare_omc_layers.py ...
```

- [ ] **Step 6: Commit**

```powershell
git add src/internvl_vit_poc/omc_dump_align.py scripts/compare_omc_layers.py tests/test_omc_dump_align.py pyproject.toml
git commit -m "Add OMC dump inventory utility"
```

---

### Task 3: Add Mapping-Based ONNX-vs-OMC Comparator

**Files:**
- Modify: `src/internvl_vit_poc/omc_dump_align.py`
- Modify: `tests/test_omc_dump_align.py`

- [ ] **Step 1: Add failing tests for mapping compare**

Append to `tests/test_omc_dump_align.py`:

```python
    def test_compare_mapped_tensors_reads_onnx_npz_and_raw_omc_file(self):
        from internvl_vit_poc.onnx_intermediate_debug import save_outputs_npz
        from internvl_vit_poc.omc_dump_align import compare_mapped_tensors

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx_npz = root / "onnx.npz"
            save_outputs_npz(onnx_npz, {"encoder.block0": np.array([1.0, 2.0], dtype=np.float32)})
            np.array([1.0, 2.1], dtype="<f4").tofile(root / "omc_output_0")
            mapping = root / "mapping.csv"
            mapping.write_text(
                "name,onnx_name,omc_path,shape,dtype,note\n"
                "block0,encoder.block0,omc_output_0,\"2\",float32,first block\n",
                encoding="utf-8",
            )

            result = compare_mapped_tensors(
                onnx_npz=onnx_npz,
                mapping_csv=mapping,
                base_dir=root,
            )

            self.assertEqual(result["summary"]["count"], 1)
            self.assertEqual(result["summary"]["ok_count"], 1)
            self.assertGreater(result["records"][0]["compare"]["cosine"], 0.99)

    def test_compare_mapped_tensors_reports_missing_onnx_name(self):
        from internvl_vit_poc.onnx_intermediate_debug import save_outputs_npz
        from internvl_vit_poc.omc_dump_align import compare_mapped_tensors

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            onnx_npz = root / "onnx.npz"
            save_outputs_npz(onnx_npz, {"present": np.array([1.0], dtype=np.float32)})
            np.array([1.0], dtype="<f4").tofile(root / "omc_output_0")
            mapping = root / "mapping.csv"
            mapping.write_text(
                "name,onnx_name,omc_path,shape,dtype,note\n"
                "missing,absent,omc_output_0,\"1\",float32,no tensor\n",
                encoding="utf-8",
            )

            result = compare_mapped_tensors(onnx_npz=onnx_npz, mapping_csv=mapping, base_dir=root)

            self.assertEqual(result["summary"]["error_count"], 1)
            self.assertIn("ONNX tensor not found", result["records"][0]["error"])
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_omc_dump_align
```

Expected:

```text
ImportError or AttributeError for compare_mapped_tensors
```

- [ ] **Step 3: Implement mapping parsing and comparison**

Append to `src/internvl_vit_poc/omc_dump_align.py` before `parse_args`:

```python
def parse_mapping_shape(value: str) -> list[int]:
    parts = [part for part in value.replace("x", ",").replace("X", ",").split(",") if part.strip()]
    shape = [int(part.strip()) for part in parts]
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError(f"Invalid shape: {value}")
    return shape


def resolve_mapping_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else base_dir / path


def load_mapping_rows(path: str | Path, base_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"name", "onnx_name", "omc_path", "shape", "dtype"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Mapping CSV missing columns: {sorted(missing)}")
        for row in reader:
            rows.append(
                {
                    "name": row["name"],
                    "onnx_name": row["onnx_name"],
                    "omc_path": resolve_mapping_path(row["omc_path"], base_dir),
                    "shape": parse_mapping_shape(row["shape"]),
                    "dtype": row["dtype"],
                    "note": row.get("note", ""),
                }
            )
    return rows


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    ok = [record for record in records if record.get("status") == "ok"]
    errors = [record for record in records if record.get("status") != "ok"]
    summary: dict[str, object] = {
        "count": len(records),
        "ok_count": len(ok),
        "error_count": len(errors),
    }
    if ok:
        cosines = [float(record["compare"]["cosine"]) for record in ok]  # type: ignore[index]
        summary["cosine_min"] = float(np.min(cosines))
        summary["cosine_mean"] = float(np.mean(cosines))
    else:
        summary["cosine_min"] = None
        summary["cosine_mean"] = None
    return summary


def compare_mapped_tensors(
    *,
    onnx_npz: str | Path,
    mapping_csv: str | Path,
    base_dir: str | Path | None = None,
) -> dict[str, object]:
    mapping_path = Path(mapping_csv)
    resolved_base = Path(base_dir) if base_dir else mapping_path.parent
    onnx_outputs = load_npz_named(onnx_npz)
    rows = load_mapping_rows(mapping_path, resolved_base)
    records: list[dict[str, object]] = []

    for row in rows:
        name = str(row["name"])
        onnx_name = str(row["onnx_name"])
        record: dict[str, object] = {
            "name": name,
            "onnx_name": onnx_name,
            "omc_path": str(row["omc_path"]),
            "shape": row["shape"],
            "dtype": row["dtype"],
            "note": row.get("note", ""),
        }
        try:
            if onnx_name not in onnx_outputs:
                raise KeyError(f"ONNX tensor not found: {onnx_name}")
            onnx_tensor = np.asarray(onnx_outputs[onnx_name], dtype=np.float32)
            omc_tensor = read_bin_tensor(row["omc_path"], shape=row["shape"], dtype=str(row["dtype"]))  # type: ignore[arg-type]
            record["status"] = "ok"
            record["compare"] = compare_tensors(omc_tensor, onnx_tensor)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = str(exc)
        records.append(record)

    return {
        "onnx_npz": str(onnx_npz),
        "mapping_csv": str(mapping_path),
        "summary": summarize_records(records),
        "records": records,
    }
```

- [ ] **Step 4: Add compare CLI**

In `parse_args`, add:

```python
    compare_parser = subparsers.add_parser("compare", help="Compare mapped ONNX npz tensors against OMC raw files.")
    compare_parser.add_argument("--onnx-npz", required=True)
    compare_parser.add_argument("--mapping-csv", required=True)
    compare_parser.add_argument("--base-dir")
    compare_parser.add_argument("--output-json")
    compare_parser.add_argument("--output-csv")
```

Add CSV writer:

```python
def save_compare_csv(path: str | Path, records: list[dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "name",
                "onnx_name",
                "omc_path",
                "shape",
                "dtype",
                "status",
                "cosine",
                "mean_abs_diff",
                "max_abs_diff",
                "error",
                "note",
            ],
        )
        writer.writeheader()
        for record in records:
            compare = record.get("compare", {})
            writer.writerow(
                {
                    "name": record.get("name"),
                    "onnx_name": record.get("onnx_name"),
                    "omc_path": record.get("omc_path"),
                    "shape": ",".join(str(dim) for dim in record.get("shape", [])),
                    "dtype": record.get("dtype"),
                    "status": record.get("status"),
                    "cosine": compare.get("cosine") if isinstance(compare, dict) else None,
                    "mean_abs_diff": compare.get("mean_abs_diff") if isinstance(compare, dict) else None,
                    "max_abs_diff": compare.get("max_abs_diff") if isinstance(compare, dict) else None,
                    "error": record.get("error"),
                    "note": record.get("note"),
                }
            )
```

In `run`, add:

```python
    if args.command == "compare":
        result = compare_mapped_tensors(
            onnx_npz=args.onnx_npz,
            mapping_csv=args.mapping_csv,
            base_dir=args.base_dir,
        )
        if args.output_json:
            save_json(args.output_json, result)
        if args.output_csv:
            save_compare_csv(args.output_csv, result["records"])  # type: ignore[arg-type]
        return result
```

- [ ] **Step 5: Run tests and CLI help**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_omc_dump_align
uv run --no-sync python scripts\compare_omc_layers.py compare --help
```

Expected:

```text
OK
usage: compare_omc_layers.py compare ...
```

- [ ] **Step 6: Commit**

```powershell
git add src/internvl_vit_poc/omc_dump_align.py tests/test_omc_dump_align.py
git commit -m "Add mapped ONNX OMC tensor comparison"
```

---

### Task 4: Add Exact One-Case Runbook For Current Windows/WSL And Device Layout

**Files:**
- Create: `docs/onnx-omc-layer-dump-runbook.md`
- Modify: `docs/index.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Create runbook**

Create `docs/onnx-omc-layer-dump-runbook.md`:

````markdown
# ONNX vs OMC Layer Dump Runbook

This runbook starts from the current state:

```text
FP32 ONNX vs INT8 ONNX visual_features cosine > 0.99
OMC final output vs ONNX cosine ~= 0.96
```

The goal is to find the first layer or block where OMC diverges from ONNX.

Environment boundary:

```text
Blue-zone:
  Windows local workspace
  optional WSL for Python/ONNX commands
  hdc-connected device

Yellow-zone:
  Windows local workspace
  Linux server for DDK/DOPT/OMG quantization and OMC compilation
  hdc-connected device

This runbook is for blue-zone layer dump alignment. It does not run DDK, DOPT,
or OMG; it uses the OMC already present on the device.
```

## 1. Pick One Case

Use the first existing case unless a worse case is known:

```text
case_id = 000000018463
input bin = quantized_v6/om_compare/inputs/000000018463_pixel_values_fp32.bin
device input bin = /data/local/tmp/internvl_om_compare/inputs/000000018463_pixel_values_fp32.bin
```

## 2. Host: List Candidate ONNX Tensors

```bash
uv run python scripts/augment_onnx_outputs.py list \
  --onnx ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --op-type Add,LayerNormalization,MatMul,Gemm \
  --name-regex "vision_model|encoder|mlp1|visual" \
  --max-outputs 300 \
  > ./quantized_v6/layer_debug/onnx_candidates.json
```

For a coarse first pass, prefer tensors like:

```text
/vision_model/embeddings/Add_output_0
/vision_model/encoder/layers.0/Add_1_output_0
/vision_model/encoder/layers.1/Add_1_output_0
...
visual_features
```

## 3. Host: Create Debug ONNX

Create:

```text
quantized_v6/layer_debug/debug_tensors.txt
```

Example:

```text
/vision_model/embeddings/Add_output_0
/vision_model/encoder/layers.0/Add_1_output_0
/vision_model/encoder/layers.1/Add_1_output_0
/vision_model/encoder/layers.2/Add_1_output_0
visual_features
```

Then run:

```bash
uv run python scripts/augment_onnx_outputs.py augment \
  --onnx ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --output-onnx ./quantized_v6/layer_debug/int8.debug.onnx \
  --tensor-file ./quantized_v6/layer_debug/debug_tensors.txt \
  --output-json ./quantized_v6/layer_debug/int8_debug_outputs.json
```

## 4. Host: Dump ONNX From The Same Input Bin

```bash
uv run python scripts/dump_onnx_outputs.py dump \
  --onnx ./quantized_v6/layer_debug/int8.debug.onnx \
  --input-bin ./quantized_v6/om_compare/inputs/000000018463_pixel_values_fp32.bin \
  --input-shape 1,3,448,448 \
  --output-npz ./quantized_v6/layer_debug/onnx/000000018463.npz \
  --output-json ./quantized_v6/layer_debug/onnx/000000018463_dump.json
```

Output:

```text
quantized_v6/layer_debug/onnx/000000018463.npz
quantized_v6/layer_debug/onnx/000000018463.npz.manifest.json
```

## 5. Device: Dump OMC For The Same Input Bin

Windows local hdc path if `hdc` is not in PATH:

```powershell
$hdc = "D:\software\deveco\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
```

Device command:

```bash
cd /data/local/tmp/internvl_om_compare
chmod +x model_run_tool_internal
export LD_LIBRARY_PATH=.
rm -rf dump_000000018463
./model_run_tool_internal \
  --model=/data/local/tmp/internvl_om_compare/split_internvit_v6_exclude_conv_int8_8.omc \
  --input=/data/local/tmp/internvl_om_compare/inputs/000000018463_pixel_values_fp32.bin \
  --output_dir=/data/local/tmp/internvl_om_compare/dump_000000018463 \
  --enable_item=2 \
  --times=1
find /data/local/tmp/internvl_om_compare/dump_000000018463 -type f | head -100
```

From Windows:

```powershell
& $hdc shell "cd /data/local/tmp/internvl_om_compare; chmod +x model_run_tool_internal; export LD_LIBRARY_PATH=.; rm -rf dump_000000018463; ./model_run_tool_internal --model=/data/local/tmp/internvl_om_compare/split_internvit_v6_exclude_conv_int8_8.omc --input=/data/local/tmp/internvl_om_compare/inputs/000000018463_pixel_values_fp32.bin --output_dir=/data/local/tmp/internvl_om_compare/dump_000000018463 --enable_item=2 --times=1"
& $hdc shell "find /data/local/tmp/internvl_om_compare/dump_000000018463 -type f | head -100"
```

## 6. Pull OMC Dump Back To Windows Host

```powershell
New-Item -ItemType Directory -Force D:\proj\internvl-cann-vit-poc\quantized_v6\layer_debug\omc | Out-Null
& $hdc file recv /data/local/tmp/internvl_om_compare/dump_000000018463 D:\proj\internvl-cann-vit-poc\quantized_v6\layer_debug\omc\000000018463
```

If comparing from WSL, keep the repo under `/mnt/d/proj/internvl-cann-vit-poc`
or copy this pulled dump directory into the WSL working copy before running the
Python comparison scripts.

## 7. Host: Inventory OMC Dump

```bash
uv run python scripts/compare_omc_layers.py inventory \
  --dump-dir ./quantized_v6/layer_debug/omc/000000018463 \
  --output-json ./quantized_v6/layer_debug/omc/000000018463_inventory.json \
  --output-csv ./quantized_v6/layer_debug/omc/000000018463_inventory.csv
```

Use the inventory byte sizes to find candidate OMC files. For fp32:

```text
element_count = byte_count / 4
```

## 8. Host: Create Mapping CSV

Create:

```text
quantized_v6/layer_debug/mapping/000000018463_mapping.csv
```

Format:

```csv
name,onnx_name,omc_path,shape,dtype,note
embeddings,/vision_model/embeddings/Add_output_0,../omc/000000018463/path/from/inventory/output_0,"1,1025,1024",float32,patch embedding
```

The `omc_path` is resolved relative to the mapping CSV directory unless it is absolute.

## 9. Host: Compare Mapped Tensors

```bash
uv run python scripts/compare_omc_layers.py compare \
  --onnx-npz ./quantized_v6/layer_debug/onnx/000000018463.npz \
  --mapping-csv ./quantized_v6/layer_debug/mapping/000000018463_mapping.csv \
  --output-json ./quantized_v6/layer_debug/reports/000000018463_onnx_vs_omc_layers.json \
  --output-csv ./quantized_v6/layer_debug/reports/000000018463_onnx_vs_omc_layers.csv
```

Interpretation:

```text
First checkpoint where cosine drops sharply = first suspicious area.
Later bad checkpoints may only be error propagation.
```
````

- [ ] **Step 2: Link runbook from docs index and AGENTS**

Add to `docs/index.md` near the quantization debug documents:

````markdown
```text
onnx-omc-layer-dump-runbook.md
```

One-case Windows/WSL host and device workflow for ONNX intermediate dump, OMC device dump,
OMC dump inventory, manual mapping, and layer-level comparison.
````

Add to `AGENTS.md` documentation map:

```text
docs/onnx-omc-layer-dump-runbook.md one-case ONNX vs OMC layer dump workflow
```

- [ ] **Step 3: Commit**

```powershell
git add docs/onnx-omc-layer-dump-runbook.md docs/index.md AGENTS.md
git commit -m "Document ONNX OMC layer dump workflow"
```

---

### Task 5: Verify End-To-End On Synthetic Local Data

**Files:**
- No new files expected beyond test artifacts under temporary directories.

- [ ] **Step 1: Run unit tests**

Run:

```powershell
uv run --no-sync python -B -m unittest tests.test_onnx_intermediate_debug tests.test_omc_dump_align
```

Expected:

```text
OK
```

- [ ] **Step 2: Run full test suite**

Run:

```powershell
uv run --no-sync python -B -m unittest discover -s tests
```

Expected:

```text
OK
```

- [ ] **Step 3: Run CLI help checks**

Run:

```powershell
uv run --no-sync python scripts\dump_onnx_outputs.py dump --help
uv run --no-sync python scripts\compare_omc_layers.py inventory --help
uv run --no-sync python scripts\compare_omc_layers.py compare --help
```

Expected:

```text
Each command prints usage and exits 0.
```

- [ ] **Step 4: Run diff checks**

Run:

```powershell
git diff --check
git status --short
```

Expected:

```text
No whitespace errors.
Only intended source, test, docs, and pyproject changes are present.
```

- [ ] **Step 5: Commit verification doc update if needed**

If any runbook command changed during testing:

```powershell
git add docs/onnx-omc-layer-dump-runbook.md
git commit -m "Refine ONNX OMC dump runbook"
```

---

### Task 6: Push And Handoff

**Files:**
- No file changes unless verification forces a runbook correction.

- [ ] **Step 1: Push branch**

Run:

```powershell
git push origin main
```

Expected:

```text
main -> main
```

- [ ] **Step 2: Final handoff summary**

Report:

```text
Commit hash
Changed files
Verification commands and results
First command the user should run on Windows/WSL host:
  uv run python scripts/dump_onnx_outputs.py dump ...
First command the user should run on device:
  ./model_run_tool_internal --enable_item=2 ...
```

---

## Execution Notes

Use one image first:

```text
000000018463
```

Use the device model already present:

```text
/data/local/tmp/internvl_om_compare/split_internvit_v6_exclude_conv_int8_8.omc
```

Do not run full `inputs_all` dump until the one-case dump and mapping flow works.

Do not require exact automatic matching between ONNX names and OMC dump names in
the first implementation. The first useful result is a correct manual mapping
for a few block-level checkpoints.
