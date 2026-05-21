# ONNX vs OMC Layer Dump Alignment Design

## Problem Statement

The current quantization chain has passed the first numerical gate:

```text
FP32 ONNX visual_features
vs
INT8 ONNX visual_features
cosine > 0.99
```

But the device-side heterogeneous OMC execution is still off:

```text
INT8/FP32 ONNX visual_features
vs
device OMC visual_features
cosine ~= 0.96
```

This means the likely problem is after INT8 ONNX generation:

```text
INT8 ONNX -> OMG/OMC compile -> device heterogeneous execution
```

The next task is not to improve quantization yet. The next task is to find the
first layer or block where OMC output diverges significantly from ONNX output.

## Goals

1. Dump comparable intermediate tensors from ONNX and device OMC for the same
   `pixel_values_fp32.bin` input.
2. Compare layer/block outputs with cosine, mean absolute error, and maximum
   absolute error.
3. Identify the earliest layer/block where error sharply increases.
4. Produce a repeatable workflow that can start with one image and scale to a
   small set of bad cases.

## Non-Goals

1. Do not rerun DOPT quantization as part of this task.
2. Do not change OMC generation parameters until we have layer-level evidence.
3. Do not compare labels or classification accuracy in this workflow.
4. Do not start by dumping every ONNX node for every image.
5. Do not assume OMC dump tensor names will match ONNX tensor names exactly.

## Current Artifacts

Host-side artifacts:

```text
FP32 ONNX model
INT8 ONNX model
quantized_v6/om_compare/inputs/*_pixel_values_fp32.bin
quantized_v6/om_compare/int8_onnx_outputs/*_visual_tokens_int8_onnx_fp32.bin
quantized_v6/om_compare/manifest.json
quantized_v6/om_compare/om_run_plan.csv
```

Environment boundary:

```text
Blue-zone current environment:
  Windows local workspace
  WSL if Linux-style Python execution is needed
  hdc-connected device

Yellow-zone environment:
  Windows local workspace
  Linux server for DDK/DOPT/OMG quantization and OMC compilation
  hdc-connected device
```

This layer-dump alignment task should run in the blue-zone environment unless a
new OMC must be generated. DDK, DOPT quantization, and OMG OMC compilation are
yellow-zone responsibilities and are not part of this implementation.

Device-side artifacts:

```text
/data/local/tmp/internvl_om_compare/
  inputs/
  int8_onnx_outputs/
  model_run_tool_internal
  OMC/OM model file
```

Observed device tool capability:

```text
./model_run_tool_internal --model=... --input=... --output_dir=... --enable_item=2
```

`--enable_item=2` is the dump switch according to the tool help.

## Key Design

Use a two-sided dump workflow:

```text
same pixel_values_fp32.bin
        |
        +--> host ONNX Runtime debug ONNX -> ONNX intermediate dump
        |
        +--> device model_run_tool_internal -> OMC intermediate dump
```

The workflow must be driven by tensor input bins, not source images. This keeps
preprocessing out of the comparison and makes the ONNX and OMC paths consume the
same bytes.

The first pass should be block-level, not node-level:

```text
patch embedding output
encoder block 0 output
encoder block 1 output
...
encoder final output
pixel_shuffle output
mlp1/projector output
visual_features
```

If the first bad block is found, the second pass narrows inside that block:

```text
LayerNorm
QKV / attention projection
attention score / softmax if visible
attention output projection
MLP fc1 / activation / fc2
residual Add
```

## Architecture

### ONNX Side

ONNX Runtime only returns graph outputs. To dump intermediate tensors, selected
intermediate tensor names are temporarily appended to the ONNX graph outputs.

Existing code already supports most of this:

```text
src/internvl_vit_poc/onnx_intermediate_debug.py
scripts/augment_onnx_outputs.py
scripts/dump_onnx_outputs.py
scripts/compare_onnx_intermediates.py
```

Required extension:

```text
scripts/dump_onnx_outputs.py must support --input-bin and --input-shape.
```

Reason:

```text
Current dump flow can consume images. The OMC path consumes pixel_values bins.
To avoid preprocessing mismatch, ONNX dump must consume the same input bin.
```

Expected ONNX dump output:

```text
quantized_v6/layer_debug/onnx/<case_id>.npz
quantized_v6/layer_debug/onnx/<case_id>.npz.manifest.json
```

The manifest preserves original ONNX tensor names.

### OMC Side

OMC is executed on the physical device using `model_run_tool_internal`.

First command shape:

```bash
cd /data/local/tmp/internvl_om_compare
export LD_LIBRARY_PATH=.
./model_run_tool_internal \
  --model=<model.omc-or-om> \
  --input=/data/local/tmp/internvl_om_compare/inputs/<case_id>_pixel_values_fp32.bin \
  --output_dir=/data/local/tmp/internvl_om_compare/dump_<case_id> \
  --enable_item=2 \
  --times=1
```

Expected OMC dump output:

```text
/data/local/tmp/internvl_om_compare/dump_<case_id>/
```

The exact file layout is tool-defined and must be inspected after the first run.

Then pull it back:

```bash
hdc file recv \
  /data/local/tmp/internvl_om_compare/dump_<case_id> \
  quantized_v6/layer_debug/omc/<case_id>
```

### Alignment Side

The OMC dump may not use ONNX tensor names. The alignment should therefore be
manifest-driven and allow manual mapping:

```csv
onnx_name,omc_path,shape,dtype,note
encoder.layers.0.output,/path/to/omc/file.bin,1,1025,1024,float32,block0
```

The first version can be semi-manual:

1. List ONNX selected tensor names and shapes.
2. List OMC dumped files and byte sizes.
3. Infer candidate shapes from byte size.
4. Compare only plausible pairs.
5. Write the confirmed mapping to CSV.

The comparison output should be sorted by execution order and include:

```text
case_id
stage/block name
onnx tensor name
omc dump file
shape
dtype
cosine
mean_abs_diff
max_abs_diff
status
```

## Data Flow

### Step 1: Choose One Case

Use one input bin first:

```text
quantized_v6/om_compare/inputs/000000018463_pixel_values_fp32.bin
```

If a known bad case exists from final-output OMC comparison, use that instead.

### Step 2: Select ONNX Checkpoints

Start with coarse checkpoints. Selection can be based on ONNX node names,
op types, and regex:

```text
vision_model
encoder
layers.<N>
mlp1
visual
```

### Step 3: Create Debug ONNX

Create a temporary ONNX with selected tensors as graph outputs.

Input:

```text
INT8 ONNX or FP32 ONNX
debug_tensors.txt
```

Output:

```text
quantized_v6/layer_debug/onnx_debug/int8.debug.onnx
```

### Step 4: Dump ONNX With Input Bin

Input:

```text
int8.debug.onnx
000000018463_pixel_values_fp32.bin
```

Output:

```text
quantized_v6/layer_debug/onnx/000000018463.npz
```

### Step 5: Dump OMC On Device

Input:

```text
device OMC/OM model
/data/local/tmp/internvl_om_compare/inputs/000000018463_pixel_values_fp32.bin
```

Output:

```text
/data/local/tmp/internvl_om_compare/dump_000000018463/
```

Then pull the dump directory back to the Windows/WSL host through `hdc file recv`.

### Step 6: Build Mapping

Input:

```text
ONNX dump manifest
OMC dump file list
OMC file sizes
```

Output:

```text
quantized_v6/layer_debug/mapping/000000018463_mapping.csv
```

### Step 7: Compare

Input:

```text
ONNX .npz
OMC dump files
mapping.csv
```

Output:

```text
quantized_v6/layer_debug/reports/000000018463_onnx_vs_omc_layers.json
quantized_v6/layer_debug/reports/000000018463_onnx_vs_omc_layers.csv
```

## Comparison Strategy

The first bad location is the first checkpoint where one of these happens:

```text
cosine drops sharply compared with the previous checkpoint
mean_abs_diff jumps sharply
max_abs_diff becomes an obvious outlier
shape/layout mismatch appears
```

Possible interpretations:

```text
patch embedding already bad
  input layout, dtype, or first Conv exclusion / quantization boundary issue

block N output first bad
  inspect block N internals next

only projector/mlp1 bad
  vision encoder probably fine; focus on pixel_shuffle/projector ops

final output bad but block outputs good
  output layout, final reshape, or dump interpretation issue
```

## Risks And Mitigations

### OMC Tensor Names Do Not Match ONNX

Mitigation:

```text
Use byte size, shape, execution order, and manual mapping CSV.
Do not require automatic exact name matching in v1.
```

### OMC Graph Is Fused

Mitigation:

```text
Start with block-level checkpoints.
Only inspect internals after finding the first bad block.
```

### OMC Dump Dtype Is Not FP32

Mitigation:

```text
Mapping CSV must include dtype.
Comparator must support fp32 and fp16 at minimum.
```

### Layout Differs

Mitigation:

```text
Comparator should report shape mismatch clearly.
If byte size matches but cosine is bad, test likely reshapes/transposes before
calling it numerical mismatch.
```

### Dump Volume Is Too Large

Mitigation:

```text
Start with one image and coarse checkpoints.
Scale only after the first bad area is known.
```

## Implementation Units

1. Extend ONNX dump CLI to accept raw input bins:

```text
--input-bin
--input-shape
--input-dtype
```

2. Add OMC dump file inventory helper:

```text
list files
byte sizes
candidate shapes by dtype
optional output JSON/CSV
```

3. Add mapped ONNX-vs-OMC comparator:

```text
--onnx-npz
--mapping-csv
--output-json
--output-csv
```

4. Add a runbook with exact one-case commands:

```text
host ONNX dump
device OMC dump
hdc recv
mapping
compare
```

## Acceptance Criteria

1. The workflow can dump ONNX intermediate outputs from a
   `pixel_values_fp32.bin` input without reading the original image.
2. The workflow can inventory an OMC dump directory pulled from the device.
3. The workflow can compare at least one manually mapped ONNX tensor and OMC
   dump tensor.
4. The report identifies cosine, mean absolute error, and max absolute error
   per mapped checkpoint.
5. The first pass uses one image and a small set of block-level checkpoints.
6. Documentation clearly states which commands run on the host and which run on
   the device through `hdc shell`.

## References

ONNX Runtime Python API supports running selected output names:

```text
session.run([output names], inputs)
```

Source:

```text
https://onnxruntime.ai/docs/api/python/api_summary.html
```

ONNX shape inference can enrich graph value information for intermediate
tensors:

```text
https://onnx.ai/onnx/api/shape_inference.html
```

Local reference:

```text
success_demo/说明文档.docx
```

The local doc records that `tools_sysdbg/model_run_tool_internal` is used to
dump every layer output on device.
