# OM vs INT8 ONNX Compare Runbook

Use this after FP32 ONNX vs INT8 ONNX has already been checked. The purpose is
to isolate whether the accuracy/numerical drop happens after OMG/OM or runtime:

```text
INT8 ONNX reference output
vs
OM output
```

The comparison target is still the ViT + projector output:

```text
input  = pixel_values   [1, 3, 448, 448] fp32
output = visual_tokens  [1, 256, 1024]   fp32 by default
```

Do not compare by feeding images directly into one side and `.bin` tensors into
the other. Both paths must use the same `pixel_values` tensor.

## Step 0: Prepare Paths

Run this in the server checkout:

```bash
cd internvl-cann-vit-poc
git pull origin main
```

Example paths used below:

```text
INT8 ONNX:
quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx

Image set:
calib_images/

Work directory:
success_demo/quant/om_compare/

OM output directory to be filled by the DDK/OM runtime tool:
success_demo/quant/om_compare/om_outputs/
```

Inputs for this step:

```text
repo checkout
INT8 ONNX file
image directory
```

Outputs for this step:

```text
none, only path confirmation
```

## Step 1: Generate Input Bins And INT8 ONNX Reference Bins

Command:

```bash
uv run python scripts/eval_om_against_onnx.py prepare \
  --onnx ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --image-dir ./calib_images \
  --work-dir ./success_demo/quant/om_compare \
  --output-name visual_features
```

If your INT8 ONNX output name is `visual_tokens`, use:

```bash
--output-name visual_tokens
```

If you only want to test a few images first:

```bash
--max-images 5
```

Inputs:

```text
./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx
./calib_images/*
```

Outputs:

```text
success_demo/quant/om_compare/manifest.json
success_demo/quant/om_compare/om_run_plan.csv
success_demo/quant/om_compare/inputs/*_pixel_values_fp32.bin
success_demo/quant/om_compare/int8_onnx_outputs/*_visual_tokens_int8_onnx_fp32.bin
success_demo/quant/om_compare/om_outputs/    # expected output directory, initially empty
```

Meaning of these files:

```text
manifest.json
  Main machine-readable manifest. It records image path, input bin path, INT8
  ONNX reference output path, expected OM output path, and tensor shapes.

om_run_plan.csv
  Human/tool-friendly list for the DDK/OM runtime step. Each row tells which
  input bin to feed to OM and what output filename should be produced.

inputs/*_pixel_values_fp32.bin
  Raw fp32 input tensor. Shape is [1, 3, 448, 448].

int8_onnx_outputs/*_visual_tokens_int8_onnx_fp32.bin
  Raw fp32 INT8 ONNX reference output. Shape is [1, 256, 1024].
```

## Step 2: Run The OM With The Same Input Bins

This step depends on the yellow-zone DDK tool you use to execute `.om`. Use the
DDK/OM model-run tool to feed each file listed in:

```text
success_demo/quant/om_compare/om_run_plan.csv
```

Each row has:

```csv
case_id,image,input_bin,expected_om_output_bin,onnx_output_bin
```

Inputs:

```text
your generated .om file
success_demo/quant/om_compare/inputs/*_pixel_values_fp32.bin
success_demo/quant/om_compare/om_run_plan.csv
```

Required output naming if possible:

```text
success_demo/quant/om_compare/om_outputs/<case_id>_visual_tokens_om_fp32.bin
```

Example:

```text
case_id = 000000018463

input:
success_demo/quant/om_compare/inputs/000000018463_pixel_values_fp32.bin

expected OM output:
success_demo/quant/om_compare/om_outputs/000000018463_visual_tokens_om_fp32.bin
```

Outputs:

```text
success_demo/quant/om_compare/om_outputs/*_visual_tokens_om_fp32.bin
```

If your DDK tool cannot control output filenames, create this mapping file:

```text
success_demo/quant/om_compare/om_output_map.csv
```

Format:

```csv
case_id,om_output_bin
000000018463,ddk_dump/output_0.bin
000000018468,ddk_dump/output_1.bin
```

Relative paths in `om_output_map.csv` are resolved from:

```text
success_demo/quant/om_compare/
```

## Step 3: Compare INT8 ONNX Reference Bins Against OM Output Bins

If OM outputs follow the default filename convention:

```bash
uv run python scripts/eval_om_against_onnx.py compare \
  --manifest ./success_demo/quant/om_compare/manifest.json \
  --om-output-dir ./success_demo/quant/om_compare/om_outputs \
  --output-json ./success_demo/quant/om_compare/int8_onnx_vs_om.json \
  --output-csv ./success_demo/quant/om_compare/int8_onnx_vs_om.csv
```

If OM outputs require a map file:

```bash
uv run python scripts/eval_om_against_onnx.py compare \
  --manifest ./success_demo/quant/om_compare/manifest.json \
  --om-output-dir ./success_demo/quant/om_compare/om_outputs \
  --om-output-map ./success_demo/quant/om_compare/om_output_map.csv \
  --output-json ./success_demo/quant/om_compare/int8_onnx_vs_om.json \
  --output-csv ./success_demo/quant/om_compare/int8_onnx_vs_om.csv
```

If the OM runtime dump is fp16 instead of fp32:

```bash
--om-output-dtype float16
```

Inputs:

```text
success_demo/quant/om_compare/manifest.json
success_demo/quant/om_compare/om_outputs/*_visual_tokens_om_fp32.bin
optional: success_demo/quant/om_compare/om_output_map.csv
```

Outputs:

```text
success_demo/quant/om_compare/int8_onnx_vs_om.json
success_demo/quant/om_compare/int8_onnx_vs_om.csv
```

## Step 4: Read The Result

Quick summary:

```bash
jq '.summary' ./success_demo/quant/om_compare/int8_onnx_vs_om.json
```

Worst cases by cosine:

```bash
jq '.records | map(select(.status=="ok")) | sort_by(.compare.cosine) | .[:20] | .[] | {case_id,image,cosine:.compare.cosine,mean_abs:.compare.mean_abs_diff,max_abs:.compare.max_abs_diff}' \
  ./success_demo/quant/om_compare/int8_onnx_vs_om.json
```

Missing OM outputs:

```bash
jq '.records[] | select(.status=="missing_om_output") | {case_id,image,error}' \
  ./success_demo/quant/om_compare/int8_onnx_vs_om.json
```

Important fields:

```text
summary.count
  Number of images in the manifest.

summary.valid_count
  Number of images with both INT8 ONNX reference output and OM output.

summary.missing_count
  Number of OM outputs not found.

summary.cosine_mean
  Average similarity between INT8 ONNX and OM outputs.

summary.cosine_min
  Worst image similarity.

summary.mean_abs_diff_mean
  Average absolute error.

summary.max_abs_diff_max
  Largest single-element absolute error.
```

## Step 5: Decide The Next Debug Target

Use these initial thresholds:

```text
cosine_mean >= 0.995 and cosine_min not far below 0.99
  OM output is numerically close to INT8 ONNX. Next debug postprocess or the
  final classification/business chain.

cosine_mean < 0.995 or a cluster of low-cosine images
  The mismatch likely happens in OMG conversion, OM runtime, or output dtype /
  layout handling.

missing_count > 0
  The OM run step is incomplete. Do not interpret cosine until all required
  output bins exist or the manifest is intentionally narrowed.
```

If OM vs INT8 ONNX is bad, pick the worst image from
`int8_onnx_vs_om.csv` and then do layer-level dump:

```text
INT8 ONNX intermediate dump
vs
OM/device layer dump
```

Start from the earliest layer where cosine drops sharply. Later bad layers may
only be error propagation.
