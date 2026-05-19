# ONNX Quantization Debug Runbook

Use this when FP32 ONNX accuracy is high but INT8/OM accuracy drops. The goal is
to separate three possible failure locations:

```text
FP32 ONNX -> INT8 ONNX quantization
INT8 ONNX -> OMG/OM conversion
OM -> device runtime execution
```

## Step 1: Batch-Compare FP32 ONNX And INT8 ONNX

Run both ONNX models on the same test images:

```bash
uv run python scripts/eval_onnx_pair.py \
  --fp32-onnx ./onnx/split_internvit_projector_fixed_b1_no_resize.onnx \
  --int8-onnx ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --image-dir ./test_images \
  --fp32-output-name visual_features \
  --int8-output-name visual_features \
  --output-json ./quantized_v6/fp32_vs_int8_eval.json \
  --output-csv ./quantized_v6/fp32_vs_int8_eval.csv
```

If the model output is classification logits, add labels:

```bash
uv run python scripts/eval_onnx_pair.py \
  --fp32-onnx ./fp32_classifier.onnx \
  --int8-onnx ./int8_classifier.onnx \
  --image-dir ./test_images \
  --labels-csv ./test_labels.csv \
  --compute-top1 \
  --output-json ./quantized_v6/fp32_vs_int8_eval.json \
  --output-csv ./quantized_v6/fp32_vs_int8_eval.csv
```

`test_labels.csv` format:

```csv
image,label
dog/0001.jpg,151
cat/0002.jpg,282
```

Interpretation:

```text
FP32 ONNX = 99, INT8 ONNX = 99, OM = 96
  Quantized ONNX is probably OK. Compare INT8 ONNX vs OM/device dump.

FP32 ONNX = 99, INT8 ONNX = 96, OM = 96
  DOPT quantization itself caused the drop. Do ONNX intermediate comparison.

FP32 ONNX is not 99
  The evaluation script, preprocessing, output node, labels, or postprocess is wrong.
```

If the output is `visual_features` / `visual_tokens`, it is an embedding rather
than classification logits. In that case the script still reports cosine and
diff between FP32 and INT8 outputs, but accuracy only makes sense if a classifier
or label postprocess is included.

## Step 2: Select Intermediate ONNX Tensors

Start with block-level outputs instead of dumping every node.

List candidate tensors:

```bash
uv run python scripts/augment_onnx_outputs.py list \
  --onnx ./onnx/split_internvit_projector_fixed_b1_no_resize.onnx \
  --op-type Add,LayerNormalization,MatMul,Gemm \
  --name-regex "vision_model|encoder|mlp1|visual" \
  --max-outputs 300 \
  > ./quantized_v6/fp32_candidates.json
```

Create a text file with tensor names to dump:

```bash
cat > ./quantized_v6/debug_tensors.txt << 'EOF'
# Put selected tensor names here, one per line.
# Prefer encoder block outputs first, then narrow down inside the first bad block.
EOF
```

If names are mostly preserved across FP32 and INT8 ONNX, use the same tensor
file for both models. If DOPT renamed tensors, create one tensor list per model
with semantically matching tensors.

## Step 3: Create Debug ONNX Files

```bash
uv run python scripts/augment_onnx_outputs.py augment \
  --onnx ./onnx/split_internvit_projector_fixed_b1_no_resize.onnx \
  --output-onnx ./quantized_v6/fp32.debug.onnx \
  --tensor-file ./quantized_v6/debug_tensors.txt \
  --output-json ./quantized_v6/fp32_debug_outputs.json

uv run python scripts/augment_onnx_outputs.py augment \
  --onnx ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --output-onnx ./quantized_v6/int8.debug.onnx \
  --tensor-file ./quantized_v6/debug_tensors.txt \
  --output-json ./quantized_v6/int8_debug_outputs.json
```

Alternative: add outputs directly by op type and regex:

```bash
uv run python scripts/augment_onnx_outputs.py augment \
  --onnx ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --output-onnx ./quantized_v6/int8.debug.onnx \
  --op-type Add,LayerNormalization,MatMul,Gemm \
  --name-regex "vision_model|encoder|mlp1|visual" \
  --max-outputs 300
```

## Step 4: Dump One Or More Images

Start with a misclassified image from `fp32_vs_int8_eval.csv`.

```bash
uv run python scripts/dump_onnx_outputs.py dump \
  --onnx ./quantized_v6/fp32.debug.onnx \
  --image ./test_images/bad_case.jpg \
  --output-npz ./quantized_v6/dumps/fp32_bad_case.npz

uv run python scripts/dump_onnx_outputs.py dump \
  --onnx ./quantized_v6/int8.debug.onnx \
  --image ./test_images/bad_case.jpg \
  --output-npz ./quantized_v6/dumps/int8_bad_case.npz
```

Each `.npz` has a sidecar manifest:

```text
fp32_bad_case.npz.manifest.json
int8_bad_case.npz.manifest.json
```

The manifest preserves original ONNX tensor names even when they contain `/`.

## Step 5: Compare Intermediate Outputs

```bash
uv run python scripts/compare_onnx_intermediates.py compare \
  --left-npz ./quantized_v6/dumps/fp32_bad_case.npz \
  --right-npz ./quantized_v6/dumps/int8_bad_case.npz \
  --output-json ./quantized_v6/dumps/fp32_vs_int8_bad_case_layers.json
```

The compare result is sorted by the worst cosine first. Use it together with
network order:

```text
First layer where cosine sharply drops = likely root-cause area.
Later layers with large error may only be error propagation.
```

Typical next quantization experiments:

```text
Use BINARY calibration generated by the exact Python preprocessing.
Increase calibration images.
Exclude the first bad block or its qkv/proj/fc weights from quantization.
Exclude the projector if the error appears after mlp1.
```

## Step 6: If INT8 ONNX Is Good But OM Is Bad

Then compare:

```text
INT8 ONNX debug outputs
vs
OM/device layer dump outputs from tools_sysdbg/model_run_tool_internal
```

The checked-in scripts compare `.npz` files. If the device dump is raw binary or
text, first convert each dumped tensor into `.npz` with the same tensor names.
Then run `scripts/compare_onnx_intermediates.py compare`.

Keep these artifacts for review:

```text
fp32_vs_int8_eval.json
fp32_vs_int8_eval.csv
debug_tensors.txt
fp32_debug_outputs.json
int8_debug_outputs.json
fp32_vs_int8_bad_case_layers.json
device dump logs and tensor files, if OM-side comparison is needed
```
