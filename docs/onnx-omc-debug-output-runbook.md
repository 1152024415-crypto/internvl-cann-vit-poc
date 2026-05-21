# ONNX vs OMC Debug Multi-Output Runbook

Use this when `model_run_tool_internal --enable_item=2` cannot produce usable
runtime dump files on the phone.

Observed phone-side failure modes:

```text
Older runtime:
  Profiling mode: maintain type 0 -> supported
  Dump mode:      maintain type 1 -> hiaiserver says "Not supported Type: 1"

Newer runtime:
  Dump is enabled.
  Model load succeeds.
  Input fill succeeds.
  Run fails with status:1.
  data_proc_tool_internal says "No profiling/dump file."
  hiaiserver fd count jumps from ~33 to ~7555, mostly anon_inode:dmabuf.
```

So the working strategy is:

```text
Do not use runtime dump.
Add selected intermediate tensors as real ONNX graph outputs.
Compile that debug ONNX to debug OMC in the yellow-zone DDK environment.
Run debug OMC with normal model_run_tool_internal inference.
Compare output_0/output_1/... against ONNX Runtime outputs in the same order.
```

This workflow still uses the same fixed input:

```text
input = pixel_values [1, 3, 448, 448] fp32
```

## 1. Pick One Case

Start with one existing input bin:

```text
case_id = 000000018463
host input = quantized_v6/om_compare/inputs/000000018463_pixel_values_fp32.bin
device input = /data/local/tmp/internvl_om_compare/inputs/000000018463_pixel_values_fp32.bin
```

## 2. Host: List Candidate ONNX Outputs

Run in the repo checkout that has the INT8 ONNX:

```bash
uv run python scripts/dump_onnx_outputs.py list \
  --onnx ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --op-type Add,LayerNormalization,MatMul,Gemm \
  --name-regex "vision_model|encoder|mlp1|visual" \
  --max-outputs 300 \
  > ./quantized_v6/layer_debug/onnx_candidates.json
```

For the first pass, choose coarse checkpoints:

```text
embedding output
encoder block 0 output
encoder block 1 output
...
encoder final output
pixel_shuffle / projector related outputs
visual_features
```

Do not start by adding every tensor. A huge multi-output model may be slow,
large, or harder for OMG to compile.

## 3. Host: Create Debug Tensor List

Create:

```text
quantized_v6/layer_debug/debug_tensors.txt
```

Example format:

```text
/vision_model/embeddings/Add_output_0
/vision_model/encoder/layers.0/Add_1_output_0
/vision_model/encoder/layers.1/Add_1_output_0
/vision_model/encoder/layers.2/Add_1_output_0
visual_features
```

Tensor names must match actual ONNX graph tensor names from Step 2.

## 4. Host: Generate Debug Multi-Output ONNX

```bash
uv run python scripts/dump_onnx_outputs.py augment \
  --onnx ./quantized_v6/split_internvit_v6_exclude_conv_int8_8.onnx \
  --output-onnx ./quantized_v6/layer_debug/split_internvit_debug_outputs.onnx \
  --tensor-file ./quantized_v6/layer_debug/debug_tensors.txt \
  --output-json ./quantized_v6/layer_debug/split_internvit_debug_outputs.augment.json
```

Important output-order rule:

```text
Original ONNX outputs remain first.
New debug tensors are appended after original outputs.
model_run_tool_internal should therefore write them as:
  output_0 = original output, usually visual_features
  output_1 = first appended debug tensor
  output_2 = second appended debug tensor
  ...
```

## 5. Host: Dump ONNX Runtime Outputs For The Same Input Bin

```bash
uv run python scripts/dump_onnx_outputs.py dump \
  --onnx ./quantized_v6/layer_debug/split_internvit_debug_outputs.onnx \
  --input-bin ./quantized_v6/om_compare/inputs/000000018463_pixel_values_fp32.bin \
  --input-shape 1,3,448,448 \
  --output-npz ./quantized_v6/layer_debug/onnx/000000018463_debug_outputs.npz \
  --output-json ./quantized_v6/layer_debug/onnx/000000018463_debug_outputs.json
```

Outputs:

```text
000000018463_debug_outputs.npz
000000018463_debug_outputs.npz.manifest.json
000000018463_debug_outputs.json
```

The JSON and manifest preserve `output_names`, which is the order used later to
map `output_N` files.

## 6. Yellow-Zone: Compile Debug ONNX To Debug OMC

Run this in the yellow-zone DDK environment, not blue-zone:

```bash
./ddk/tools/tools_omg/omg \
  --model ./quantized_v6/layer_debug/split_internvit_debug_outputs.onnx \
  --output ./quantized_v6/layer_debug/split_internvit_debug_outputs \
  --framework 5 \
  --input_shape='pixel_values:1,3,448,448' \
  --compress_conf=./quantized_v6/split_internvit_v6_exclude_conv_int8_8_param \
  2>&1 | tee ./quantized_v6/layer_debug/split_internvit_debug_outputs_omg.log
```

Expected output:

```text
quantized_v6/layer_debug/split_internvit_debug_outputs.omc
quantized_v6/layer_debug/split_internvit_debug_outputs_omg.log
```

The debug ONNX only adds graph outputs. It should not change weights or the DOPT
quantization parameter directory.

## 7. Device: Run Debug OMC Without Dump Mode

Send the debug OMC to the device:

```powershell
$hdc = "D:\software\deveco\DevEco Studio\sdk\default\openharmony\toolchains\hdc.exe"
& $hdc file send .\quantized_v6\layer_debug\split_internvit_debug_outputs.omc /data/local/tmp/internvl_om_compare/split_internvit_debug_outputs.omc
```

Run normal inference:

```powershell
& $hdc shell "cd /data/local/tmp/internvl_om_compare; export LD_LIBRARY_PATH=.; rm -rf debug_outputs_000000018463; mkdir -p debug_outputs_000000018463; ./model_run_tool_internal --model=/data/local/tmp/internvl_om_compare/split_internvit_debug_outputs.omc --input=/data/local/tmp/internvl_om_compare/inputs/000000018463_pixel_values_fp32.bin --output_dir=/data/local/tmp/internvl_om_compare/debug_outputs_000000018463 --times=1"
```

Inspect outputs:

```powershell
& $hdc shell "ls -la /data/local/tmp/internvl_om_compare/debug_outputs_000000018463"
```

Expected:

```text
output_0
output_1
output_2
...
```

## 8. Pull Debug OMC Outputs Back

```powershell
& $hdc file recv /data/local/tmp/internvl_om_compare/debug_outputs_000000018463 D:\proj\internvl-cann-vit-poc\quantized_v6\layer_debug\omc_debug_outputs\000000018463
```

## 9. Host: Compare ONNX Outputs Against OMC output_N Files

```bash
uv run python scripts/compare_omc_layers.py compare-indexed \
  --onnx-npz ./quantized_v6/layer_debug/onnx/000000018463_debug_outputs.npz \
  --om-output-dir ./quantized_v6/layer_debug/omc_debug_outputs/000000018463 \
  --output-names-json ./quantized_v6/layer_debug/onnx/000000018463_debug_outputs.json \
  --cosine-threshold 0.99 \
  --output-json ./quantized_v6/layer_debug/reports/000000018463_debug_outputs_compare.json \
  --output-csv ./quantized_v6/layer_debug/reports/000000018463_debug_outputs_compare.csv
```

Read:

```bash
jq '.summary, .first_bad' ./quantized_v6/layer_debug/reports/000000018463_debug_outputs_compare.json
```

Interpretation:

```text
first_bad = first output_N with missing file, shape mismatch, or cosine below threshold
```

That output tells us where to narrow next. If block 5 is the first bad coarse
checkpoint, build a second debug ONNX with internal tensors from block 5 only.

## 10. Why This Replaces Runtime Dump

`--enable_item=2` is not reliable enough for this InternVL ViT+projector model.
Depending on the phone image, it either fails before model load or enables dump
and then fails during execution.

Old failure:

```text
AiOmService::maintain type: 1, action: 1
OmManagerFactory::GetManager: Not supported Type: 1
```

New failure, captured on DEL 6.0.0.271 / API 26:

```text
Dump is enabled.
Inference: loading model split_internvit_v6_exclude_conv_int8_8 succeeded.
Inference: filling input tensors succeeded.
Run model failed. status:1.
No profiling/dump file.
```

Key hilog lines to inspect:

```text
DirectEngine Model Execute failed: modelId = 0 payload:streamId=65535 taskId=65535 errCode=0x30
rtClientExecuteModel: modelId = 0 execute error, errno = 0x56080030
npu_graph_executor_client.cc Execute: rtModelExecute failed: rtRet=0x0x56080030
npucl_dump_binary.h DumpBinaryFile: write op output file failed
SetResult: model maintaince failed, error = 0x56060004
FdLeakInfo: hiaiserver fd handler leaked, fdcnt:7567
FdLeakDumpState: leakType: dmabuf not support
```

Faultlog files can contain an additional hiaiserver native crash:

```text
/data/log/faultlog/faultlogger/cppcrash-hiaiserver-*.log
/data/log/faultlog/temp/cppcrash-*
/data/log/faultlog/temp/crashjsonstack-*

Reason: Signal:SIGSEGV(SEGV_MAPERR)
#00 /vendor/lib64/libai_infra_rpc_server.so
```

Normal inference works:

```text
model_run_tool_internal --times=1
```

So debug multi-output OMC uses the reliable path and makes intermediate tensors
ordinary model outputs instead of relying on unsupported runtime dump.

After any failed `--enable_item=2` run, check and usually reboot the phone before
continuing:

```bash
pidof hiaiserver
ls /proc/<hiaiserver_pid>/fd | wc -l
cat /proc/<hiaiserver_pid>/limits
```

If fd count is thousands or near `Max open files = 32768`, reboot before running
more tests. Otherwise normal inference may start failing with `dup failed,
errno:24`.
