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
