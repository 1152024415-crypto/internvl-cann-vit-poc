# Documentation Index

This repository keeps detailed context in focused documents instead of one
large instruction file.

## Current Mainline

Start here:

```text
yellow-zone-agent-handoff.md
current-status.md
stage-4-vit-projector-chain.md
```

It describes the current ViT + projector path:

```text
image -> vision_model -> pixel_shuffle -> mlp1 -> visual_tokens
```

For fast lookup, use:

```text
search-guide.md
```

## Stage Documents

```text
stage-1-model-split-baseline.md
```

PyTorch loading, model splitting, fixed `1 x 3 x 448 x 448` preprocessing, and
baseline tensors.

```text
stage-2-onnx-export.md
```

ONNX export choices, opset 18, FlashAttention disabled, static position
embedding, ONNX Runtime validation.

```text
stage-3-onnx-to-om.md
```

CANN Kit OMG setup, ONNX to OM conversion, and current runtime verification
limits.

```text
stage-4-vit-projector-chain.md
```

The current deployment artifact chain and verification numbers for dog/cat
images.

```text
stage-5-int8-quantization-runbook.md
```

INT8 DOPT quantization and OMG conversion path based on the successful demo.

```text
onnx-quantization-debug-runbook.md
```

How to compare FP32 and INT8 ONNX accuracy, dump ONNX intermediate tensors, and
decide whether to debug DOPT quantization or OM/device execution.

## Artifact Documents

```text
release-artifacts.md
```

GitHub Release asset workflow and large-file policy.

```text
yellow-zone-agent-handoff.md
```

One-page entrypoint for a yellow-zone AI/engineer. It gives the read order,
current OM-regeneration requirement, platform-plugin clarification, deliverables,
device gates, and a ready-to-use prompt.

```text
yellow-zone-validation-runbook.md
```

Step-by-step yellow-zone DevEco/HarmonyOS device validation guide.

```text
yellow-zone-onnx-to-om-ddk-runbook.md
```

Yellow-zone DDK/OMG ONNX-to-OM regeneration guide with required logs,
static-shape gates, precheck, OM-to-JSON inspection, and upload checklist.

```text
next-steps.md
```

Active next steps and known risks.
