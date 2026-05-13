# Documentation Index

This repository keeps detailed context in focused documents instead of one
large instruction file.

## Current Mainline

Start here:

```text
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

## Artifact Documents

```text
release-artifacts.md
```

GitHub Release asset workflow and large-file policy.

```text
next-steps.md
```

Active next steps and known risks.
