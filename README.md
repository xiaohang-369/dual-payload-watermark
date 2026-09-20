# Dual-Payload Watermarking

Network V2 dual-payload watermarking system for color recovery and 64-bit watermark/message recovery from a single grayscale carrier.

## Current setup

- Input: `256×256` RGB images
- Architecture: `architecture_version=v2`
- Message: 64-bit binary payload
- Recovery: blind decoding from the received carrier only

## Core modules

- `Ec`: Color Encoder
- `Ew`: Watermark Encoder
- `Dc`: Color Decoder
- `Dw`: Watermark Decoder


## Training baseline

- Optimizer: Adam
- Learning rate: `1e-4`
- Objectives: RGB, message, carrier, and range losses
- Channel: float clean, identity, no clamp

## Current status

Network V2 is implemented. The single-H100 FP32 baseline configuration is ready at [configs/v2_clean_baseline_h100.json](configs/v2_clean_baseline_h100.json). It uses the full DIV2K split, logical batch size 16, random 64-bit training messages, and a 50,000 optimizer-step budget.

## Entry points

- `train.py`: training
- `evaluate.py`: checkpoint evaluation
- `diagnose_watermark.py`: watermark diagnostics

## Repository structure

```text
dual_payload/                         Network V2 models, system, data, losses, and training logic
tests/                                Network V2 tests
configs/v2_clean_baseline_h100.json   Formal single-H100 FP32 baseline configuration
train.py                              Training entry point
evaluate.py                           Evaluation entry point
diagnose_watermark.py                 Diagnostic entry point
```
