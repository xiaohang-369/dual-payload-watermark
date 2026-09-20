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

The complete architecture specification is in [docs/network2/network2_architecture.md](docs/network2/network2_architecture.md).

## Training baseline

- Optimizer: Adam
- Learning rate: `1e-4`
- Objectives: RGB, message, carrier, and range losses
- Channel: float clean, identity, no clamp

## Current status

Network V2 is implemented. The formal experiment scale, logical batch size, gradient accumulation, epoch/step budget, and server paths are still being determined.

The current parameter design draft is [configs/v2_clean_baseline_tbd.md](configs/v2_clean_baseline_tbd.md). It is a planning template, not a ready-to-run formal training configuration.

## Entry points

- `train.py`: training
- `evaluate.py`: checkpoint evaluation
- `diagnose_watermark.py`: watermark diagnostics

## Repository structure

```text
dual_payload/                         Network V2 models, system, data, losses, and training logic
tests/                                Network V2 tests
configs/v2_clean_baseline_tbd.md      Baseline parameter and TBD design draft
docs/network2/                        Network V2 architecture specification
train.py                              Training entry point
evaluate.py                           Evaluation entry point
diagnose_watermark.py                 Diagnostic entry point
```
