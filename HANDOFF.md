# Network V2 Handoff

## Current Git state

- Active branch: `feature/network-v2-clean`
- `main` and `baseline-v1` preserve the Network V1 history.
- Network V2 requires `architecture_version=v2` and refuses Network V1 checkpoints.

## Network V2 core

- Input: `256×256` RGB images
- Payload: 64-bit binary message
- Recovery: blind color and message recovery from the received grayscale carrier
- Modules: Color Encoder (`Ec`), Watermark Encoder (`Ew`), Color Decoder (`Dc`), and Watermark Decoder (`Dw`)
- Architecture version: `v2`


## Confirmed baseline parameters

The formal single-H100 configuration is [configs/v2_clean_baseline_h100.json](configs/v2_clean_baseline_h100.json). Key values are:

- Seed: `2026`
- Image size: `256`
- Message length: 64 bits
- `delta_c=2/255`, `delta_w=2/255`, `eps=1e-8`
- Adam, learning rate `1e-4`, weight decay `0`, gradient clipping `1`
- Loss weights: RGB `1.0`, message `1.0`, carrier `1.0`, range `0.1`; chroma and luma `0.0`
- Float clean channel: no quantization, identity, no clamp
- Full DIV2K: 800 train images and 100 validation images
- Logical/effective batch size: `16`; micro-batch size: `16`
- Gradient accumulation steps: `1`; DataLoader workers: `8`
- Budget: at most `50,000` optimizer steps (`epochs=1000`, 50 steps per complete epoch)
- Messages: fresh random 64-bit messages during training and deterministic messages during validation
- Runtime precision: FP32; BF16, AMP, and TF32 are disabled
- Output: `/data/zwc/zyh/experiments/dual-payload-watermark/network-v2/v2_clean_div2k_full_fp32_b16_ga1_seed2026_run01`

## Batch semantics

- `batch_size` is the logical/effective batch size and the DataLoader batch size.
- `gradient_accumulation_steps` splits one logical batch into micro-batches.
- Gradients do not accumulate across DataLoader batches.
- Each DataLoader iteration performs one `optimizer.step()`.

## Server deployment principles

- Clone and use `feature/network-v2-clean`, not the default historical branch.
- Use one H100 selected through `CUDA_VISIBLE_DEVICES`; the current trainer is single-GPU.
- Train data: `/data/zwc/Data/DIV2K/DIV2K_train_HR`.
- Validation data: `/data/zwc/Data/DIV2K/DIV2K_valid_HR`.
- The configured output directory must be new or empty for a fresh run.
- Do not inherit local Windows absolute paths.
- Start with `python train.py --config configs/v2_clean_baseline_h100.json --gradient-accumulation-steps 1`.

## Cleanup status

The current V2 branch has removed the old V1 configs, Network1 documentation, `joint_10x20_v1` assets, and legacy analysis scripts.
