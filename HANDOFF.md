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

The authoritative architecture specification is [docs/network2/network2_architecture.md](docs/network2/network2_architecture.md).

## Confirmed baseline parameters

The working design record is [configs/v2_clean_baseline_tbd.md](configs/v2_clean_baseline_tbd.md). Key confirmed values are:

- Seed: `2026`
- Image size: `256`
- Message length: 64 bits
- `delta_c=2/255`, `delta_w=2/255`, `eps=1e-8`
- Adam, learning rate `1e-4`, weight decay `0`, gradient clipping `1`
- Loss weights: RGB `1.0`, message `1.0`, carrier `1.0`, range `0.1`; chroma and luma `0.0`
- Float clean channel: no quantization, identity, no clamp

## Batch semantics

- `batch_size` is the logical/effective batch size and the DataLoader batch size.
- `gradient_accumulation_steps` splits one logical batch into micro-batches.
- Gradients do not accumulate across DataLoader batches.
- Each DataLoader iteration performs one `optimizer.step()`.

## Current TBD

- Training and validation data scale
- Logical/effective batch size
- Gradient accumulation and resulting micro-batch size
- `epochs`, `max_steps`, and total optimizer-step budget
- `num_workers`
- Train, validation, and output paths
- Random-message or fixed-message experiment organization
- BF16, AMP, and TF32 strategy

## Server deployment principles

- Clone and use `feature/network-v2-clean`, not the default historical branch.
- Determine all dataset and output paths after deployment to the server.
- Do not inherit local Windows absolute paths.
- The final formal training config has not been generated yet.

## Cleanup status

The current V2 branch has removed the old V1 configs, Network1 documentation, `joint_10x20_v1` assets, and legacy analysis scripts.
