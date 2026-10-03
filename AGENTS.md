# Medical V3 project constraints

The current user request supersedes iteration 001. Implement the complete frozen Protocol v1 with mandatory 256-bit payloads. Preserve v2clean Ec/Ew/Dc/Dw bodies and 39/9/16 bands. No runtime legacy mode, partial legacy checkpoint loader, medical preprocessing decisions, or real-data training. Report algorithm/spec conflicts rather than inventing alternatives. Synthetic engineering tests are not research results.

Current entry points: README.md and HANDOFF.md. Training is joint_256 from scratch only; evaluation is frozen protocol_eval. No Stage A, checkpoint initialization, or migration. All four core loss weights must be finite and positive. specs/001 and specs/002 preserve historical records; their former training flow is superseded.

Algorithm and protocol authority: 医疗图像双载荷分权限可逆灰度共享方案_Protocol_v1.md in this repository. Code, README and old specs are not algorithm authorities. Do not modify the frozen algorithm.

<!-- SPECKIT START -->
Current data preparation plan: [PAD-UFES-20 prepared v1](specs/003-pad-prepared-v1/plan.md). The current user explicitly authorizes this frozen preprocessing and patient split on real data; model training remains prohibited.
<!-- SPECKIT END -->
