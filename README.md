# Medical V3：双载荷分权限灰度共享

本主线固定使用 **256-bit 消息**，保留 v2clean 的 Ec/Ew/Dc/Dw 主体、8×8 DCT 和 39/9/16 分频。

**算法与协议唯一权威：** [Protocol v1](医疗图像双载荷分权限可逆灰度共享方案_Protocol_v1.md)。本 README 记录当前工程入口，不改变冻结算法。`specs/001-message-bits-stages` 和 `specs/002-medical-protocol-v1` 保留历史决策与实施记录，其中的旧训练流程已被当前决定取代，不再作为训练指引。

`architecture_version="v3"` 标识当前运行时配置版本；它不表示更换了 v2clean 主体。运行时不支持旧消息长度、旧配置自动迁移或旧检查点部分加载。256-bit 接口、完整 Protocol v1 和主链中的两路密钥机制均已实现。

## 安装与检查

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest -q
```

采用 FP32。正式设备为后续服务器环境；当前工程验证只使用本地 CPU 合成数据。密码原语由 `cryptography` 的 [AESGCM](https://cryptography.io/en/stable/hazmat/primitives/aead/) 和 [Ed25519](https://cryptography.io/en/stable/hazmat/primitives/asymmetric/ed25519/) 提供。

## 模块

| 文件 | 职责 |
| --- | --- |
| `models.py`, `system.py`, `transforms.py` | 原四网络主体；Ew 输入 256；Dw 输出 256；Dc/Dw 提供 DCT 后继续执行接口 |
| `config.py`, `stages.py` | 严格配置校验、联合训练/冻结评估模式和 optimizer 参数集合 |
| `checkpoints.py` | 完整 V3 检查点严格加载、FP32/DCT buffer 检查、文件原始字节 ModelID |
| `data.py` | 已处理工作图与 manifest；随机训练消息和固定验证消息 |
| `crypto.py` | 分用途 HKDF、AES-256-GCM、MSB-first 位序、签名摘要 |
| `keyed_permutation.py` | PRF、分层 Fisher–Yates、15 候选、结构筛选/失真门槛、selector、逆排列 |
| `package.py` | H0 152 bytes、selector 1024、G 262144、签名 64；共 263384 bytes |
| `protocol.py` | Algorithm 1 发布、Algorithm 2 颜色恢复、Algorithm 3 患者恢复 |
| `training.py` | 从头联合训练循环及冻结协议评估入口 |

模块位于 `dual_payload/`。签名为 **普通 Ed25519 签署 SHA256(DPWSIG01 || H0 || selector || G)**，不是 Ed25519ph。

`decode_package()` 仅按固定边界拆分原始字节，**不代表认证成功**。恢复统一经过 `verify_package()`：原始字节验签 → 头部合法性 → ModelID → 有限 FP32 灰度；失败即停止。`ProtocolV1` 还核对所加载模型的接口和冻结状态。

## 训练与评估配置

- **Training: joint_256 from scratch**
- **Evaluation: protocol_eval**

所有输入 JSON 必须显式包含 `model.message_bits: 256` 和 `train.stage`。缺失或非法值直接拒绝。

| 配置 | stage | 训练网络 | 置换 |
| --- | --- | --- | --- |
| `configs/medical_joint_256.json` | `joint_256` | 四网络；总损失一次 backward | 关闭 |
| `configs/medical_protocol_eval.json` | `protocol_eval` | 无；全 eval，requires_grad=False | 开启完整协议 |

`joint_256` 要求 `rgb`、`message`、`carrier`、`range` 四项 loss 权重均为有限正值，0、负值、NaN、Inf 均拒绝；`chroma/luma` 辅助权重可为 0。四网络共同计算总损失，执行一次 backward 和一次 optimizer.step；训练不调用密钥排列、AES-GCM 或签名。配置中的学习率、batch size、epoch 和损失权重继续保留当前工程占位值，尚未确定为 PAD-UFES-20 正式超参数。

`candidate_count=15` 固定。`beta_c/beta_m/min_moved_c/min_moved_m` 保留 `null`，本轮没有替研究者选择数值。joint_256 关闭置换时允许待定；**protocol_eval 配置在填入合法值之前会明确报错**。beta 有限非负，min_moved_c 在 1..39，min_moved_m 在 1..9。

训练命令模板（本轮未执行真实训练）：

```bash
.venv/bin/python train.py --config configs/medical_joint_256.json --output-dir /path/to/new-joint-run
```

先在配置填好工作图 manifest。正式训练每次新建四网络和 optimizer，使用原网络初始化规则，epoch/global_step 从 0 开始；不加载已有权重。`--init-from` 已从 CLI 删除，传入即报未知参数。没有 checkpoint fallback、旧权重迁移或自动 resume 入口。现有循环没有 scheduler，本次未新增调度策略。训练保存 `last.pt`、`best.pt`、`best_message_ber.pt`、配置和验证指标；跨中断精确续训不在本轮接口中。

显式 `--smoke --max-steps 1` 使用合成数据做工程自测。没有 `--smoke` 时，数据缺失直接报错。

### Epoch 历史与 overfit8 工程检查

每次 epoch 的 validation 完成后，输出目录追加一行 `metrics.jsonl` 并立即 flush/fsync。字段固定为：

- `epoch`：与 checkpoint 一致的 **0-based** 编号；`global_step`：累计 optimizer step 数。
- `train_samples`：该轮实际训练的样本次数；`train_loss`：`rgb/chroma/luma/message/carrier/range/total`，按 batch 样本数加权平均。
- `validation`：原样保存 `validate()` 返回的全部指标和 validation loss；`mode`：`joint_256` 或 `overfit8`。

每行 UTF-8 JSON 使用 `ensure_ascii=False`、`allow_nan=False`。若 `--max-steps` 提前结束当轮，沿用现有验证和 checkpoint 边界，历史只统计已执行的训练样本。`validation.json` 仍每轮覆盖，last/best 保存条件及 stdout step 日志保持不变；输出目录仍必须新建或为空，不支持 resume。

显式 `--overfit8` 从配置的 `train_manifest` 中选择 8 行：`sorted(random.Random(config["seed"]).sample(range(N), 8))`，少于 8 行报错。`overfit8_selection.json` 记录 seed、原 manifest 和选中行的 0-based 原始索引、原 path，以及存在时的 patient_id/img_id；path 相对于原 manifest。

overfit8 通过 `Subset(ManifestDataset(training=False, seed=config["seed"]), indices)` 保留原行索引：复用 `fixed_message(原行索引, seed)`，原 manifest 若已有显式 message 则沿用它。训练和 validation 共享相同 8 个固定 image-message pair，跨读取、重建和 epoch 稳定。仅此模式允许二者重合；与 `--smoke` 互斥。

所有训练 checkpoint 新增 `overfit8: true/false`，核心 schema 与加载接口不变。正常训练仍使用 `ManifestDataset(training=True)` 每次随机产生 message，train/val 隔离检查不变。overfit8 是工程 sanity check，不能把它的 checkpoint 或拟合结果当作正式研究结果。本轮只用临时合成图片测试了此入口，未启动真实 overfit 训练。

protocol_eval 命令模板（先填写合法协议参数）：

```bash
.venv/bin/python evaluate.py --config configs/medical_protocol_eval.json \
  --checkpoint /path/to/joint-run/best.pt --manifest /path/to/validation.json \
  --kc-file /path/to/kc.bin --km-file /path/to/km.bin \
  --signing-key-file /path/to/signing-seed.bin --token-file /path/to/token.bin \
  --output-dir /path/to/new-protocol-evaluation
```

密钥输入都是外部已有的原始字节文件：KC/KM 各 32 bytes，签名私钥 seed 32 bytes，Token 16 bytes。评估命令用给定 Token 测试每张图，每次发布使用新的随机参数。它输出 `.dpw` 包和逐图指标，不在日志保存密钥或 Token。实际业务的逐患者 Token 由上游传给 Python 发布 API。

评估报告包括 BER、整条 256-bit 成功、GCM 成功、灰度 PSNR/SSIM、授权 RGB、同消息无排列基线、新增失真、越界比例、两路逐图回退比例。尚未设定科研验收门槛；报告不自动宣布配置通过。

## 独立授权 API

```python
from dual_payload.protocol import ProtocolV1

engine = ProtocolV1(checkpoint_path, device="cpu")
package = engine.publish(rgb_fp32, patient_token, kc, km, signing_key, parameters)
rgb_hat = engine.recover_color(package, kc, trusted_public_key)       # 无需 KM
patient_token = engine.recover_patient(package, km, trusted_public_key)  # 无需 KC
```

`rgb_fp32` 为 `[1,3,256,256]`，范围 `[0,1]`。模型使用文件加载时的同一字节快照绑定 ModelID；不要修改实例内权重、buffer 或模式。重传直接重传原包，不能重用旧 nonce 重新加密。公开 G 和恢复 RGB 都不 clamp。患者认证失败抛出 `InvalidTag`，不返回候选明文。错误 KC 可能产生有限 RGB，签名通过不代表 KC 正确。

## 数据接口

manifest 示例：

```json
{"samples": [{"path": "prepared/image.png", "patient_id": "upstream-id"}]}
```

路径相对于 manifest。图片必须已经是 **RGB 256×256**，读取只将 8-bit RGB 转为 `[0,1]` FP32。拒绝其他尺寸/模式；没有 resize/crop/padding/增强或 EXIF 自动旋转。训练每次产生随机 256-bit；验证可在每条样本提供 `message` 数组（恰好 256 个 0/1），否则按 seed 与行号固定生成。

训练/验证文件重叠会报错；若提供 patient_id，也拒绝已有 ID 重叠。完整的数据准备与分组验收由独立工具负责，见 [数据工具说明](tools/README.md)。

### PAD-UFES-20 prepared v1

用户已冻结数据准备规则：RGB 原通道；RGBA 必须 alpha 全 255 后去 A；256×256 不 resize，两边均小于 256 用 BICUBIC，其余 LANCZOS；输出 RGB 256×256 PNG。不 crop/pad/EXIF 自动旋转/ICC 变换/线性化/增强/归一化。正式 `data.py` 保持工作图读取接口。

独立入口 `tools/prepare_pad_ufes20.py` 使用 seed=2026，固定 train/val/test 患者数 961/206/206；基于多诊断患者统计向量优化 70/15/15 组成，所有 lesion 按 `(patient_id, lesion_id)` 计数。源数据只读，输出目录禁止覆盖，完整验收后发布。输出清单路径相对 manifest，可整体迁移。

数据准备不启动训练；batch/lr/epoch、augmentation 和研究质量门槛未在此选择。

## 证据边界

- 系数数组上的 inverse(forward(c)) 可逐值一致；FP32 IDCT→DCT 往返仍有舍入误差。
- beta 限制的是置换新增失真，不能证明医学灰度质量。
- 受控 decoder 的载荷链路测试验证认证接口，不代表未训练网络能恢复 256-bit。
- 真实 256-bit 训练、无钥/错钥/学习型攻击、临床质量、CUDA/H100 与独立实现互通仍待验证。
- 尚未启动真实医疗数据训练。代码测试通过不等于 256-bit 可靠恢复、颜色保护效果或医学灰度质量已经验证。
- 冻结方案文档保留原文；当前工程入口见本 README 与 HANDOFF。历史 implementation-report 中的测试数字仅描述当时状态。
