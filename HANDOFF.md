# 双载荷灰度系统：新对话交接

更新日期：2026-09-09。本文记录当前代码与已收到的实验结果，不代表网络已经达到性能要求。新对话请先阅读本文，再检查实际代码和报告；不要重新搭建项目或直接启动训练。

## 1. 用户要求与当前目标

- 用中文沟通。用户希望指出设计和表达中的错误，不能一味认可。
- 用户明确要求：需要运行程序、安装环境、下载数据、执行测试或训练时，给出 PowerShell 命令，由用户亲自执行。可以阅读代码、日志、配置，并按任务编辑代码。
- 工作目录：`D:\02_代码项目（AI、课程与竞赛）\paper`。
- 当前只做 **float + identity clean**，先做好无攻击颜色恢复与水印提取，再处理量化和攻击。不要现在加入 JPEG、噪声、FFT、Transformer 或 attention。
- 不覆盖原训练检查点。不把诊断实验权重误当成正式训练续训权重。
- 本次交接仅新增本文，没有改网络、配置、权重或实验结果；没有运行测试和训练。

长期目标：将 RGB 图像的颜色信息和独立 64-bit 消息同时嵌入单通道灰度载体，接收端仅凭载体恢复 RGB 和消息，后续研究常见攻击下的鲁棒性。

## 2. 当前代码目录：一个文件对应一个职责

以下路径均相对于上述项目根目录；以磁盘上的源文件为准。

```text
paper/
├── HANDOFF.md                    本交接说明
├── README.md                     环境、网络约定、训练/评估/诊断使用说明
├── pyproject.toml                项目依赖、打包与 pytest 设置
├── .gitignore                    排除环境、缓存、数据集与训练输出
├── train.py                      正式 clean 训练入口
├── evaluate.py                   检查点评估入口
├── diagnose_watermark.py         水印诊断入口，inspect / overfit 两种模式
├── configs/
│   ├── clean_float.json          当前 float + identity 基线配置
│   └── clean_8bit.json           后续 STE 8-bit clean 配置，目前未进入此阶段
├── dual_payload/
│   ├── __init__.py               Python 包初始化
│   ├── transforms.py             固定颜色转换、8×8 DCT、频带投影与 RMS Cap
│   ├── models.py                 四个网络及残差块，网络结构全部在这里
│   ├── system.py                 四网络串联、通道与盲解码接口
│   ├── channel.py                量化、identity 攻击及 AttackResult
│   ├── losses.py                 RGB/色度/亮度/消息/载体/越界损失
│   ├── metrics.py                PSNR、SSIM、BER、整条消息成功率等
│   ├── data.py                   图片读取、预处理、固定验证消息、合成 smoke 数据
│   ├── config.py                 默认参数、严格 JSON 配置合并与检查
│   ├── training.py               训练、评估、日志、预览图、检查点与续训实现
│   └── diagnostics.py            消息敏感性、梯度检查、小样本水印过拟合
├── tests/
│   ├── conftest.py               测试随机种子和线程设置
│   ├── test_transforms.py        颜色转换、DCT、投影与幅度预算测试
│   ├── test_models.py            网络输出、梯度、初始化与 clean 分离关系测试
│   ├── test_channel.py           identity、量化及相关行为测试
│   ├── test_data_metrics.py      数据与指标测试
│   ├── test_training.py          训练/续训一致性、评估和保护逻辑测试
│   └── test_diagnostics.py       诊断不改权重、消息库、过拟合隔离与输出保护测试
├── datasets/DIV2K/               已下载并解压的数据集，不需要重新下载
├── runs/                         正式训练与独立诊断结果
└── .venv/                        已安装的 Python 环境
```

原有测试用户反馈 `29 passed in 11.46s`；随后新增了 7 个诊断测试。用户已运行 inspect 和 overfit，但对话未收到新增测试的完整 pytest 结果，不能宣称全部新测试通过。缓存文件存在也不能当作测试通过证据。

## 3. 数学约定与完整链路

```text
RGB → 固定 YCbCr → Color Encoder → S
64-bit message + S → Watermark Encoder → X_float
→ Quantization → Attack Layer → attacked_image
                            ├→ Color Decoder → 固定颜色逆变换 → RGB_hat
                            └→ Watermark Decoder → 64 logits
```

`DualPayloadSystem.decode(attacked_image)` 不接收原 RGB、真实消息、目标或有效区域掩码；两个 Decoder 接收同一个 attacked_image。

### 输入与颜色转换

输入 float32、gamma-coded sRGB，范围 `[0,1]`，形状 B×3×H×W；H/W 是 8 的正整数倍。不进行 linear RGB 转换，不做额外均值方差标准化。

```text
Y  = 0.299R + 0.587G + 0.114B
Cb = (B - Y) / 1.772
Cr = (R - Y) / 1.402

R = Y + 1.402Cr
B = Y + 1.772Cb
G = Y - (0.114×1.772/0.587)Cb - (0.299×1.402/0.587)Cr
```

Cb/Cr 为零中心，不带 +0.5 偏置。绿色逆变换使用代数推导系数，而不是提前截断的小数。

### DCT 频带与预算

使用正交归一化、左上角对齐、不重叠 8×8 block-DCT，系数索引为 u×8+v，其中 u,v=0…7。

| 投影 | 系数范围 | 个数 | 用途 |
| --- | --- | ---: | --- |
| P0 | u+v≤2 或 u+v≥11 | 16 | 保留频带 |
| Pw | u+v=3 或 4 | 9 | 水印残差频带 |
| Pc | 5≤u+v≤10 | 39 | 颜色残差频带 |

水印 Decoder 的 9 个系数顺序固定为 `(0,3),(1,2),(2,1),(3,0),(0,4),(1,3),(2,2),(3,1),(4,0)`。

RMS Cap 对每张图使用一个标量：

```text
r = sqrt(mean(V²) + eps)
gain = min(1, delta/r)
residual = gain × V
```

当前 δc=2/255；δw=2/255 是暂定实验初值，不是已经论证的最优预算；eps=1e-8。投影后没有逐像素激活或裁剪。RMS 预算不保证每个像素的幅度或载体范围合法。触发 Cap 不等于所有梯度都变成零。

## 4. 四个网络的真实实现

正式默认主干均为 64 通道、8 个残差块，总可训练参数 2,469,701。没有 Transformer、attention、归一化层或主干上下采样。

残差块：`x + Conv3×3(activation(Conv3×3(x)))`，相加后不再激活。前三个网络使用 ReLU，Watermark Decoder 使用 LeakyReLU(0.1)。

### Color Encoder（Ec）

```text
concat(Y,Cb,Cr)
→ Conv3×3(3→64)，此处不接激活
→ 8×ReLU 残差块
→ 线性 Conv3×3(64→1)，残差头零初始化
→ Pc 投影 → RMS Cap(δc) → Δc
S = Y + Δc
```

### Watermark Encoder（Ew）

```text
Bw = S - Pc(S)
图像支路：Bw → Conv3×3(1→64) → ReLU
消息支路：64 bit 映射到 ±1 → Linear64→128 → ReLU → Linear128→64
消息 embedding 广播到整张特征图
concat(图像特征, 消息特征) → Conv3×3(128→64) → ReLU
→ 8×ReLU 残差块 → 线性 Conv3×3(64→1)，残差头零初始化
→ Pw 投影 → RMS Cap(δw) → Δw
X_float = S + Δw
```

消息只有广播，没有空间位置编码或独立空间消息图。平坦宿主上，广播消息经卷积得到的内部近常量成分可能被 Pw 去掉，边界和纹理可产生其他响应；这是待检查的表达限制，不是已证实的唯一根因。

### Color Decoder（Dc）

```text
Z  = attacked_image - Pw(attacked_image)
Zc = Pc(attacked_image)
concat(Z,Zc) → Conv3×3(2→64) → ReLU → 8×ReLU 残差块
├→ Conv3×3(64→2) → Cb_hat, Cr_hat
└→ Conv3×3(64→1) → ΔY_raw
Y_hat = Z + (Pc+Pw)(ΔY_raw)
RGB_hat = 固定 YCbCr 逆变换(Y_hat,Cb_hat,Cr_hat)
```

不是自由输出 RGB 的网络。两个输出头非零 Xavier 初始化。P0 亮度约束在最终 RGB 裁剪之前成立到数值误差范围；显示时的 RGB clamp 可能破坏该代数关系。

### Watermark Decoder（Dw）

```text
attacked_image
→ 固定 9 个 DCT 卷积核，kernel=8、stride=8，无可训练参数但保留输入梯度
→ Conv3×3(9→64) → LeakyReLU(0.1)
→ 8×LeakyReLU 残差块
→ 线性 Conv1×1(64→64)
→ 全局空间平均 GAP → 64 个 raw logits
```

没有 sigmoid 输出；训练使用 BCEWithLogits，判 bit 时用 `logits >= 0`。没有纠错码。该网络主干 Kaiming normal，bit head Xavier；其他主干采用 PyTorch 默认 fan-in 尺度，偏置为零。

### clean 分离关系的边界

理想精确投影下：Ew 的 Bw 不依赖 Δc；Dc 的两路输入消除 Δw；Dw 的 Pw 系数消除 Δc，但宿主 Y 的 Pw 系数仍然存在。单通道载体不是纯水印残差。

量化、clamp、攻击可能破坏这种理想分离，尤其几何变换通常不与固定 block-DCT 投影交换。不能直接把 clean 的结论推广到攻击后。

## 5. 通道、损失与训练状态

当前 `quantization_mode=none, clamp_enabled=false, attack_mode=identity`，因此实际通道恒等，不是把图像乘 0。

代码已有 none / ste8 / real8；STE 对 rounding 使用近似恒等梯度，clamp 仍保留真实梯度；real8 只供评估。实际攻击仅实现 identity，非 identity 会报错，不会静默跳过。未来仍需实现具体攻击、空间对齐目标和掩码指标，并非改一个开关就能完成鲁棒训练。

AttackResult 包含 attacked_image、targets 字典、valid_mask、attack_info。当前 targets 是原 RGB、Y、Cb/Cr，mask 全 1，目标不送入 Decoder。

默认正式联合损失：

```text
1.0 × L1(RGB_hat, RGB)
+ 1.0 × BCEWithLogits(logits, message)
+ 1.0 × MSE(X_float, Y)
+ 0.1 × mean(relu(-X_float)² + relu(X_float-1)²)
```

色度 L1 和亮度 L1 会记录，但权重均为 0；RGB 损失在裁剪前计算。范围损失只做软约束。

当前正式实验是四网络从头联合 clean 训练，并没有实际执行历史讨论中的 Color-only 预训练分阶段流程。不要把曾建议的训练顺序误记为已完成实验。

默认 seed=2026，FP32 Adam、lr=1e-4、weight_decay=0、grad_clip=1、batch_size=2、256×256、num_workers=0，未启用 AMP/TF32。训练消息每 batch 重新随机生成，与图片索引独立；验证消息固定。训练随机裁剪、验证中心裁剪，均先短边缩放。

指标包含载体 PSNR/SSIM、RGB raw 与 clipped PSNR/SSIM、BER、Bit Accuracy、64-bit 整条消息成功率、越界比例和残差 RMS。预览图依次是原 RGB、灰度载体、恢复 RGB；只在展示导出时转 8-bit。

正式检查点包含网络、优化器、配置、随机状态、训练位置。`--resume` 恢复同一 run，`--max-steps` 是总步数；`--init-from` 只加载权重进入新 run。`best.pt` 按验证总损失选择，不等于 RGB PSNR 或 BER 最优。

## 6. 环境和数据，不要重复安装下载

- Windows / PowerShell；环境 Python 位于 `.venv\Scripts\python.exe`。
- 用户日志已确认 torch=2.14.0+cu132、device=cuda；已知显卡 RTX 4060 Laptop，8 GB。
- 原有 29 项测试通过，正式主干 synthetic smoke 3 步跑通。
- DIV2K：`datasets\DIV2K\DIV2K_train_HR` 共 800 张；`datasets\DIV2K\DIV2K_valid_HR` 共 100 张。
- 原先下载中断问题已解决，压缩包已续传完成并解压。不需要再处理下载。

## 7. 已完成实验与证据

以下实验路径均位于 `D:\02_代码项目（AI、课程与竞赛）\paper\runs\`。

### A. 正式联合 float clean：1000 步

目录：`clean_none_20260909_151508_769857`。重要文件：last.pt、best.pt、config.json、manifest.json、metrics.jsonl、preview.png。

| global_step | 载体 PSNR | RGB PSNR（raw） | BER | 整条消息成功率 |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 39.32 dB | 18.07 dB | 49.52% | 0% |
| 400 | 39.12 dB | 21.16 dB | 50.27% | 0% |
| 800 | 39.10 dB | 22.37 dB | 49.89% | 0% |
| 1000 | 39.11 dB | 22.00 dB | 48.84% | 0% |

1000 步验证消息 BCE=0.69407，接近概率全部为 0.5 的 ln(2)≈0.69315。δc/δw RMS 接近 2/255。载体越界约 0.509%，恢复 RGB 越界约 1.522%。颜色有进步但预览仍有错色；颜色指标提高也不证明 Δc 传输了有效色度，需要后续去残差等消融。

### B. inspect：同图换消息与梯度检查

目录：`watermark_inspect_20260909_154226_773288`。完整结果 `report.json`，另有 manifest.json、metrics.jsonl。使用原正式 1000 步 last.pt，4 张训练图片，每张 4 条相同消息库，无权重更新。

- 换消息后限幅残差变化 RMS≈0.0051～0.0060，logits 变化 RMS≈0.0105～0.0126。
- 正确配对 BER≈50.4%～51.2%；BCE 比错配标签略低，差不足 0.001。
- 所查残差全部触发 Cap，平均缩放系数约 0.38～0.48；消息影响未被完全消除。
- 消息损失对 Ew 的梯度范数≈0.00720，对消息 MLP≈0.000837，对 Dw≈0.24437；均有限且非零。
- 本次联合梯度总范数≈0.97345，预计 clip gain=1，因此这个 batch 没触发全局裁剪；不能推断历史所有 batch。
- 消息损失对 Ec 梯度≈3e-10、对 Dc 无梯度，符合理想 clean 分离关系到数值误差。

结论：没有发现消息通路完全断开或完全被投影删除；但有响应和梯度不代表能可靠解码，根因尚未定位。

### C. 小样本 watermark-only 过拟合：4 图 × 4 消息，500 步

目录：`watermark_overfit_20260909_154345_580268`。重要文件：report.json、manifest.json、metrics.jsonl、diagnostic_weights.pt。

实验从正式 1000 步 last.pt 加载权重；冻结 Ec/Dc、固定中心裁剪并缓存 S，只训练 Ew/Dw。使用 4 张图与 4 条固定消息的全部 16 个组合，每图不是唯一标签；另有 4 条不重叠消息作同图留出评估。

**与正式联合训练不同**：仅优化消息 BCE；没有 RGB/载体/范围损失；Adam 重新建立，沿用 lr=1e-4、wd=0、clip=1，但只裁剪水印参数；δw 不变。该实验用于诊断，不是联合训练性能结果。

| 项目 | 第 0 步 | 第 500 步 |
| --- | ---: | ---: |
| 拟合消息 BER | 50.59% | 17.87% |
| 拟合消息 BCE | 0.6942 | 0.3553 |
| 拟合消息整条成功率 | 0% | 25% |
| 留出消息 BER（同图） | 54.20% | 52.05% |
| 留出消息 BCE（同图） | 0.6990 | 3.3473 |
| 留出消息整条成功率 | 0% | 0% |

完全不看图、只学每位先验的拟合 BER 对照=31.25%、BCE=0.54784。拟合效果超过对照，说明学到部分消息相关区分；但没有把 16 个组合全部拟合好。25% 表示 4/16 个组合成功，不能仅凭汇总断定是哪一条消息或哪张图。

第 450 步拟合 BER=16.31%，末步回到 17.87%；后期记录的梯度范数约 3.4～15，多次触发裁剪。说明有波动，但不等于证明梯度爆炸或学习率是根因。

仅区分 4 条预设消息只需约 2 bit 的选择信息，不能证明任意 64-bit 容量。留出消息只有 4 条且在同图上，不能作为正式泛化测试；高 BCE 伴随近随机 BER 表示存在更自信的错误预测。

诊断权重使用 `diagnostic_format_version=1`，不是正式 `format_version=1`，不能交给 train.py 的 --resume/--init-from。该诊断入口未实现中途续训。

## 8. 当前判断：不要提前下结论

已知：训练/前向/反向可运行；水印信号受消息影响；有限消息库上可学到部分区分；正式随机消息提取尚未成功。

尚未知：广播消息＋频带投影是否是主要表达瓶颈，宿主 Pw 干扰占多大作用，Decoder/GAP 的限制，以及初始化、优化与 RMS Cap 的相对影响。不能声称已证明 DCT 不行、FFT 更好、预算一定太小、梯度被完全截断，或单纯换 attention 就能解决。

后续攻击目标、几何同步、掩码、几何变换后的频带泄漏问题仍未冻结。之前讨论过 Swin/ConvNeXt/NAF、cross-attention，以及 HiDDeN/RivaGAN 等比较方法，但这些均未加入当前实现。

## 9. 下一步：上一轮刚建议，尚未收到执行结果

> 2026-09-10 更新：本节建议的单图实验及等曝光对照已经完成；结果与新的随机消息诊断见第 11 节。本节保留为当时的实验依据。

建议做 **1 张图 × 4 条消息、500 步** 的独立对照，去掉多宿主变化，其他设置不改。必须从原正式 last.pt 开始，而不是从刚才诊断权重续训。

先问用户是否已有该实验输出；没有时给下列命令，由用户运行，不自行启动：

```powershell
Set-Location -LiteralPath 'D:\02_代码项目（AI、课程与竞赛）\paper'
.\.venv\Scripts\python.exe diagnose_watermark.py --mode overfit --checkpoint ".\runs\clean_none_20260909_151508_769857\last.pt" --data-dir ".\datasets\DIV2K\DIV2K_train_HR" --device cuda --images 1 --messages 4 --steps 500
```

如果单图能拟合、多图不行，宿主变化值得进一步检查，但不是单次实验已证明根因；若单图也不能拟合，优先检查消息表达、解码和优化。可进一步增加逐图片/逐消息指标，分清哪些组合失败，而不是只看平均 BER。若提出架构修改，先解释证据、最小改动和对照设计，不同时改变多个因素。

## 10. 新对话建议阅读顺序

1. 本文和 README.md。
2. dual_payload/models.py、transforms.py、system.py、channel.py。
3. configs/clean_float.json、losses.py、training.py、diagnostics.py。
4. 两份诊断 report.json、正式训练 metrics.jsonl 与 config.json。
5. 如果用户已完成单图实验，优先读新结果，再决定下一步。

当前没有必要重装环境、重下 DIV2K、重写四网络或直接继续正式长训练。

## 11. 2026-09-10：单图结果与随机消息诊断

### 已完成的单图固定码本实验

`watermark_overfit_20260909_171704_422347` 使用原正式 `last.pt@1000`，固定 `0001.png`，只训练 Ew/Dw，在 1 张图与 4 条固定拟合消息上运行 500 步。第 200 步拟合 BER 已为 0、整串成功率为 100%；第 500 步拟合 BCE 为 `2.13e-6`。但同图 4 条留出消息的最终 BER 为 `48.83%`、BCE 为 `10.76`、整串成功率为 0。这证明当前通路能在固定宿主上记住 4 个码字，不证明任意 64-bit 泛化。

`watermark_overfit_20260910_131422_374525` 是 1 图×4消息×125步的等组合曝光对照。每个图像—消息组合出现 62 或 63 次，与旧 4图×4消息×500步相同。单图结果为拟合 BER `7.03%`、BCE `0.1794`，优于四图的 `17.87%`、`0.3553`；但两者整串成功率均为 25%，单图留出消息 BER 仍为 `48.05%`。因此宿主变化是强嫌疑，但固定消息记忆仍不能回答正式训练所需的随机消息泛化。

### 随机消息模式与已完成结果

`diagnose_watermark.py --mode random_messages` 固定一张图片并缓存 S，冻结 Ec/Dc，只训练 Ew/Dw。每一步从独立 CPU RNG 生成全局互异的新 64-bit 消息，并严格排除另一独立 RNG 预先生成的固定验证库。默认验证库 256 条，分批前向；报告匹配验证消息的 BCE、BER、整串成功率、64 位逐位 BER，以及错配标签和移除水印残差的负对照。

用户执行测试得到 `41 passed`。1000 步运行 `watermark_random_messages_20260910_141045_164122` 的匹配验证 BER 从 `49.5850%` 降至 `44.5496%`，错配标签和移除水印分别为 `49.7253%`、`49.9207%`。5000 步独立重跑 `watermark_random_messages_20260910_142828_770669` 在公共 step 0/1/500/1000 上逐值复现；最终匹配 BER `41.8640%`、BCE `0.666081`，错配 `49.8962%`，移除水印 `49.5056%`，整串成功率仍为 0。最终 62/64 位 BER 低于 50%，说明是广泛但很弱的未见消息信号，不是只记四个码字。

硬 BER 在 step 3500 已达到 `41.8640%`，之后到 step 5000 仅在约 `41.86%`～`42.34%` 波动；BCE 仍缓慢下降。因此“1000 步不够”只解释了部分问题，继续单纯延长同一单宿主随机流的诊断收益较低。当前证据转向广播消息、Pw/RMS 预算、宿主 Pw 与 Dw/GAP 整条表示效率不足，但尚不能唯一归因某个模块。

### 20 条固定码本桥接实验（代码已改，待用户测试和运行）

为区分“只能记 4 条”与“可以记更大的有限码本”，overfit 现支持显式的独立验证库、分批验证、拟合/留出两侧的错配标签与移除水印对照、逐位 BER，以及每个图像—消息组合的实际曝光次数。显式 `--validation-messages 256` 使用与随机消息实验相同的验证 seed；训练 20 条仍固定为 seed+700000，前 4 条与旧 4 条实验相同。

```powershell
Set-Location -LiteralPath 'D:\02_代码项目（AI、课程与竞赛）\paper'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe diagnose_watermark.py --mode overfit --checkpoint ".\runs\clean_none_20260909_151508_769857\last.pt" --data-dir ".\datasets\DIV2K\DIV2K_train_HR" --device cuda --images 1 --messages 4 --validation-messages 256 --steps 500 --batch-size 2 --eval-batch-size 2 --log-every 100
.\.venv\Scripts\python.exe diagnose_watermark.py --mode overfit --checkpoint ".\runs\clean_none_20260909_151508_769857\last.pt" --data-dir ".\datasets\DIV2K\DIV2K_train_HR" --device cuda --images 1 --messages 20 --validation-messages 256 --steps 2500 --batch-size 2 --eval-batch-size 2 --log-every 250
```

先补跑 4 码字是为了让它也使用同一 256 条验证库；旧实验只有 4 条 held-out。两次固定码本实验每条训练消息都恰好曝光 250 次。首轮不做梯度累计，否则会同时改变有效 batch 或 Adam 更新/曝光口径。20 码字的主对照还包括随机消息运行的 step 2500（同为 2500 次 Adam 更新、5000 个训练样本，BER `43.4631%`），而不是只与随机 step 5000 比。

若 fit BER 接近 0 且 20/20 整串正确、held-out 与负对照仍约 50%，说明能记 20 条但没有通用规律；若 fit 在后段仍明显大于 10%，说明当前训练条件下 4→20 已出现码本干扰或优化瓶颈；若 fit 良好且 held-out 明显优于负对照并接近或优于随机 step 2500，则固定码本多样性促进了按位规律。必须同时参考 `message_prior_without_image`：20 条随机码本的逐位多数类预测本身就可能把 fit BER 降到约 41%，不能把这一水平误认为学会传消息。

所有输出仍进入新的 `runs/watermark_模式_时间戳/`，诊断权重保持独立格式，不能用于正式 `--resume/--init-from`。
