# 双载荷灰度系统：Clean V1

> `feature/network-v2-clean` 当前实现以 `docs/network2/network2_architecture.md` 为唯一 Network V2 架构规格；本文其余 V1 结构说明保留为基线背景，不能覆盖该规格。

按当前确定的四网络方案实现：把颜色信息和 64-bit 消息写入单通道灰度载体，再恢复颜色并提取消息。

当前阶段是 **float + identity clean**。攻击层已有固定位置和接口，仅实现恒等操作；尚未实现 JPEG、噪声、模糊或几何攻击。不使用 FFT、Transformer、额外 attention 或原图旁路通道。

代码和测试用例已编写。**依赖安装、测试与训练由用户执行。** 用户已反馈原有 29 项测试通过、CUDA smoke 跑通，并完成 DIV2K float clean 1000 步实验；水印 BER 仍接近 50%，不能视为性能达标。`inspect`、固定码本 overfit 及单图等曝光对照已执行；本次新增的随机消息诊断及其测试仍待用户运行。合成数据 smoke 仅用于检查代码链路。

## 环境安装

在 PowerShell 中进入项目目录：

```powershell
Set-Location -LiteralPath 'D:\02_代码项目（AI、课程与竞赛）\paper'
```

当前项目已有 `.venv` 目录，用户已完成依赖安装，无需重复安装。以下安装步骤保留供重建环境使用；若在另一台机器上使用，先运行 `uv venv .venv --python 3.11`。

安装项目及测试依赖，自动选择适合本机驱动的 PyTorch 后端：

```powershell
uv pip install --python .venv\Scripts\python.exe --torch-backend auto -e ".[test]"
```

安装方式参考 [uv 的 PyTorch 官方集成说明](https://docs.astral.sh/uv/guides/integration/pytorch/)。CUDA 包体积较大。本机为 RTX 4060 Laptop GPU，8 GB 显存；用户提供的训练日志已确认 PyTorch 使用 CUDA。

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

如只需要 CPU 环境，把安装命令中的 `--torch-backend auto` 改为 `--torch-backend cpu`。

## 先验证代码

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe train.py --config configs/clean_float.json --smoke
```

第一条检查颜色转换、DCT 投影、RMS Cap、梯度、原设计的 clean 隔离关系、量化、消息指标、数据读取和断点续训。测试使用小主干以控制运行时间。

Network V2 的 smoke 固定使用 256×256、batch size 1 和一个训练 step；不能为了加速 smoke 缩小 Dw 输入或改变其 `Linear(1024,256)`。它会记录设备、参数数目、损失和验证指标，检查一次 forward/loss/backward/optimizer step 以及保存加载链路，不能把结果当成自然图像上的性能结论。

每次 smoke 使用新的 `runs/smoke_...` 目录。若手动指定非空输出目录，程序会拒绝覆盖。

## 正式 clean 训练

需要独立的训练图片目录和验证图片目录；目前没有自动下载或选择数据集。支持递归读取 PNG、JPEG、BMP、TIFF 和 WebP，检查路径重叠，不会自动把同一个目录当训练集和验证集。

```powershell
.\.venv\Scripts\python.exe train.py --config configs/clean_float.json --train-dir 'D:\datasets\train' --val-dir 'D:\datasets\val'
```

请把示例目录改为实际目录。图片经过 EXIF 方向处理、RGB 转换、短边缩放和方形裁剪；这些是输入数据预处理，编码后的攻击层仍为 identity。训练按 epoch/sample 固定裁剪随机数，验证用中心裁剪；RGB 为 gamma-coded sRGB 的 `[0,1]` 数值，不做 linear RGB 转换或额外均值方差标准化。

训练消息在每个 batch 重新随机生成，与图片索引独立。验证消息按验证样本索引和 seed 固定，便于跨 epoch 比较。只有明确传入 `--smoke` 时才会使用合成数据；缺少真实数据路径时，正式训练会报错。

默认 FP32、Adam、学习率 `1e-4`、batch size 2、20 epochs、梯度范数上限 1；这些是实验初值，不是已经调优的训练参数。Windows 默认 `num_workers=0`。暂不启用 AMP 或 TF32。

需要限定试运行长度时：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/clean_float.json --train-dir 'D:\datasets\train' --val-dir 'D:\datasets\val' --max-steps 50
```

`--max-steps` 表示整个 run 的总更新步数上限。`--epochs`、`--batch-size`、`--image-size`、`--device cpu` 也可覆盖配置。图像尺寸必须是 8 的倍数。

## 水印停滞诊断（先 inspect，后决定是否 overfit）

入口为 `diagnose_watermark.py`，实现位于 `dual_payload/diagnostics.py`。不修改四网络结构、DCT、幅度预算或正式训练配置；不安装或下载任何东西。只接受 `none + identity + 不 clamp` 的检查点。

先由用户运行测试，再检查当前 1000 步权重：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe diagnose_watermark.py --mode inspect --checkpoint ".\runs\clean_none_20260909_151508_769857\last.pt" --data-dir ".\datasets\DIV2K\DIV2K_train_HR" --device cuda --images 4 --messages 4
```

`inspect` 不更新任何权重，也不保存模型文件。它读取训练目录排序后的前 4 张图，固定中心裁剪，默认保持检查点的 256×256 尺寸。每张图使用同一组 4 条不同随机消息，逐条前向以控制显存。报告包括：

- 同图换消息后，消息 embedding、原始候选残差、Pw 投影、限幅后残差和 logits 的变化 RMS；这里是跨消息去均值后的 RMS，不是解码正确率。
- 正确配对消息、同图错配消息标签、去掉水印残差三种情况下的 BCE、BER 和整条消息成功率。错配只用于对照，不用于训练。
- 候选/投影/残差 RMS、RMS Cap 的缩放系数、触发比例和宿主 Pw 能量。
- 固定第一条消息仅翻转第 0 位，检查残差/logits 差异；它不是全部 64 位敏感性评估。
- 对同一个小 batch 分别做消息 BCE 和联合总损失的反向传播，报告各模块梯度范数、候选/限幅后残差的激活梯度，以及按原配置做全局梯度裁剪时的预计缩放系数。不会实际裁剪或更新参数，结束清空梯度。

输出自动保存到新的 `runs/watermark_inspect_时间戳/`：`manifest.json` 记录源检查点、配置、图片和消息，`metrics.jsonl` 记录过程，`report.json` 保存完整结果。任何已存在的 `--output-dir` 都会被拒绝，包括原训练目录。请把报告路径或终端输出发回来，先判断问题再继续。

如诊断后决定做小样本过拟合，再显式运行（**现在不必同时运行**）：

```powershell
.\.venv\Scripts\python.exe diagnose_watermark.py --mode overfit --checkpoint ".\runs\clean_none_20260909_151508_769857\last.pt" --data-dir ".\datasets\DIV2K\DIV2K_train_HR" --device cuda --images 4 --messages 4 --steps 500
```

该实验从源权重开始，冻结 Color Encoder/Decoder，缓存固定 S，只更新 Watermark Encoder/Decoder；所选图片与固定消息的全部笛卡尔积组合都进入拟合集，每张图不再绑定唯一消息。组合按固定随机数逐轮打乱并平衡取样，默认 batch size 2。留出消息数默认等于 `--messages`；也可用 `--validation-messages` 指定更大的独立固定验证库。显式验证库使用与 `random_messages` 相同的验证 seed，因此可以直接比较，但仍是同图评估，不等于独立图片测试集性能。

为了隔离水印分支，**该实验只优化消息 BCE，不使用 RGB、载体或范围损失**；保留原 δw、学习率、weight decay 和梯度范数上限，但 Adam 优化器重新建立，裁剪对象仅为水印分支。这是独立诊断目标，不替代四网络联合训练，也不承诺图像质量。默认每 50 步以及第 0/1/末步评估完整拟合消息库和留出消息库，必须显式提供 `--steps` 才能启动训练。评估同时报告匹配、错配标签、移除水印、逐位 BER，并记录训练样本总曝光数及图像—消息组合的最小/最大曝光次数。

结果存入新的 `runs/watermark_overfit_时间戳/`。`diagnostic_weights.pt` 使用独立诊断格式，不能传给正式 `train.py --resume/--init-from`；原始 `last.pt/best.pt` 不变。当前入口不支持诊断中途续训。先看拟合集 BER 能否下降，再看留出消息；即使小样本成功，也需要回到联合训练和独立验证验证泛化。

少量固定消息的每一位可能不平衡，模型只学每位的多数值也可能让拟合集 BER 低于 50%。因此过拟合日志同时提供 `message_prior_without_image`：完全不看载体、仅按拟合消息库每位频率预测的对照。不能把刚降到这个对照水平误判为学会传消息，应观察能否明显超过它，并区分拟合消息与留出消息表现。

连接“4 条固定码字”和“每步全随机消息”的 20 条固定码本对照如下：

```powershell
# 推荐先补一个与新实验使用同一 256 条验证消息的 4 码字基线
.\.venv\Scripts\python.exe diagnose_watermark.py --mode overfit --checkpoint ".\runs\clean_none_20260909_151508_769857\last.pt" --data-dir ".\datasets\DIV2K\DIV2K_train_HR" --device cuda --images 1 --messages 4 --validation-messages 256 --steps 500 --batch-size 2 --eval-batch-size 2 --log-every 100

# 20 码字主实验
.\.venv\Scripts\python.exe diagnose_watermark.py --mode overfit --checkpoint ".\runs\clean_none_20260909_151508_769857\last.pt" --data-dir ".\datasets\DIV2K\DIV2K_train_HR" --device cuda --images 1 --messages 20 --validation-messages 256 --steps 2500 --batch-size 2 --eval-batch-size 2 --log-every 250
```

20 码字实验共有 `2500×2=5000` 次训练样本曝光，平衡调度使每条消息恰好出现 250 次，与 4 码字实验的逐消息曝光量一致。补跑 4 码字基线的原因是旧结果只有 4 条 held-out，不能和 256 条验证库直接量化比较。首轮保持真实 batch 2，不做梯度累计；累计会改变有效 batch 和 Adam 更新口径，作为后续独立优化对照处理。20 码字的另一个主比较对象是随机消息实验的 **step 2500**（相同 2500 次 Adam 更新、5000 个训练样本、同一 256 条验证消息），不是只比较随机实验的 step 5000。

判读注意：残差达到 RMS 上限不等于没有梯度；非零梯度也不保证可学。初始零残差头会使消息 MLP 第一轮梯度为零，这是初始化预期。不要仅凭一个梯度数值或几条消息就断定根因。

### 单宿主随机消息诊断

固定四消息 overfit 可以检验通路能否记住一个小码本，却不能检验任意 64-bit 消息。`random_messages` 模式固定一张图片和对应的灰度宿主，每个优化步骤生成一批新的、互不重复的随机消息；这些训练消息严格排除一组独立固定的验证消息。颜色编码器/解码器冻结，宿主只计算一次，只训练 Watermark Encoder/Decoder，目标仍仅为消息 BCE。

默认验证库包含 256 条消息。训练与验证使用两个独立 CPU 随机数生成器，因此改变验证频率不会改变训练消息流；验证按小批次执行，避免一次复制 256 份 256×256 特征到显存。报告保存总体 BCE、BER、64-bit 整串成功率、logit RMS、总错误数以及 64 个位置各自的 BER，并同时记录错配标签和移除水印残差的负对照。验证仍使用同一张宿主图，所以它只回答“固定宿主下能否学习可迁移到未见消息的编码规则”，不代表跨图片泛化。

先运行完整测试；通过后再启动 1000 步诊断：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe diagnose_watermark.py --mode random_messages --checkpoint ".\runs\clean_none_20260909_151508_769857\last.pt" --data-dir ".\datasets\DIV2K\DIV2K_train_HR" --device cuda --images 1 --steps 1000 --batch-size 2 --log-every 100
```

`--messages` 属于固定码本的 inspect/overfit，在随机模式必须省略。随机模式验证库默认 256 条；诊断性缩小可用 `--validation-messages`，验证显存批量可用 `--eval-batch-size`（默认等于训练 batch size）。结果自动写入新的 `runs/watermark_random_messages_时间戳/`，已有目录仍会被拒绝。

判读以 `random_message_validation.validation_messages_same_image` 为准，不以单个随机训练 batch 的即时指标为准：

- 只有当匹配消息的 BER 和 BCE 持续优于错配标签、移除水印两个对照，而且大多数 bit 的 BER 都同步改善时，才能说明出现了可迁移的消息信号；再增加宿主数量检查宿主干扰。仅仅略低于 50% 不足以证明通用 64-bit 规则。
- 若训练 1000 步后验证 BER 仍在约 48%～52%，说明没有消息泛化；此时 BCE 接近 `ln(2)` 表示近似不确定预测，BCE 很高则表示自信地预测错误。两种情况都应优先检查消息注入、Pw 投影后的表达及 Watermark Decoder/GAP。
- 若训练 batch 指标下降而固定验证不下降，仍属于对近期样本的适配，不算消息泛化成功。

该模式也只保存独立格式的 `diagnostic_weights.pt`，不能交给正式 `train.py --resume/--init-from`。

## 固定网络与数学约定

```text
RGB, message -> Color Encoder -> S -> Watermark Encoder -> X_float
    -> Quantization -> Attack(identity) -> attacked_image
    -> Color Decoder / Watermark Decoder
```

两个 Decoder 使用同一个 `attacked_image` 张量。`DualPayloadSystem.decode(attacked_image)` 是独立的盲解码接口，不读取原 RGB、消息、目标、掩码或攻击参数。

- Y = 0.299R + 0.587G + 0.114B；Cb = (B−Y)/1.772；Cr = (R−Y)/1.402。Cb/Cr 零中心，不加 0.5。
- 固定逆变换的绿色系数由上述正向公式直接推导，采用未截断精度，对应文档中约 `0.344136`、`0.714136` 的系数。
- DCT 为左上角对齐、正交归一化的非重叠 8×8 block-DCT。`P0` 包含 16 个系数，`Pw` 包含 9 个系数，`Pc` 包含 39 个系数。固定核为 buffer，无可训练权重，输入梯度保留。
- `δc=2/255` 沿用原设计；**`δw=2/255` 是本次代码的暂定实验初值**。二者均在配置中显式记录，不作为损失权重使用。
- RMS Cap 逐样本计算一个缩放标量，`eps=1e-8`；投影后的残差没有逐像素激活或裁剪。RMS 预算不保证逐像素值合法。
- 原始配置每个主干为 64 通道、8 个残差块，无归一化，无额外上下采样；RB 加法后不接激活。Color Encoder stem 没有激活。
- 两个 Encoder 残差头权重和偏置为 0，因此初始 `S=X=Y`。第一步 Encoder 主干梯度为 0 属于该初始化的预期行为；头更新后主干才能获得梯度。
- Color Decoder 的两个输出头非零 Xavier 初始化、偏置 0；先预测 Cb/Cr 和原始亮度残差，再投影到 `Pc+Pw`，加回 `Z=(I−Pw)attacked_image`，最后固定转换 RGB。没有改成自由 RGB 输出头。
- Watermark Encoder 消息分支为 `64 -> 128 -> 64`，广播后与图像特征融合。Watermark Decoder 使用原顺序的 9 个 DCT 核、LeakyReLU(0.1)、Kaiming 主干初始化、Xavier bit head 和 GAP，输出 64 个 raw logits。
- 其他主干采用与 PyTorch Conv2d/Linear 默认一致的 fan-in 初始化尺度，偏置 0；这些未在原文完全指定的初始化细节在代码中显式固定。

消息二值化规则为 `logits >= 0`。不引入纠错码，消息长度就是 64 bit。原设计在平坦区域的消息表达能力限制仍保留，需要后续实验评估。

## 当前 clean 目标与损失

通道接口返回 `attacked_image`、目标字典、`valid_mask` 和 `attack_info`。当前目标就是原始 RGB、Y、Cb/Cr，mask 全 1。目标与 mask 只用于训练和评测。

默认损失为：

```text
1.0 * L1(rgb_hat, original_rgb)
+ 1.0 * BCEWithLogits(logits, message)
+ 1.0 * MSE(X_float, Y)
+ 0.1 * mean(relu(-X_float)^2 + relu(X_float - 1)^2)
```

额外色度 L1、亮度 L1 已提供，默认权重为 0。所有权重在 `loss` 配置中；它们是可调初值。范围损失是软约束，不会偷偷裁剪浮点载体。RGB 重建损失也使用未裁剪的输出，避免掩盖越界问题。消息损失直接使用 [PyTorch BCEWithLogits](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)，不在网络输出端重复 sigmoid。

`CleanLoss` 和当前指标明确拒绝非 identity/非全有效掩码。后续启用攻击时，应实现对应攻击及目标和指标规则；完整链路的位置和两个 Decoder 的外部输入接口已经保留。

## 输出、评估与续训

运行后产生：

| 文件 | 含义 |
| --- | --- |
| `config.json` | 本次实际生效参数，含 CLI/smoke 覆盖值 |
| `manifest.json` | 排序后的训练/验证图片路径 |
| `metrics.jsonl` | 训练损失和逐轮验证指标 |
| `last.pt` | 模型、优化器、进度、随机状态和配置 |
| `best.pt` | 验证集总损失最低的检查点 |
| `preview.png` | 原 RGB、灰度载体、恢复 RGB，按从左到右排列 |

预览图只在导出时裁剪到 `[0,1]` 并转为 8-bit；这种导出不参与训练，不改变 float clean 的计算。`best.pt` 的选择规则仅为验证总损失最小，不能等同于最终 PSNR/BER 最优折中。

指标包括载体 PSNR/SSIM、RGB 恢复 PSNR/SSIM、额外注明 clipped 的 RGB 指标、BER、Bit Accuracy、64-bit 整条消息成功率、载体/输出越界像素比例及两种残差的 RMS。先在每张图计算再按图像数平均；固定 `data_range=1`。PSNR 为避免 JSON 出现无穷值设 120 dB 上限，包括完全一致的图像。

SSIM 为单尺度、11×11 Gaussian 窗口、sigma=1.5、valid 窗口、RGB 通道均值；小于 11 的尺寸使用可容纳的奇数窗口。默认同时记录 raw 和 clipped RGB 结果，不能把两者混用。尚未加入 LPIPS，也未实现有掩码的几何攻击指标。

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint 'runs\实际运行目录\best.pt' --data-dir 'D:\datasets\test'
```

测试目录请使用独立留出的数据，不用测试指标调损失权重。

继续同一次运行，包含优化器和训练进度：

```powershell
.\.venv\Scripts\python.exe train.py --resume 'runs\实际运行目录\last.pt' --epochs 40
```

如果上次设置了步数上限，续训时要增大，例如 `--max-steps 100`。中途因步数上限停止时，会记录下一个 batch；在相同设备/运行环境和未更改的数据下，设计为恢复该位置的消息、shuffle 和裁剪状态。测试用例包含 CPU 中途续训与连续训练的一致性检查，实际运行结果尚待用户执行测试确认。恢复时检查配置和文件列表；不要修改同路径图像的内容。

`--resume` 禁止改变网络、预算、通道、损失、数据和优化器设置。切换实验阶段用 `--init-from`，只加载模型权重并建立新 run：

```powershell
.\.venv\Scripts\python.exe train.py --config configs/clean_8bit.json --init-from 'runs\float运行目录\best.pt' --train-dir 'D:\datasets\train' --val-dir 'D:\datasets\val'
```

这一步用于后续 **8-bit clean**，不需要现在运行。它执行 clamp 和 STE rounding，攻击仍为 identity。`clean_8bit.json` 未指定的值继承项目默认配置，不会自动继承旧 run 的自定义损失、预算或数据参数；如有自定义值，应显式写入新配置。

真实 8-bit 数值评估：

```powershell
.\.venv\Scripts\python.exe evaluate.py --checkpoint 'runs\实际运行目录\best.pt' --data-dir 'D:\datasets\test' --quantization-mode real8
```

`real8` 指 `round(255*clamp(x,0,1))/255`，不等于 JPEG 编解码。测试用例会比较它与单通道 8-bit PNG 保存/读取的数值。STE 与 real8 使用相同的量化规则，STE 只近似 rounding 的反向传播；clamp 保留真实梯度。`real8` 禁止参与有梯度的训练。

## 当前尚需实验确定

当前使用 DIV2K 的 800 张训练图片和 100 张验证图片，位于 `datasets/DIV2K/DIV2K_train_HR` 和 `datasets/DIV2K/DIV2K_valid_HR`。δw、损失权重、学习率和训练时长仍是初始实验配置；1000 步 clean 结果需要先排查水印分支，不直接进入攻击训练。攻击类型、强度、概率、几何同步和攻击后颜色目标留到该阶段设计。
