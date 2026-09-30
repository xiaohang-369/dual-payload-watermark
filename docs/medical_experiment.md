# 医学主实验运行说明

当前代码支持：数据清单 → 原权重核验 → 训练集量化标定 → 医疗训练 → 冻结权重和 Profile → 真实 PNG 文件评测。
本机检查使用合成图片和测试参数；没有运行医学主训练或 H100 检查，没有医学恢复质量结论。
协议、网络接口及 PNG 格式见 [medical_v1.md](medical_v1.md)。

## 1. 已确定的数据用途

| 数据 | 用途 | 分组方式 |
| --- | --- | --- |
| [PAD-UFES-20 v1](https://data.mendeley.com/datasets/zr7vgbcyr2/1) | 主训练、内部验证、内部测试 | 按患者分组；共享病灶或完全重复图像连接的患者合并后划分 |
| [Derm7pt](https://github.com/jeremykawahara/derm7pt) | 外部评测 | 同一病例保持外部用途，临床照片和皮肤镜图像分别报告 |

PAD 读取 `patient_id`、`lesion_id`、`img_id`、`diagnostic`，与[数据论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/)中的关联字段一致。
按文件名递归查找解压目录中的 PNG，缺失或重名报错。划分比例和 seed 必须显式提供，比例指患者合并组占比；不保证图像数量或诊断分布恰好符合比例。清单同时记录各诊断数量，正式运行前检查分布。

Derm7pt 根目录为包含 `images/` 和 `meta/` 的解压目录，读取 `meta/meta.csv` 中的 `case_num`、`clinic`、`derm`、`diagnosis`。
若附带官方 `train_indexes.csv`、`valid_indexes.csv`、`test_indexes.csv`，检查三个文件的完整覆盖和互斥性，保留原划分供追溯；本实验中这些病例全部用于外部评测，不参与模型选择、训练或步长标定。
元数据未提供的视图记入 `missing_views`；指定了文件名但文件不存在会报错，不能悄悄排除。Derm7pt 图像获取条件见[官方页面](https://derm.cs.sfu.ca/Download.html)，GitHub 仓库不能代替图像包。

所有图像采用收发端共同的 256×256 保留视野预处理。清单保存源文件和工作图 SHA-256、有效区域、分区和来源元数据摘要。读取时再次核验。跨分区患者、病灶及完全重复图像会被拒绝；这不构成近重复图像或未知跨数据集患者身份的完整核验。

以下路径和大写变量均为待替换项。数据、密钥、权重和实验输出放在源码目录之外，先创建记录文件的父目录。

```sh
python -m dual_payload.medical.experiment manifest \
  --pad-root /absolute/data/PAD-UFES-20 \
  --pad-metadata /absolute/data/PAD-UFES-20/metadata.csv \
  --derm-root /absolute/data/derm7pt/release_v0 \
  --derm-metadata /absolute/data/derm7pt/release_v0/meta/meta.csv \
  --fractions TRAIN_FRACTION VALID_FRACTION TEST_FRACTION \
  --seed SPLIT_SEED --output /absolute/experiment/manifest.json
```

如 Derm7pt 尚未到位，可以暂时省略两个 `--derm-*` 参数，先准备 PAD。之后新增含 Derm7pt 的清单时保持原 PAD 图片、元数据、比例和 seed；外部评测会检查 PAD 记录及划分完全一致。内部测试仍使用训练时冻结的原清单。

## 2. 训练配置和原权重核验

复制 [medical_train.template.json](../configs/medical_train.template.json) 到实验目录后填写：

- `manifest`、`roots.pad`，以及需要时的 `roots.derm7pt`。
- 三个独立 `rms_limits`，取值必须显式给定且在 `(0,1]`。
- `initialization`：首次通常为 `kind=v2`，填写原 checkpoint 路径及核验后的 SHA-256。后续阶段使用 `kind=medical` 和上一阶段 checkpoint。
- `calibration` 指向下一步生成的校准记录。
- `contract` 中的 Profile ID、量化 ID、固定交织 seed。训练配置没有最终权重摘要要求。
- `stage`、`epochs`、`max_steps`、逻辑 `batch_size`、`micro_batch_size`、`lr`、`save_every`、阶段损失权重及新 `output` 目录。

模板的 `null` 和空损失配置不能直接启动训练。模板中的 seed、线程数等是可修改的起始设置，不代表已经验证的正式训练参数。

```sh
python -m dual_payload.medical.experiment audit-weights \
  --config /absolute/experiment/train.json \
  --output /absolute/experiment/weight-audit.json
```

V2 核验要求 checkpoint 含 `architecture_version=v2`、`config.model` 和完整 `model` 状态，先严格加载原网络，再迁移 Ec 和可兼容的 Dc 部分，记录逐层结果。新 Ew/Dw、Dc 输入层和亮度修正头重新初始化。摘要只能确认文件身份；原实验训练记录及数据来源仍需单独核验。

## 3. 仅用训练集标定 39 个步长

```sh
python -m dual_payload.medical.experiment calibrate \
  --config /absolute/experiment/train.json \
  --quantile CALIBRATION_QUANTILE --minimum-rms MINIMUM_RESIDUAL_RMS \
  --output /absolute/experiment/calibration.json
```

工具逐频率统计 Ec 残差 DCT 系数的绝对值分位数，以 `quantile(abs(coeff))/7` 为该频率步长，记录裁剪比例、系数量化 MSE、残差 RMS、训练清单和 Ec 状态摘要。
分位数和退化判定阈值必须显式给定。使用磁盘临时数组存放系数；验证和测试图片不会进入标定。
零初始化 Ec、非有限系数或没有有效范围的频率会被拒绝，不以任意常数替代步长。

训练会检查初始 Ec、Ec 幅度和校准记录匹配。阶段切换后若 Ec 改变，需要针对新的初始化重新标定；续跑同一训练阶段保持原步长不变。

## 4. H100 联合训练主实验

### H100 80 GB 主实验入口

主实验配置为 `configs/medical_h100_80gb_joint.json`，`stage=joint`。Ec/Ew/Dc/Dw 从第一步同时更新，不冻结网络，不做阶段切换。
共同设置为 256×256、FP32、有效 batch 16、micro-batch 4、AdamW、权重衰减 `1e-4`、梯度裁剪 1。按用户最新要求关闭数据增强，训练集只做固定的等比缩放和补边。
原 Ec 和颜色嵌入 RMS 为 `2/255`，患者嵌入 RMS 为 `1/255`；实际原 checkpoint 的 Ec 幅度必须匹配。

| 模式 | 更新网络 | 学习率 | 上限 |
| --- | --- | --- | --- |
| joint | Ec、Ew、Dc、Dw | 已迁移层 `1e-5`，新增层 `1e-4` | 120 轮或 20,000 次更新 |

学习率按整个训练的更新总数计算：先预热 200 次更新，再余弦下降到 `1e-6`，预热期间四网络仍同时训练。
轮数和更新次数任一达到上限即结束。总损失包含 RGB L1 权重 1、两路 BCE 各 1、灰度 MSE 1000、越界 MSE 1000。
这些是首轮工程设置，尚未在 H100 医学数据上验证。主实验不使用旋转、翻转或颜色增强。

路径到位后，在服务器环境执行：

```sh
python -u -m dual_payload.medical.main_experiment \
  --pad-root /absolute/data/PAD-UFES-20 \
  --pad-metadata /absolute/data/PAD-UFES-20/metadata.csv \
  --checkpoint /absolute/checkpoints/original-v2.pt \
  --sha256 VERIFIED_ORIGINAL_CHECKPOINT_SHA256 \
  --output /absolute/experiments/medical-main-01
```

程序要求可用的完整 H100 80 GB，拒绝本机 CPU 和显存不足的 MIG 分区；按患者合并组 70/15/15、seed 2026 创建 PAD 清单。
先核验原 V2 文件摘要和 Ec 幅度，再用训练集标定一次步长（绝对系数分位数 0.999，退化 RMS 阈值 `1e-6`）。随后直接进入一轮完整联合训练；步长保持固定，Ec 继续学习适应该量化配置，并记录裁剪比例。
每轮和每 500 次更新进行网络验证；当前最佳权重按验证总损失选择。日志包含各参数组学习率和 CUDA 峰值 allocated/reserved 字节数。正式业务成功率由第 6 节的独立文件评测确认。

中断后使用同一命令恢复；保留整个输出目录。训练权重保存在 `joint/`，配置为 `train-joint.json`；已完成训练通过带摘要的完成记录识别，未完成训练恢复 `joint/last.pt`。并发运行同一目录会被文件锁拒绝。
若要用 micro-batch 2，初次启动时加 `--micro-batch-size 2`，有效 batch 仍为 16。已启动目录的配置不得静默修改；出现显存不足时保留该目录，在新目录中明确调整配置。
模板中只保留真实路径、初始化文件摘要和公共 ID 的空位；启动器会填入这些字段。Profile ID/量化 ID 默认 1，可以通过命令参数指定，最终发布仍遵守不可重绑定规则。

### 训练内部计算

主实验在同一批图像上同时计算颜色重建、两路载荷提取、灰度失真和越界损失，一次反向传播后更新四网络。

训练中的量化和 8 bit 输出使用 STE：前向遵守实际取整规则，反向使用近似梯度。AES、LDPC 样本生成不参与反向传播；Dc 训练使用已知残差，不能用这一路的 RGB 损失宣称解密已成功。`joint` 的 Ec 梯度来自残差 STE 路径，RGB 损失不反传到载体路径。
两路 BCE/BER 排除 512 bit 布局填充；内容损失统计有效区域，越界损失统计整张图。

每张训练样本生成随机 16 B 测试 Token、图像 ID 和临时业务密钥，使用共同 nonce 登记接口完成真实加密。Token 用来验证完整传输，不代表恢复了真实患者身份或临床关联表。
验证密文帧按样本、配置及量化明文缓存于 `validation_frames/`；相同输入复用缓存，Ec 更新导致量化明文变化时生成新项。因此改变 Ec 的阶段需预留缓存空间。

在 H100 环境安装 `.[test,medical]` 后，也可直接使用准备好的联合配置启动训练：

```sh
python -m dual_payload.medical.experiment train \
  --config /absolute/experiment/train-joint.json
```

训练中断后恢复同一目录：

```sh
python -m dual_payload.medical.experiment train \
  --config /absolute/experiment/train-joint.json \
  --resume /absolute/runs/main/last.pt --resume-sha256 VERIFIED_CHECKPOINT_SHA256
```

命令结果返回 checkpoint 摘要。`run.json` 保存配置、数据/校准/代码摘要和设备依赖版本；`metrics.jsonl` 保存训练与验证指标；`last.pt` 保存优化器、游标和 Torch RNG，`best.pt` 按验证总损失选择。
恢复要求原输出目录、相同配置/代码/数据/校准记录。保留 nonce 数据库和验证缓存。训练密文继续使用操作系统随机源，因此断点恢复不承诺与未中断训练逐比特相同。
修改配置或切换阶段时创建新目录，用 `initialization.kind=medical` 加载已核验的前阶段权重。
当前实现为单设备 FP32，支持 Adam/AdamW、按初始化来源划分学习率、预热和余弦调度、微批次梯度累积；尚未在 H100 上实测吞吐或显存。`augmentation=dihedral` 开启几何增强，`none` 关闭；通用模板默认关闭。

## 5. 冻结模型、Profile 和评价口径

在查看正式测试结果前填写 [medical_evaluation.template.json](../configs/medical_evaluation.template.json) 中的所有门槛：两路整包成功率、颜色恢复成功率、联合成功率、灰度和恢复 RGB 内容区域 PSNR。
先用验证集选择 checkpoint，再导出正式材料：

```sh
python -m dual_payload.medical.experiment export \
  --checkpoint /absolute/runs/main/best.pt --sha256 VERIFIED_CHECKPOINT_SHA256 \
  --policy /absolute/experiment/evaluation-policy.json \
  --profile-id NEW_PUBLIC_PROFILE_ID \
  --output /absolute/exports/main
```

导出四个组件权重、摘要、排列文件、正式 Profile、训练清单、冻结评价口径及来源记录。不能先填写虚构最终权重来启动训练。
导出目录必须不存在。同一个公开 Profile ID 不可绑定不同配置或权重；新 checkpoint 需要新的 ID。不同导出目录之间也应遵守这一发布约定，不能利用新建目录重用 ID。
`--profile-id` 可以在导出时指定新的发布 ID，省略时使用训练配置中的 ID；它不要求重新训练。不手改已冻结文件。

## 6. 真实文件批量评测

```sh
python -m dual_payload.medical.experiment evaluate \
  --export /absolute/exports/main --manifest /absolute/experiment/manifest.json \
  --pad-root /absolute/data/PAD-UFES-20 --split test \
  --device cuda:0 --output /absolute/results/pad-test

python -m dual_payload.medical.experiment evaluate \
  --export /absolute/exports/main --manifest /absolute/experiment/manifest-with-derm.json \
  --pad-root /absolute/data/PAD-UFES-20 \
  --derm-root /absolute/data/derm7pt/release_v0 --split external \
  --device cuda:0 --output /absolute/results/derm-external
```

每张图片执行“生成真实 PNG → 新接收进程 → 验签 → Dw → 逆排列/LDPC/解密 → Dc”。接收命令只传 PNG、公共配置、模型、可信医院公钥和授权密钥；原图、原始亮度、参考残差不传入接收端。
评测父进程保留真实比特用于评分，接收进程输出自身 logits/译码信息。独立进程用于检查数据依赖，并非操作系统级访问隔离。

结果文件包括：

- `samples.jsonl`：每个样本的文件、认证、两个业务状态、失败原因和指标。
- `summary.json`：总体以及两种模态分别汇总，含最差质量样本及失败列表。
- PNG 和各接收目录：成功时的 Token、浮点 RGB、整数 RGB，及接收诊断输出。
- `private/`：本次评测的临时业务密钥、nonce 数据库和发送参考帧。评测签名密钥临时生成，不代表医院真实签发记录。

报告口径：

1. 认证、两路整包、联合成功率及颜色恢复成功率以清单中的全部评测图片为分母。认证失败、读取失败和进程错误均保留。
2. 两路纠错前 BER 排除布局填充；码块失败根据译码信息位与发送真值逐块比较，包含零填充检查。无法测量时记 `null`，并报告可测样本数，不当作零误码。
3. 整包精确成功、AEAD 验证成功及 RGB 恢复成功分别统计。颜色包失败时不生成部分 RGB。
4. 灰度对照工作图亮度。RGB 质量只对颜色包精确成功且 Dc 成功的样本统计，同时给出样本数；零成功样本的质量是 `null`，不能通过验收。
5. 有效内容区域和整图分别报告 PSNR、SSIM、MAE/MSE、最大及 99% 分位误差、越界比例；RGB 另报亮度/色度误差和 CIE76 色差。浮点输出与实际整数 RGB 分开，PSNR 上限 120 dB，验收使用整数 RGB 内容区域 PSNR。
6. 外部总体与各模态都检查冻结门槛；命令返回 0 表示全部通过，2 表示未通过。模型依赖或配置错误会明确报错。

最终需要分别验收程序可执行、真实载荷可靠提取、医学图像恢复质量。合成测试只支持第一项和协议正确性，不能替代后两项。
