# 医学主实验运行说明

当前代码支持：数据清单 → 随机初始化 → 训练集量化步长初始化 → 四网络联合训练 → 冻结权重和 Profile → 真实 PNG 文件评测。
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

## 2. 从头训练的配置与初始化

复制 [medical_train.template.json](../configs/medical_train.template.json) 到实验目录后填写：

- `manifest`、`roots.pad`，以及需要时的 `roots.derm7pt`。
- 三个独立 `rms_limits`，取值必须显式给定且在 `(0,1]`。
- `initialization`：主实验固定为 `{"kind":"scratch","path":null,"sha256":null}`，不需要已有权重。`seed=2026` 控制四网络随机初始化。
- `calibration` 指向下一步生成的校准记录。
- `contract` 中的 Profile ID、量化 ID、固定交织 seed。训练配置没有最终权重摘要要求。
- `stage`、`epochs`、`max_steps`、逻辑 `batch_size`、`micro_batch_size`、`lr`、`save_every`、损失权重及新 `output` 目录。

模板的 `null` 和空损失配置不能直接启动训练。模板中的 seed、线程数等是可修改的起始设置，不代表已经验证的正式训练参数。

```sh
python -m dual_payload.medical.experiment audit-weights \
  --config /absolute/experiment/train.json \
  --output /absolute/experiment/weight-audit.json
```

上面的可选审计命令记录初始网络摘要，不加载外部权重。Ec 残差头和 Ew 两路嵌入头使用标准差 `1e-4` 的小幅随机初始化；Dc 亮度修正头权重和偏置为零；其余卷积使用 Xavier 初始化。四网络使用同一 seed 重建，标定与训练不依赖调用前的随机数状态。

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

这一步统计的是随机初始化 Ec 的数值范围，用于设置起始量化尺度，不代表已经学到有效的颜色编码。训练检查初始 Ec、seed、Ec 幅度和标定记录一致；39 个步长全程固定，持续记录截断比例和量化误差。更换初始化或 seed 时使用新的实验目录重新标定；续跑保持原步长不变。

## 4. H100 联合训练主实验

### H100 80 GB 主实验入口

主入口必须通过 `--config` 指定 JSON；该文件是训练超参数的唯一来源，代码不再维护一份重复预设。当前预设为 `configs/medical_h100_80gb_joint.json`，`stage=joint`。Ec/Ew/Dc/Dw 从第一步同时更新，不冻结网络，不做阶段切换。
当前预设设置为 256×256、FP32、有效 batch 16、micro-batch 4、AdamW、权重衰减 `1e-4`、梯度裁剪 1。按用户最新要求关闭数据增强，训练集只做固定的等比缩放和补边。
Ec 残差和颜色嵌入 RMS 上限为 `2/255`，患者嵌入 RMS 上限为 `1/255`。

| 模式 | 更新网络 | 学习率 | 上限 |
| --- | --- | --- | --- |
| joint | Ec、Ew、Dc、Dw | 全部参数统一 `1e-4` | 120 轮或 20,000 次更新 |

学习率按整个训练的更新总数计算：先预热 200 次更新，再余弦下降到 `1e-6`，预热期间四网络仍同时训练。
轮数和更新次数任一达到上限即结束。总损失包含 RGB L1 权重 1、两路 BCE 各 1、灰度 MSE 1000、越界 MSE 1000。
这些是首轮工程设置，尚未在 H100 医学数据上验证。主实验不使用旋转、翻转或颜色增强。

本次实验根目录固定为 `/data/zwc/zyh/experiments/v3clean-main-run01`。数据路径到位后，在服务器仓库根目录执行以下完整命令（程序自动创建实验根目录及训练所需子目录）：

输入读取支持完全不透明的 RGBA，并按嵌入的 RGB ICC 转换到 sRGB；详细规则见 [工作图预处理](medical_v1.md#5-工作图和-png)。原始图片不改写，manifest 保留原文件摘要及归一化后的工作图摘要。用户提供的服务器检查输出显示 2298 张图片中有 1440 张完全不透明 RGBA，以及 2 张带 `Google Skia` RGB ICC 的图片；两张 ICC 图片的模式均为 RGB。此统计来自服务器检查输出，本地合成测试不替代服务器实际解码验证。

同步新版代码后，可先在服务器仓库根目录执行以下只读检查，遍历实际预处理；该命令不生成实验文件或启动训练：

```sh
python - <<'PY'
from pathlib import Path
from dual_payload.medical.preprocess import prepare_work_image

paths = sorted(Path('/data/zwc/zyh/data/PAD-UFES-20/images').rglob('*.png'))
passed = 0
for path in paths:
    try:
        prepare_work_image(path)
        passed += 1
    except Exception as exc:
        print(f'FAIL {path}: {exc}')
print(f'预处理通过 {passed}/{len(paths)}')
raise SystemExit(0 if paths and passed == len(paths) else 1)
PY
```

主训练完整启动命令：

```sh
python -u -m dual_payload.medical.main_experiment \
  --config configs/medical_h100_80gb_joint.json \
  --pad-root /data/zwc/zyh/data/PAD-UFES-20 \
  --pad-metadata /data/zwc/zyh/data/PAD-UFES-20/metadata.csv \
  --output /data/zwc/zyh/experiments/v3clean-main-run01
```

程序检查 JSON 中 `device` 指定的 GPU，要求可用的完整 H100 80 GB，拒绝本机 CPU 和显存不足的 MIG 分区；按患者合并组 70/15/15 创建 PAD 清单。清单划分、初始标定和训练统一使用 JSON 的 `seed`（当前预设为 2026）。
先按 seed 随机初始化四网络，再用训练集初始化一次步长（绝对系数分位数 0.999，退化 RMS 阈值 `1e-6`）。随后直接进入一轮完整联合训练；步长保持固定，Ec 继续学习适应该量化配置，并记录裁剪比例。
每轮和每 `save_every` 次更新进行网络验证（当前预设 500）；当前最佳权重按验证总损失选择。日志包含各参数组学习率和 CUDA 峰值 allocated/reserved 字节数。正式业务成功率由第 6 节的独立文件评测确认。

启动器按数据参数和 `--output` 填入运行路径，覆盖 JSON 的这些路径占位；无需手写生成文件：

| 实际配置字段或文件 | 本次位置 |
| --- | --- |
| `roots.pad` | `/data/zwc/zyh/data/PAD-UFES-20` |
| `manifest` | `/data/zwc/zyh/experiments/v3clean-main-run01/manifest.json` |
| `calibration` | `/data/zwc/zyh/experiments/v3clean-main-run01/calibration.json` |
| `output` | `/data/zwc/zyh/experiments/v3clean-main-run01/joint` |
| 实际生效配置 | `/data/zwc/zyh/experiments/v3clean-main-run01/train-joint.json` |

`epochs`、`lr`、`batch_size`、`loss_weights` 等超参数直接取自指定 JSON。`micro_batch_size` 必须为正整数且不大于 JSON 中的 `batch_size`；无需整除 batch，最后一个 micro-batch 按实际样本数累计梯度。需要临时调整时，显式传入 `--micro-batch-size 2` 才覆盖 JSON；不传就保留文件值，`batch_size` 始终由 JSON 决定。
Profile ID/量化 ID 优先使用显式 `--profile-id` / `--quantization-id`，否则保留 JSON 值，仅 `null` 时补为 1。主入口保留 `scratch`、`joint`、无增强和 FP32 校验；`initialization.path` 和 `initialization.sha256` 必须保持 `null`。最终发布仍遵守 ID 不可重绑定规则。

中断后使用同一命令和配置恢复，保留整个输出目录。完整生效配置纳入 `main-inputs.json` 一致性记录，并与 `train-joint.json`、checkpoint 校验；修改超参数、代码、数据或标定后不得混用原目录。程序不会覆盖配置或删除旧结果来绕过检查。已完成训练通过带摘要的完成记录识别，未完成训练恢复 `joint/last.pt`；并发运行同一目录会被文件锁拒绝。若已有结果不匹配，应先保留原目录并处理不一致原因；本次命令仍使用上述固定目录。

### 训练内部计算

主实验在同一批图像上同时计算颜色重建、两路载荷提取、灰度失真和越界损失，一次反向传播后更新四网络。

训练中的量化和 8 bit 输出使用 STE：前向遵守实际取整规则，反向使用近似梯度。AES、LDPC 样本生成不参与反向传播；Dc 训练使用已知残差，不能用这一路的 RGB 损失宣称解密已成功。`joint` 的 Ec 梯度来自残差 STE 路径，RGB 损失不反传到载体路径。
两路 BCE/BER 排除 512 bit 布局填充；内容损失统计有效区域，越界损失统计整张图。

每张训练样本生成随机 16 B 测试 Token、图像 ID 和临时业务密钥，使用共同 nonce 登记接口完成真实加密。Token 用来验证完整传输，不代表恢复了真实患者身份或临床关联表。
验证密文帧按样本、配置及量化明文缓存于 `validation_frames/`；相同输入复用缓存，Ec 更新导致量化明文变化时生成新项。联合训练期间需预留缓存空间。

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

命令结果返回 checkpoint 摘要。`run.json` 保存配置、数据/校准/代码摘要和设备依赖版本；`metrics.jsonl` 保存训练与验证指标；`last.pt` 在首次更新前就保存 step 0，之后保存优化器、游标和 Torch RNG，`best.pt` 按验证总损失选择。
恢复要求原输出目录、相同配置/代码/数据/校准记录。保留 nonce 数据库和验证缓存。训练密文继续使用操作系统随机源，因此断点恢复不承诺与未中断训练逐比特相同。
修改配置时创建新目录。通用工具可显式加载同结构医疗 checkpoint 开始另一轮训练；主实验入口始终从头初始化，续跑只读取本次实验目录中的检查点。
当前实现为单设备 FP32，支持 Adam/AdamW、统一学习率、预热和余弦调度、微批次梯度累积；尚未在 H100 上实测吞吐或显存。`augmentation=dihedral` 开启几何增强，`none` 关闭；通用模板默认关闭。

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
