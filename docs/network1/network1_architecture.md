# Network1：源码级结构冻结

## 定义与证据范围

【源码确认】Network1 是 `dual_payload/system.py:11-49` 中 `DualPayloadSystem` 实例化并串联的四个可训练网络：`ColorEncoder (Ec)`、`WatermarkEncoder (Ew)`、`ColorDecoder (Dc)`、`WatermarkDecoder (Dw)`，加固定 YCbCr / DCT 频带投影、RMS cap 和传输通道。源码分别见 `dual_payload/models.py:40-121`、`dual_payload/transforms.py:14-90`、`dual_payload/channel.py:17-52`。`train.py:1-5` → `dual_payload/training.py:419-471,529-567` → `DualPayloadSystem.forward` → `CleanLoss.forward` (`dual_payload/losses.py:13-26`)。没有共享的可训练 backbone；四网络各自持有 `ResidualBlock`。没有 BatchNorm、LayerNorm、attention、学习式 DCT 或学习式 projector。

【源码确认】固定实验输入为 `[B,3,256,256]` float32 RGB 与 `[B,64]` 二值 message。四组 RGB 权重 1/2/3/4 采用相同 `model` 和 `channel` 配置；分别对应 `configs/joint_10x20_clean_v1.json`、`joint_10x20_rgb2_v1.json`、`joint_10x20_rgb3_v1.json`、`joint_10x20_rgb4_v1.json`。这些是 Network1 的 loss-weight ablation，不是四个网络。`config.py:9-15,103-110` 给出默认模型及固定协议约束。实验运行记录仅用于交叉核对，不取代当前源码。

## ResidualBlock（所有四个网络共用类）

源码：`dual_payload/models.py:11-19`。每个 block 为 `Conv2d(C,C,3,stride=1,padding=1)` → ReLU（仅 Dw 使用 `LeakyReLU(0.1)`）→ `Conv2d(C,C,3,stride=1,padding=1)` → 与原输入逐元素相加。没有第二层之后的激活、归一化、残差缩放、投影 skip、pooling 或通道/空间尺寸变化。64 通道每块 73,856 参数（两层各 36,928）；每网络 8 块。图：

```text
x ──────────────────────────┐
│                           │
Conv3×3(C→C,pad=1)          │
│                           │
ReLU / Dw: LeakyReLU(0.1)   │
│                           │
Conv3×3(C→C,pad=1)          │
│                           │
└──────────── + ────────────┘ → output
```

## Ec — ColorEncoder：593,217 参数

源码：`dual_payload/models.py:40-55`。输入 `y,cb,cr` 各 `[B,1,256,256]`，按通道拼成 `[B,3,256,256]`；不是直接把 RGB 张量传给 Ec，但由 RGB 经固定变换得到 Y/Cb/Cr。

| 顺序 | 层/操作 | 输出 | 参数 |
|---|---|---|---:|
| 1 | Conv2d(3,64,3,stride=1,padding=1)，无 stem 激活 | `[B,64,256,256]` | 1,792 |
| 2 | ResidualBlock(64, ReLU) ×8；每块两层 3×3 Conv 与 identity skip | `[B,64,256,256]` | 590,848 |
| 3 | Conv2d(64,1,3,stride=1,padding=1)，零初始化 head | candidate `[B,1,256,256]` | 577 |
| 4 | 固定 `Pc=DCT⁻¹(mask_c·DCT(candidate))`，`rms_cap(Pc,δc)` | residual `[B,1,256,256]` | 0 |
| 5 | `carrier=y+residual` | s `[B,1,256,256]` | 0 |

CNN 最长路径 18 层 3×3、无下采样；其纯 CNN 理论感受野约 37×37。块 DCT 在 8×8 块内混合信息，此数值不能当成全链路严格感受野。没有 CNN 多尺度特征。

## Ew — WatermarkEncoder：682,433 参数

源码：`dual_payload/models.py:58-83`。输入 `s:[B,1,256,256]`、`message:[B,64]`；验证 message 仅含 0/1。`bw=s-Pc(s)` 移除颜色频带。message 先变为 `2m-1`。

| 顺序 | 层/操作 | 输出 | 参数 |
|---|---|---|---:|
| 1a | Linear(64,128) → ReLU → Linear(128,64) | embedding `[B,64]` | 16,576 |
| 1b | `embedding[:,:,None,None].expand(...)`，广播而非学习式空间 reshape | message_map `[B,64,256,256]` | 0 |
| 2a | `bw=s-Pc(s)`；Conv2d(1,64,3,pad=1) → ReLU | image feature `[B,64,256,256]` | 640 |
| 2b | 拼接 image feature 与 message_map；Conv2d(128,64,3,pad=1) → ReLU | fused `[B,64,256,256]` | 73,792 |
| 3 | ResidualBlock(64, ReLU) ×8 | `[B,64,256,256]` | 590,848 |
| 4 | Conv2d(64,1,3,pad=1)，零初始化 head | candidate `[B,1,256,256]` | 577 |
| 5 | 固定 `Pw(candidate)` → `rms_cap(...,δw)` → 加到 s | residual、最终 carrier 各 `[B,1,256,256]` | 0 |

图像支路到 candidate 的最长纯 CNN 路径 19 层 3×3、约 39×39 感受野；message 以全局常值的 64 通道 map 提供给每个位置。没有卷积下采样、多尺度或 learned upsample。

## Dw — WatermarkDecoder：600,256 参数

源码：`dual_payload/models.py:109-121`。输入为通道输出 `attacked_image:[B,1,256,256]`，不是训练时原 message。固定 DCT 前端从每个不重叠 8×8 块抽取水印频带的 9 个系数，得到 `[B,9,32,32]`。

| 顺序 | 层/操作 | 输出 | 参数 |
|---|---|---|---:|
| 1 | 固定 9 个 8×8 DCT kernel，stride=8，无 padding | `[B,9,32,32]` | 0 |
| 2 | Conv2d(9,64,3,pad=1) → LeakyReLU(0.1) | `[B,64,32,32]` | 5,248 |
| 3 | ResidualBlock(64, LeakyReLU(0.1)) ×8 | `[B,64,32,32]` | 590,848 |
| 4 | Conv2d(64,64,1) | evidence `[B,64,32,32]` | 4,160 |
| 5 | `mean(dim=(2,3))` 全局平均 | raw logits `[B,64]` | 0 |

【根据源码推导】3×3 stem 加 16 个 3×3 block 卷积在 32×32 网格上的理论最长路径约 35×35 个块，超过整张 32×32 特征图；全局平均也汇总所有空间位置。因此不能称 Dw 缺少全局聚合或显然只看局部。无可训练逐级下采样，只有固定 DCT 步长 8 的前端降采样；没有全连接输出层。logits 不经过 sigmoid，直接用于 BCE-with-logits。

## Dc — ColorDecoder：593,795 参数

源码：`dual_payload/models.py:86-106`。输入是最终通道图 `x:[B,1,256,256]`。先算 `z=x-Pw(x)` 与 `zc=Pc(x)`，两者各一通道；拼成 `[B,2,256,256]`。

| 顺序 | 层/操作 | 输出 | 参数 |
|---|---|---|---:|
| 1 | Conv2d(2,64,3,pad=1) → ReLU | `[B,64,256,256]` | 1,216 |
| 2 | ResidualBlock(64, ReLU) ×8 | `[B,64,256,256]` | 590,848 |
| 3a | Conv2d(64,2,3,pad=1)，分为 cb/cr | cb、cr 各 `[B,1,256,256]` | 1,154 |
| 3b | Conv2d(64,1,3,pad=1) | raw luma `[B,1,256,256]` | 577 |
| 4 | `correction=Pcw(raw)`，`y_hat=z+correction` | y_hat `[B,1,256,256]` | 0 |
| 5 | 固定 `ycbcr_to_rgb(y_hat,cb_hat,cr_hat)`，无 clamp | rgb_hat `[B,3,256,256]` | 0 |

Dc 的两条 head 均从同一个 64 通道特征图出发。CNN 主干全程 256×256，最长至 head 为 18 层 3×3，纯 CNN 理论感受野约 37×37。没有 encoder-decoder、多尺度融合、U-Net skip、attention、dilation 或 learned frequency-domain feature；但输入与亮度修正明确使用**固定频域投影**，不能称整个 Dc 完全没有频域处理。`y_hat=z+correction` 是显式亮度残差旁路，block 内也有 identity skip；不能说 Dc 没有任何 skip。

## 固定频带和幅度机制

源码：`dual_payload/transforms.py:10-12,31-90`。`BlockDCT` 在每个不重叠 8×8 块上以固定正交基做 stride-8 卷积，逆变换用相同基的转置卷积。DCT kernels、bool masks、9 个索引均为 `register_buffer`，不是可训练参数；每个网络各持有一份。索引为 `u*8+v`。

| band | 条件 | 系数数/64 | 用途 |
|---|---|---:|---|
| c | `5≤u+v≤10` | 39 | Ec 颜色扰动、Dc 颜色频带输入 |
| w | `3≤u+v≤4` | 9 | Ew 扰动、Dw 9 核输入、Dc 去水印 |
| 0 | `u+v≤2` 或 `u+v≥11` | 16 | 已定义固定 mask；主 forward 未调用 |
| cw | `3≤u+v≤10` | 48 | Dc 亮度修正 |

`rms_cap(v,δ,eps)=v·min(δ/sqrt(mean(v²)+eps),1)`，按**每张图**的全通道/空间元素算一个缩放因子；不是逐像素裁剪，不保证单像素 `|v|≤δ`。Ec/Ew 的 δ 分别为 `2/255≈0.0078431373`，eps=`1e-8`。`dual_payload/models.py:54,82`、`dual_payload/config.py:12`。

## 参数核对

用项目 `.venv/Scripts/python.exe` 构造 `DualPayloadSystem` 并实际求和 `numel()`：Ec 593,217；Ew 682,433；Dw 600,256；Dc 593,795；合计 **2,469,701 trainable / 0 non-trainable Parameter**。DCT buffer 存在但不计入 Parameter。与 `dual_payload/training.py:514-518` 所用的历史日志统计口径一致。
