# Network1：loss、梯度路径与结构分析

## 源码入口与真实目标

【源码确认】`train.py:1-5` 调 `run_cli(train_main)`；`dual_payload/training.py:419-471` 读取配置、实例化 `DualPayloadSystem` / `CleanLoss` / Adam；`training.py:529-567` 用 `model(rgb,message)`、`criterion(output,message)`、`losses["total"].backward()` 更新四网络。`dual_payload/system.py:30-49` 是训练前向；`dual_payload/losses.py:13-26` 是所有六项 loss 的唯一正式组合。当前 clean/RGB1-4 配置为 `quantization_mode=none`、`clamp_enabled=false`、`attack_mode=identity`，见四份 `configs/joint_10x20_*.json` 和 `dual_payload/channel.py:35-52`。

| loss | 源码表达式（`dual_payload/losses.py:16-25`） | 目标/说明 | RGB1/2/3/4 权重 |
|---|---|---|---|
| RGB | `F.l1_loss(output["rgb_hat"],output["target_rgb"])` | Dc 的 RGB vs 通道传递的原始 `rgb`；**L1** | 1/2/3/4 |
| message | `F.binary_cross_entropy_with_logits(output["logits"],message.float())` | Dw logits vs 输入 bits | 1 |
| carrier | `F.mse_loss(x,output["y"])`，其中 `x=output["x_float"]` | **最终浮点 carrier `x_float` 与原始 RGB 计算的 `y`**；不是 `s`，也不是 `x_quantized` | 1 |
| range | `(relu(-x)^2+relu(x-1)^2).mean()` | 最终 `x_float` 的越界平方惩罚 | 0.1 |
| luma | `F.l1_loss(output["y_hat"],output["target_luma"])` | Dc 的 `y_hat` vs 原始 `y`；**L1** | 0 |
| chroma | `F.l1_loss(cat((output["cb_hat"],output["cr_hat"]),1),output["target_chroma"])` | Dc 直接预测的 Cb/Cr vs 原始 `cat(cb,cr)`；**L1** | 0 |

`target_rgb/target_luma/target_chroma` 来自 `system.py:38,46-47`；identity attack 将它们原样返回（`channel.py:17-19,47-52`）。luma/chroma 并非对 `rgb_hat` 再做 RGB→YCbCr 计算：它们直接使用 Dc 的三个分支输出。`rgb_hat` 则由同一三个分支经固定逆变换合成，所以数值上可对应，但比较路径不同。六项每次 `CleanLoss.forward` 都计算并返回；`training.py:543,563-564,569-571,576-577` 可将它们记入日志。权重为 0 的 luma/chroma 在求和时乘零，不向参数提供有效训练梯度；计算这两项本身和记录数值不改变优化目标。

原始 y 由 `rgb_to_ycbcr(rgb)` 计算：`y=0.299r+0.587g+0.114b`（`transforms.py:14-20`）。`s=y+Δc`，`x_float=s+Δw=y+Δc+Δw`（`models.py:54-55,82-83`；`system.py:42-45`）。因此 carrier loss 精确表达为 `MSE(y+Δc+Δw,y)`。在理想正交 DCT 下 c/w 频带不重叠，这个 MSE 分解为两扰动的平方能量和；它约束**两种**载荷，而非单独水印扰动。`2/255` 是两条扰动分别的每图 RMS 上限，合成 carrier 没有额外的联合 RMS cap（`transforms.py:31-38`）。range loss 若 `x_float` 全在 `[0,1]` 内，值与局部梯度均为零；否则可给 Ec/Ew 梯度。

## Loss → Network 梯度路径

下表的 `✓` 表示计算图中有反向路径，`×` 表示没有，`✓*` 表示计算图有路径但固定正交频带在理想精确算术下使跨分支导数为零。此结论由前向依赖和线性投影代数推导，**本次没有运行 backward**；实际浮点误差可能造成极小的非零交叉梯度。Dc/Dw 各自的 head 是分开的。

| loss | Ec | Ew | Dw | Dc | 证据/路径 |
|---|:---:|:---:|:---:|:---:|---|
| RGB | ✓ | ✓* | × | ✓ | `rgb_hat←Dc(x)←x=s+Δw←Ew,Ec`；`models.py:98-105`, `system.py:35-45` |
| message | ✓* | ✓ | ✓ | × | `logits←Dw(x)←x←Ew,Ec`；`models.py:119-121` |
| carrier | ✓ | ✓ | × | × | `MSE(x_float,y)`；`losses.py:16,22` |
| range | ✓ | ✓ | × | × | `range(x_float)`；`losses.py:23`；当前样本可为零梯度 |
| luma（权重 0） | ✓ | ✓* | × | ✓ | `y_hat←Dc(x)`；计算但不进入有效 total |
| chroma（权重 0） | ✓ | ✓* | × | ✓ | `cb_hat/cr_hat←Dc(x)`；计算但不进入有效 total |

为何 `✓*`：`P_cP_w=0`。Ec 的 `Δc` 在 c 频带，Ew 的 `bw=s-P_c(s)` 会消去它；最终 Dw 的固定 w-band DCT 也看不到 c 频带。故 message loss 对 Ec 的理想解析导数为零。Ew 的 `Δw` 在 w 频带，Dc 的 `z=x-P_w(x)` 和 `zc=P_c(x)` 都消去它；故 RGB/luma/chroma 对 Ew 的理想解析导数为零。**注意**：`P_w(x)` 消去的是最终 carrier 的整个 w 频带，连原始 y 在该频带的系数也一并移除；Dw 看到的 w 频带则包含原始 y 的系数和水印扰动。这两点影响结构分析。`models.py:77-82,98-105,119-121`；`transforms.py:53-61,80-90`。

## 结构能力：事实与推断分开

1. **单通道与预算。**【源码事实】传输的是 `[B,1,H,W]` 的 `x_float=y+Δc+Δw`，不是三通道 RGB；c/w 两路扰动各自受 `2/255` 每图 RMS 限制；`system.py:35-45`, `models.py:54-55,82-83`。【结构推断】在很小的 carrier 变化内承载 Cb/Cr 和 64 bits 可能限制可恢复颜色细节；不能仅由结构推出实际 PSNR 上限或宣称 30–31 dB 是理论极限。

2. **Dc 的尺度与上下文。**【源码事实】Dc 用 64 通道、8 个 3×3 ResidualBlock，主 CNN 始终在 256×256；纯 CNN 最长路径感受野约 37×37；有 block identity skip 与 `y_hat=z+correction` 旁路，但没有多尺度 encoder-decoder、pooling、U-Net skip、attention 或 dilation（`models.py:86-106`）。【结构推断】大范围颜色一致性及跨远距离区域的语义推断可能受局部感受野限制。固定 8×8 DCT 带来块内混合，并未构成 learned global context。

3. **被移除的亮度频带。**【源码事实】Dc 的 `z=x-P_w(x)` 去掉整个 w 频带，`zc=P_c(x)` 单独提供 c 频带；luma_head 的修正被 `P_cw` 限在 c+w 频带（`models.py:98-104`）。【结构推断】Dc 必须从相邻区域/其他频带推测原始 y 的 w-band 细节；这可能带来局部纹理或高频重建误差。与此同时 `y_hat` 在低/极高 `0` band 直接继承 z；不能把所有亮度信息都称为丢失。

4. **颜色表示方式。**【源码事实】Ec 输入 Y/Cb/Cr，但只输出 1 通道 c-band 扰动；Dc 从 2 通道 `(z,zc)` 预测 2 通道色度，两个色度 head 来自同一特征图（`models.py:52-55,98-105`）。【结构推断】一个受预算约束的 c-band residual 要编码两个色度分量，信息容量与可逆性可能成为瓶颈；模型可能依赖亮度对常见颜色的统计先验，复杂或罕见配色更难恢复。源码没有给出可证明的信息容量上界。

5. **颜色/水印交互。**【源码事实】c/w 固定掩码互斥；Ew 输入移除 c 频带，Dc 输入移除 w 频带，但 carrier loss 和 range loss 同时约束 `x=y+Δc+Δw`（`transforms.py:53-61`, `models.py:77-83,98-105`, `losses.py:22-23`）。【结构推断】两路在频带上避免直接重叠，但共享单通道动态范围、carrier 损失预算和训练优化目标；存在间接竞争，不能说它们在所有意义上完全独立。

6. **Dw 的聚合与瓶颈。**【源码事实】Dw 使用固定 8×8 DCT stride-8 得到 32×32×9，随后 8 个 block 和空间全局均值产出 64 logits；其最长卷积路径在块网格上理论可覆盖整图（`models.py:109-121`, `transforms.py:85-90`）。【结构推断】Dw 并非显然“偏浅到看不到全局”；固定只取 9 个频率和最终平均可能限制局部位置信息表达，但源码无法证明其是当前 RGB PSNR 的瓶颈。

7. **频域处理。**【源码事实】Ec、Ew、Dc、Dw 均使用固定 DCT 或 projector；Dc 的可训练主干仍是单尺度空间卷积，没有学习式频域特征融合（`models.py:40-121`）。【结构推断】固定频带约束可能帮助隔离两路，也限制网络自选有效频谱；要判断其对 RGB 误差的贡献，需要后续受控实验，当前审计不作因果结论。

## 未来可研究位置（仅定位，不设计 Network2）

P1 Ec 的单通道色度编码；P2 Dc 的颜色/亮度恢复主干；P3 `(z,zc)` 的特征使用与融合；P4 跨尺度上下文；P5 固定频带下的细节恢复；P6 carrier 与各自 RMS 预算；P7 两种载荷在损失和动态范围中的间接交互。以上是审计得到的待检验方向，不是修改方案。

## 尚不能确认

- 单次随机未训练前向和源码审计不能证明任何一项是 30–31 dB 饱和的主因，也不能给出可达到的 PSNR 上界。
- 历史运行是否逐字对应当前源码，单靠日志参数量与配置无法证明；`network1_manifest.json` 冻结的是**当前工作区源码**，不是历史 checkpoint 的完整 provenance。
- 当前 `configs/joint_10x20_clean_v1.json` 和 `configs/joint_10x20_rgb2_v1.json` **没有** `experiment` 字段；而对应 `runs/joint_10x20_clean_v1_full/config.json`、`runs/joint_10x20_rgb2_v1/config.json` 的历史存档含 10×20 `experiment`。当前 `training.py:421,452-460` 依配置中的 `experiment` 判定是否走固定协议。因此不能把这两份当前配置文件单独作为 RGB1/RGB2 固定协议的直接证据；历史存档可支持其曾按该协议运行，但无法仅凭存档证明当时源码与现源码字节相同。四份当前配置与历史记录中的 `model` / `channel` 设置一致，故这项差异不改变本次 Network1 **结构**定义。未修改任一配置。
- 当前目录没有 `.git`，因此无可记录的 Git commit/status；源码 SHA256 是本次冻结的可核验依据。
