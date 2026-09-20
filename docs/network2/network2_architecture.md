# Network V2 正式架构规格

状态：**Network V2 第一版实现冻结**。

本文是 Ec、Ew、Dc、Dw 的唯一正式架构规格。后文中的原理说明和历史措辞不得覆盖本节；如后文仍出现“建议”“可以”等非强制表述，以本节的冻结决策为准。任何改变均需先更新本文并重新审查，不能在实现代码中依据聊天记录或外部实现自行补全。

## 0. 全局冻结决策

### 0.1 Ew Host 分支必须使用 StopGrad

Ew 的 Host Frequency Encoder 固定使用：

\[
D_S=\operatorname{DCT}_8(\operatorname{StopGrad}(S))
\]

实现语义等价于：

```python
Ds = dct(S.detach())
```

StopGrad 只阻断 Host 条件分支对 Ec 的反向梯度。Ew 仍根据宿主内容生成水印，并且 `X=S+Delta_w` 的主残差路径保持可微；不得 detach `Delta_w`、`X` 或进入 Dw 的 attacked tensor。

### 0.2 TransformerBlock 统一实现

四个网络共享同一套 `TransformerBlock`、MDTA、GDFN 和 LayerNorm 实现代码，但每个网络、每个 block 都持有自己的参数；共享实现不等于共享权重。

统一规则：

- 输入/输出布局为 BCHW；
- LayerNorm 前将 BCHW 重排为 `B x HW x C`，只沿最后的 channel 维归一化，完成后恢复 BCHW；
- LayerNorm `eps=1e-5`；
- Ec、Dc 使用 WithBias LayerNorm；Ew、Dw 使用 BiasFree LayerNorm；
- WithBias LayerNorm 包含 weight 和 bias；BiasFree LayerNorm 只包含 weight；
- attention temperature 为每个 head 一个独立可训练参数，shape 为 `[heads,1,1]`，初始化为 1；
- Q、K reshape 为 `[B,heads,C/heads,HW]` 后，使用 `dim=-1` 做 L2 normalize；
- attention matrix 在最后一维执行 softmax；
- 所有普通 `3x3 Conv` 固定为 `stride=1, padding=1, dilation=1`；
- 所有 `1x1 Conv` 固定为 `stride=1, padding=0, dilation=1`；
- 未明确标成 depthwise 的卷积使用 `groups=1`；
- depthwise convolution 的 `groups` 等于该层输入通道数，因此 MDTA QKV depthwise conv 使用 `groups=3C`，GDFN depthwise conv 使用 `groups=2hidden`，64-channel 局部融合 depthwise conv 使用 `groups=64`；
- MDTA 和 GDFN 均使用 Pre-Norm 和 block 内 residual add；
- 不使用 Dropout、DropPath、BatchNorm 或位置编码。

### 0.3 初始化冻结

除以下特例外，所有 Conv2d 和 Linear 的 weight 使用 Xavier uniform，bias 初始化为 0：

- Ec 最终 `Conv3x3, 24->1` head：weight 和 bias 全零；
- Ew 最终 `Conv1x1, 32->9` coefficient head：weight 使用 `Normal(mean=0,std=1e-4)`，bias 为 0；
- Dc 的 Chroma Head 和 Luma Head：Xavier uniform，bias 为 0；
- Dw 的两个输出卷积和两个 Linear：Xavier uniform，bias 为 0。

所有 LayerNorm weight 初始化为 1；WithBias LayerNorm bias 初始化为 0；所有 attention temperature 初始化为 1。Ew coefficient head 不得零初始化。

### 0.4 RMSCap 原样复用现有定义

Ec 和 Ew 必须复用同一个逐样本 RMSCap：

\[
r=\sqrt{\operatorname{mean}(x^2,\operatorname{dim}=(1,2,3),\operatorname{keepdim}=\mathrm{True})+10^{-8}}
\]

\[
g=\min\left(\frac{\delta}{r},1\right),\qquad
\operatorname{RMSCap}(x,\delta)=x\cdot g
\]

Ec 使用 `delta_c`，Ew 使用 `delta_w`。不得改成逐像素 clamp、逐通道 cap 或另一套幅度限制。

### 0.5 Python 外部接口冻结

保留现有类名：

- `ColorEncoder`（Ec）；
- `WatermarkEncoder`（Ew）；
- `ColorDecoder`（Dc）；
- `WatermarkDecoder`（Dw）。

Ec 保留以下返回 key：

```text
candidate = Conv3x3(24->1) head 的原始空间输出
residual  = Pc(candidate) 再经过 RMSCap 后的 Delta_c
carrier   = S = Y + Delta_c
```

Ew 保留并扩展以下返回 key：

```text
candidate    = scatter + IDCT 后、RMSCap 前的空间残差
coefficients = 9-channel A_w
dct_residual = scatter 后的 64-channel DCT 系数图
residual     = RMSCap 后的 Delta_w
carrier      = X = S + Delta_w
```

Dc 必须继续提供 `rgb`、`y`、`cb`、`cr`、`raw_luma_delta`、`luma_delta`、`z`、`zc`。其中 `z=Z0+Zc=X'-Pw(X')`，`zc=Zc=Pc(X')`，以保持现有 system、loss、metrics 和诊断语义。

Dw 直接返回 `[B,64]` raw logits，不返回 sigmoid 概率。

### 0.6 Ew 与 Dw 不共享权重

Ew 与 Dw 只在以下结构语义上对应：

- 使用相同的 9 个频点；
- 使用相同的频点顺序；
- 使用相同的 `32x32` DCT block grid；
- Ew 产生二维 message pattern，Dw 从二维 evidence pattern 恢复消息。

二者不共享任何 Conv、Linear、LayerNorm、attention temperature 或 TransformerBlock 参数，也不要求逐层互逆。

### 0.7 架构版本和 checkpoint 隔离

Network V2 配置必须显式包含：

```yaml
model:
  architecture_version: v2
```

规则：

- 缺少 `architecture_version` 的旧配置或旧 checkpoint 一律视为 `v1`；
- V2 checkpoint 必须在保存的 config 中显式记录 `architecture_version=v2`；
- V2 禁止 resume 或 init-from V1 checkpoint；
- evaluate 必须在构造模型和加载 state_dict 前检查 architecture version；
- 不得用 `strict=False` 绕过 V1/V2 不兼容；
- V2 正式实验从头训练；
- V1 的 `channels=64`、`blocks=8` 是 V1 结构参数，不属于必须保持不变的实验控制变量。

### 0.8 固定尺寸和测试边界

完整 Ew、Dw 和完整 system 固定使用 256x256 输入与 64-bit message。Dw 的正式路径必须保持：

```text
B x 1 x 256 x 256
-> B x 9 x 32 x 32
-> B x 1 x 32 x 32
-> flatten B x 1024
-> Linear(1024,256)
-> Linear(256,64)
```

不得为了兼容 16x16 或 32x32 测试加入 AdaptivePool、动态 Linear，或改变 32x32 message grid。小尺寸输入只能用于与固定 flatten 无关的局部模块测试。

# Ec v2 定稿方案

## 1. 设计目标

Ec v2 仍然完成：

\[
(Y,C_b,C_r)\rightarrow \Delta_c
\]

接口保持不变：

\[
s=Y+\Delta_c
\]

最终传输仍然只有：

\[
x=Y+\Delta_c+\Delta_w
\]

Ec 的内部特征不会传给 Dc 或 Dw。

设计借鉴 Restormer 的：

- MDTA；
- GDFN；
- 多尺度编码解码；
- PixelUnshuffle/PixelShuffle；
- 编码器到解码器的内部 skip。

官方依据：[Restormer论文](https://openaccess.thecvf.com/content/CVPR2022/papers/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.pdf) · [官方实现](https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py)

---

# 2. 完整数据流

```text
Y [B,1,256,256] ── Conv3×3, 1→8 ───────┐
                                        ├─ concat
CbCr [B,2,256,256] ─ Conv3×3, 2→16 ────┘
                                        ↓
                              [B,24,256,256]
                                        ↓
                              Conv3×3, 24→24
                                        ↓
                         Encoder Level 1：TB×1
                              [B,24,256,256]
                                        │
                                  保存 skip E1
                                        ↓
                                  Downsample
                                        ↓
                         Encoder Level 2：TB×1
                              [B,48,128,128]
                                        │
                                  保存 skip E2
                                        ↓
                                  Downsample
                                        ↓
                         Encoder Level 3：TB×2
                               [B,96,64,64]
                                        │
                                  保存 skip E3
                                        ↓
                                  Downsample
                                        ↓
                            Latent：TB×3
                              [B,192,32,32]
                                        ↓
                                   Upsample
                                        ↓
                              [B,96,64,64]
                                        ↓
                              concat(E3)
                                        ↓
                            Conv1×1, 192→96
                                        ↓
                        Decoder Level 3：TB×2
                                        ↓
                                   Upsample
                                        ↓
                              [B,48,128,128]
                                        ↓
                              concat(E2)
                                        ↓
                             Conv1×1, 96→48
                                        ↓
                        Decoder Level 2：TB×1
                                        ↓
                                   Upsample
                                        ↓
                              [B,24,256,256]
                                        ↓
                              concat(E1)
                                        ↓
                             Conv1×1, 48→24
                                        ↓
                        Decoder Level 1：TB×1
                                        ↓
                           Refinement TB×1
                                        ↓
                      zero-init Conv3×3, 24→1
                                        ↓
                                 candidate
                                        ↓
                        Pc(candidate) + RMSCap
                                        ↓
                                      Δc
                                        ↓
                                  s = Y + Δc
```

这里的 skip 全部发生在 Ec 内部。

---

# 3. 为什么用四层和三次下采样

三次下采样后：

\[
256\rightarrow128\rightarrow64\rightarrow32
\]

最终 32×32 恰好对应原图的 8×8 区域：

\[
256/8=32
\]

而当前颜色载荷最终也是通过不重叠的 8×8 DCT 块写入。因此最深层的一个位置大致对应一个 DCT 块，结构上比停在 64×64 更符合当前频域输出机制。

高分辨率 skip 负责保留：

- 边缘；
- 精确空间位置；
- 纹理；
- DCT 块之间的细节。

低分辨率 Transformer 负责建模更大范围的颜色关系。

---

# 4. 通道和 Block 配置

| 阶段 | 分辨率 | 通道 | Block 数 | Heads |
|---|---:|---:|---:|---:|
| Encoder L1 | 256×256 | 24 | 1 | 1 |
| Encoder L2 | 128×128 | 48 | 1 | 2 |
| Encoder L3 | 64×64 | 96 | 2 | 4 |
| Latent L4 | 32×32 | 192 | 3 | 8 |
| Decoder L3 | 64×64 | 96 | 2 | 4 |
| Decoder L2 | 128×128 | 48 | 1 | 2 |
| Decoder L1 | 256×256 | 24 | 1 | 1 |
| Refinement | 256×256 | 24 | 1 | 1 |

每个 head 始终处理 24 个通道：

\[
24/1=48/2=96/4=192/8=24
\]

这样每层的 attention head 宽度一致。

选择基础通道 24 而不是 48 或64，是因为：

- Ec 最终只输出一个通道；
- 当前固定使用 FP32；
- 需要在 8GB 显存上训练；
- 多尺度结构本身已经明显增加参数容量；
- 后续还要升级 Dc、Ew 和 Dw。

估算 Ec v2 约 **288 万参数**，旧 Ec 约 59 万参数。但由于大量计算转移到低分辨率，并使用 depth-wise convolution，估算 256×256 前向约为 **9.6 GMAC**；旧 Ec 的 16 个全分辨率 64通道卷积约为 **38.8 GMAC**。

---

# 5. Transformer Block 精确定义

每个 Block 使用 Pre-Norm：

\[
X_1=X+\operatorname{MDTA}(\operatorname{LN}(X))
\]

\[
X_2=X_1+\operatorname{GDFN}(\operatorname{LN}(X_1))
\]

LayerNorm 第一版选择 **WithBias LayerNorm**。

## MDTA

```text
输入 X
  ↓
WithBias LayerNorm
  ↓
1×1 Conv：C → 3C
  ↓
3×3 Depth-wise Conv，groups=3C
  ↓
拆分 Q、K、V
  ↓
reshape：[B, heads, C/heads, HW]
  ↓
Q、K 在 HW 维做 L2 normalize
  ↓
QKᵀ × learnable temperature
  ↓
Softmax
  ↓
Attention × V
  ↓
恢复 BCHW
  ↓
1×1 Conv：C → C
  ↓
与输入相加
```

MDTA 建立的是输入相关的通道关系，不构造 `HW×HW` 空间注意力。

## GDFN

采用官方的 `ffn_expansion_factor=2.66`：

```text
输入 X
  ↓
WithBias LayerNorm
  ↓
1×1 Conv：C → 2×hidden
  ↓
3×3 Depth-wise Conv
  ↓
拆成 A、B
  ↓
GELU(A) × B
  ↓
1×1 Conv：hidden → C
  ↓
与输入相加
```

其中：

\[
hidden=\lfloor2.66C\rfloor
\]

第一版不加入：

- Dropout；
- DropPath；
- 位置编码；
- BatchNorm；
- 额外通道注意力。

这样结构最接近官方 Restormer Block，也方便做论文说明。

---

# 6. 上下采样

采用官方方式：

```text
Downsample(C):
Conv3×3(C → C/2)
→ PixelUnshuffle(2)
→ 输出通道 2C
```

例如：

```text
[B,24,256,256]
→ Conv 24→12
→ PixelUnshuffle
→ [B,48,128,128]
```

上采样：

```text
Upsample(C):
Conv3×3(C → 2C)
→ PixelShuffle(2)
→ 输出通道 C/2
```

PixelUnshuffle不会像普通池化一样直接删除采样点，更适合生成精确的隐藏扰动。

---

# 7. 输入分支

采用浅层双分支：

```text
Y    → Conv 1→8
CbCr → Conv 2→16
```

原因是：

- Y 是载体内容和空间结构；
- Cb/Cr 是需要编码的主要颜色载荷；
- 每个输入平面大致分配8个特征通道。

两路经过一层卷积后立即融合，不建立两个独立深层网络。这样既保留任务角色差异，也能尽早学习：

\[
Y\text{内容}\leftrightarrow C_b/C_r\text{写入方式}
\]

---

# 8. 输出头

输出端为：

```text
Refinement feature
→ Conv3×3(24→1)
→ candidate
→ Pc(candidate)
→ RMSCap(delta_c)
→ Δc
```

要求：

- head 权重零初始化；
- 不使用 sigmoid；
- 不使用 tanh；
- 不在 head 内加入 Y；
- `carrier=Y+Δc` 仍在外部完成。

零初始化保证初始状态：

\[
\Delta_c=0,\qquad s=Y
\]

---

# 9. 历史 Ec 单网络实验说明（非 Network V2 正式协议）

本节记录的是仅替换 Ec、其余网络仍使用 V1 的早期单网络消融设想，不属于 Network V2 四网络实现或正式实验协议，不得据此在 V2 中保留原 Dc、Ew 或 Dw。Network V2 的实现与实验以本文第 0 节冻结决策、四网络定稿结构及当前受保护实验控制变量为准。

第一轮只验证 Ec 架构，因此保持：

- `delta_c=2/255`；
- `delta_w=2/255`；
- 固定颜色频带 `Pc`；
- 原 Dc；
- 原 Ew/Dw；
- 原损失权重；
- 相同训练数据；
- 相同步数；
- 相同 PSNR 计算方式。

对比：

| 实验 | Ec | 其他网络 |
|---|---|---|
| Baseline | 旧 8 ResBlocks | 原结构 |
| Ec v2 | 新 Restormer Ec | 原结构 |

Dc 结构保持一致，但不能冻结，因为它需要适应新 Ec 的颜色编码方式。

第一轮重点记录：

- RGB PSNR、SSIM；
- Y、Cb、Cr 分别的恢复误差；
- carrier PSNR；
- `delta_c_rms`；
- BER；
- candidate 中被 `Pc` 保留下来的能量比例；
- 显存、运行时间和参数量。

Ec v2 第一版固定采用上述结构：结构来源清楚，符合 Restormer 的核心实现，也针对当前 8×8 DCT 隐藏颜色任务做了明确适配。



# Ew v2 网络结构

总体拆成四部分：

\[
\boxed{
E_w=
\text{Message Pattern Encoder}
+
\text{Host Frequency Encoder}
+
\text{Fusion Restormer}
+
\text{Coefficient Head}
}
\]

输入：

\[
S\in\mathbb R^{B\times1\times256\times256},
\qquad
m\in\{0,1\}^{B\times64}
\]

输出：

\[
\Delta_w\in\mathbb R^{B\times1\times256\times256}
\]

最终载体：

\[
X=S+\Delta_w
\]

---

## 一、完整结构图

```text
                 64-bit message m
                         │
                         ▼
              Message Pattern Encoder
                         │
                         ▼
                   B×64×32×32
                         │
                         │
S: B×1×256×256           │
        │                │
        ▼                │
 Fixed 8×8 Block DCT     │
        │                │
        ▼                │
   B×64×32×32            │
        │                │
        ▼                │
Host Frequency Encoder   │
        │                │
        ▼                │
   B×64×32×32            │
        └───────┬────────┘
                ▼
          Concatenation
                │
                ▼
         B×128×32×32
                │
                ▼
       1×1 Fusion Conv
                │
                ▼
          B×64×32×32
                │
                ▼
      Restormer Block ×4
                │
                ▼
          B×64×32×32
                │
                ▼
       Coefficient Head
                │
                ▼
           B×9×32×32
                │
                ▼
  Scatter到9个固定DCT频点
                │
                ▼
        固定 Block IDCT
                │
                ▼
      B×1×256×256 raw Δw
                │
                ▼
          RMSCap(δw)
                │
                ▼
               Δw
                │
                ▼
             X=S+Δw
```

---

# 二、Message Pattern Encoder

WOFA 使用两层全连接，将 bit 消息转换成二维灰度噪声模式。我们保留这个核心思想，但把二维模式放在你的 DCT 块网格上。[WOFA，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Watermarking_One_for_All_A_Robust_Watermarking_Scheme_Against_Partial_CVPR_2025_paper.pdf)

## 结构

首先把消息从 \(\{0,1\}\) 转为 \(\{-1,1\}\)：

\[
m_s=2m-1
\]

然后：

```text
B×64
  │
Linear 64→256
  │
GELU
  │
Linear 256→1024
  │
reshape
  │
B×1×32×32
  │
Conv 3×3, 1→32
  │
GELU
  │
Conv 3×3, 32→64
  │
B×64×32×32
```

写成公式：

\[
M_0=
\operatorname{reshape}_{1\times32\times32}
\left(
W_2\operatorname{GELU}(W_1m_s)
\right)
\]

\[
F_m=
\operatorname{Conv}_{3\times3}^{32\rightarrow64}
\left(
\operatorname{GELU}
\left(
\operatorname{Conv}_{3\times3}^{1\rightarrow32}(M_0)
\right)
\right)
\]

最终：

\[
F_m\in\mathbb R^{B\times64\times32\times32}
\]

## 为什么不用旧版的 64 通道广播

旧版每个像素收到相同的 64 维向量。

新版先产生一个真正不同位置具有不同数值的：

\[
1\times32\times32
\]

消息模式，再将它编码成 64 通道特征。

这样每个 DCT 块都会获得不同但相互关联的消息编码。

---

# 三、Host Frequency Encoder

输入是 Ec 产生的载体：

\[
S=Y+\Delta_c
\]

直接做固定 8×8 Block DCT：

\[
D_S=\operatorname{DCT}_8(S)
\]

输出：

\[
D_S\in\mathbb R^{B\times64\times32\times32}
\]

## 结构

```text
S: B×1×256×256
        │
固定8×8 Block DCT
        │
B×64×32×32
        │
Conv 3×3, 64→64
        │
GELU
        │
Conv 3×3, 64→64
        │
B×64×32×32
```

即：

\[
F_s=
\operatorname{Conv}_{3\times3}
\left(
\operatorname{GELU}
\left(
\operatorname{Conv}_{3\times3}(D_S)
\right)
\right)
\]

这里固定输入完整 64 个 DCT 系数，而不是继续使用：

\[
S-P_c(S)
\]

因为输出端已经严格限制为 \(P_w\)，输入端没有必要删除原始图像的 \(P_c(Y)\) 纹理。

为了防止联合训练时水印分支通过该输入反向推动 Ec，Host Frequency Encoder 的输入固定写成：

\[
D_S=\operatorname{DCT}_8(\operatorname{StopGrad}(S))
\]

Ew 仍然能观察 \(S\)，但不会通过内容分析分支改变 Ec。这里只 detach Host 条件分支；`X=S+Delta_w` 主路径不 detach。

---

# 四、Feature Fusion

消息特征：

\[
F_m\in\mathbb R^{B\times64\times32\times32}
\]

图像特征：

\[
F_s\in\mathbb R^{B\times64\times32\times32}
\]

首先拼接：

\[
F_{cat}=\operatorname{Concat}(F_s,F_m)
\]

得到：

\[
F_{cat}\in\mathbb R^{B\times128\times32\times32}
\]

然后：

```text
B×128×32×32
      │
Conv 1×1, 128→64
      │
Conv 3×3 Depth-wise, 64→64
      │
GELU
      │
Conv 1×1, 64→64
      │
      +  前面的1×1融合结果
      │
B×64×32×32
```

公式：

\[
F_0=\operatorname{Conv}_{1\times1}^{128\rightarrow64}(F_{cat})
\]

\[
F_f=
F_0+
\operatorname{Conv}_{1\times1}
\left(
\operatorname{GELU}
\left(
\operatorname{DWConv}_{3\times3}(F_0)
\right)
\right)
\]

这里使用 Depth-wise Conv，让每个 DCT 块先吸收附近块的信息。

---

# 五、主体使用 Restormer Block ×4

主体不再使用旧的 8 个普通 ResBlock。

固定配置：

| 参数 | 设计 |
|---|---:|
| 特征通道 | 64 |
| Restormer Block 数量 | 4 |
| Attention heads | 4 |
| GDFN expansion ratio | 2.66 |
| LayerNorm | Bias-free LayerNorm |
| 下采样 | 无 |
| 上采样 | 无 |

```text
Ff: B×64×32×32
        │
Restormer Block
        │
Restormer Block
        │
Restormer Block
        │
Restormer Block
        │
        + Ff
        │
Fc: B×64×32×32
```

即：

\[
F_c=F_f+\mathcal R_4(\mathcal R_3(\mathcal R_2(\mathcal R_1(F_f))))
\]

每个 Restormer Block：

```text
Input
  │
Bias-free LayerNorm
  │
MDTA
  │
Residual Add
  │
Bias-free LayerNorm
  │
GDFN
  │
Residual Add
  │
Output
```

Restormer 的 MDTA 通过通道注意力建立全局关系，GDFN 使用门控和深度卷积筛选信息。[Restormer，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Zamir_Restormer_Efficient_Transformer_for_High-Resolution_Image_Restoration_CVPR_2022_paper.html)

## 为什么只用 4 个 Block

这里处理的是：

\[
32\times32
\]

的 DCT 块网格，而不是 256×256 原图。

消息的全局组合已经在两层 FC 中完成，Restormer 主要负责：

- 消息特征和载体内容的结合；
- 不同 DCT 块之间的信息协调；
- 根据纹理决定系数分布。

因此没有必要一开始堆很深。

---

# 六、为什么不使用多尺度 U-Net

Ew 已经在 1/8 分辨率的块网格工作：

\[
256\times256\rightarrow32\times32
\]

如果继续下采样：

\[
32\rightarrow16\rightarrow8
\]

容易丢失具体 DCT 块的位置关系。

因此 Ew v2 不采用：

- Encoder-Decoder 下采样；
- PixelShuffle；
- 跨层 U-Net skip。

这里保留的是：

- Restormer Block 内部 residual；
- Fusion 部分 residual；
- 整个主体的 long residual；
- 最终的载体残差连接 \(X=S+\Delta_w\)。

---

# 七、Coefficient Head

经过 Restormer 后：

\[
F_c\in\mathbb R^{B\times64\times32\times32}
\]

输出头：

```text
B×64×32×32
      │
Conv 3×3, 64→32
      │
GELU
      │
Conv 1×1, 32→9
      │
B×9×32×32
```

公式：

\[
A_w=
\operatorname{Conv}_{1\times1}^{32\rightarrow9}
\left(
\operatorname{GELU}
\left(
\operatorname{Conv}_{3\times3}^{64\rightarrow32}(F_c)
\right)
\right)
\]

其中：

\[
A_w\in\mathbb R^{B\times9\times32\times32}
\]

代表每个 DCT 块的 9 个水印增量系数。

最后使用 `1×1 Conv` 做 9 个频率通道的组合，也参考了 TrustMark 使用轻量 `1×1` 后处理改善编码质量的思路。[TrustMark，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Bui_TrustMark_Robust_Watermarking_and_Watermark_Removal_for_Arbitrary_Resolution_Images_ICCV_2025_paper.html)

输出头不使用：

- Sigmoid；
- Tanh；
- ReLU。

因为水印系数必须允许正负值，强度由后面的 RMSCap 控制。

输出层固定采用很小的非零初始化：

\[
W_{\text{head}}\sim\mathcal N(0,10^{-4})
\]

其 bias 固定为 0。这样初始扰动很小，同时第一步消息分支和主干就能收到梯度；该 head 不得零初始化。

---

# 八、固定 DCT 输出层

对每个块建立 64 个系数位置：

\[
C_w\in\mathbb R^{B\times64\times32\times32}
\]

把 \(A_w\) 的 9 个通道放到固定水印坐标：

\[
(0,3),(1,2),(2,1),(3,0),
\]

\[
(0,4),(1,3),(2,2),(3,1),(4,0)
\]

其余 55 个系数全部为零。

然后：

\[
\Delta_w^{raw}
=
\operatorname{IDCT}_8(C_w)
\]

得到：

\[
\Delta_w^{raw}\in
\mathbb R^{B\times1\times256\times256}
\]

再执行：

\[
\Delta_w=
\operatorname{RMSCap}
\left(
\Delta_w^{raw},\delta_w
\right)
\]

最后：

\[
X=S+\Delta_w
\]

这样可以严格保证：

\[
P_c(\Delta_w)=0
\]

并且：

\[
P_0(\Delta_w)=0
\]

即 Ew 只能写入 \(P_w\)。

---

# 九、最终逐层表

| 阶段 | 层 | 输出尺寸 |
|---|---|---|
| 消息输入 | \(m\) | `B×64` |
| 消息编码 | Linear 64→256 + GELU | `B×256` |
| 消息编码 | Linear 256→1024 | `B×1024` |
| 重排 | reshape | `B×1×32×32` |
| 消息 Stem | Conv 3×3, 1→32 + GELU | `B×32×32×32` |
| 消息 Stem | Conv 3×3, 32→64 | `B×64×32×32` |
| 载体输入 | \(S\) | `B×1×256×256` |
| 固定变换 | Block DCT | `B×64×32×32` |
| 载体 Stem | Conv 3×3, 64→64 + GELU | `B×64×32×32` |
| 载体 Stem | Conv 3×3, 64→64 | `B×64×32×32` |
| 融合 | concat | `B×128×32×32` |
| 融合 | Conv 1×1, 128→64 | `B×64×32×32` |
| 局部融合 | DWConv 3×3 + GELU + Conv 1×1 | `B×64×32×32` |
| 主体 | Restormer Block ×4 | `B×64×32×32` |
| 输出 Head | Conv 3×3, 64→32 + GELU | `B×32×32×32` |
| 输出 Head | Conv 1×1, 32→9 | `B×9×32×32` |
| 固定层 | Scatter 9→64 | `B×64×32×32` |
| 固定层 | Block IDCT | `B×1×256×256` |
| 约束 | RMSCap | `B×1×256×256` |
| 残差输出 | \(X=S+\Delta_w\) | `B×1×256×256` |

# 最终结论

Ew 第一版固定结构是：

\[
\boxed{
\text{WOFA式二维消息编码}
+
\text{DCT块域载体特征}
+
\text{4个Restormer Block}
+
\text{9通道频率输出头}
+
\text{固定IDCT与RMSCap}
}
\]

它没有照搬某一篇论文，而是让每个参考负责自己擅长的部分：

- WOFA：64-bit 到二维消息模式；
- Restormer：局部与全局特征融合；
- TrustMark：轻量输出细化思想；
- 你的原始系统：固定 DCT 频带隔离和 RMS 预算。



# Dc v2 完整定稿方案

Dc v2 定义为：

\[
\boxed{
D_c^{v2}
=
\text{固定频带分解}
+
\text{双分支输入编码}
+
\text{多尺度Restormer U-Net}
+
\text{色度/亮度双输出头}
}
\]

输入：

\[
X'\in\mathbb R^{B\times1\times256\times256}
\]

输出：

\[
\hat Y,\hat C_b,\hat C_r,\hat{RGB}
\]

Dc 只接收传输通道输出 \(X'\)，不接收 Ec/Ew 的内部特征、原始图像、消息或标签。

---

## 1. 完整数据流

```text
输入 X'
[B,1,256,256]
        │
        ▼
固定 8×8 Block DCT
        │
        ▼
D = DCT8(X')
[B,64,32,32]
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
固定 P0 投影                           固定 Pc 投影
        │                                      │
        ▼                                      ▼
Z0 = P0(X')                            Zc = Pc(X')
[B,1,256,256]                          [B,1,256,256]
        │                                      │
        ▼                                      ▼
Conv3×3：1→8                          Conv3×3：1→16
        │                                      │
        ▼                                      ▼
F0：[B,8,256,256]                    Fc：[B,16,256,256]
        │                                      │
        └──────────────────┬───────────────────┘
                           │
                           ▼
                    通道维 Concat
                           │
                           ▼
                   [B,24,256,256]
                           │
                           ▼
                  Conv3×3：24→24
                           │
                           ▼
                   [B,24,256,256]
                           │
                           ▼
                Encoder Level 1
                Restormer Block ×1
                channels=24, heads=1
                           │
                           ▼
                  E1：[B,24,256,256]
                           │
                      保存 Skip E1
                           │
                           ▼
                       Downsample
            Conv3×3：24→12
            PixelUnshuffle(2)
                           │
                           ▼
                  [B,48,128,128]
                           │
                           ▼
                Encoder Level 2
                Restormer Block ×1
                channels=48, heads=2
                           │
                           ▼
                  E2：[B,48,128,128]
                           │
                      保存 Skip E2
                           │
                           ▼
                       Downsample
            Conv3×3：48→24
            PixelUnshuffle(2)
                           │
                           ▼
                   [B,96,64,64]
                           │
                           ▼
                Encoder Level 3
                Restormer Block ×2
                channels=96, heads=4
                           │
                           ▼
                   E3：[B,96,64,64]
                           │
                      保存 Skip E3
                           │
                           ▼
                       Downsample
            Conv3×3：96→48
            PixelUnshuffle(2)
                           │
                           ▼
                  [B,192,32,32]
                           │
                           ▼
                     Latent Level
                Restormer Block ×3
                channels=192, heads=8
                           │
                           ▼
                  [B,192,32,32]
                           │
                           ▼
                        Upsample
            Conv3×3：192→384
            PixelShuffle(2)
                           │
                           ▼
                   [B,96,64,64]
                           │
                     concat(E3)
                           │
                           ▼
                  [B,192,64,64]
                           │
                  Conv1×1：192→96
                           │
                           ▼
                Decoder Level 3
                Restormer Block ×2
                channels=96, heads=4
                           │
                           ▼
                   [B,96,64,64]
                           │
                           ▼
                        Upsample
            Conv3×3：96→192
            PixelShuffle(2)
                           │
                           ▼
                  [B,48,128,128]
                           │
                     concat(E2)
                           │
                           ▼
                  [B,96,128,128]
                           │
                   Conv1×1：96→48
                           │
                           ▼
                Decoder Level 2
                Restormer Block ×1
                channels=48, heads=2
                           │
                           ▼
                  [B,48,128,128]
                           │
                           ▼
                        Upsample
            Conv3×3：48→96
            PixelShuffle(2)
                           │
                           ▼
                  [B,24,256,256]
                           │
                     concat(E1)
                           │
                           ▼
                  [B,48,256,256]
                           │
                   Conv1×1：48→24
                           │
                           ▼
                Decoder Level 1
                Restormer Block ×1
                channels=24, heads=1
                           │
                           ▼
                  [B,24,256,256]
                           │
                           ▼
                  Refinement TB ×1
                  channels=24, heads=1
                           │
                           ▼
                 Fout：[B,24,256,256]
                    ┌──────┴───────┐
                    │              │
                    ▼              ▼
               Chroma Head      Luma Head
               Conv3×3          Conv3×3
                 24→2             24→1
                    │              │
                    ▼              ▼
             Cb_hat,Cr_hat     raw_luma_delta
                                   │
                                   ▼
                              固定 Pcw 投影
                                   │
                                   ▼
                         luma_delta=Pcw(raw)
                                   │
                                   ▼
                    Y_hat=Z0+Zc+luma_delta
                    │              │
                    └──────┬───────┘
                           │
                           ▼
                   固定 YCbCr→RGB
                           │
                           ▼
                  RGB_hat：[B,3,256,256]
```

---

## 2. 固定频带输入

首先执行一次：

\[
D=\operatorname{DCT}_8(X')
\]

\[
D\in\mathbb R^{B\times64\times32\times32}
\]

当前三个频带满足：

\[
P_0+P_c+P_w=I
\]

### 2.1 结构频带

\[
Z_0=P_0(X')
\]

即：

\[
Z_0
=
\operatorname{IDCT}_8(M_0\odot D)
\]

理想 clean 情况下：

\[
Z_0=P_0(Y)
\]

因为：

\[
P_0(\Delta_c)=P_0(\Delta_w)=0
\]

\(Z_0\) 负责提供：

- 大尺度亮度；
- 基本形状；
- 边缘和纹理参考；
- 未承载载荷的可靠结构信息。

### 2.2 颜色载荷频带

\[
Z_c=P_c(X')
\]

即：

\[
Z_c
=
\operatorname{IDCT}_8(M_c\odot D)
\]

理想情况下：

\[
Z_c=P_c(Y)+\Delta_c
\]

它同时包含：

- 原始亮度在 \(P_c\) 中的系数；
- Ec 写入的颜色载荷 \(\Delta_c\)。

### 2.3 亮度旁路

定义：

\[
Z=Z_0+Z_c
\]

由于三个频带完备且互斥：

\[
Z_0+Z_c=X'-P_w(X')
\]

所以它与原 Dc 的亮度旁路完全等价。

---

## 3. 双分支输入编码

### 3.1 结构分支

```text
Z0：[B,1,256,256]
→ Conv3×3, padding=1, 1→8
→ F0：[B,8,256,256]
```

公式：

\[
F_0=
\operatorname{Conv}_{3\times3}^{1\rightarrow8}(Z_0)
\]

### 3.2 颜色分支

```text
Zc：[B,1,256,256]
→ Conv3×3, padding=1, 1→16
→ Fc：[B,16,256,256]
```

公式：

\[
F_c=
\operatorname{Conv}_{3\times3}^{1\rightarrow16}(Z_c)
\]

颜色分支使用更多通道，是因为：

- 一个 \(Z_c\) 平面承载两个目标色度平面；
- 颜色残差幅度很小；
- 需要区分 \(P_c(Y)\) 与 \(\Delta_c\)；
- 结构信息已经通过固定亮度旁路得到较强保护。

### 3.3 浅层融合

\[
F_{\mathrm{cat}}
=
\operatorname{Concat}(F_0,F_c)
\]

\[
F_{\mathrm{cat}}
\in
\mathbb R^{B\times24\times256\times256}
\]

然后：

\[
F_{\mathrm{in}}
=
\operatorname{Conv}_{3\times3}^{24\rightarrow24}
(F_{\mathrm{cat}})
\]

得到：

\[
F_{\mathrm{in}}
\in
\mathbb R^{B\times24\times256\times256}
\]

三个浅层卷积后均不加入 ReLU/GELU，避免在进入 Restormer 前截断输入中的负频率响应。

---

## 4. 主干配置

| 阶段 | 分辨率 | 通道 | Block数 | Heads |
|---|---:|---:|---:|---:|
| 双分支输入融合 | 256×256 | 24 | — | — |
| Encoder L1 | 256×256 | 24 | 1 | 1 |
| Encoder L2 | 128×128 | 48 | 1 | 2 |
| Encoder L3 | 64×64 | 96 | 2 | 4 |
| Latent L4 | 32×32 | 192 | 3 | 8 |
| Decoder L3 | 64×64 | 96 | 2 | 4 |
| Decoder L2 | 128×128 | 48 | 1 | 2 |
| Decoder L1 | 256×256 | 24 | 1 | 1 |
| Refinement | 256×256 | 24 | 1 | 1 |

每个 head 始终包含24个通道：

\[
24/1=48/2=96/4=192/8=24
\]

Dc v2 与 Ec v2 使用相同的尺度、通道数和 Block配置，但不共享参数或内部特征。

---

## 5. Restormer Block

每个 Block 使用 Pre-Norm：

\[
X_1=X+\operatorname{MDTA}(\operatorname{LN}(X))
\]

\[
X_2=X_1+\operatorname{GDFN}(\operatorname{LN}(X_1))
\]

LayerNorm 选择：

\[
\boxed{\text{WithBias LayerNorm}}
\]

### 5.1 MDTA

```text
输入 X
→ WithBias LayerNorm
→ Conv1×1：C→3C
→ DWConv3×3，groups=3C
→ 拆分 Q、K、V
→ reshape：[B,heads,C/heads,HW]
→ Q、K沿HW维L2 normalize
→ QKᵀ × learnable temperature
→ Softmax
→ Attention × V
→ reshape回BCHW
→ Conv1×1：C→C
→ 与输入残差相加
```

MDTA 生成的是通道关系矩阵：

\[
\frac Ch\times\frac Ch
\]

而不是：

\[
HW\times HW
\]

因此能用于高分辨率特征。

### 5.2 GDFN

```text
输入
→ WithBias LayerNorm
→ Conv1×1：C→2hidden
→ DWConv3×3
→ 拆分 A、B
→ GELU(A) × B
→ Conv1×1：hidden→C
→ 与输入残差相加
```

其中：

\[
hidden=\lfloor2.66C\rfloor
\]

第一版不使用：

- Dropout；
- DropPath；
- BatchNorm；
- 位置编码；
- SE；
- 额外空间注意力；
- FFT模块；
- Cross-attention。

---

## 6. 下采样

定义：

```text
Downsample(C):
Conv3×3：C→C/2
→ PixelUnshuffle(2)
→ 输出通道2C
→ 分辨率减半
```

具体为：

```text
24×256×256
→ Conv 24→12
→ PixelUnshuffle
→ 48×128×128
```

```text
48×128×128
→ Conv 48→24
→ PixelUnshuffle
→ 96×64×64
```

```text
96×64×64
→ Conv 96→48
→ PixelUnshuffle
→ 192×32×32
```

---

## 7. 上采样

定义：

```text
Upsample(C):
Conv3×3：C→2C
→ PixelShuffle(2)
→ 输出通道C/2
→ 分辨率扩大两倍
```

具体为：

```text
192×32×32
→ Conv 192→384
→ PixelShuffle
→ 96×64×64
```

```text
96×64×64
→ Conv 96→192
→ PixelShuffle
→ 48×128×128
```

```text
48×128×128
→ Conv 48→96
→ PixelShuffle
→ 24×256×256
```

---

## 8. Skip连接

Dc 内部保存：

\[
E_1\in\mathbb R^{B\times24\times256\times256}
\]

\[
E_2\in\mathbb R^{B\times48\times128\times128}
\]

\[
E_3\in\mathbb R^{B\times96\times64\times64}
\]

解码阶段执行：

```text
上采样特征
→ concat对应Encoder特征
→ Conv1×1压缩通道
→ Restormer Block
```

三个 skip 只属于 Dc 内部，不与 Ec 建立跨网络连接。

---

## 9. 为什么潜层是32×32

三次下采样：

\[
256\rightarrow128\rightarrow64\rightarrow32
\]

固定 DCT 块网格也是：

\[
256/8=32
\]

因此潜层的一个位置大致对应原图中的一个8×8 DCT块。

这使网络可以在潜层统一建模：

- 颜色载荷的块间关系；
- 大范围颜色一致性；
- 结构与色度之间的联系；
- 不同图像区域之间的颜色传播。

但 Dc 不直接读取39通道 DCT 系数，保持第一版结构简洁。

---

## 10. 色度输出头

解码和 Refinement 后：

\[
F_{\mathrm{out}}
\in
\mathbb R^{B\times24\times256\times256}
\]

色度头：

```text
Fout
→ Conv3×3：24→2
→ split
→ Cb_hat、Cr_hat
```

公式：

\[
[\hat C_b,\hat C_r]
=
H_c(F_{\mathrm{out}})
\]

不使用：

- Sigmoid；
- Tanh；
- ReLU；
- Clamp。

因为 Cb/Cr 为零中心值，需要允许正负输出。

---

## 11. 亮度修复头

亮度头：

```text
Fout
→ Conv3×3：24→1
→ raw_luma_delta
→ 固定Pcw投影
→ luma_delta
```

公式：

\[
R_y=H_y(F_{\mathrm{out}})
\]

\[
\Delta\hat Y=P_{cw}(R_y)
\]

最终亮度：

\[
\hat Y=Z_0+Z_c+\Delta\hat Y
\]

因为：

\[
Z_0+Z_c=Y-P_w(Y)+\Delta_c
\]

所以网络需要学习：

\[
\Delta\hat Y\approx P_w(Y)-\Delta_c
\]

其中：

- 在 \(P_w\) 中恢复原始亮度；
- 在 \(P_c\) 中消除 Ec 的颜色残差；
- 不修改 \(P_0\)。

因此严格保证：

\[
P_0(\hat Y)=P_0(X')
\]

---

## 12. 固定 RGB 重建

\[
\hat R=\hat Y+1.402\hat C_r
\]

\[
\hat B=\hat Y+1.772\hat C_b
\]

\[
\hat G
=
\hat Y
-
\frac{0.114\times1.772}{0.587}\hat C_b
-
\frac{0.299\times1.402}{0.587}\hat C_r
\]

最终：

\[
\hat{RGB}
=
\operatorname{Concat}(\hat R,\hat G,\hat B)
\]

网络内部不执行 RGB clamp。

---

## 13. 初始化

| 模块 | 初始化 |
|---|---|
| 双输入 Stem | Xavier uniform，bias=0 |
| 24→24融合卷积 | Xavier uniform，bias=0 |
| 上下采样卷积 | Xavier uniform，bias=0 |
| Skip后的1×1卷积 | Xavier uniform，bias=0 |
| MDTA/GDFN Conv | Xavier uniform，bias=0 |
| Chroma Head | Xavier uniform，bias=0 |
| Luma Head | Xavier uniform，bias=0 |
| LayerNorm weight | 1 |
| LayerNorm bias | 0 |
| Attention temperature | 1 |

输出头不使用零初始化，保持原 Dc 的初始化逻辑，并保证训练开始时亮度和色度路径都能向共享主干传播梯度。

---

## 14. 完整逐层表

| 阶段 | 层 | 输出尺寸 |
|---|---|---|
| 输入 | \(X'\) | `B×1×256×256` |
| 固定变换 | Block DCT | `B×64×32×32` |
| 结构频带 | \(Z_0=P_0(X')\) | `B×1×256×256` |
| 颜色频带 | \(Z_c=P_c(X')\) | `B×1×256×256` |
| 结构Stem | Conv3×3, 1→8 | `B×8×256×256` |
| 颜色Stem | Conv3×3, 1→16 | `B×16×256×256` |
| 分支拼接 | concat | `B×24×256×256` |
| 浅层融合 | Conv3×3, 24→24 | `B×24×256×256` |
| Encoder L1 | TB×1, heads=1 | `B×24×256×256` |
| Down 1 | Conv 24→12 + Unshuffle | `B×48×128×128` |
| Encoder L2 | TB×1, heads=2 | `B×48×128×128` |
| Down 2 | Conv 48→24 + Unshuffle | `B×96×64×64` |
| Encoder L3 | TB×2, heads=4 | `B×96×64×64` |
| Down 3 | Conv 96→48 + Unshuffle | `B×192×32×32` |
| Latent | TB×3, heads=8 | `B×192×32×32` |
| Up 3 | Conv 192→384 + Shuffle | `B×96×64×64` |
| Skip 3 | concat E3 + Conv 192→96 | `B×96×64×64` |
| Decoder L3 | TB×2, heads=4 | `B×96×64×64` |
| Up 2 | Conv 96→192 + Shuffle | `B×48×128×128` |
| Skip 2 | concat E2 + Conv 96→48 | `B×48×128×128` |
| Decoder L2 | TB×1, heads=2 | `B×48×128×128` |
| Up 1 | Conv 48→96 + Shuffle | `B×24×256×256` |
| Skip 1 | concat E1 + Conv 48→24 | `B×24×256×256` |
| Decoder L1 | TB×1, heads=1 | `B×24×256×256` |
| Refinement | TB×1, heads=1 | `B×24×256×256` |
| 色度头 | Conv3×3, 24→2 | `B×2×256×256` |
| 亮度头 | Conv3×3, 24→1 | `B×1×256×256` |
| 亮度约束 | \(P_{cw}\) | `B×1×256×256` |
| 亮度输出 | \(Z_0+Z_c+\Delta\hat Y\) | `B×1×256×256` |
| 固定逆变换 | YCbCr→RGB | `B×3×256×256` |

---

## 15. 最终数学表达

\[
D=\operatorname{DCT}_8(X')
\]

\[
Z_0=\operatorname{IDCT}_8(M_0\odot D)
\]

\[
Z_c=\operatorname{IDCT}_8(M_c\odot D)
\]

\[
F_0=\operatorname{Conv}_{1\rightarrow8}(Z_0)
\]

\[
F_c=\operatorname{Conv}_{1\rightarrow16}(Z_c)
\]

\[
F_{\mathrm{in}}
=
\operatorname{Conv}_{24\rightarrow24}
\left(
\operatorname{Concat}(F_0,F_c)
\right)
\]

\[
F_{\mathrm{out}}
=
\operatorname{RestormerUNet}(F_{\mathrm{in}})
\]

\[
[\hat C_b,\hat C_r]=H_c(F_{\mathrm{out}})
\]

\[
\Delta\hat Y=P_{cw}(H_y(F_{\mathrm{out}}))
\]

\[
\hat Y=Z_0+Z_c+\Delta\hat Y
\]

\[
\hat{RGB}
=
\operatorname{YCbCrToRGB}
(\hat Y,\hat C_b,\hat C_r)
\]

最终结构可以概括为：

```text
P0(X') → 8通道结构特征 ──┐
                          ├→ 24通道融合
Pc(X') → 16通道颜色特征 ─┘
                          │
                          ▼
                多尺度Restormer U-Net
                          │
                ┌─────────┴─────────┐
                │                   │
             Cb/Cr Head         Luma Head
                │                   │
          Cb_hat,Cr_hat     Pcw修正 + Z0 + Zc
                │                   │
                └─────────┬─────────┘
                          ▼
                      YCbCr→RGB
```

这版 Dc v2 保留了原系统的固定频带可解释性，同时解决原 Dc 单尺度、局部感受野和颜色载荷不突出的问题，没有引入额外39通道分支或复杂注意力模块。




2. Dw v2 完整数据流
输入 X'
[B,1,256,256]
        │
        ▼
固定 8×8 Block DCT
        │
        ▼
D = DCT8(X')
[B,64,32,32]
        │
        ▼
按照 Ew 相同顺序
Gather 9个 Pw 系数
        │
        ▼
W：[B,9,32,32]
        │
        ▼
Conv1×1：9→32
        │
       GELU
        │
        ▼
Conv3×3：32→64
        │
        ▼
F0：[B,64,32,32]
        │
        ▼
Restormer Block ×4
channels=64
heads=4
Bias-free LayerNorm
GDFN expansion=2.66
        │
        ▼
Fr：[B,64,32,32]
        │
        ├───────────────┐
        │               │
        ▼               │
与 F0 做 long residual ◄┘
        │
        ▼
Fc：[B,64,32,32]
        │
        ▼
Conv3×3：64→32
        │
       GELU
        │
        ▼
Conv3×3：32→1
        │
        ▼
恢复消息模式 Q
[B,1,32,32]
        │
        ▼
Flatten
        │
        ▼
q：[B,1024]
        │
        ▼
Linear：1024→256
        │
       GELU
        │
        ▼
Linear：256→64
        │
        ▼
logits：[B,64]
        │
        ▼
logits ≥ 0
        │
        ▼
64-bit消息预测
3. 固定 DCT 输入
输入：
\[
X'\in\mathbb R^{B\times1\times256\times256}
\]固定 Block DCT：
\[
D=\operatorname{DCT}_8(X')
\]\[
D\in\mathbb R^{B\times64\times32\times32}
\]提取顺序固定为：
\[
(0,3),(1,2),(2,1),(3,0),
\]\[
(0,4),(1,3),(2,2),(3,1),(4,0)
\]得到：
\[
W=\operatorname{Gather}_w(D)
\]\[
W\in\mathbb R^{B\times9\times32\times32}
\]该顺序必须与 Ew Coefficient Head 的9个输出通道完全相同。
固定 DCT：
- 没有训练参数；
- 不 detach；
- 保留对 \(X'\) 的梯度；
- 允许消息损失反向更新 Ew。
4. 系数输入编码器
Ew 的 Coefficient Head 是：
64→32→9
Dw 使用反向层次：
9→32→64
具体结构：
W：[B,9,32,32]
→ Conv1×1, 9→32
→ GELU
→ Conv3×3, 32→64
→ F0：[B,64,32,32]
公式：
\[
F_1=
\operatorname{GELU}
\left(
\operatorname{Conv}_{1\times1}^{9\rightarrow32}(W)
\right)
\]\[
F_0=
\operatorname{Conv}_{3\times3}^{32\rightarrow64}(F_1)
\]其中：
- 1×1 Conv 负责混合9个频点；
- 3×3 Conv 负责整合相邻 DCT 块；
- 第二层后不加激活，直接进入 Pre-Norm Restormer。
5. Restormer 主体
主体配置：
参数	设置
分辨率	32×32
通道数	64
Block数	4
Heads	4
每个head通道	16
LayerNorm	Bias-free
GDFN expansion	2.66
下采样	无
上采样	无
Dropout	无
DropPath	无


定义：
\[
F_r=
\mathcal R_4
\left(
\mathcal R_3
\left(
\mathcal R_2
\left(
\mathcal R_1(F_0)
\right)
\right)
\right)
\]主体 long residual：
\[
F_c=F_0+F_r
\]\[
F_c\in\mathbb R^{B\times64\times32\times32}
\]作用包括：
- 联合不同 DCT 块中的弱消息证据；
- 学习抑制宿主 \(P_w(Y)\)；
- 建立9个频率之间的关系；
- 恢复全局消息编码结构；
- 适应 RMSCap 带来的尺度变化。
6. Restormer Block
每个 Block 使用：
\[
X_1=X+\operatorname{MDTA}(\operatorname{LN}(X))
\]\[
X_2=X_1+\operatorname{GDFN}(\operatorname{LN}(X_1))
\]MDTA
输入 X
→ Bias-free LayerNorm
→ Conv1×1：64→192
→ DWConv3×3，groups=192
→ 拆分 Q、K、V
→ reshape：[B,4,16,1024]
→ Q、K沿空间维L2 normalize
→ QKᵀ × learnable temperature
→ Softmax
→ Attention × V
→ 恢复BCHW
→ Conv1×1：64→64
→ Residual Add
GDFN
输入
→ Bias-free LayerNorm
→ Conv1×1：64→340
→ DWConv3×3，groups=340
→ 拆分为两个170通道分支
→ GELU(A) × B
→ Conv1×1：170→64
→ Residual Add
其中：
\[
hidden=\lfloor2.66\times64\rfloor=170
\]7. 消息模式恢复头
Ew 的 Message Stem 是：
1→32→64
Dw 使用对应的反向结构：
64→32→1
具体为：
Fc：[B,64,32,32]
→ Conv3×3, 64→32
→ GELU
→ Conv3×3, 32→1
→ Q：[B,1,32,32]
公式：
\[
Q=
\operatorname{Conv}_{3\times3}^{32\rightarrow1}
\left(
\operatorname{GELU}
\left(
\operatorname{Conv}_{3\times3}^{64\rightarrow32}(F_c)
\right)
\right)
\]最后一层不使用激活函数，使恢复模式可以同时包含正值和负值。
8. 消息逆映射 MLP
将 \(Q\) 展平：
\[
q=\operatorname{Flatten}(Q)
\]\[
q\in\mathbb R^{B\times1024}
\]第一层：
\[
h=
\operatorname{GELU}
\left(
W_1q+b_1
\right)
\]\[
h\in\mathbb R^{B\times256}
\]第二层：
\[
\ell=W_2h+b_2
\]\[
\ell\in\mathbb R^{B\times64}
\]对应结构：
1×32×32
→ Flatten 1024
→ Linear 1024→256
→ GELU
→ Linear 256→64
→ raw logits
最终不使用 sigmoid。
9. 初始化方案
模块	初始化
Conv1×1, 9→32	Xavier uniform
Conv3×3, 32→64	Xavier uniform
Restormer卷积	Xavier uniform
Pattern Conv 64→32	Xavier uniform
Pattern Conv 32→1	Xavier uniform
Linear 1024→256	Xavier uniform
Linear 256→64	Xavier uniform
所有bias	0
LayerNorm weight	1
Attention temperature	1


所有输出层采用非零初始化，保证消息损失从第一步就能传到：
- Message MLP；
- Pattern Head；
- Restormer 主体；
- 系数输入编码器；
- Ew。
10. 完整逐层表
阶段	层	输出尺寸
输入	\(X'\)	B×1×256×256
固定变换	Block DCT	B×64×32×32
频率选择	Gather 9个 \(P_w\) 系数	B×9×32×32
系数编码	Conv1×1, 9→32 + GELU	B×32×32×32
系数编码	Conv3×3, 32→64	B×64×32×32
主体	Restormer Block×4	B×64×32×32
主体旁路	Long residual	B×64×32×32
Pattern Head	Conv3×3, 64→32 + GELU	B×32×32×32
Pattern Head	Conv3×3, 32→1	B×1×32×32
展平	Flatten	B×1024
Message MLP	Linear 1024→256 + GELU	B×256
Logit Head	Linear 256→64	B×64


11. 第一版明确不加入的部分
第一版不加入：
- \(P_0\) 宿主上下文分支；
- 全64个DCT系数输入；
- 下采样或 U-Net；
- bit-query Cross-attention；
- 显式位置编码；
- FFT模块；
- Mamba模块；
- 纠错码；
- Sigmoid输出；
- 中间 Pattern Loss。
这些可以作为后续独立消融，不能与 Dw v2 主干同时加入，否则难以判断收益来源。
12. 适用边界
该结构固定使用：
\[
256\times256
\rightarrow
32\times32
\rightarrow
1024
\]因此当前版本只支持固定256×256输入。
此外，它默认 DCT 块网格保持对齐。裁剪、旋转、缩放和透视变换会改变块位置，不能仅靠当前 Dw 结构解决。未来进入几何攻击阶段时，需要单独设计同步或几何校正模块。
13. 最终定稿
完整数学表达：
\[
D=\operatorname{DCT}_8(X')
\]\[
W=\operatorname{Gather}_w(D)
\]\[
F_0=
\operatorname{Conv}_{3\times3}^{32\rightarrow64}
\left(
\operatorname{GELU}
\left(
\operatorname{Conv}_{1\times1}^{9\rightarrow32}(W)
\right)
\right)
\]\[
F_c=
F_0+
\mathcal R_4
\left(
\mathcal R_3
\left(
\mathcal R_2
\left(
\mathcal R_1(F_0)
\right)
\right)
\right)
\]\[
Q=
\operatorname{Conv}_{3\times3}^{32\rightarrow1}
\left(
\operatorname{GELU}
\left(
\operatorname{Conv}_{3\times3}^{64\rightarrow32}(F_c)
\right)
\right)
\]\[
\ell=
W_2
\operatorname{GELU}
\left(
W_1\operatorname{Flatten}(Q)+b_1
\right)
+b_2
\]最终结构：
\[
\boxed{
X'
\rightarrow
\operatorname{DCT}_8
\rightarrow
9
\rightarrow32
\rightarrow64
\rightarrow\mathrm{TB}\times4
\rightarrow32
\rightarrow1
\rightarrow1024
\rightarrow256
\rightarrow64
}
\]预计参数量约为 0.5M。网络主体全部运行在32×32分辨率，显存与计算量相对 Ec/Dc 很小，同时与 Ew v2 的消息编码路径具有明确、完整的层级对应关系。
