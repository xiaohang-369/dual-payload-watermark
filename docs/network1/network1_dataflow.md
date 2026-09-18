# Network1：RGB 与 64-bit message 的完整数据流

尺寸以固定实验 B=1、256×256 为例；一般 B 保持不变。源码入口：`train.py:1-5` → `dual_payload/training.py:468-471,529-543` → `dual_payload/system.py:30-49`。下表的来源和去向均由当前源码确认。

```text
RGB [B,3,256,256] ──rgb_to_ycbcr──→ y,cb,cr 各 [B,1,256,256]
                                    │
                                    └─concat(y,cb,cr)─→ Ec CNN─→ candidate_c [B,1,256,256]
                                                          └─Pc─→RMSCap(δc)─→ Δc
                         y ──────────────────────────────────────────── + ─→ s（颜色 carrier）
                                                                      │
message [B,64]─→2m-1─→Linear─ReLU─Linear─→ embedding [B,64]─→broadcast map [B,64,256,256]
                                                                      │
s ─→ s-Pc(s)=bw ─→ image stem ────────────────────────────────concat─→ Ew fusion/body/head
                                                                        └─Pw─→RMSCap(δw)─→ Δw
s ───────────────────────────────────────────────────────────────────────────────────── + ─→ x_float
                                                                                              │
                                                     TransmissionChannel: none/no clamp/identity
                                                                                              │
                                                                                attacked_image [B,1,256,256]
                                                                                       /                 \
                                      Dc: z=x-Pw(x), zc=Pc(x) ─→ CNN ─→ cb,cr,raw_y       Dw: DCT w(9 maps)
                                               y_hat=z+Pcw(raw_y) ─→ fixed YCbCr→RGB      └─CNN─→64 maps
                                                          rgb_hat [B,3,256,256]              └─global mean
                                                                                               logits [B,64]
```

| 节点/实际 tensor 名 | shape | 来源和操作 | 去向/证据 |
|---|---|---|---|
| `rgb` | `[B,3,256,256]` | 输入，float32、有限、[0,1] | `rgb_to_ycbcr`；`system.py:30-35` |
| `message` | `[B,64]` | 输入，二值 | Ew、BCE；`models.py:72-79`, `losses.py:21` |
| `y`, `cb`, `cr` | 各 `[B,1,256,256]` | `y=.299r+.587g+.114b`；`cb=(b-y)/1.772`；`cr=(r-y)/1.402`，无 0.5 偏置；`transforms.py:14-20` | Ec；原始目标 y/chroma；`system.py:35-38` |
| Ec CNN 输入 | `[B,3,256,256]` | `cat(y,cb,cr)` | Ec stem；`models.py:53` |
| `color["candidate"]` | `[B,1,256,256]` | Ec head 原始空间图 | `Pc`；`models.py:53-55` |
| `color["residual"]`, `delta_c` | `[B,1,256,256]` | `RMSCap(Pc(candidate),δc)` | 与 y 相加；`models.py:54-55`, `system.py:45` |
| `color["carrier"]`, `s` | `[B,1,256,256]` | `y+delta_c`，颜色已嵌入的单通道浮点图 | Ew 输入；`models.py:55`, `system.py:36-37,42` |
| `bw` | `[B,1,256,256]` | `s-Pc(s)`，Ew 图像分支输入 | image_stem；`models.py:77,80` |
| `embedding` | `[B,64]` | `Linear(64,128)→ReLU→Linear(128,64)` 处理 `2message-1` | 广播；`models.py:64,78` |
| `message_map` | `[B,64,256,256]` | embedding 增添空间维并 `expand`，所有位置共享同一 64 维码 | 与 image_stem(bw) 拼接；`models.py:79-80` |
| Ew 拼接/融合 | `[B,128,256,256]` → `[B,64,256,256]` | 64 图像特征 + 64 message map → 3×3 Conv/ReLU | Ew 8 blocks/head；`models.py:65-68,80-81` |
| `water["candidate"]` | `[B,1,256,256]` | Ew head 原始空间图 | `Pw`；`models.py:81-83` |
| `water["residual"]`, `delta_w` | `[B,1,256,256]` | `RMSCap(Pw(candidate),δw)` | 与 s 相加；`models.py:82-83`, `system.py:45` |
| `water["carrier"]`, `x_float` | `[B,1,256,256]` | `s+delta_w=y+delta_c+delta_w` | 通道、carrier/range loss；`system.py:37-43`, `losses.py:16,22-23` |
| `x_quantized` | `[B,1,256,256]` | 正式配置 `quantization_mode=none`, `clamp_enabled=false`，数值上即 `x_float` | identity attack；`channel.py:35-52` |
| `attacked_image` | `[B,1,256,256]` | identity attack 返回 `x_quantized` | **同一个** tensor 分给 Dc 与 Dw；`system.py:23-28,39-44` |
| Dc `z`, `zc` | 各 `[B,1,256,256]` | `z=x-Pw(x)`，`zc=Pc(x)` | 拼成 `[B,2,256,256]`；`models.py:98-100` |
| Dc `features` | `[B,64,256,256]` | 2→64 stem + 8 residual blocks | 两个 head；`models.py:90-93,100` |
| `cb_hat`, `cr_hat` | 各 `[B,1,256,256]` | chroma_head 2 通道分裂 | 固定逆变换、chroma loss；`models.py:101,105` |
| `raw_luma_delta`, `luma_delta` | 各 `[B,1,256,256]` | luma_head → `Pcw(raw)` | `y_hat=z+luma_delta`；`models.py:102-106` |
| `y_hat` | `[B,1,256,256]` | z 与亮度修正相加 | RGB 逆变换、luma loss；`models.py:104-105` |
| `rgb_hat` | `[B,3,256,256]` | `ycbcr_to_rgb(y_hat,cb_hat,cr_hat)`，无 clamp | RGB loss；`transforms.py:23-28`, `losses.py:18` |
| Dw DCT 前端 | `[B,9,32,32]` | 从 x 各 8×8 块取 9 个 w-band DCT 系数 | Dw stem；`transforms.py:85-90`, `models.py:120` |
| Dw `evidence` | `[B,64,32,32]` | stem + 8 residual blocks + 1×1 head | 空间均值；`models.py:113-121` |
| `logits` | `[B,64]` | 每 bit 的 32×32 evidence 均值，raw logits | BCE-with-logits；`models.py:120-121`, `losses.py:21` |

数学式（`P_c/P_w/P_cw` 为同一固定正交 8×8 DCT 的不同频带投影；`Cap` 为每图 RMS 限制）：

```text
(y,cb,cr) = T(rgb)
u_c = Ec_CNN(cat(y,cb,cr))
Δc = Cap(P_c(u_c), 2/255); s = y + Δc
bw = s - P_c(s)
e = MLP(2m-1); M = broadcast(e,H,W)
u_w = Ew_CNN(cat(image_stem(bw),M))
Δw = Cap(P_w(u_w), 2/255); x_float = s + Δw
x = Channel(x_float)                 # 当前 none/no-clamp/identity
z = x - P_w(x); zc = P_c(x)
(cb_hat,cr_hat), raw = Dc_CNN(cat(z,zc))
y_hat = z + P_cw(raw); rgb_hat = T_inverse(y_hat,cb_hat,cr_hat)
logits = spatial_mean(Dw_CNN(DCT_w_9(x)))
```

【根据源码推导】颜色信息通过 Ec 的单通道 c-band 扰动 `Δc` 被加到原始亮度 `y` 上；RGB 不作为三通道载体发送。水印信息经 MLP 广播和 Ew CNN 变成 w-band 扰动 `Δw`，再加到颜色 carrier `s`。两频带正交，由固定 mask 强制分开，并非两个单独传输的 carrier。`bw` 理想情况下移除 `Δc`；Dc 的 `z` 理想情况下移除 `Δw`。边界、浮点运算及预算缩放以实际实现为准。

## 真实 B=1 前向 shape trace

用 `.venv/Scripts/python.exe`、CPU、float32、`model.eval()`、`torch.inference_mode()`、随机合法 RGB/二值 message 与 forward hooks 实测一次，无训练/反向传播。全部记录节点的 dtype 为 `torch.float32`、device=`cpu`、`torch.isfinite` 为 true（无 NaN/Inf）。这是未加载训练 checkpoint 的结构检查，不是保真度评估。

| hook/输出 | 实测 shape |
|---|---|
| input RGB / message | `[1,3,256,256]` / `[1,64]` |
| Ec stem / body / head | `[1,64,256,256]` / `[1,64,256,256]` / `[1,1,256,256]` |
| Ec DCT coefficients / candidate / residual / s | `[1,64,32,32]` / `[1,1,256,256]` / `[1,1,256,256]` / `[1,1,256,256]` |
| Ew message_branch / image_stem / fusion / body / head | `[1,64]` / `[1,64,256,256]` / `[1,64,256,256]` / `[1,64,256,256]` / `[1,1,256,256]` |
| Ew candidate / residual / x_float | 各 `[1,1,256,256]` |
| x_quantized / attacked_image | 各 `[1,1,256,256]` |
| Dc stem / body / chroma_head / luma_head | `[1,64,256,256]` / `[1,64,256,256]` / `[1,2,256,256]` / `[1,1,256,256]` |
| Dc z / zc / y_hat / cb_hat / cr_hat / rgb_hat | `[1,1,256,256]` / `[1,1,256,256]` / `[1,1,256,256]` / `[1,1,256,256]` / `[1,1,256,256]` / `[1,3,256,256]` |
| Dw DCT w / stem / body / head / logits | `[1,9,32,32]` / `[1,64,32,32]` / `[1,64,32,32]` / `[1,64,32,32]` / `[1,64]` |
