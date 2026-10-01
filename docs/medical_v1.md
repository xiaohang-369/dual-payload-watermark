# 医疗首版实现约定

本文记录代码已经采用的约定和运行方法。功能依据为用户给定的修正统一方案。
4 bit、16 B Token、LDPC 1024→1536、236／2 通道、双输入 Dc 和 PNG 医院整图签名均已进入代码。

## 1. 版本与字节格式

- 网络版本：`medical-v1`，复用原 Restormer 的 24／48／96／192 主干。
- 业务协议版本：1。支路标识：颜色 1，患者 2。
- 所有多字节整数为大端；每个字节按最高位优先转成比特。
- 颜色系数：按块行、块列、`8u+v` 递增排列；每字节先高后低两个 4 bit 补码数。
- 患者频率保持原顺序：`[3,10,17,24,4,11,18,25,32]`。

颜色内部头使用 `>4sHHHHIII`：`MCLR`、版本 1、量化 ID、高 256、宽 256、系数数目 39936、有效位数 159744、零保留字段。24 B 头加 19968 B 系数，共 19992 B。

公开业务头使用 `>BBH16sI8s`：版本、支路、传输 Profile ID、16 B 随机图像 ID、加密包长度、8 B 零保留字段。
加密包为 `nonce(12 B) || ciphertext || tag(16 B)`。

AAD 为以下原始字节的拼接：

```text
b"medical-share/aead/v1\x00" || branch_label || public_header
branch_label = b"COLOR\x00" 或 b"PATIENT\x00"
```

AES-256-GCM 使用两把独立的 32 B 密钥。`NonceStore` 先用操作系统随机源生成 nonce，再由 SQLite 唯一约束持久预留，之后才加密；进程退出不会撤销已用 nonce。
同一密钥的所有发送者必须共用该登记文件。不得在保留原密钥时重置、回滚登记文件或改用另一份登记文件。
登记文件只保存密钥 SHA-256 和 nonce。密钥不进入 Profile、PNG 或状态日志。

## 2. LDPC、填充与布局

实际使用 Sionna 2.1.0 配套 `LDPC5GEncoder` / `LDPC5GDecoder`：BG2、提升因子 104、RV0、无 QAM 交织和 HARQ。
解码为 `boxplus-phi`、flooding、20 次迭代、幅值 20、正 logits 表示 bit 1；不预先硬判决。
API 和符号依据：[编码器](https://nvlabs.github.io/sionna/v2.1.0/phy/api/fec/ldpc/sionna.phy.fec.ldpc.LDPC5GEncoder.html)、[解码器](https://nvlabs.github.io/sionna/v2.1.0/phy/api/fec/ldpc/sionna.phy.fec.ldpc.LDPC5GDecoder.html)。

| 项目 | 颜色 | 患者 |
| --- | ---: | ---: |
| 业务帧字节数 | 20052 | 76 |
| 纠错前零填充字节数 | 44 | 52 |
| LDPC 块数 | 157 | 1 |
| 码字比特数 | 241152 | 1536 |
| 布局零填充比特数 | 512 | 512 |
| 网络布局 | 236×32×32 | 2×32×32 |

交织定义 `out[i] = codeword[permutation[i]]`。先交织、末尾补 512 个零，再按通道、行、列 reshape。
解码先去掉布局填充，再逆排列，最后做 LDPC；解码后的数据填充必须为零。
两条固定排列以 PCG64 和登记的 seed 首次生成，正逆排列都保存为大端 uint32 原始文件。读取时校验摘要、完整排列和逆映射；接收端不重生成。

## 3. 公共 Profile

从 `configs/medical_v1.template.json` 填写独立工作副本。以下字段必须显式填写：

- `profile_id`、`quantization_id`：1–65535。
- `quantization_steps`：39 个正且适用于 FP32 的校准步长。
- `rms_limits.color_residual`：原 Ec 残差限制。
- `rms_limits.color_ciphertext`、`rms_limits.patient_ciphertext`：两路嵌入限制。
- `weights.ec/ew/dc/dw`：各权重文件的 SHA-256。

登记命令：

```sh
.venv/bin/python -m dual_payload.medical.cli register-profile \
  --config /absolute/public/medical-config.json \
  --registry /absolute/public/profiles
```

结果目录为 `<registry>/<profile_id>/`，包含 `profile.json`、`profile.sha256` 和四份排列文件。
JSON 规范化使用 ASCII、排序键、无空白分隔符，拒绝重复键和非有限数值。
登记器拒绝同一 ID 绑定另一配置；加载时检查摘要，签名还绑定整个 Profile 摘要。
变更量化、权重或任何配置必须使用新的 Profile ID，并向接收者可信分发整份目录。
不要手工改写已登记目录；接收端只读取调用者指定的可信配置和模型，不按 PNG 字段选择路径或下载代码。

正式模板的未定字段为 `null`，不能直接运行。程序测试使用 `tests/medical/conftest.py` 中 `purpose=test` 的配置，并显式传 `allow_test=True` 或 `--allow-test-profile`。
该测试配置的步长和幅度仅用于程序检查。

## 4. 权重

每个权重文件为以下字典；读取使用 `torch.load(..., weights_only=True)`：

```python
torch.save({
    "architecture": "medical-v1",
    "component": "ec",  # 或 ew、dc、dw
    "state_dict": model.state_dict(),
}, "/absolute/models/ec.pt")
```

加载会先校验同一份文件字节的 SHA-256，再严格检查版本、组件和所有张量键及形状。
主实验四网络从头联合训练，不加载已有基线权重。Ec 残差头和 Ew 嵌入头为小幅随机初始化，Dc 亮度修正头为零初始化。初始残差用于设定量化尺度，不代表有效颜色编码。训练入口及续跑方式见 [medical_experiment.md](medical_experiment.md)。

## 5. 工作图和 PNG

输入约定为单帧 RGB8 sRGB，无 ICC 和透明。先应用 EXIF 方向，再把长边缩放到 256；短边按最近值半值取偶确定，至少为 1。使用 Pillow bicubic，居中 edge 补边保留完整视野。
补边参与嵌入，保持 32×32 个 DCT 块。有效内容 `(x,y,w,h)` 写入受签名保护的认证头。
恢复对象为这个工作图；评价应同时报告内容区域和整张工作图的结果。

载体与最终 RGB 都采用 `roundEven(255 * clip(x,0,1))`。
接收端保留 `rgb_float.npy` 供越界与误差统计，另输出 `rgb.png`。
`ste_gray8` 提供相同整数前向的训练辅助；真实收发始终写入并重读 PNG。

接收只允许 PNG 签名、一个首部 IHDR、连续 IDAT、至多一个认证 iTXt 和结尾 IEND；CRC 必须正确，文件最多 1 MiB。
IHDR 固定为 256×256、8 bit、灰度、非隔行。拒绝尾随数据、动画、透明、调色板、EXIF、ICC、gamma、其他文字或显示解释字段，以及重复认证项。
像素解码和验签使用一次读取的同一字节快照；验签通过后由同一像素数组直接得到 `G8/255`。

## 6. 医院签名

认证二进制头为 `>BBBBHHH32s32s32s16s16s4H`，共 146 B，依次为：

1. 认证版本 1、Ed25519 算法号 1、gray8 像素格式号 1、零保留字节。
2. 宽 256、高 256、传输 Profile ID。
3. Profile SHA-256、传输模型 ID、颜色编解码模型 ID，各 32 B。
4. 图像 ID 16 B、医院 key ID 16 B。
5. 有效内容矩形的 x、y、宽、高，各 2 B。

传输模型 ID 为 `SHA256(ew_digest || dw_digest)`，颜色模型 ID 为 `SHA256(ec_digest || dc_digest)`；使用二进制摘要拼接。
医院 key ID 为 Ed25519 原始 32 B 公钥的 SHA-256 前 16 B，只用于查询调用者提供的可信公钥。

签名对象为 `b"medical-share/sign/v1\x00" || auth_header || G8_row_major`。
Ed25519 签名固定 64 B，内容采用标准带 padding、无空白 Base64 编码 `auth_header || signature`，写入 `medical_auth_v1` iTXt。
iTXt 压缩标志和方法均为 0，语言和译名为空。认证块增加 312 B，发送报告单独记录这一开销和 PNG 总大小。
PNG 无秘密密钥、明文患者数据或具体医生身份。

## 7. 收发命令

路径示例均需替换为已有的可信材料；密钥为原始 32 B 文件，Token 为原始 16 B 文件，医院公私钥为 Ed25519 PEM。

```sh
.venv/bin/python -m dual_payload.medical.cli send \
  --profile /absolute/public/profiles/1 \
  --ec /absolute/models/ec.pt --ew /absolute/models/ew.pt \
  --input /absolute/images/source.png --output /absolute/results/shared.png \
  --color-key /absolute/keys/color.key --patient-key /absolute/keys/patient.key \
  --token /absolute/private/patient.token \
  --hospital-private-key /absolute/keys/hospital-private.pem \
  --nonce-store /absolute/private/nonces.sqlite

.venv/bin/python -m dual_payload.medical.cli receive \
  --profile /absolute/public/profiles/1 \
  --dw /absolute/models/dw.pt --dc /absolute/models/dc.pt \
  --input /absolute/results/shared.png --output /absolute/results/received \
  --hospital-public-key /absolute/trust/hospital-public.pem \
  --color-key /absolute/keys/color.key --patient-key /absolute/keys/patient.key
```

接收可以省略任一路密钥；两路均无密钥时只验签，不加载网络。仅患者权限不需要 Dc。
接收目录和发送 PNG 必须不存在，防止覆盖已有结果。
CLI 返回 0 表示认证通过且所有请求分支成功；认证或请求分支失败返回 2。
部署配置、依赖或模型错误明确报错。

| 整图状态 | 业务行为 |
| --- | --- |
| `AUTH_MISSING` | 缺认证项，两路 `AUTH_BLOCKED` |
| `TRUST_KEY_MISSING` | 无可信医院公钥，两路停止 |
| `AUTH_FAILED` | 验签失败，两路停止 |
| `AUTH_FORMAT_ERROR` | 不支持的格式、冲突或配置不符，两路停止 |
| `AUTHENTIC` | 处理拥有密钥的分支 |

分支状态为 `KEY_MISSING`、`DECODE_FAILED`、`DECRYPT_FAILED`、`OK`。解密失败不返回部分明文。
成功颜色路保存浮点 RGB 和 8 bit RGB，成功患者路保存 16 B `patient.token`。`status.json` 记录状态和有效区域，不记录 Token。

## 8. 当前检查的边界

- 协议成功路径使用实际 AES-GCM、实际 Sionna LDPC 和理想 logits，验证精确往返、权限组合与失败隔离。
- 网络检查覆盖真实四网络形状、频带限制、双输入 Dc、原主干复用与梯度。
- 独立进程检查让真实发送端输出 PNG 后，删除发送端专用文件，从新工作目录启动接收 CLI；接收进程只有 PNG、公共配置、Dw/Dc 和授权材料。
- 独立进程使用未训练网络，业务包应失败；它证明代码和依赖可以执行，未证明可靠提取。

尚未核验原服务器权重、未启动医疗训练、未确定最终步长与幅度、未测业务成功率与医学恢复质量。
这些结论必须由后续真实文件主实验给出。

数据清单、标定、医疗训练和正式评测工具现已补齐，运行方法见 [medical_experiment.md](medical_experiment.md)。合成数据短程训练不计作医学实验结果。
