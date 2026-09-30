# Dual-Payload Watermark

以原 Ec / Ew / Dc / Dw 和 DCT、Restormer 主干为基础，增加医疗灰度共享首版代码：

- Ec 颜色残差 → 39 频率系数 → 4 bit 打包 → AES-256-GCM。
- 独立加密的 16 B PatientToken；两路使用独立业务密钥。
- Sionna 2.1.0 的 5G NR LDPC 1024→1536、固定交织、236／2 通道布局。
- Ew 以亮度和两路密文比特生成载体；保存真实单通道 8 bit PNG。
- Ed25519 医院整图签名置于一个 `medical_auth_v1` iTXt 项。
- 接收端先验签，再执行 Dw、纠错和独立解密；Dc 使用实际灰度与解密残差恢复 RGB。

**已接通协议、医疗训练与真实文件评测工具，并用合成数据检查短程训练和断点恢复。未启动医学主训练或 H100 检查，尚无医学载荷可靠提取率或 RGB 质量结论。**

## 安装与检查

需要 Python 3.11+。在仓库目录执行：

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[test,medical]'
.venv/bin/python -m pytest tests/medical -q
.venv/bin/python -m pytest -q
.venv/bin/python -m dual_payload.medical.cli --help
```

检查包含真实 LDPC 和加密、PNG 文件往返、认证与权限失败路径、四网络前向及梯度、独立接收进程。
权限成功路径通过测试专用理想 logits 隔离验证协议；它不代表未训练的 Dw 已能提取载荷。

## 配置和收发

[实现约定与命令](docs/medical_v1.md)记录了二进制格式、Profile 登记、输入预处理和收发用法。
[正式配置模板](configs/medical_v1.template.json)的 39 个量化步长、三种幅度和权重摘要保持空值；缺失时程序拒绝登记和运行。
测试配置仅位于 `tests/medical/conftest.py`，需显式允许 test profile，不是正式实验参数。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| `dual_payload/medical/models.py` | 新四网络接口，复用原 Ec 和 Restormer |
| `dual_payload/medical/protocol.py` | 字节、比特、量化、打包 |
| `dual_payload/medical/crypto.py` | 两路认证加密、持久 nonce 登记 |
| `dual_payload/medical/profile.py` | 公共配置、交织文件及不可重绑定登记 |
| `dual_payload/medical/ldpc.py` | 真实 LDPC、填充和空间布局 |
| `dual_payload/medical/png.py` | 严格 PNG 解析与医院签名 |
| `dual_payload/medical/pipeline.py` | `Sender.send` 与 `Receiver.receive` |
| `dual_payload/medical/cli.py` | 独立发送与接收命令 |
| `dual_payload/medical/data.py` | PAD 患者分组清单与 Derm7pt 外部数据 |
| `dual_payload/medical/calibration.py` | 仅训练集残差的 39 频率步长标定 |
| `dual_payload/medical/training.py` | 医疗训练、验证、梯度累积与断点恢复 |
| `dual_payload/medical/evaluation.py` | 冻结导出与独立进程真实文件评测 |
| `dual_payload/medical/experiment.py` | 主实验工具命令 |

## 医学主实验

PAD-UFES-20 用于主训练及内部验证/测试，Derm7pt 用于外部评测，临床和皮肤镜图像分别报告。
[运行说明](docs/medical_experiment.md)提供数据清单、权重核验、校准、H100 短程检查、训练、冻结与评测命令。
[训练模板](configs/medical_train.template.json)与最终 Profile 独立；[评价模板](configs/medical_evaluation.template.json)要求测试前填写门槛。
工具不自动下载数据，也不填入未经核验的训练权重、步长和幅度。

单卡 H100 80 GB 的主实验参数见 `configs/medical_h100_80gb_joint.json`：四网络从第一步一起训练，关闭数据增强。
`python -m dual_payload.medical.main_experiment --help` 提供主实验启动入口，接入实际数据和原 V2 checkpoint 后完成一次标定并启动联合训练，支持断点恢复。

## 原 V2 基线

`dual_payload/models.py`、`system.py`、`train.py`、`evaluate.py` 和 `diagnose_watermark.py` 保留原 64 bit V2 基线接口。
`configs/v2_clean_baseline_h100.json` 属于原 DIV2K 基线，不能用于新医疗协议。
医疗模型使用独立的 `architecture=medical-v1` 标识，严格加载对应权重，使用 `dual_payload.medical.experiment` 的独立训练入口。
