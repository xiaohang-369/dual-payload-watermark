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
| `dual_payload/models.py` | 公共 Restormer、归一化、颜色残差编码组件 |
| `dual_payload/transforms.py` | 公共 DCT、颜色转换和 RMS 幅度限制 |
| `dual_payload/metrics.py` | 公共 PSNR、SSIM 图像指标 |
| `dual_payload/medical/models.py` | 医疗四网络接口 |
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
| `dual_payload/medical/main_experiment.py` | H100 从零联合训练和自动续跑入口 |

## 医学主实验

PAD-UFES-20 用于主训练及内部验证/测试，Derm7pt 用于外部评测，临床和皮肤镜图像分别报告。
[运行说明](docs/medical_experiment.md)提供数据清单、随机初始化、步长标定、联合训练、冻结与评测命令。
[训练模板](configs/medical_train.template.json)与最终 Profile 独立；[评价模板](configs/medical_evaluation.template.json)要求测试前填写门槛。
工具不自动下载数据；主实验从头训练，步长由训练集初始化。

单卡 H100 80 GB 的主实验参数见 `configs/medical_h100_80gb_joint.json`：四网络从第一步一起训练，关闭数据增强。
主入口必须传入 `--config`，指定 JSON 是训练超参数的唯一来源；数据清单、量化标定和输出路径自动生成，支持一致性校验后的断点恢复。无需外部 checkpoint。在服务器仓库根目录启动本次实验：

```sh
python -u -m dual_payload.medical.main_experiment \
  --config configs/medical_h100_80gb_joint.json \
  --pad-root /data/zwc/zyh/data/PAD-UFES-20 \
  --pad-metadata /data/zwc/zyh/data/PAD-UFES-20/metadata.csv \
  --output /data/zwc/zyh/experiments/v3clean-main-run01
```

程序自动创建所需目录；`manifest.json`、`calibration.json`、`train-joint.json` 保存在实验根目录，实际配置的 `output` 为其下的 `joint`。只有显式传入的 micro-batch/公共 ID 选项覆盖 JSON。同目录续训必须通过配置、代码和输入记录一致性检查，不能覆盖或删除旧结果来绕过。

## 当前入口与版本

训练使用 `python -m dual_payload.medical.main_experiment`；标定、冻结和评测使用 `python -m dual_payload.medical.experiment`；文件收发使用 `python -m dual_payload.medical.cli`（或安装后的 `medical-share`）。
医疗模型和文件协议仍使用 `architecture=medical-v1` 标识，V3 为项目分支名。历史基线入口与配置已移除，公共网络、变换和图像指标保留。清理范围见 [交接说明](HANDOFF.md)。
