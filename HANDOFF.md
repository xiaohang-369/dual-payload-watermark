# 医疗灰度共享代码交接

更新：2026-10-01。工作目录位于桌面 `dual-payload-watermark`，分支为 `V3`。

## 当前主实验约定

- PAD-UFES-20 主训练和内部验证/测试；按患者合并组 70/15/15，seed 2026。Derm7pt 保留作外部评测。
- 单卡 H100 80 GB，固定 256×256 等比缩放和补边；关闭数据增强。
- 源图读取接受 RGB 和 alpha 全为 255 的 RGBA；后者去掉 alpha 保留 RGB 数值。有 RGB ICC 时按嵌入 profile 转为 sRGB，无 ICC 时按 sRGB 解释。原始文件不改写，Profile/manifest 记录新版预处理规则；真实透明及无效 ICC 仍拒绝。
- Ec/Ew/Dc/Dw 从第一步同时训练。主实验从头随机初始化，不依赖已有基线权重，不使用 A/B/C 分阶段训练。
- Ec 残差头和 Ew 两路嵌入头使用标准差 `1e-4` 的小幅随机初始化；Dc 亮度修正头零初始化；其他卷积使用 Xavier 初始化。
- 初始化显式使用同一个 seed，标定与训练重建完全相同的初始四网络，不受外部 RNG 状态影响。
- 仅训练分区参与 39 频率步长初始化：绝对系数 0.999 分位数除以 7，保存 FP32 步长并全程固定。随机初始残差仅用于设置数值尺度，不代表已经学会颜色编码。
- 从第一步使用真实 4 bit 残差取整和 8 bit 灰度取整，反向使用 STE。AES/LDPC 不参与求导；Dc 的 RGB 监督使用已知量化残差，灰度输入截断梯度。
- 有效 batch 16、micro-batch 4、FP32、AdamW；统一基础学习率 `1e-4`、200 次预热、余弦降到 `1e-6`；上限 120 轮或 20,000 次更新。损失及幅度见配置文件，这些仍是待医学实验验证的起始参数。

## 启动与续跑

主入口：`python -m dual_payload.medical.main_experiment`。
必填 `--config`、`--pad-root`、`--pad-metadata`、`--output`；没有外部权重参数。完整启动命令见 [运行说明](docs/medical_experiment.md)。
主配置：`configs/medical_h100_80gb_joint.json`，由 `--config` 显式传入，是训练超参数的唯一来源；已删除 Python `h100_config()` 重复预设。`initialization.kind=scratch`，path/SHA-256 保持 null。seed 统一从 JSON 读取，micro-batch 按实际 batch 大小校验，只有显式 CLI 选项覆盖文件值。
本次 `--output /data/zwc/zyh/experiments/v3clean-main-run01`；程序自动创建目录，在根目录生成 manifest/calibration/train-joint JSON，实际训练输出为其下 `joint`。完整生效配置参与续训一致性校验，不得覆盖或删除旧结果来绕过检查。

程序创建清单、初始步长和配置，随后联合训练；中断后使用同一命令及输出目录恢复。
训练前保存 step 0 检查点，支持首次更新/验证前中断后的恢复；训练后保存 `last.pt`、`best.pt` 和完成记录。
续跑核验配置、代码、清单及标定记录；不能把旧初始化方案的实验目录当成本次从头训练目录。
通用工具保留同结构医疗 checkpoint 的显式加载和本次实验的断点恢复；已移除旧基线迁移函数及默认配置。
历史 64 bit 基线脚本和专用模块已删除；保留并整理 V3 使用的公共网络计算组件。
数据、密钥、权重和输出均放在源码目录之外。

## 收发与验收

实现位于 `dual_payload/medical/`，协议和模型接口仍使用 `medical-v1` 标识；V3 是项目分支，不改动线上协议版本。
已连接 39 频率 4 bit 打包、两路 AES-256-GCM、Sionna 2.1.0 LDPC 1024→1536、固定交织及 236/2 通道载荷。
Ew 输入亮度与两路载荷；Dc 输入实际灰度和解密后的反量化残差。
真实 gray8 PNG 保存重读，Ed25519 签名，接收先验签及独立授权。实现约定见 [medical_v1.md](docs/medical_v1.md)。
周期性网络验证记录 loss/BER/量化误差，按验证总损失选择 checkpoint。
完整业务成功率由独立接收进程的真实 PNG 文件评测确认；冻结 Profile 和评价门槛在正式测试前完成。

## 检查与证据边界

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m dual_payload.medical.main_experiment --help
```

本轮在删除前用 AST 检查所有医疗模块、公共模块与测试的导入关系，并检查子进程和命令入口引用。固定 seed 下，清理前后四网络全部参数摘要、前向输出与 PSNR/SSIM 输出摘要一致。

2026-10-01 清理后全量测试：`82 passed`（72.22 秒），无警告。旧基线专用测试随功能删除，公共组件测试保留并整理。所有 22 个包子模块和全部测试的本地导入可解析；三个模块入口帮助、安装后的 `medical-share` 以及 `git diff --check` 均通过。

2026-10-01 配置入口修复后全量测试：`91 passed`（68.26 秒）；主入口 `--help` 和 `git diff --check` 通过。新增回归验证 JSON 的 epochs/lr/batch_size/loss_weights 进入实际配置及合成训练 checkpoint，seed 贯通清单和标定，显式 CLI 覆盖、自动建目录、非法配置拒绝及同目录参数变化不改写旧文件。仅运行合成程序测试，未启动医学主训练。

2026-10-01 源图格式适配后全量测试：`102 passed`（68.84 秒）。新增 11 项测试覆盖不透明 RGBA 与 RGB 像素一致、EXIF 方向、真实 ICC 颜色变换、无效 ICC/真实透明/多帧拒绝、manifest 与 Dataset 一致读取及原文件摘要保留。服务器检查输出报告 2298 张源图、1440 张不透明 RGBA、2 张 RGB 模式的 Google Skia ICC 图片；本地未读取服务器原图，全部源图能否通过新版预处理仍需服务器只读检查。未启动医学主训练。

本机 `.venv` 的 editable 注册文件被系统标记为 hidden，导致仓库外无法导入；已使用 `uv pip install --python .venv/bin/python --no-deps --no-build-isolation --offline .` 安装当前源码构建的普通包，未更改依赖版本。仓库外已验证可导入全部模块，安装包不含删除的旧模块，安装的全部 23 个 Python 文件与已测试源码逐字节一致。该普通安装不会自动跟随后续源码变化，更新源码后需重新安装；服务器按 README 在自身环境安装。

合成测试覆盖量化/打包、密码/LDPC、签名和失败路径、网络前向梯度、无外部权重的联合训练、初始化一致性、首次更新前中断后的恢复、续跑和独立接收。
医学数据训练和 H100 显存/吞吐尚未实测；程序测试通过不代表载荷可靠提取或医学恢复质量达标。
本次服务器实验根目录已指定为 `/data/zwc/zyh/experiments/v3clean-main-run01`；本轮只修正配置入口和运行说明，不连接服务器、不上传 GitHub、不下载数据、不启动医学主训练。

## 历史文件清理清单

以下只涉及源码、配置和测试；没有删除数据集、权重或实验结果。

| 删除文件 | 理由 |
| --- | --- |
| `train.py`、`evaluate.py`、`diagnose_watermark.py` | 历史基线训练、评估和诊断入口，已由医疗模块入口替代 |
| `configs/v2_clean_baseline_h100.json` | 旧 DIV2K、64 bit 训练配置 |
| `dual_payload/training.py`、`dual_payload/diagnostics.py`、`dual_payload/system.py` | 仅服务旧基线；不在 V3 导入链中 |
| `dual_payload/config.py`、`dual_payload/data.py`、`dual_payload/channel.py`、`dual_payload/losses.py` | 旧配置、裁剪加载器、模拟信道和单消息损失；V3 使用医疗模块自己的实现 |
| `tests/test_training.py`、`tests/test_diagnostics.py`、`tests/test_channel.py` | 仅验证已删除的基线功能 |
| `tests/test_data_metrics.py` | 移除旧加载器/配置/64 bit 指标检查；公共指标检查移到 `tests/test_metrics.py` |

| 保留或整理文件 | 理由 |
| --- | --- |
| `dual_payload/models.py` | 保留 Restormer、归一化、卷积和初始化函数；颜色残差主干明确命名为 `ColorResidualEncoder`；删除旧 Ew/Dw/Dc 专用类 |
| `dual_payload/transforms.py` | 医疗网络、量化、协议与收发共同依赖 DCT、颜色转换及 RMS 限制 |
| `dual_payload/metrics.py` | 医疗评测依赖 PSNR/SSIM；删除旧系统专用 `compute_metrics` |
| `dual_payload/medical/`、`dual_payload/__init__.py` | V3 主训练、标定、续跑、收发、认证和评测的完整导入链 |
| 四份 `configs/medical*.json` | 当前训练、Profile 和评价模板 |
| `tests/medical/`、`tests/conftest.py`、`tests/test_transforms.py` | 保留医疗流程与公共变换测试；补回独立频带可分离性检查 |
| `tests/test_models.py`、`tests/test_metrics.py` | 整理为公共组件、医疗权重严格往返、网络参数独立性和图像指标检查 |
| `README.md`、本文件、`docs/medical_experiment.md`、`docs/medical_v1.md` | 仍是有效文档；清理旧入口说明，保留当前运行与协议约定 |
| `pyproject.toml` | 保留包发现、依赖和有效 `medical-share` 入口 |

仓库没有单独的过时基线文档文件；过时说明位于 README 和本文件，已就地修正。代码摘要随清理变化；已有实验产物原样保留，同目录续训的代码一致性检查仍会拒绝混用清理前后的源码。
