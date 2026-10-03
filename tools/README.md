# PAD-UFES-20 原始数据审计

独立脚本仅依赖 Python 标准库与 Pillow，不导入 `dual_payload`，不执行训练或正式预处理。

```bash
.venv/bin/python tools/audit_pad_ufes20.py \
  --metadata /absolute/path/to/metadata.csv \
  --image-dir /absolute/path/to/images \
  --output-dir /absolute/path/to/new-audit
```

图片目录递归扫描；输出目录必须在图片目录之外，不得包含 metadata，且必须不存在或为空。已有报告不会被覆盖。所有输入仅以读取方式打开。

## 输出

- `audit_summary.json`：完整统计、计数口径和全部异常明细。
- `image_properties.csv`：每个实际图片文件一行，包含未引用图片；损坏图片保留行，无法读取的属性为空。附加 format、status、metadata_record_numbers、error、lesion_keys 及八个 RGBA alpha 属性字段。
- `patient_summary.csv`：每位患者的唯一图像数、唯一病灶数与诊断集合。
- `lesion_summary.csv`：每个 `(patient_id, lesion_id)` 一行，保留原始两个 ID，并添加 JSON 数组形式的 lesion_key；同一 key 的多诊断保留全部值。
- `diagnostic_summary.csv`：六个预期类别、原样保留的未知类别，以及存在缺失时的空诊断行。
- `audit_report.md`：全部关键统计、alpha 汇总与透明像素占比前 20 张、完整几何异常列表；其他每类异常显示前 20 条，全部明细在 JSON 中。

metadata 只接受精确字段名 `patient_id`、`lesion_id`、`img_id`、`diagnostic`，允许其他字段。缺字段、重复表头或记录列数错误时，停止关联与统计，仅写 JSON 和 Markdown 的结构错误报告，列出真实表头，不猜映射。UTF-8 BOM 可以读取；不自动猜测编码或分隔符。

## 计数与对应规则

- 保留 ID 的前导零、大小写和非空白字符，不自动修剪空格。仅空值/纯空白计为缺失，`NA`、`NULL` 等字符串不自动转为空值。
- metadata 行数与唯一 img_id 数分开。患者/病灶/诊断表的 image_count 是唯一非空 img_id 数，含缺文件记录。缺失分组 ID 不形成虚构患者/病灶。
- diagnostic 的 image_percent 分母是所有唯一非空 img_id。若同一图像有冲突诊断，会在多个类别中出现；报告冲突，比例可能超过 100%。病灶/患者同样可能出现在多个诊断类别中。
- 原始 lesion_id 保留不变；逻辑病灶标识为 `(patient_id, lesion_id)`，仅两个 ID 均非空时构造。lesion_summary、images_per_lesion、所有 lesion_count 均按这个二元组统计，不从 img_id 猜测或补齐。
- `raw_unique_lesion_ids`、`unique_patient_lesion_pairs`、`reused_lesion_ids_across_patients` 分开计数。跨患者复用 raw lesion_id 属于标识语义现象，不作为错误，明细在 observations 中；这不修改原始 metadata。
- 同一 lesion_key 内多个非空诊断才列为 `lesion_diagnostic_conflicts`。缺失诊断单独报告，患者多诊断仍为组成事实。
- 用 `img_id` 与磁盘文件名精确对应，不自动加 `.png`，不做大小写匹配或路径截取。同名文件全部列为歧义；大小写相近文件仅作候选提示。
- 对全部可解码图片逐文件统计，即使图片未被引用或格式/扩展名异常。用 PNG 扩展名、PNG 签名、metadata 引用或 Pillow 识别结果确定图片清单；其他无法识别文件另列，不伪装为可读图片。
- 先 `Pillow.verify()`，重新打开后完整 `load()`；不转换模式、不按 EXIF 旋转、不改像素或原文件。Pillow 的读取限制和警告保留在报告，不自动关闭保护。
- 不跟随目录符号链接，明确列出被跳过的目录并将状态标为 incomplete；文件符号链接只读并单列。无法遍历目录时审计失败，不声称完成。
- 图片统计不因重复 metadata 膨胀。CSV 中单个关联值为原字符串，多个值为 JSON 数组，空集合为空单元格；JSON 报告中的集合始终为数组。
- lesion_key 用 JSON 数组编码，避免拼接字符串产生冲突；image_properties 的 lesion_keys 为二元组列表，保持原始配对关系。
- 分位数使用 `(n-1)*p` 的线性插值；空总体的统计量为 null，比例为 0。percent 字段单位为百分比。

## RGBA alpha 与几何审计

直接读取 RGBA 的 A 通道直方图，不转换为 RGB、不合成背景。逐图输出 `alpha_min`、`alpha_max`、`alpha_unique_count`、`non_opaque_pixel_count`（alpha<255）、`fully_transparent_pixel_count`（alpha=0）、`partially_transparent_pixel_count`（0<alpha<255）、`total_pixel_count` 和 `non_opaque_ratio`（0..1）。非 RGBA 图片的这些字段为空。

`rgba_alpha` 汇总完全不透明、有任何透明像素、有部分透明值、有 alpha=0 的图片数；`alpha_min_range`/`alpha_max_range` 分别为逐图最小/最大值的范围。只对存在透明像素的图片按 non_opaque_ratio 降序列出前 20 张。无 RGBA 时范围为空，数量为 0。

`geometry` 完整列出 `aspect_ratio_outside_0_9_to_1_1`（不含 0.9 和 1.1 边界）及 `short_side_lt256`（不含 256 边界）。JSON 与 Markdown 均保存全部列表，不选择预处理策略。

状态与进程退出码：`complete_clean -> 0`；`complete_with_findings -> 1`（不等于数据不可用）；invalid/schema/incomplete 及未知状态 `-> 2`。有透明像素或几何异常时作为 findings，单纯跨患者复用 lesion_id 不触发 findings。使用 subprocess 验证真正的进程退出码。运行期间应保持输入数据稳定。

## 合成测试

```bash
.venv/bin/python -m pytest -q tests/test_audit_pad_ufes20.py
```

测试只在临时目录创建合成图片，并检查输入文件哈希保持不变。本工具不创建 train/val/test，不选择或执行 resize、crop、padding、augmentation、normalization，不保存处理后的图片，不修改模型、Protocol、configs 或正式 `data.py`。

## 正式数据准备 v1（独立入口）

用户已冻结 preprocessing v1 与 patient split；使用 `prepare_pad_ufes20.py` 实际生成，前述 audit 脚本继续只读。

```bash
.venv/bin/python tools/prepare_pad_ufes20.py \
  --dataset-root /Users/zhuyanghang/Desktop/PAD-UFES-20 \
  --metadata /Users/zhuyanghang/Desktop/PAD-UFES-20/raw/metadata.csv \
  --image-dir /Users/zhuyanghang/Desktop/PAD-UFES-20/raw/images/all \
  --output-dir /Users/zhuyanghang/Desktop/PAD-UFES-20/prepared_v1

.venv/bin/python -m pytest -q tests/test_prepare_pad_ufes20.py tests/test_audit_pad_ufes20.py
```

- 固定 seed=2026；正式入口要求 2298 images、1373 patients，train/val/test 患者数严格 961/206/206。不导入训练模块。
- RGB 保留通道；RGBA 在逐像素确认 alpha 全为 255 后去 A。原尺寸 256×256 不 resize；两边均<256 用 BICUBIC，其余 LANCZOS。输出 8-bit RGB PNG，保留 img_id。
- 不自动旋转/颜色变换/线性化/裁剪/补边/增强/归一化。源 EXIF/ICC 不复制到输出文件。was_upsampled 表示任一维需要放大，与选择哪个滤波器分别记录。
- 20 维患者向量包含总图片/病灶、六类图片/病灶/患者 presence。所有病灶按 (patient_id, lesion_id) 计数。以等权的相对目标平方误差优化，稀有类别 greedy 后做固定 seed 患者交换，最多 30 轮或无改善即停止；不宣称全局最优。
- 每组保持六类存在，MEL 患者数在理论目标 floor/ceil 范围内。每次实际运行还用逆序输入重复计算并核验完全相同的分组与优化记录。
- 所有源图先完成 alpha/mode/格式检查；在临时目录生成和验收后发布。输出目录已存在即拒绝覆盖，失败清理本次临时输出。
- 输出 `images/`、`manifests/{train,val,test}.json`、`reports/{split_summary.json,split_report.md,source_integrity.json}`、`preprocessing_manifest.csv`。manifest 的 `../images/img_id` 相对 manifest 文件，可整体搬迁，兼容现有 `ManifestDataset`。
- 验证所有原始 dataset 文件的 SHA-256 与文件集合不变（含 metadata、压缩包和旧报告；仅排除本次生成目录），输出文件逐张读回并校验 SHA-256、RGB 和 256×256、三组完整覆盖及患者隔离。

准备脚本成功 exit 0；输入/验收/I/O 失败 exit 2。运行准备命令不启动训练，也不设置 batch/lr/epoch。
