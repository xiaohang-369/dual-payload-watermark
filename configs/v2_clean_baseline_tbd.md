# Network V2 clean baseline：TBD 实验设计模板

## 状态

`v2_clean_baseline_tbd.json` 只固定已经确认的 Network V2 参数。它不是正式训练配置，目前不得用于启动正式训练。

当前配置加载器会先载入 `DEFAULT_CONFIG`，再用 JSON 中出现的字段覆盖默认值。因此，JSON 中未出现的字段仍会在运行时取得代码默认值。那些默认值只是解析器回退值，不代表本模板已经确认它们。

## 已固定参数

- seed：`2026`
- architecture version：`v2`
- delta_c：`2/255`
- delta_w：`2/255`
- eps：`1e-8`
- image size：`256`
- channel：float、无量化、无 clamp、identity、无攻击增强
- loss weights：RGB `1.0`、chroma `0.0`、luma `0.0`、message `1.0`、carrier `1.0`、range `0.1`
- learning rate：`1e-4`
- weight decay：`0.0`
- gradient clipping：`1.0`
- message bits：源码固定为 `64`，不是普通训练配置字段
- optimizer：源码固定为 `torch.optim.Adam`，不是配置字段
- 当前目标中不使用 VGG、LPIPS、GAN 或 adversarial loss

这里的 `float` 指当前 `quantization_mode=none` 的浮点 clean channel。当前实现仍是 FP32；是否为 H100 引入 BF16、AMP 或 TF32 尚未决定。

## TBD：路径

- `data.train_dir`
- `data.val_dir`
- `train.output_dir`

JSON 中这三个路径暂时写为 `null`。其中 `train_dir=null` 和 `val_dir=null` 会阻止不带真实数据路径的正式训练；`output_dir=null` 在现有训练代码中表示运行时自动创建输出目录，因此它不能被理解为最终输出路径已经确定。

## TBD：数据规模

- train image count
- validation image count
- 是否使用完整 DIV2K
- 是否使用固定图片子集

这些不是当前普通训练配置的合法字段，只记录在本文档中。

## TBD：消息与 experiment 组织

当前尚未确定：

- 是否固定 message bank
- 是否每个 batch 随机生成 message
- 是否采用固定图像乘固定消息的笛卡尔积
- `experiment.mode`
- `experiment.asset_manifest`
- `experiment.image_manifest`
- `experiment.message_bank`
- `experiment.message_metadata`
- `experiment.image_count`
- `experiment.message_count`
- `experiment.pair_count`
- `experiment.crop_policy`

只有 `experiment.message_bits=64` 已确定。

当前解析器允许顶层 `experiment` 为 `null`，但这会使训练走普通数据集路径，并在训练阶段随机生成消息；它不能表示“消息组织方式仍为 TBD”。因此模板 JSON 不写 `experiment` 对象，正式训练前必须根据最终协议补齐。

一旦 `experiment` 是对象，当前解析器要求下面全部字段同时存在：

```json
{
  "mode": "<non-empty string>",
  "asset_manifest": "<non-empty string>",
  "image_manifest": "<non-empty string>",
  "message_bank": "<non-empty string>",
  "message_metadata": "<non-empty string>",
  "image_count": "<positive integer>",
  "message_count": "<positive integer>",
  "message_bits": 64,
  "pair_count": "<positive integer>",
  "crop_policy": "<supported non-empty string>"
}
```

上述示例仅说明类型，带尖括号的值不能复制到训练 JSON。当前解析器还要求 `pair_count = image_count * message_count`，并且现有实现只接受 `crop_policy=fixed_center_crop_256`。如果正式实验决定使用其他 crop policy，需要先单独评审实现；本模板不猜测该决定。

## TBD：batch 与显存

- `data.batch_size`：logical/effective batch size
- `gradient_accumulation_steps`：一个 logical batch 内的 micro-batch 切分数
- 实际 micro-batch size
- `data.num_workers`

`gradient_accumulation_steps` 不是 JSON 配置字段，而是训练 CLI 参数和 checkpoint 元数据。不要把它加入 JSON。

派生关系为：

```text
micro-batch size approximately equals logical batch size / gradient_accumulation_steps
```

实际边界使用 `training.py` 中的整数切分规则。梯度不会跨 DataLoader batch 累积，每个 DataLoader iteration 执行一次 `optimizer.step()`。

## TBD：训练预算

- `train.epochs`
- `train.max_steps`
- total optimizer-step budget
- `train.log_every`

`train.max_steps=null` 在当前代码中表示“不设置 step 上限”，不是通用 TBD 标记，因此模板 JSON 不显式写入它。

## TBD：运行设备与 H100 精度策略

- `device`
- BF16
- AMP
- TF32

当前配置解析器没有 BF16、AMP 或 TF32 字段。当前训练实现不启用 AMP/BF16，并明确关闭 TF32。这些选择只能在实验方案确认后另行处理，不能通过本模板中的未知字段表达。

## 当前解析器中的 null 和未知字段规则

可以通过配置校验的 `null`：

- `data.train_dir`
- `data.val_dir`
- `train.output_dir`
- `train.max_steps`
- 顶层 `experiment`

不能写 `null`：

- `seed`
- `device`
- `model` 内字段
- `channel` 内字段
- `loss` 内字段
- `data.image_size`
- `data.batch_size`
- `data.num_workers`
- `train.epochs`
- `train.lr`
- `train.weight_decay`
- `train.grad_clip`
- `train.log_every`
- 非空 `experiment` 对象中的任何必需字段

解析器不允许额外说明字段，因此不能在 JSON 中加入 `_tbd`、`notes`、`micro_batch_size`、`message_bits` 或 `optimizer` 等未知字段。所有这类信息只记录在本文档中。

## 正式训练前必须完成

至少必须确认并写入或通过 CLI 明确传入：

1. train/validation 数据路径和数据规模协议；
2. 普通随机消息协议或完整的 `experiment` 固定消息协议；
3. logical/effective `batch_size`；
4. `gradient_accumulation_steps` 及由此得到的实际 micro-batch；
5. `num_workers`；
6. `epochs`、`max_steps` 和总 optimizer-step budget；
7. 输出目录；
8. H100 上的 FP32/BF16、AMP 和 TF32 策略；
9. 运行设备和必要的日志频率。

不得从旧 Windows 实验继承 batch 8、accumulation 8、200 epochs、10 images、20 messages 或 200 pairs。
