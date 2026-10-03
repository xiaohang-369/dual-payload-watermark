# Medical V3 continuation

1. 完整阅读 `医疗图像双载荷分权限可逆灰度共享方案_Protocol_v1.md`。它是算法与协议唯一权威；代码/README/旧 specs 不得覆盖它。
2. 当前主线固定 256-bit，无 legacy 配置、部分加载或 64-bit 新运行入口。`architecture_version=v3` 是工程 schema，四网络主体仍为 v2clean。
3. 当前入口见 README。`specs/001-message-bits-stages` 与 `specs/002-medical-protocol-v1` 保留历史记录，旧训练流程已被当前决定取代。
4. 协议入口 `ProtocolV1` 从完整 V3 检查点精确文件字节绑定 ModelID；先验签，独立 KC/KM 恢复。不要改动冻结的候选生成、编号、选择、字节格式和签名摘要规则。
5. 正式训练只有 `joint_256`，四网络从头初始化、全部进入同一 optimizer，一次总损失 backward；不加载已有权重，不开启密钥排列或密码协议。`--init-from` 已删除。四项核心 loss（rgb/message/carrier/range）权重必须有限且 >0。训练后冻结模型，只在 `protocol_eval` 启用完整协议。
6. 模型工程验证仍只使用合成数据，没有连接服务器或启动真实数据训练。用户现已授权本地真实 PAD-UFES-20 数据准备；数据准备与合成测试均不构成真实解码质量、机密性或医学质量证据。
7. active configs 只有 `medical_joint_256.json` 和 `medical_protocol_eval.json`。beta/min_moved 保留 null，协议评估未填写就拒绝。数据仍只收已经处理好的 RGB 256×256 工作图，不增加预处理。
8. 用户已冻结 PAD-UFES-20 preprocessing v1 与 patient split，独立工具和验收说明见 `tools/README.md`、`specs/003-pad-prepared-v1/`。seed=2026，患者配额 961/206/206，lesion_key=(patient_id, lesion_id)；禁止在正式 `data.py` 内重复预处理。augmentation、正式训练超参数、beta/min_moved 和医学质量门槛仍由用户另行确定。当前不迁移旧权重，不实现导入工具；精确 resume 与 scheduler 均未实现，本次不扩展。正式训练和保护效果实验尚未开展。
