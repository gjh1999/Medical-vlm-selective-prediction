# 交互分割的提示鲁棒性与可靠停止

2026-11-17 新建的并行探索路线，与 `../medical-vlm-hierarchy/` 并行保留。

核心问题：用户点击有偏移、遗漏或标签错误时，能否在减少交互次数的同时，限制“分割未达标却自动停止”的比例？先冻结模型研究停止决策，暂不训练新分割网络。

当前判断：值得做小规模验证，但“提示扰动 + 不确定性”和“交互早停”都有直接前作，不能作为独立创新点。候选切口是**有噪声交互下、以完整轨迹为单位校准的选择性停止**；新颖性仍需全文复核和真实模型实验支持。

- [研究计划](研究计划.md)：问题定义、数据、对照、指标、两周推进与终止条件。
- [文献差异表](文献差异表.md)：直接竞争工作、核验范围及来源。
- [实验记录](实验记录.md)：本轮统计可行性计算与实际执行状态。
- [机器可读协议](protocol.json)：v0.2，补充固定源码版本、输入契约及策略选择顺序；[v0.1](protocol-v0.1.json) 保留。
- [模型与数据审计](模型与数据审计.md)：已确认 logits 历史输入、灰度/坐标契约及 MedScribble 实际文件数量。
- `stopping.py`：完整轨迹停止回放、一般失败数的二项上界、有限策略族校准；不包含分割模型。
- `sample_size.py`：纯 Python 的零失败样本量计算，不需要 GPU 或第三方依赖。

从仓库根目录运行（每次换一个不存在的输出路径）：

```powershell
python research/interactive-segmentation-reliability/sample_size.py --output research/interactive-segmentation-reliability/outputs/20261117/E00-sample-size-01
```

2026-11-18：源码审计固定 revision `182c44975f77749b559974ce8db558c8bde57788`；11 项停止器测试通过，并完成一次明确标为合成数据的轨迹回放。结果验证了首次越阈值、转人工和候选族校准路径，也显示构造的分布变化会破坏停止可靠性，不能作为真实方法性能证据。详见实验记录与 `reports/` 汇总。

本机只做检索、协议和轻量统计准备；GPU 推理仍计划放在家中 RTX 5080。尚未下载此方向的数据或权重、未运行分割模型，真实 Dice、显存、延迟和节省点击数均未测量。Kvasir/arXiv 本地获取出现证书验证失败；沙箱外重试未获批准，未绕过校验。

本地软件验证与合成回放（从仓库根执行）：

```powershell
python -m unittest discover -s research/interactive-segmentation-reliability -p test_stopping.py -v
python research/interactive-segmentation-reliability/synthetic_replay.py --output research/interactive-segmentation-reliability/outputs/20261118/E00-synthetic-replay-02
```

第二条命令是未来复现示例，`-02` 尚未执行。已执行结果位于 `E00-synthetic-replay-01`。所有尝试使用新输出目录。`source_audit.py --output 新目录` 用于允许联网时归档官方来源，任何失败写入 audit.json；不会安装 GPU 依赖或下载模型。

下一步：审计 ScribblePrompt 的训练来源和 Kvasir 数据身份/近重复，固定代码与权重哈希，在开发集先跑 50 张、1/3/5 次点击的配对轨迹。数据和模型审计通过前不发布泛化或风险保证结论。原图、权重和逐样本输出沿用根目录 `.gitignore`，只提交代码、协议和汇总记录。
