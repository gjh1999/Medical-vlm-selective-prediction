# 医学 VLM 层级分类与拒识

已选研究方向：有限标注和长尾条件下，在具体病种、上位类别和完全拒识之间作可靠决策。

当前状态：2026-11-17 已执行 E00 环境与官方数据源预检，下载 HAM v4.0 元数据并生成三套病灶级划分；层级参考基线和指标核心通过 8 项边界测试。尚未下载原图、完成近重复审计或执行模型推理。

实测数据：10,015 图像 / 7,470 病灶；calibration 1,869 病灶，其中 df 18、nv 1,351。当前划分没有跨病灶组交叉；尚不能声称患者级独立。完整证据见[实验记录](实验记录.md)。

正式 GPU 为家中的 RTX 5080，详见 [5080 运行说明](5080运行说明.md)。本机不安装 GPU 环境。

- [研究计划](研究计划.md)：研究问题、最近工作、数据/树/划分、基线、候选方法、统计边界、实验清单和八周里程碑。
- [本地网页计划](研究计划.html)：与研究计划内容一致，可离线浏览、打印 PDF。
- [协议与执行状态](protocol.json)：v1.1 更新实测状态和目标主机；[v1.0](protocol-v1.0.json) 保留原始设计。科学评价协议未改变，目前不是训练配置。
- [实验记录模板](实验记录模板.md)：首次实测后建立本目录的实验记录.md，并记录所有尝试。
- [后续工作约定](AGENTS.md)：新会话先从这里恢复上下文。

下一步顺序：最近方法全文复核 → 原站数据获取/许可/去重与分组 → 类别树核对 → 冻结特征 → 基线与新增信号验证 → 第 2 周决定是否继续方法开发。

用户后续要求每 30 分钟同步 GitHub，覆盖此前的 Git 限制。目标为 `https://github.com/gjh1999/Medical-vlm-selective-prediction.git`。同步脚本位于根目录 `scripts/`；注册脚本必须先成功推送再启用任务。实际状态以任务 `MCCAI-Research-GitHub-Sync` 及根目录 `.sync-state/last-success.json` 为准；需要本机开机且注册用户已登录。原图、权重、样本 manifest/split 与 outputs 默认不入库。

网页由 `python build_plan.py` 从 Markdown 生成，不依赖外部库。以下命令可在当前主机或 5080 主机执行：

```powershell
python preflight.py --network --output outputs/new-preflight-01
python prepare_ham.py --output outputs/new-ham-01
python -m unittest -v test_core.py
```

各 output 必须使用不存在的独立路径。`--download-images` 才会下载约 2.77GB 原图 ZIP。PAD 预检接口返回 404，入口需修正，尚未开展该数据的分组审计。参考核心不是完整 B0–B6 复现；正式特征提取、温度校准、margin/entropy、提示分歧和外部验证仍待实施。
