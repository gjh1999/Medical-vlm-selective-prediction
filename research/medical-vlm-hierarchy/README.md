# 医学 VLM 层级分类与拒识

已选研究方向：有限标注和长尾条件下，在具体病种、上位类别和完全拒识之间作可靠决策。

当前状态：2026-11-17 已完成 HAM v4.0 与 PAD v1 官方元数据下载和校验，分别生成三套病灶级/保守患者连通组划分。参考核心、数据审计及特征输入校验合计 13 项测试通过。尚未下载原图或执行模型推理；5080 特征提取代码未在目标环境验证。

HAM：10,015 图像 / 7,470 病灶；calibration 1,869 病灶，其中 df 18、nv 1,351，尚不能声称患者级独立。PAD：2,298 图像 / 1,373 患者；250 个 raw lesion_id 跨患者重用，其中 226 个存在标签差异。当前将共享 raw ID 的患者保守连接成 1,132 个 split_group，三套划分均无患者/raw ID/连通组交叉。1,891 个 patient+lesion 处理键不能当成已核实的真实病灶数。完整证据见[实验记录](实验记录.md)。

正式 GPU 为家中的 RTX 5080，详见 [5080 运行说明](5080运行说明.md)。本机不安装 GPU 环境。

- [研究计划](研究计划.md)：研究问题、最近工作、数据/树/划分、基线、候选方法、统计边界、实验清单和八周里程碑。
- [本地网页计划](研究计划.html)：与研究计划内容一致，可离线浏览、打印 PDF。
- [协议与执行状态](protocol.json)：v1.2 对 PAD 身份歧义采用保守组级划分；[v1.0](protocol-v1.0.json) 与 [v1.1](protocol-v1.1.json) 保留历史。风险/指标不变，PAD 划分协议有实质修订。
- [实验记录模板](实验记录模板.md)：首次实测后建立本目录的实验记录.md，并记录所有尝试。
- [后续工作约定](AGENTS.md)：新会话先从这里恢复上下文。

下一步顺序：最近方法全文复核 → 原站数据获取/许可/去重与分组 → 类别树核对 → 冻结特征 → 基线与新增信号验证 → 第 2 周决定是否继续方法开发。

GitHub：`https://github.com/gjh1999/Medical-vlm-selective-prediction.git`。已加入全局项目列表，每天 17:00 执行 WIP 备份；原每 30 分钟项目同步任务保留。较大变更验证后也主动提交推送。任务需要本机开机且注册用户已登录。原图、权重、样本 manifest/split 与 outputs 不入库。

网页由 `python build_plan.py` 从 Markdown 生成，不依赖外部库。以下命令可在当前主机或 5080 主机执行：

```powershell
python preflight.py --network --output outputs/new-preflight-01
python prepare_ham.py --output outputs/new-ham-01
python prepare_pad.py --output outputs/new-pad-01
python -m unittest -v test_core.py
```

各 output 必须使用不存在的独立路径。`prepare_ham.py --download-images` 才会下载约 2.77GB 原图 ZIP。PAD API 已修正，不再使用旧 `/versions/1` 路径。`prepare_pad.py --source-dir 旧run --output 新run` 可复用已校验的元数据；它不下载 PAD 原图。

准备机仅安装独立 `.venv` 中的 Pillow 13.0.0，无 torch/CUDA 安装。完整软件测试命令（从仓库根目录）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s research/medical-vlm-hierarchy -p 'test_*.py' -v
```

`audit_images.py` 对已解压图像做逐图 SHA256、EXIF 统一后的像素哈希和 dHash 候选审计，不静默去重。`extract_features.py` 只读取完整且未篡改的审计清单，并核对每个图像与权重哈希；它可以生成待去重审阅的数据缓存，但这类缓存不可直接当作已完成无泄漏评价。当前只验证了输入契约和 CLI，未验证 PyTorch/OpenCLIP 推理路径。参考核心也不是完整 B0–B6 复现；温度校准、提示文本缓存、分歧信号和外部模型验证仍待实施。
