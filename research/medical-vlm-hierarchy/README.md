# 医学 VLM 层级分类与拒识

已选研究方向：有限标注和长尾条件下，在具体病种、上位类别和完全拒识之间作可靠决策。

截至 2026-11-18，HAM 的 10,015 张原图全部解码，发现 2 对像素完全相同的图像，均在同一病灶内。三个划分中跨集合的精确重复均为 0。按每病灶只取一张代表图，种子 17、29、43 仍分别有 1,082、1,113、1,057 对跨集合近重复候选待审。PAD 已完成元数据与校准样本量审计。[HAM 聚合审计](reports/E00-ham-duplicates-01.json)

冻结特征入口已用 8 张 HAM 训练代表图在 CPU 上跑通。OpenAI CLIP ViT-B/16 批量提取与官方加载器逐图复算的最大逐分量差为 3.13×10⁻⁷。分类性能和 5080 上的推理成本待测。[单图复算结果](reports/E00-ham-clip-cpu-reference-01.json)

HAM：10,015 图像 / 7,470 病灶；calibration 1,869 病灶，其中 df 18、nv 1,351，尚不能声称患者级独立。PAD：2,298 图像 / 1,373 患者；250 个 raw lesion_id 跨患者重用，其中 226 个存在标签差异。当前将共享 raw ID 的患者保守连接成 1,132 个 split_group，三套划分均无患者/raw ID/连通组交叉。1,891 个 patient+lesion 处理键不能当成已核实的真实病灶数。完整证据见[实验记录](实验记录.md)。

正式 GPU 为家中的 RTX 5080，详见 [5080 运行说明](5080运行说明.md)。本机不安装 GPU 环境。

PAD 每次划分可用 283 个校准组。按 v1.2 的 16 个候选策略计算，即使零错误，叶风险与非根风险也分别要求至少 62 和 126 个独立接受组。预算降到 100 组时，两项约束无法同时通过。每次仅 9 组含 MEL，三个种子合计涉及 16 个不同的含 MEL 组。[校准样本量与重叠统计](reports/E00-pad-capacity-02.json)

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
python audit_pad_capacity.py --split-dir outputs/new-pad-01 --output outputs/new-pad-capacity-01
python -m unittest -v test_core.py
```

各 output 必须使用不存在的独立路径。`prepare_ham.py --download-images` 才会下载约 2.77GB 原图 ZIP。PAD API 已修正，不再使用旧 `/versions/1` 路径。`prepare_pad.py --source-dir 旧run --output 新run` 可复用已校验的元数据；它不下载 PAD 原图。

仓库根目录的审计环境使用 Pillow 13.0.0。CPU 特征小批次使用本次输出目录内独立的 PyTorch CPU 环境。完整软件测试命令从仓库根目录执行。

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s research/medical-vlm-hierarchy -p 'test_*.py' -v
```

`audit_images.py` 记录文件、像素哈希和 dHash 候选，`summarize_ham_duplicates.py` 复用同一轮解码结果统计三个划分。`extract_features.py --smoke-only` 用一批训练代表图验证入口，`verify_clip_smoke.py` 可对 OpenAI CLIP 向量逐张复算。近重复的身份审阅、BiomedCLIP 实机推理及 B0–B6 的性能比较仍待完成。
