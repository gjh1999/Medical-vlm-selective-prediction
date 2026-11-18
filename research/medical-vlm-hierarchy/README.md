# 医学 VLM 层级分类与拒识

已选研究方向：有限标注和长尾条件下，在具体病种、上位类别和完全拒识之间作可靠决策。

HAM 的 10,015 张原图全部解码。公开审阅确认的 18 对重复图在旧划分中有 13、12、11 对跨集合。v1.3 将确认重复合并，将另两对未决图分到同一集合。三个种子下这两类配对均无跨集合。修订后仍有 1,045、1,076、1,015 对跨集合代表图命中 dHash 筛查。保守敏感性分析排除涉及跨集合候选的 817 个组，主分析保留全部图像。[修订前后审计](reports/E00-ham-reviewed-summary-01.json) · [候选抽查](reports/E00-ham-visual-review-01.json) · [敏感性分析](reports/E00-ham-sensitivity-01.json)

两个冻结模型已在同一批 128 个 HAM 开发组上完成首轮小试。固定两种提示下，OpenAI CLIP 判对 12 组，BiomedCLIP 判对 15 组；训练集多数类基线判对 91 组。两个模型的父子分歧错例排序 AUROC 均低于普通熵，配对区间均包含零。独立逐图前向与批量缓存的最大 logit 差均小于 0.00002。正式比较、校准和 5080 成本待测。[开发集小试](reports/E01-ham-dev-pilot-01.json) · [独立复算](reports/E01-ham-dev-pilot-reference-01.json)

HAM 有 7,470 个原始病灶 ID、7,452 个分析单位和 7,450 个划分组。种子 17 的校准集有 1,859 个分析单位，其中 df 18、nv 1,344。患者级独立性未知。PAD 有 2,298 张图像、1,373 位患者。250 个 raw lesion_id 跨患者重用，其中 226 个有标签差异。共享 raw ID 的患者保守连接成 1,132 个划分组，三套划分均无患者、raw ID 或连通组交叉。1,891 个 patient+lesion 处理键的生物学身份仍待核实。[实验记录](实验记录.md)

另取 56 个互不重叠、各类等量的开发组，用每类 8 个训练组构建图像类中心。OpenAI CLIP 的正确数由文本基线的 11 提升至 29，BiomedCLIP 由 16 提升至 22。此结果支持在后续评测中纳入同预算的图像分类基线。项目 Climbing 在 1,280 次真实开发预测和 20 次构造边界比较中与作者实现逐节点一致。[少样本对照](reports/E01-ham-balanced-probe-01.json) · [Climbing 对齐](reports/E01-hsc-climbing-match-01.json)

正式 GPU 为家中的 RTX 5080，详见 [5080 运行说明](5080运行说明.md)。本机不安装 GPU 环境。

PAD 每次划分可用 283 个校准组。按 v1.2 的 16 个候选策略计算，即使零错误，叶风险与非根风险也分别要求至少 62 和 126 个独立接受组。预算降到 100 组时，两项约束无法同时通过。每次仅 9 组含 MEL，三个种子合计涉及 16 个不同的含 MEL 组。[校准样本量与重叠统计](reports/E00-pad-capacity-02.json)

- [研究计划](研究计划.md)：研究问题、最近工作、数据/树/划分、基线、候选方法、统计边界、实验清单和八周里程碑。
- [本地网页计划](研究计划.html)：与研究计划内容一致，可离线浏览、打印 PDF。
- [协议与执行状态](protocol.json)。v1.3 修订 HAM 身份分组。历版配置见 [v1.2](protocol-v1.2.json)、[v1.1](protocol-v1.1.json) 与 [v1.0](protocol-v1.0.json)。
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

`audit_images.py` 记录文件、像素哈希和 dHash 候选，`summarize_ham_duplicates.py` 复用同一轮解码结果统计三个划分。`prepare_biomedclip.py` 可下载固定版权重及离线文本配置，`extract_features.py --smoke-only` 用一批训练代表图验证入口。`ham_dev_pilot.py` 用固定的两种提示和树草案运行开发集对照，`verify_ham_dev_pilot.py` 逐图复算并核对错例排序。命令见[实验记录](实验记录.md)。
