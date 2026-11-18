# 医学 VLM 层级分类与拒识

已选研究方向：有限标注和长尾条件下，在具体病种、上位类别和完全拒识之间作可靠决策。

**当前决定（11 月 19 日）。** 暂缓开发新的层级门控。对 96 个预先留出的 HAM 开发组，加入父子分歧后，图像类中心的错例排序 AUROC 在 OpenAI CLIP 上由 0.703 升到 0.788，在 BiomedCLIP 上由 0.761 变为 0.759。同一套 OpenAI CLIP 风险模型在 56 个类别均衡开发组上的差为 −0.070。将训练支持、风险拟合和评价一并重抽 2,000 次，四种骨干与分类器组合的差值区间都跨零。正式比较将保留同标注预算的图像类中心、基本置信度与 Climbing。[预先固定的对照](reports/E02-ham-error-increment-02.json) · [父文本错配](reports/E02-ham-parent-alignment-01.json) · [重抽样](reports/E02-ham-error-bootstrap-01.json)

基础共形对照已在开发组完成。名义覆盖 90% 时，两种图像类中心的预测集合在同一批 96 组上都包含 88 个真值。压缩为一个树节点后，OpenAI CLIP 输出正确非根节点 0 次，BiomedCLIP 为 49 次。正式结果须同时报告集合覆盖、根拒识和正确非根输出。[共形集合及祖先投影](reports/E01-ham-dev-conformal-01.json)

HAM 的 10,015 张原图全部解码。公开审阅确认的 18 对重复图在旧划分中有 13、12、11 对跨集合。v1.3 处理了作者的 18 对确认及 2 对未决关联。全库外观检索和几何核对又找到 5 对需同集合处理的图，v1.4 将其跨集合数从 3、5、5 降到零。仍有 1,045、1,076、1,015 对跨集合代表图命中 dHash 筛查。保守敏感性分析扣除涉及跨集合候选的 817 个组，主分析保留全部原图。[v1.3 审计](reports/E00-ham-reviewed-summary-01.json) · [v1.4 审计](reports/E00-ham-scene-summary-01.json) · [当前敏感性分母](reports/E00-ham-scene-sensitivity-01.json)

进一步的全库纹理检索与缩略图候选合并后找回 17/18 个作者确认对，先前单用缩略图找回 6/18。只检查每图首位纹理邻居，就在 v1.4 的不同组间找到 7 对强几何对应。三个种子分别有 6、5、5 对跨集合，历史 336 个小试组均未涉及。正式留出评价应等待合并候选的全量复核与分组验收。[检索对照](reports/E00-ham-retrieval-cascade-01.json) · [首位候选的划分影响](reports/E00-ham-word-impact-rank1-01.json)

两个冻结模型已在同一批 128 个 HAM 开发组上完成首轮小试。固定两种提示下，OpenAI CLIP 判对 12 组，BiomedCLIP 判对 15 组；训练集多数类基线判对 91 组。两个模型的父子分歧错例排序 AUROC 均低于普通熵，配对区间均包含零。独立逐图前向与批量缓存的最大 logit 差均小于 0.00002。正式比较、校准和 5080 成本待测。[开发集小试](reports/E01-ham-dev-pilot-01.json) · [独立复算](reports/E01-ham-dev-pilot-reference-01.json)

HAM 有 7,470 个原始病灶 ID、7,452 个分析单位和 7,445 个划分组。种子 17 校准集有 1,859 个分析单位、1,858 个划分组。患者级独立性未知。PAD 有 2,298 张图像、1,373 位患者，按共享原始病灶 ID 保守连接成 1,132 个划分组。两个固定公开镜像的 2,298 张 PNG 逐字节一致。16 对像素完全相同，其中 13 对对应不同诊断；仅看每个处理键的代表图，仍有 7 对。三个划分均无跨集合精确重复。[PAD 图像审计](reports/E00-pad-image-summary-01.json) · [标签冲突与敏感性分母](reports/E00-pad-pixel-conflicts-01.json)

另取 56 个互不重叠、各类等量的开发组，用每类 8 个训练组构建图像类中心。OpenAI CLIP 的正确数由文本基线的 11 提升至 29，BiomedCLIP 由 16 提升至 22。此结果支持在后续评测中纳入同预算的图像分类基线。项目 Climbing 在 1,280 次真实开发预测和 20 次构造边界比较中与作者实现逐节点一致。[少样本对照](reports/E01-ham-balanced-probe-01.json) · [Climbing 对齐](reports/E01-hsc-climbing-match-01.json)

正式 GPU 为家中的 RTX 5080，详见 [5080 运行说明](5080运行说明.md)。本机不安装 GPU 环境。

两篇原始数据论文的插图与本地数据有可复核的对应。逐一检查相关的 11 个图像区域后，HAM 的 2 个划分组、PAD 的 6 个划分组进入公开图版敏感性名单；种子 29 的测试集中分别涉及 2 组、1 组。BiomedCLIP 实际训练时是否读到这些图，逐图成员清单仍待核实。[公开图版审计](reports/E00-paper-panels-01.json) · [三种子名单统计](reports/E00-paper-sensitivity-01.json)

PAD 每次划分可用 283 个校准组。按 v1.2 的 16 个候选策略计算，即使零错误，叶风险与非根风险也分别要求至少 62 和 126 个独立接受组。预算降到 100 组时，两项约束无法同时通过。每次仅 9 组含 MEL，三个种子合计涉及 16 个不同的含 MEL 组。[校准样本量与重叠统计](reports/E00-pad-capacity-02.json)

跨库共有叶类 MEL、NEV、BCC 只覆盖 PAD 的 904/1,891 张代表图。每次校准中有 161 个组含共享叶类，只有 113 个组完全由共享叶类构成。按现行 5% 非根风险目标，即使接受全部 161 组，一个观测错误也超出可认证范围。Bowen 病在两个数据集中分别并入 akiec 与 SCC，跨库对照须保留这项标签粒度差异。[标签映射](configs/ham-pad-crosswalk-v1.json) · [分母和容量](reports/E00-external-label-support-02.json)

相同组大小还用于检验逐图二项上界的适用条件。假设真实错误率 6%、同组图像的对错完全一致，把 473 张代表图当作独立观测会以 0.594% 的概率通过单项 5% 风险约束，高于 0.15625% 的单项预算。这是设定相关性后的计算结果。实际模型的组内相关和外部风险待测。[相关性情景与独立复算](reports/E00-pad-cluster-risk-01.json)

- [研究计划](研究计划.md)：研究问题、最近工作、数据/树/划分、基线、候选方法、统计边界、实验清单和八周里程碑。
- [本地网页计划](研究计划.html)：与研究计划内容一致，可离线浏览、打印 PDF。
- [协议与执行状态](protocol.json)。v1.4 增加图像几何关联的同集合约束。历史的开发集小试使用 [v1.3](protocol-v1.3.json)。更早版本为 [v1.2](protocol-v1.2.json)、[v1.1](protocol-v1.1.json) 与 [v1.0](protocol-v1.0.json)。
- [实验记录](实验记录.md)收录实测命令、输出目录、失败尝试、复核和研究决定。
- [后续工作约定](AGENTS.md)：新会话先从这里恢复上下文。

正式运行从 v1.4 数据重建和 5080 双骨干特征验收开始。完整数据上的排序、校准、留出评估和耗时仍待测量。[数据流程重复验证](reports/E00-ham-v14-rebuild-01.json)

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

AISTATS 2026 作者的三种随机化层级共形构造已在七叶树上与本地穷举逐例对齐，共 5,472 次集合输出一致。BiomedCLIP 图像类中心允许输出两个节点后，自然开发集返回全部七个诊断的次数从 27–29 / 96 降到 3–8 / 96。病例覆盖及均衡组结果见[对照汇总](reports/E01-ham-crsvp-02.json)。
