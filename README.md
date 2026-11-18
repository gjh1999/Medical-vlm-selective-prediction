# 医学视觉研究选题与计划

- [五个课题的导师评审与实验顺序](research/导师组会-2026-11-18.md)（2026-11-18）。包括主线、各方向的强基线、已发现的预算和数据限制，以及继续或停题的条件。
- [固定追加预算下的皮肤镜采集](research/paired-dermoscopy-acquisition/README.md)。核查同一病灶的双视图配对，检验追加前能否预测净纠错收益。

- [新增并行方向：跨模态异常检测中的证据校准](research/cross-modal-evidence-calibration/README.md)：医学正常组织误报、目标正常样本预算与迁移失效边界；独立子目录启动探索。

- [视觉顶会选题网页报告](docs/vision-report/index.html)：12 篇论文、9 项数据资源、4 个候选方向。
- [已选方向的研究计划](research/medical-vlm-hierarchy/研究计划.md)：医学 VLM 的层级分类与拒识，v1.0。
- [研究计划网页](research/medical-vlm-hierarchy/研究计划.html)：离线阅读及打印。
- [方向项目入口](research/medical-vlm-hierarchy/README.md)：后续实验与交接从此处开始。
- [少样本分割的支持集质量与预算](research/fewshot-support-budget/README.md)：2026-11-17 新建并完成首轮查新与源码预检，重点探索新增标注与修正支持标签的预算分配，尚无模型性能结果。
- [交互分割的提示鲁棒性与可靠停止](research/interactive-segmentation-reliability/README.md)：并行探索路线；首轮文献核查、轨迹级停止协议与样本量可行性计算，尚未执行分割模型实验。

医学 VLM 主线已解码 HAM 全部 10,015 张原图。18 对作者审阅确认的重复在旧划分中分别有 13、12、11 对跨集合。v1.4 增加了 5 对场景约束；v1.5 对全部 235,405 个候选作几何核查，再将 23 对强场景对应约束到同一集合。后者原有的 18、14、18 对跨集合对应均已归零，保留全部 7,452 个分析单元。PAD 有 13 对图像像素完全相同、诊断标签却不同，已记录对校准分母的影响。PAD 每次有 283 个校准组，仅 9 组含黑色素瘤。[实验记录](research/medical-vlm-hierarchy/实验记录.md) · [5080 主机运行说明](research/medical-vlm-hierarchy/5080运行说明.md)

在两个自然开发批合计 192 组上，加入父子分歧后，原模型的错例排序 AUROC 增量为 OpenAI CLIP +0.082、BiomedCLIP −0.005。同一批已用病例按相同预算重分训练支持与风险拟合后，512 种训练组合中有 30 种在两个骨干及两种评价构成上均有正增益。两个骨干的三类策略在 224 组开发校准演练中均未达到预设的双风险目标，暂缓新增门控。正式检验待家中 RTX 5080 与类别树审阅。[训练组合分布](research/medical-vlm-hierarchy/reports/E02-ham-training-allocations.png) · [成对风险演练](research/medical-vlm-hierarchy/reports/E03-ham-paired-risk-01.json) · [跨划分监督使用审计](research/medical-vlm-hierarchy/reports/E00-ham-development-exposure-01.json)

GitHub 仓库：[Medical-vlm-selective-prediction](https://github.com/gjh1999/Medical-vlm-selective-prediction)。`scripts/sync-github.ps1` 同步代码、配置、文档与实验摘要；`scripts/register-github-sync.ps1` 在成功首次推送后注册每 30 分钟执行的 Windows 任务。原始数据、权重与逐样本缓存不入库。状态以本机任务 `MCCAI-Research-GitHub-Sync` 和 `.sync-state/last-success.json` 为准；任务需要本机开机且注册用户已登录，日志在 `.sync-state/sync.log`。

本项目已登记到用户 Codex 全局自动同步列表，每天本地时间 17:00 进行 WIP 备份；较大变更验证后也主动提交推送。定时备份不代表实验完成或测试通过。
