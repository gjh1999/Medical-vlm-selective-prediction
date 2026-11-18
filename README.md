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

在 96 个新 HAM 开发组上，父子分歧加入普通置信度后，图像类中心的错例排序 AUROC 在 OpenAI CLIP 上提高 0.085，在 BiomedCLIP 上变化 −0.001。换为类别均衡开发组，OpenAI CLIP 的增量为 −0.070。纳入训练、拟合和评价变动后，四种组合的重抽样区间都跨零。暂缓新增门控，保留少样本图像类中心和置信度强基线，正式结果待 5080 运行。[双模型对照与复核](research/medical-vlm-hierarchy/README.md)

GitHub 仓库：[Medical-vlm-selective-prediction](https://github.com/gjh1999/Medical-vlm-selective-prediction)。`scripts/sync-github.ps1` 同步代码、配置、文档与实验摘要；`scripts/register-github-sync.ps1` 在成功首次推送后注册每 30 分钟执行的 Windows 任务。原始数据、权重与逐样本缓存不入库。状态以本机任务 `MCCAI-Research-GitHub-Sync` 和 `.sync-state/last-success.json` 为准；任务需要本机开机且注册用户已登录，日志在 `.sync-state/sync.log`。

主线补做了第二批自然开发组和同预算的训练病例对照。两批合计 192 组，原训练配置下加入父子分歧的错例排序增量为 OpenAI CLIP +0.082、BiomedCLIP −0.005。换一套支持组及风险拟合组，增量变为 +0.025、+0.075，在均衡开发组上则两者都为负。另已确定哪些早期开发病例会落入另两份正式测试集。[四格对照](research/medical-vlm-hierarchy/reports/E02-ham-crossed-fit-02.json) · [跨划分审计](research/medical-vlm-hierarchy/reports/E00-ham-development-exposure-01.json)

本项目已登记到用户 Codex 全局自动同步列表，每天本地时间 17:00 进行 WIP 备份；较大变更验证后也主动提交推送。定时备份不代表实验完成或测试通过。
