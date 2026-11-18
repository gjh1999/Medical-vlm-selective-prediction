# 医学视觉研究选题与计划

- [新增并行方向：跨模态异常检测中的证据校准](research/cross-modal-evidence-calibration/README.md)：医学正常组织误报、目标正常样本预算与迁移失效边界；独立子目录启动探索。

- [视觉顶会选题网页报告](docs/vision-report/index.html)：12 篇论文、9 项数据资源、4 个候选方向。
- [已选方向的研究计划](research/medical-vlm-hierarchy/研究计划.md)：医学 VLM 的层级分类与拒识，v1.0。
- [研究计划网页](research/medical-vlm-hierarchy/研究计划.html)：离线阅读及打印。
- [方向项目入口](research/medical-vlm-hierarchy/README.md)：后续实验与交接从此处开始。
- [少样本分割的支持集质量与预算](research/fewshot-support-budget/README.md)：2026-11-17 新建并完成首轮查新与源码预检，重点探索新增标注与修正支持标签的预算分配，尚无模型性能结果。
- [交互分割的提示鲁棒性与可靠停止](research/interactive-segmentation-reliability/README.md)：并行探索路线；首轮文献核查、轨迹级停止协议与样本量可行性计算，尚未执行分割模型实验。

医学 VLM 第一阶段已校验 HAM 与 PAD 元数据。为处理跨患者重复的病灶 ID，PAD 采用保守连通组划分。每次有 283 个校准组，其中仅 9 组含黑色素瘤。按现行 16 个候选策略计算，即使零错误，5% 非根风险目标也至少需要 126 个独立接受组。15 项软件测试通过，5080 模型推理尚待执行。详见[实验记录](research/medical-vlm-hierarchy/实验记录.md)和[5080 主机运行说明](research/medical-vlm-hierarchy/5080运行说明.md)。

GitHub 仓库：[Medical-vlm-selective-prediction](https://github.com/gjh1999/Medical-vlm-selective-prediction)。`scripts/sync-github.ps1` 同步代码、配置、文档与实验摘要；`scripts/register-github-sync.ps1` 在成功首次推送后注册每 30 分钟执行的 Windows 任务。原始数据、权重与逐样本缓存不入库。状态以本机任务 `MCCAI-Research-GitHub-Sync` 和 `.sync-state/last-success.json` 为准；任务需要本机开机且注册用户已登录，日志在 `.sync-state/sync.log`。

本项目已登记到用户 Codex 全局自动同步列表，每天本地时间 17:00 进行 WIP 备份；较大变更验证后也主动提交推送。定时备份不代表实验完成或测试通过。
