# LLM Agent 面试备考知识库

> 三仓合一的 LLM Agent 面试备考体系：**模型层（能力地板）· 编排层（可靠性上限）· 验证层（进生产门票）**——从八股概念、公式推导，到场景设计题、Agent 层手撕代码、可观测调试，一套打完。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `备考总纲-LLM-Agent.md` | **总纲（先读这个）**：三层知识树、考点矩阵 Top20、公式速查 8 组（KV Cache/DPO/GRPO/RRF/NDCG 等）、六大硬核专题、Harness 十七 stage 设计地图、30 天冲刺计划 |
| `Agent搭建面试专项.md` | **搭建专项**：0→1 十步搭建模板 + 六层架构图；六大场景设计题（Coding / Deep Research / 客服 / 数据分析 / Computer Use / 长时程）；Agent 层手撕代码 T1–T6；可观测与调试专题（turn-span / 轨迹二分 / 裁撤实验）；组件选型决策链；高频追问 Top30 速答卡 |
| `personal-agent-research.md` | 个人 AI Agent 系统调研报告（口径核查基准，冲突数字以此为准） |
| `LLM-Agent-Interview-Guide/` | 上游题库（Apache-2.0）+ 本仓新增 **10 章逐题详解**（Foundation / Inference / FineTuning / RAG / Agent / Safety / HotTopics / Coding / SystemDesign / RealQuestions 共 218 题） |
| `llm-interview-questions/` | 上游题库（CC BY-SA 4.0）+ 本仓新增 **9 模块深度详解**（架构 / 推理优化 / 训练微调 / RLHF 对齐 / RAG 与重排 / 向量与 KG / Agent 与 Prompt / 评测监控 / 多模态共 100 题） |
| `learn-claude-code/` | 上游课程（MIT）+ 本仓新增 **17 stage 逐段详解**（docs/zh/detailed-s01-s17：Agent Loop → 工具/权限/Hooks → 上下文四解法 → 记忆/任务/Cron/团队 → MCP → 集成 Harness → Workflow 运行时 → Goal Loop） |

## 怎么用

1. **快速摸底**：总纲第二章「考点矩阵 Top20」自测，标出弱点。
2. **系统备考**：总纲第五章 30 天冲刺计划（概念周 → 代码周 → 真题周）；搭建专项第七章 7 天微计划可插入 D21–D24 区间。
3. **面试临场**：总纲 1.2 公式速查（白板可默写）+ 专项第六章 Top30 速答卡（一问一句）。
4. **设计题**：专项第一章十步模板先背——需求三问（可验证/可逆/频率）→ 单多 agent → loop → 工具面 → 上下文 → 记忆 → 权限 → 验证 → 评测 → 可观测。

## 更新计划

后续持续更新 LLM Agent 相关面试经验：面经复盘、新基准数字口径（SWE-bench / GAIA / OSWorld 轨迹）、新考点（MCP 生态、Computer Use、长时程自动化）与裁撤实验记录。

## 致谢与上游项目

本站在三个优秀开源项目的基础上新增逐题详解与整合文档，三个子目录**完整保留各自上游 LICENSE**：

| 上游项目 | 许可证 | 本仓新增 |
|---|---|---|
| [Lau-Jonathan/LLM-Agent-Interview-Guide](https://github.com/Lau-Jonathan/LLM-Agent-Interview-Guide) | Apache-2.0 | 10 章 Detailed-Answers（218 题五段式详解） |
| [MisterBooo/llm-interview-questions](https://github.com/MisterBooo/llm-interview-questions) | CC BY-SA 4.0 | 9 个 detailed-\* 模块（100 题深度详解，含 2025-11 口径纠偏注记） |
| [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) | MIT | docs/zh/detailed-s01–s17（17 stage 源码级精读 + 动手实验） |

## License

- 本仓库原创内容（总纲、专项、调研报告、各 detailed 详解文档）：[CC BY-SA 4.0](./LICENSE)
- 三个子目录内容：遵循各自目录内保留的上游许可证（Apache-2.0 / CC BY-SA 4.0 / MIT）
