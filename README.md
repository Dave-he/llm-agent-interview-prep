# LLM Agent 面试备考知识库

> 三仓合一的 LLM Agent 面试备考体系：**模型层（能力地板）· 编排层（可靠性上限）· 验证层（进生产门票）**——从八股概念、公式推导，到场景设计题、Agent 层手撕代码、可观测调试，一套打完。
>
> **在线交互版**：[LLM Agent 面试修炼场](https://dave-he.github.io/llm-agent-interview-prep/)——知识树查阅 · 闪卡自评 · 37 题挑战测验 · 场景/手撕揭示 · 7 天计划打卡（进度存本地浏览器）。

## 仓库结构

| 路径 | 内容 |
|---|---|
| `备考总纲-LLM-Agent.md` | **总纲（先读这个）**：三层知识树、考点矩阵 Top20、公式速查 8 组（KV Cache/DPO/GRPO/RRF/NDCG 等）、六大硬核专题、Harness 十七 stage 设计地图、30 天冲刺计划 |
| `学习路线图-LLM到Agent.md` | **学习路线图（从零系统学按这个走）**：开篇"能力圈边界与面试关注点"（三档划分：做不了只考推导算账 / 看着简单必须精 / 判断层拉差距；Claude 的 90/10；两个典型答错方向）；Stage 0–8 循序渐进（地基 → Transformer 底层 → 推理工程 → 训练对齐 → RAG → Agent 核心 → 协作生态 → 生产化 → 毕业项目三选一）；每 Stage 三栏验收：弄懂（材料坐标）/ 弄明白（打勾验收单）/ 实现好（动手产出：numpy attention、KV 计算器、LoRA、DPO、最小 RAG、agent-mini、MCP 接入、毕业项目清单）；附里程碑总表与材料映射表、上游三仓同步机制 |
| `Agent搭建面试专项.md` | **搭建专项**：0→1 十步搭建模板 + 六层架构图；六大场景设计题（Coding / Deep Research / 客服 / 数据分析 / Computer Use / 长时程）；Agent 层手撕代码 T1–T6；可观测与调试专题（turn-span / 轨迹二分 / 裁撤实验）；组件选型决策链；高频追问 Top30 速答卡 |
| `personal-agent-research.md` | 个人 AI Agent 系统调研报告（口径核查基准，冲突数字以此为准） |
| `docs/` | **在线修炼场**（GitHub Pages 静态站，零构建零依赖）：index.html + data.js（知识树/考点/公式/场景/手撕/计划数据）+ quiz.js（37 题题库）+ app.js（六视图交互与 localStorage 进度） |
| `LLM-Agent-Interview-Guide/` | 上游题库（Apache-2.0）+ 本仓新增 **10 章逐题详解**（Foundation / Inference / FineTuning / RAG / Agent / Safety / HotTopics / Coding / SystemDesign / RealQuestions 共 218 题） |
| `llm-interview-questions/` | 上游题库（CC BY-SA 4.0）+ 本仓新增 **9 模块深度详解**（架构 / 推理优化 / 训练微调 / RLHF 对齐 / RAG 与重排 / 向量与 KG / Agent 与 Prompt / 评测监控 / 多模态共 100 题） |
| `learn-claude-code/` | 上游课程（MIT）+ 本仓新增 **17 stage 逐段详解**（docs/zh/detailed-s01-s05 / s06-s09 / s10-s13 / s14-s17 四分片：Agent Loop → 工具/权限/Hooks → 上下文四解法 → 记忆/任务/Cron/团队 → MCP → 集成 Harness → Workflow 运行时 → Goal Loop） |

## 怎么用

0. **从零系统学**：按《学习路线图-LLM到Agent.md》Stage 0→8 顺序走（约 11.5 周），每层"弄懂→弄明白→实现好"三关全过才上楼；路线图管学习顺序，下面的计划管应试排期。
1. **快速摸底**：总纲第二章「考点矩阵 Top20」自测，标出弱点。
2. **系统备考**：总纲第五章 30 天冲刺计划（概念周 → 代码周 → 真题周）；搭建专项第七章 7 天微计划可插入 D21–D24 区间。
3. **面试临场**：总纲 1.2 公式速查（白板可默写）+ 专项第六章 Top30 速答卡（一问一句）。
4. **设计题**：专项第一章十步模板先背——需求三问（可验证/可逆/频率）→ 单多 agent → loop → 工具面 → 上下文 → 记忆 → 权限 → 验证 → 评测 → 可观测。

## 在线修炼场

**https://dave-he.github.io/llm-agent-interview-prep/** —— 把上面的知识库做成了可交互复习小游戏（GitHub Pages 静态站），六大玩法：

| 视图 | 玩法 |
|---|---|
| **总览** | 四指标仪表盘：闪卡掌握度 / 测验正确率 / 计划打卡 / 学习天数 + 智能建议（自动引导先清哪块）+ 临场表达三动作口诀（听题归类四型 · 万能四段 · 主动交棒） |
| **知识树** | 3 层 18 节点，支持关键词搜索（GQA / 注入 / compaction / DPO…），点击节点展开背诵骨架 |
| **闪卡速查** | 考点 20 + 公式 8 翻面卡：先自述再看答案，三级自评「认识 / 模糊 / 不会」，可按薄弱卡过滤、打乱顺序，掌握进度条实时统计 |
| **挑战测验** | 37 题四选一：全量 / 随机 20 / 错题重刷三种模式，键盘 1–4 作答、Enter 下一题，每题带解析与易错提示，答错自动进错题池 |
| **场景 · 手撕** | P1–P6 场景设计题（3 分钟骨架 + 追问树 + 陷阱）、T1–T6 手撕代码（思路 + 代码 + 易错自查）、Top30 速答卡（先自己说 10 秒再点开对照） |
| **复习计划** | 专项 7 天微计划：任务 + 产出双打卡，进度条可视化，通关后进模考节奏 |

**使用说明**：

- 复习进度（自评/错题/打卡）存本地浏览器 localStorage，不上传任何数据。
- 键盘速查：测验中 `1–4` 选答案、`Enter` 下一题；闪卡点击卡面翻面。
- 零构建零依赖纯静态（`docs/` 下 4 个文件：index.html / data.js / quiz.js / app.js），改完 push `main` 即自动发布。

## 更新计划

后续持续更新 LLM Agent 相关面试经验：面经复盘、新基准数字口径（SWE-bench / GAIA / OSWorld 轨迹）、新考点（MCP 生态、Computer Use、长时程自动化）与裁撤实验记录。

## 致谢与上游项目

本站在三个优秀开源项目的基础上新增逐题详解与整合文档，三个子目录**完整保留各自上游 LICENSE**：

| 上游项目 | 许可证 | 本仓新增 |
|---|---|---|
| [Lau-Jonathan/LLM-Agent-Interview-Guide](https://github.com/Lau-Jonathan/LLM-Agent-Interview-Guide) | Apache-2.0 | 10 章 Detailed-Answers（218 题五段式详解） |
| [MisterBooo/llm-interview-questions](https://github.com/MisterBooo/llm-interview-questions) | CC BY-SA 4.0 | 9 个 detailed-\* 模块（100 题深度详解，含 2025-11 口径纠偏注记） |
| [shareAI-lab/learn-claude-code](https://github.com/shareAI-lab/learn-claude-code) | MIT | docs/zh/detailed-s01-s05 / s06-s09 / s10-s13 / s14-s17 四分片（17 stage 源码级精读 + 动手实验） |

## License

- 本仓库原创内容（总纲、专项、调研报告、各 detailed 详解文档）：[CC BY-SA 4.0](./LICENSE)
- 三个子目录内容：遵循各自目录内保留的上游许可证（Apache-2.0 / CC BY-SA 4.0 / MIT）
