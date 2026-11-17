# LLM-Agent 面试备考总纲（三仓合一）

> 版本 v1.0 · 整合日 2026-11-13
> **统一口径声明**：三个仓库 + 一份调研报告整合而成；任何定量声称与时间线冲突时，《个人 AI Agent 系统全面调研报告（事实核查修订版 v1.1）》（`personal-agent-research.md`）为核查基准。仓2 文件自带"口径注记（2025-11 / 2026 新考点）"的，一律采用其修正口径。本总纲是浓缩转述版，不复制整段原文；展开精读回到第六章附录列出的对应源文件。
>
> **素材**：调研报告 1 份（250 行）｜仓1 LLM-Agent-Interview-Guide 10 章 218 题详解｜仓2 llm-interview-questions 9 模块 100 题详解｜仓3 learn-claude-code 17 stage 工程 + 4 份逐 stage 详解｜专项深挖 1 份（《Agent搭建面试专项.md》：场景设计题 + Agent 层手撕代码 + 可观测调试 + 选型链，见附录 6.7）。

---

## 一、开篇：知识地图与一页纸公式速查

### 1.1 三层知识树

```text
LLM-Agent 面试知识树
├── 模型层 —— 能力地板（模型决定下限）
│   ├── Transformer 与注意力：QKV 推导、√d_k 方差解释、多头参数恒等式
│   ├── 位置编码：RoPE 复数旋转 / 外推与 YaRN；Norm 与 FFN（Pre-Norm、RMSNorm、SwiGLU）
│   ├── 分词：BPE 训练-推理全流程、WordPiece/Unigram 对比、中文特殊性
│   ├── KV Cache：合法性、显存公式、MHA→MQA→GQA→MLA 演进、Flash/Paged/投机解码
│   ├── 推理工程：解码策略、PTQ/QAT（GPTQ/AWQ）、TP/PP、延迟拆解、框架选型
│   ├── PEFT 与对齐：LoRA/QLoRA 族、RLHF 三阶段、PPO→DPO→GRPO(→DAPO)
│   ├── 前沿架构：MoE 细粒度专家、DeepSeek-V3/R1、Test-Time Compute、多模态 VLM
│   └── 来源：仓1 第 01/02/03/07 章；仓2 architecture・inference・training-finetuning・rlhf・multimodal
│
├── 编排层 —— 可靠性上限（harness 决定上限）
│   ├── Agent Loop：while(stop_reason==tool_use)、终止三分类、观察回填
│   ├── 上下文工程：隔离(躲)/按需(等)/压缩(挤)/外存(搬) 四解法；context rot
│   ├── 工具与协议：Function Calling 全链路、工具描述质量、MCP host 四步
│   ├── 状态系统：TodoWrite 状态机、任务依赖图、记忆四件套、后台任务、cron
│   ├── 协作形态：子代理 / 持久团队 / workflow 运行时；编排确定性 + 模型智能
│   └── 来源：仓1 第 04/05/09 章；仓2 rag・vector・agent-prompt；仓3 s01–s16
│
└── 验证层 —— 进生产的门票（面试与落地里最被低估的一层）
    ├── 评测基准：SWE-bench 轨迹、GAIA/OSWorld/τ²-bench/AgentDojo 分层选型
    ├── 评测方法：自动 vs 人工、LLM-as-judge 偏差与 BT 校准、NDCG/MRR 算例
    ├── 安全：注入与越狱分类、防御纵深三层、记忆投毒、幻觉探测
    ├── 运行时验证：goal gate 自验证、对抗验证（每条 finding 一个验证者）、CI 挂评测
    └── 来源：仓1 第 06/08/10 章；仓2 evaluation-monitoring；仓3 s15/s16/s17
```

**三层树的答题用法**：任何开放题先报层——"这属于模型层翘地板，还是编排层抬上限，还是验证层给门票"。调研报告考点 19 的标准展开：**模型能力是地板，harness（上下文管理、工具设计、错误处理、验证回路）是天花板**；同等模型下，工具描述质量、错误恢复、验证信号可得性的差异远大于"换模型"。

### 1.2 一页纸核心公式速查（背熟到白板可写）

**(1) KV Cache 显存**（仓1 02-Inference 题2；仓1 01 题7 同源）

```text
KV = 2(K+V) × b × n_kv × L × s × d_head × sizeof(dtype)    【Q 不缓存；MLA 无 ×2】
每 token 每层 = 2·n_kv·d_head·dtype；全模型 ×L；×序列长 s ×batch
```

| 锚点（必会复算） | 数值 |
|---|---|
| LLaMA-2-7B MHA（L=32, n_kv=32, d=128）@ 4k | 16 KiB/层/token → 512 KiB/token → **2.0 GiB**（十进制 ≈2.15 GB） |
| LLaMA-3-70B GQA-8（L=80, n_kv=8）@ 4k | **1.25 GiB/序列**（320 KiB/token） |
| 反事实 70B MHA @ 4k | 10 GiB/序列 → GQA 省 8 倍（64/8） |
| DeepSeek MLA（d_c=512 + d_h^R=64, L=61） | 576 元素 = 1152 B/层/token（无 ×2）→ **68.6 KiB/token → 0.268 GiB@4k**；= GQA-8 的 **28%**、7B-MHA 的 **7%**（论文口径 KV 降 93.3%、吞吐 5.76×） |
| 128k 上下文（70B 档） | MHA≈320 GiB → GQA≈40 GiB → MLA≈8.4 GiB（报数先声明 GiB/GB 口径） |
| 容量规划 | A100-80G + 70B INT4：权重约 36–40 GB，剩 ~40 GB ÷ 1.25 GiB/4k = **约 30 路并发** |

**(2) FLOPs**（仓1 01 题5 / 08 题3 口径）

- 单次前向 ≈ **2 × 参数量 × token 数**（乘加记 2 FLOPs）。
- 训练（前向+反向）≈ **6ND**（N 参数、D token 数；前向 2N + 反向 4N）。
- 注意力核心项：prefill 全序列 ≈ 2·T²·d（score 矩阵），decode 每步 = 当前 qkv 投影 6d + 读旧 KV 的 4td——**缓存省投影、不省注意力读取**（decode 带宽瓶颈根源）。

**(3) DPO 目标**（仓1 03 题8；仓2 rlhf 题2/3）

```text
L_DPO(θ) = −E[ log σ( β·log(π_θ(yw|x)/π_ref(yw|x)) − β·log(π_θ(yl|x)/π_ref(yl|x)) ) ]
推导主线：KL 约束 RLHF 目标 → 拉格朗日 → π* = π_ref·e^{r/β}/Z → 反解 r = β·log(π*/π_ref)+β·logZ
→ 代入 Bradley-Terry 成对 sigmoid → Z(x) 在奖励差中相消 → π* 换成可训练 π_θ。
β 双重角色：约束端越大越贴 π_ref；损失端 sigmoid 越饱和、更新越保守。常用 β∈[0.05,0.5]，0.1 最常见。
隐式难例加权：错序越狠权重越接近 1 → 对标注噪声也"更用力学"，这是 DPO 噪声敏感的机制根源。
```

**(4) GRPO 目标**（仓1 03 题10/11；DeepSeekMath 原式口径）

```text
优势：Â_i = (r_i − mean{r_1..r_G}) / std{r_1..r_G}   （组均值即 V(x) 的蒙特卡洛基线，critic 消失）
目标：J = E[ (1/G)Σ_i (1/|y_i|) Σ_t ( min(ρ·Â, clip(ρ,1±ε)·Â) − β·D̂^KL_t ) ]
ρ = π_θ/π_old；KL 用 k3 无偏估计：D̂_t = π_ref/π_θ − log(π_ref/π_θ) − 1 ≥ 0（逐样本非负）
版本口径：DeepSeekMath 归一化分母 (1/G)(1/|y_i|) = 原式；全局 token 归一 1/Σ|y_i| = DAPO 修正，勿混写。
G=64 常见（小规模 4–32）；显存账：可训练量减半 ~16B→8B/可训参数（7B：约 250GB→130GB 量级）。
坑：全对/全错组 std→0，该组零梯度 → DAPO 动态采样过滤；R1/DAPO 后期 β=0（去 KL）。
R1 配方 = GRPO + RLVR（可验证规则奖励）+ 少量 SFT 冷启动。
```

**(5) RRF 融合**（仓1 04 题5；仓2 rag-doc-rerank）

```text
RRF(d) = Σ_{r∈R} 1/(k + rank_r(d))   k=60（Cormack 2009 缺省）、rank 从 1 起
用 rank 不用 score：BM25 无上界、cos∈[-1,1]，量纲不可比；rank 免校准、抗异常。
手算例（面试现场会算=加分）：A 路 d1 第1、d2 第3；B 路 d2 第1、d1 第5：
RRF(d1)=1/61+1/65≈0.03177；RRF(d2)=1/63+1/61≈0.03226 → d2 胜（"两路都靠前"压过"单路第一"）
k→∞ 趋近投票制；k=0 只看各路 top1。
```

**(6) GQA/MLA 一句话锚**（仓1 01 题7）

```text
演进统一叙事：瓶颈在"每 token 的 KV 元素数"（decode 带宽）。
MQA：n_kv=1，cache 缩 h 倍但质量掉；GQA：n_kv=g，LLaMA-70B g=8 → 1/8，质量接近无损（可由 MHA 权重 mean-pool uptraining）；
MLA：跳出多头框架做低秩压缩，只缓存 c_kv(512)+k_rope(64)=576 元素/层；
难点=RoPE 与压缩耦合 → k 拆 content（可压）+ rope（单独存每头 64 维）；质量略优于 GQA（DeepSeek-V2 论文）。
```

**(7) 检索评测**（仓2 evaluation-monitoring 题1/8；仓2 rag 题）

```text
DCG@k = Σ_{i≤k} (2^rel_i − 1)/log2(i+1)；NDCG@k = DCG@k / IDCG@k（理想排序，查询内归一到 [0,1]）
算例：输出 [3,0,3] k=3 → DCG=7+0+3.5=10.5，IDCG=7+7×0.631+0.5=11.92 → NDCG≈0.88；
rel=3 掉到第 3 位 → 0×1+7×0.631+7×0.5=7.92 → ≈0.66（掉位惩罚快于乱序）。
MRR：首个相关结果排名倒数的均值（"只认第一命中"）。口径注记：log2(i+1) 与 log2(i+2) 索引平移写法并存，跨系统比分先固定口径。
```

**(8) Agent 可靠性算术**（调研报告考点 10）

```text
链式衰减：单步成功率 p、串行 n 步 → P=p^n；0.95^20 ≈ 0.36。
对策四件：验证信号优先（可执行测试/退出码）→ checkpoint 与回滚 → 人审卡点 → 缩小任务粒度。
```

### 1.3 三仓去重规则（同一考点怎么读三套材料）

- **概念与标准表述跟仓1**：218 题是主口径，逐题五段式（考察意图/深度解析/30 秒框架/追问/知识串联），背"30 秒框架"即可。
- **深度算例与纠偏跟仓2**：凡仓2 标"口径注记（2025-11）/纠偏"的内容都是新口径，面试引用比仓1 原文更保险——Contextual Retrieval 35%–49%–67% 官方数字、"检索提升 5 法"实为分桶 6 手段、模型路由已转向小 LLM、HNSW 删除=逻辑标记+压实、多信号直接加权需同尺度化、NDCG 算例（IDCG@3=11.92，0.88/0.66）。
- **工程落点跟仓3**：任何"你真做过吗"的追问，用仓3 stage 的源码级细节与"裁撤实验"（删掉某机制会发生什么）作答——这是面试区分度的最大来源。
- **冲突数字以调研报告核查口径收束**：Claude Code 面世 2025-02（非 2024）、SWE-bench 原始集 vs Verified 口径分家、mem0=Apache-2.0、DeepSeek-V3.2 价 $0.28/$0.42（缓存命中 $0.028）、ECM（"Agent 生态 USB-C"＝MCP）时间 2024-11-25。
- **重复内容三处只讲一遍的位置**：RRF/NDCG 在 1.2 与 D 组；KV 公式在 1.2 与 B 组；多智能体取舍在第二章矩阵与 E4/4.3 组四。用交叉引用代替重讲。

---

## 二、考点矩阵：Top20 × 三仓素材对照

> 列：考点（调研报告 Top20）｜一句话要害｜仓1 章节｜仓2 文件｜仓3 stage 实战。五星=2026 区分度考点（MCP、上下文工程、多智能体、可靠性、注入防御——与传统八股拉开差距的五个）。

| # | 考点 | 一句话要害 | 仓1 章节 | 仓2 文件 | 仓3 stage |
|---|---|---|---|---|---|
| 1 ⭐ | Agent vs Chatbot vs Workflow | Chatbot 单轮无状态；Workflow 控制流写死在代码；Agent 控制流由模型输出决定。金句：Workflow=编排确定性，Agent=模型智能 | 05 题1 | agent-prompt 01/07 | s01、s12、s16 |
| 2 ⭐ | Agent Loop 完整循环 | while(出现 tool_use){执行→观察回填}；终止=模型自停/步数预算/人工中断；终止条件与 max_turns 是工程必须项 | 05 题1（60 行最小实现） | agent-prompt 03/04 | s01、s15 |
| 3 ⭐ | ReAct 为何有效 | Reasoning 与 Acting 交错生成：思考轨迹引导行动选择，行动结果修正推理；grounding 到观察减少幻觉 | 05 题2 | agent-prompt 01 | s01（tool_result 回 user 角色即 Observation 槽位） |
| 4 ⭐ | Function Calling | 传 JSON Schema→模型产结构化调用→**执行在客户端**→tool message 回填；并行调用=一次响应多 tool_use 块 | 05 题3 | agent-prompt 08 | s02（dispatch map） |
| 5 ⭐（2026 必考） | MCP | 解 M×N 集成为协议：host/client/server 三层 + tools/resources/prompts 三原语；MCP 管"工具从哪来"，FC 管"模型怎么调"，互补不互替 | 05 题4/5；07 题9 | — | s14（host 四步：发现/命名空间/宿主授权/每轮重建池） |
| 6 ⭐ | Context Engineering | 管"什么信息以什么形式在什么时点进入有限窗口"；核心敌人 context rot；实践=系统提示/结果裁剪/历史压缩/检索注入/子代理隔离 | 07 题7 | agent-prompt 03 | s07（按需披露）、s08（四步压缩）、s11（通知注入） |
| 7 | 记忆系统 | 短期=窗口+scratchpad；长期=外存；四件套 store/recall/consolidate/forget；方案谱系 mem0(抽取式)/Letta(分层自编辑)/Zep(时序图) | 05 题6 | agent-prompt 02 | s09（守门+召回预算）、s10（状态外置） |
| 8 | RAG 与 Agentic RAG | RAG 是检索增强模式，Agent 是多步系统；Agentic RAG=检索成为可决策工具（多跳/改写/自评再检索） | 04 题6/8 | rag-doc-rerank 03/04 | s07（索引与正文分离同构） |
| 9 ⭐ | 多智能体取舍 | 收益=并行+上下文隔离+异构分工；代价=成本倍增+失败级联+handoff 有损；默认先单 agent+子代理，显著瓶颈再升级 | 05 题7/8 | agent-prompt 05 | s06（一次性隔离）、s13（持久团队）、s16（编排替代） |
| 10 ⭐ | 可靠性瓶颈 | 链式衰减 0.95^20≈36%；错误自反思的噪声滚雪球；对策=验证信号/checkpoint/人审/缩粒度 | 09 题5 | evaluation-monitoring 04 | s10/s11/s15/s17 |
| 11 | 评测基准 | SWE-bench(编码)/GAIA(助理)/OSWorld(GUI)/WebArena(网页)/τ²-bench(策略)/AgentDojo(注入)；分数仅初筛，必须私域评测集复验 | 06 题5/6；07 题5 | evaluation-monitoring 全 | s16（每条 finding 对抗验证）、s17（退出码证据） |
| 12 ⭐ | Prompt Injection | 外部内容携带恶意指令劫持模型；agent 有工具权+上下文内不可分辩指令与数据故更险；答案=无银弹+纵深分层 | 06 题1/2/3 | evaluation-monitoring 03 | s03（三闸门）、s04、s14（hint 不可信）、s15（摘要降级） |
| 13 | 沙箱与权限 | 默认只读、写/网络按需；代码进容器/worktree 禁网；凭证分权；四档权限=同一条管线的配置差异 | 09 题5 | agent-prompt 08 | s03、s13（worktree≠沙箱）、s14（deny-by-default） |
| 14 | 成本优化 | 模型路由（per-step）+KV/前缀缓存+上下文压缩+子代理避免重复读+预算上限降级 | 02 题20 | inference-optimization 04/05 | s15（RecoveryState/Budget） |
| 15 | 微调 vs 提示 vs RAG | 默认顺序：prompt/context 工程 → RAG/工具（外部知识可更新可审计）→ 微调（格式/风格/降本/私密标注）；微调不解决知识过时 | 03 题14–18 | training-finetuning 07 | s07（技能按需加载=外部知识） |
| 16 | 论文脉络 | ReAct=交错推理行动 → Reflexion=语言化自我反思 → Voyager=技能库+自动课程 → Generative Agents=记忆流+反思+计划 | 05 题2；07 题5/6 | agent-prompt 01 | s05/s10（反思与计划的工程化落点） |
| 17 | 规划技术谱系 | CoT 线性→ToT 树搜索→任务分解+反思迭代；工程主流=plan-first + 执行中重规划（以工具形态出现） | 05 题1；09 题5 | agent-prompt 01/06 | s05（todo nudge）、s10（依赖图）、s17（block 理由=反思信号） |
| 18 | HITL | 分级审批（高危阻断）+checkpoint 后批量人审+ interrupt-on-external-effect + 可回滚是放权前提 | 09 题5 | agent-prompt 08 | s03（ask_user）、s12（定时轮禁交互审批）、s15 |
| 19 ⭐ | harness 天花板论 | 模型是地板、harness 是天花板；30 行核心循环之外全是增值工程 | 05/09 全章 | agent-prompt 04/06 | s15（3330 行集成全组件即证据） |
| 20 | 前沿：CU/长时程/个人 Agent | GUI 原生操作扩覆盖面；30h 级长时程=compaction+checkpoint+自验证；个人主线=端侧模型+本地记忆；评测从能力分转向安全与过程分 | 07 题5/6/8/9 | multimodal 01 | s17（goal gate 自验证） |

**去重说明**：三仓重叠内容已按"概念→仓1、深度算例→仓2、工程落点→仓3"分工；RAG 评估（仓2 eval）与 Agent 评测（仓1 06、调研考点 11）同源不同面，第三节分别在 D6/E6 讲，不再重复。

### 2.1 高频组合题（连环追问的"通关链"，跨考点串答）

面试官常把 2–3 个考点串成一条链；以下 8 条链各给一版连贯答法（每条 breaking point 标在末尾）：

1. **KV 链**：KV cache 原理 → 显存公式三锚点 → 怎么减（按因子列菜单）→ decode 为什么慢（带宽）→ 并发容量怎么算（30 路反例）。收尾必须落到"报数先声明口径"。
2. **对齐链**：RLHF 三阶段 → PPO 四组件 → critic 之罪 → GRPO 组均值 → 显存账（250→130GB）→ DAPO 修正 → 什么场景还选 DPO。
3. **RAG 链**：三阶段 → 失败三层归因 → chunking 权衡 → 混合检索+RRF → 重排必要性 → NDCG 算例 → 在线监控闭环。
4. **Agent 本质链**：Agent vs Workflow → loop 四要素 → FC 客户端执行 → MCP 与 FC 关系 → 工具描述质量（harness 天花板论收束）。
5. **上下文链**：context rot → 四解法（隔离/按需/压缩/外存）→ s08 四步序为何固定 → 摘要防注入 → 长时程配方收尾（compress+checkpoint+自验证）。
6. **多智能体链**：何时多 agent → 三代价 → subagent vs 持久团队 → 通信/认领/协议落点 → 为什么框架共识是"编排确定性+模型智能"。
7. **安全链**：注入为何 agent 更险 → 为什么没有根治 → 三闸门权限 → 摘要/记忆两侧的"当数据不当指令"→ 上线九条。
8. **评测链**：SWE-bench 轨迹 → 基准分层 → 自建评测三件套 → LLM-as-judge 三病 → 能力分转过程分（2026 收尾语）。

用法提示：每条链首问 30 秒、中段 15–30 秒一层、收尾落一句"区分度语句"；链内任一环展开深度由面试官追问决定，不主动灌满。

---

## 三、六大硬核专题精讲（要点 + 公式，每题 150–300 字）

**通用答题骨架（六专题共用）**：先一句定义/公式（让面试官抓到锚点）→ 再给机制或推导链（允许被打断，链条清晰第一）→ 然后代一个**可复算的数字或可跑的例子**（锚点数字专治"背书感"）→ 最后主动收口到工程取舍（每个亮点都配一句"代价是什么"）。以下每题按"要点 → 追问 → 易错"三段组织：要点段 150–300 字必背，追问段是二次防线，易错段保底不扣分。

### 专题 A · Attention 与位置编码

**A1 Self-Attention 推导、√d_k、多头**（仓1 01 题3/4/6；08-Coding 题1/2）
 softmax(QK^T/√d_k)V。√d_k 的双解释：①方差——q·k 是 d 个独立分量内积，方差随 d 增大，除以 √d 拉回单位方差，防 softmax 饱和梯度消失；②数值稳定——饱和 softmax 输出 one-hot，梯度为 0。多头要点：参数恒等式——单大头 4d²，h 个小头每个 d²/h×3 合计仍 3d²，加 d×d 的 W^O 共 4d²，**参数与 FLOPs 均不变**；收益是 h 个低秩子空间集成（句法/位置/induction head 有实证），代价是 KV cache 随头数线性膨胀（2·n·d/层），这是 MQA/GQA/MLA 的出发点。经验：d_k≥64 再谈多头收益（LLaMA h=32/d_k=128）。
- 追问：为什么 W^O 是 d×d？（concat 后一次头间混合，数学上等价于 h 个独立 d×(d/h) 的 O 再 concat）；h 太少行不行？（h<4 头分工不足同样掉点）。
- 追问：多头与 LoRA 的关系？（多头本质=并行低秩子空间，LoRA 是逆境版低秩增量——同源思想）。
- 易错：说多头"防过拟合/增参数"当场扣分；参数恒等式当场算不出很掉分。

**A2 RoPE 与长上下文扩展**（仓1 01 题8/9；07 题7）
 RoPE 把 q,k 按 token 位置旋转（复数视角 q_m◦e^{imθ}），内积天然只依赖相对位置 m−n——绝对位置编码的形式、相对位置的语义。外推问题：超出训练长度后旋转角"跨界"，主流解法是位置插值/NTK/YaRN 缩放（2025-11 口径：GPT-5.1 API 400K、Gemini 3 Pro 1M、Claude 200K；Qwen3 32K 原生 YaRN 扩至 128K，"256K 仅部分新型号"打存疑标）。答题钩子：MLA 特殊在于 RoPE 不可压缩（位置项从低秩潜码恢复不出来），必须把 rope 部分单独存——把 A2 和 B3 串起来答是加分结构。
- 追问：ALiBi 呢？（把线性 bias 直接加在 attention logits 上、外推靠斜率惩罚，主流 LLM 仍选 RoPE+缩放、因为可插值且不牺牲近程表达）。
- 追问：长上下文的三大工程账？（attention 计算 O(n²)、KV 显存线性、context rot 利用率衰减——第三条接 E 组上下文工程）。
- 易错：混淆"外推（不训练硬撑）/插值（缩位置)/训练扩展（续训长文本）"三条路径的前提与代价。

**A3 Norm / FFN / Decoder-only**（仓1 01 题10–17；仓2 architecture 题02/07/09）
 Pre-Norm vs Post-Norm：Pre 把归一化放在残差主干上，梯度路径无衰减，深模型可训性好——现代大模型全 Pre；Post-Norm 原版有效但需 warmup 精调。RMSNorm 去掉均值中心化，只做方差缩放，省一次 reduce 仍稳。SwiGLU：门控线性单元，d_ff=8/3·d 时参数量与 4d 的标准 FFN 等价（LLaMA-7B d_ff=11008）。Decoder-only 胜出的四点：因果 mask 下每个 token 都出新训练信号（稠密监督）、结构与目标天然统一、KV cache 复用前缀利于推理缓存、工程简单可 scale。
- 追问：FFN 中间维度为什么常是 4d？（经验最优+SwiGLU 等参换算后 LLaMA 系列取 8/3·d 系列）；4d 是铁律吗？（MoE、优化过的小模型有偏离）。
- 易错：答 Pre-Norm 好却讲不出"梯度经过 norm 不经过原始主干"的路径差别。

**A4 MoE 与前沿架构**（仓1 07 题1/3/4；仓2 architecture 题14）
 MoE 把 FFN 替换为 N 个专家 + router（top-k 稀疏激活）：总参数巨大、每 token 激活小——"容量与算力解耦"。训练三大挑战：负载不均（需要 aux loss 或 DeepSeek-V3 的 aux-loss-free bias 调节）、专家崩塌、all-to-all 通信与显存。DeepSeek-V3 三件套：MLA 注意力 + 细粒度专家/共享专家 + 无辅助损失负载均衡；R1 = V3 基座 + GRPO/RLVR。Test-Time Compute（o1/R1）：把算力从预训练搬到推理时采样与验证，长 CoT 是其可见形态。VLM 一句话：ViT patch 嵌入 + projector 对齐进 LLM 词嵌入空间（仓2 multimodal 01/05/06 有结构图）。
- 追问：MoE 推理部署的额外账？（专家并行 all-to-all 延迟、专家显存不均衡、动态 batch 分桶）。
- 追问：o1 类模型对 serving 的冲击？（输出长度不可预估→调度与预算策略要变，link 到 B8）。
- 易错：把 MoE 说成"减少参数"——是减少**激活**参数，总参数反而在涨。
- 追问：细粒度专家为什么配共享专家？（通用计算沉淀在少量共享专家，避免细粒度化后常识被切碎）。

**A5 分词与词表专考**（仓1 01 题18–24；08 题4）
 BPE 训练=按词型计数找最高频相邻对贪心合并（词频加权样例：low×5/lower×2/lowest×2 时第 3 次合并变为 low+</w>）→推理=预查表切分，O(长度)。WordPiece 用似然增益选合并、带 `##` 前缀标续词；Unigram 反向（从大词表按概率剪枝），中文=byte-level BPE 兜底生僻字。词表大小 32k–128k：大词表压缩率高（省 token、省上下文）但 embedding 占参数比上升、稀有词欠训练。中文大小换算：中文约 1 token≈1–1.5 字、英文 1 token≈0.75 词——报价与上下文估算必用。
- 追问：为什么中文模型常更大词表？（汉字没有空格切分、字节级回退代价高）。
- 易错：BPE 合并列展示时忽略"并列时取计数排序先插入者"这类实现细节（仓1 有可跑代码）。

---

### 专题 B · KV Cache 与推理优化

**B1 KV Cache 原理与边界**（仓1 02 题1）
 合法性关键：因果注意力下 k_j、v_j 只依赖 x_j 本身，与谁在 query 无关，跨步不变 → 每层缓存 K/V，每步只算新 token 的 q/k/v 追加。算账（单层单头、历史长 t）：无 cache 每步重算历史投影 4td + 当前 6d + 注意力 4td；有 cache 剩 6d + 4td——**省掉的是 O(td) 重复投影，省不掉的是对历史 KV 的注意力读取**，后者随上下文线性增多 → decode 是 memory/带宽 bound，这也是"KV cache 是不是越多层越糟"的答案。两阶段形态：prefill 并行建缓存（compute-bound，GEMM）、decode 逐 token 读缓存（bandwidth-bound）。
- 追问：训练时需要吗？（不需要，teacher forcing 一次并行算完——KV cache 是**推理态**概念）；Q 为什么不缓存（历史 Q 已消费完）。
- 追问：缓存 RoPE 前还是后？（常规缓存**已旋转**的 K；MLA 是例外——内容/位置解耦）。
- 易错：说"KV cache 加速了注意力计算"——FLOPs 没减，减的是重复投影；说 prefill 靠它加速——prefill 的加速点在 GEMM 并行度。

**B2 显存公式与三锚点**（见 1.2(1)；仓1 02 题2）
 白板流程：写出 KV=2·b·n_kv·L·s·d_head·bytes → 代三锚点（7B-MHA 2.0GiB/70B-GQA 1.25GiB/MLA 0.268GiB @4k）→ 反推容量规划（80G 卡 70B INT4 约 30 路并发）。追问预演：为什么有 2（K+V 各一份，答 QKV 三份出局）；INT8 KV 能省一半吗（显存能，但数值范围随位置漂移，动态量化难，要 per-channel scale）；batch 与 KV 线性 → batch 上限由 KV 预算决定。GiB/GB 混报是高频扣分点。
- 追问：报数时必说的口径三件套——二进制 GiB vs 十进制 GB、d_model 与 n_kv·d_head 在 GQA 下不等、MLA 的 576 是"每层每 token 总量"不是每头。

**B3 MHA→MQA→GQA→MLA 演进**（仓1 01 题7；仓2 architecture 题04）
 见 1.2(6)。追问预演：MQA 质量为何差（所有 Q 头挤同一 KV 子空间，头分工被强制同态）；GQA 的 uptraining（已训 MHA checkpoint 上把 K/V 权重 mean-pool 到 g 组，免重训）；prefill 受益吗（省写入但 prefill 是 compute-bound，收益占比不同）；MLA 还能再压吗（可叠 KV 量化但低秩已压去冗余，收益空间小）。DeepSeek-V3.2 的 DSA 稀疏注意力是这条线的 2025 续章（调研报告表五）。
- 追问：MQA 是 GQA 特例吗？（是，g=1）；为什么 GQA 论文强调"质量在实验误差内、显著优于 MQA"（g 份 KV 头保留了表达多样性）。
- 易错：把 MLA 的"低秩"与 LoRA 混为一谈（前者压缩 KV 缓存、后者低秩参数增量——思想同源、对象不同）。

**B4 Flash Attention / PagedAttention / Batching / 投机解码**（仓1 02 题4/5/6/7）
 FlashAttention：IO 感知分块 + online softmax 递推，中间矩阵不落 HBM，显存 O(n) 且少读写——**计算量没减，减的是显存访问**，长序列省得多。PagedAttention（vLLM）：把 KV cache 按 block 分页管理，消内部/外部碎片，同预算下并发与吞吐大涨，允许 prefix 共享。连续 batching：请求级动态进出，不再整批等待最长序列。投机（speculative）解码：小草稿模型起草、大模型并行验证，接受率决定加速倍数，**数学上分布不变**。追问：Flash 与 Paged 不冲突——一个优化算子内访存、一个优化管理器层显存。
- 追问：投机解码什么时候失效（草稿-目标分布差大→接受率低；高并发 batch 已满载时验证拆分反而亏）。
- 易错：把 Flash 说成"近似/加速数学"（精确计算，只是重排访存）。

**B5 解码策略**（仓1 02 题9–13）
 greedy（确定、重复）；beam（离线翻译好，开放式生成钝化）；temperature 缩放 logits（T→0 贪心、T 大发散）；top-k 截断固定候选数、top-p 截断动态概率质量——两者常叠用；重复惩罚（frequency/presence）治复读。Agent 场景口径：工具调用温度调低、JSON 输出用约束解码保证格式。
- 追问：top-p 对长尾分布为什么更鲁棒（保留概率质量而非固定个数）；beam 为什么伤多样性（似然高≠人类偏好，文本退化）。
- 易错：把 beam 说成开放式生成默认——现代对话/agent 一律采样系。

**B6 量化方法**（仓1 02 题14–19；仓2 inference 题04/06）
 PTQ（训后）vs QAT（训中插伪量化）。GPTQ：逐层基于二阶信息（OBS 思想）逐列量化并用 Hessian 误差补偿，GPU 侧离线长但结果好。AWQ：观察"激活大的权重通道更重要"，per-channel scaling 保护 salient 通道，编译期一次完成、推理 kernel 简单。GGUF：llama.cpp 消费侧统一格式。尺寸估算：13B FP16≈26GB → INT8≈13GB → INT4≈7GB 量级（仓2 06 题有逐位算例）。
- 追问：GPTQ 与 AWQ 的哲学差别（GPTQ=权重误差最小化；AWQ=激活分布保护；现役部署常 GPTQ 起家后失败场景用 AWQ 类混合）。
- 易错：INT4 估算漏 group-scale 开销；KV 量化忘说"动态范围漂移"。

**B7 KV Cache 减少手段="砍公式因子"**（仓1 02 题3）
 答题框架：公式 KV=2·L·n_kv·d_h·s·bytes 的每一项对应一类手段——n_kv（GQA/MLA，架构层）、s（滑动窗口/流式丢弃/context 压缩）、d_h 与存储形态（MLA 低秩潜码）、L（跨层 KV 共享/YOCO 类思路）、dtype（KV 量化 INT8/FP8）。先说瓶颈（decode 带宽+显存容量），再按因子列菜单，最后补一句取舍（滑窗伤远端信息、量化伤精度、GQA 是免费的）。
- 追问：交叉层共享 KV 的代价？（层间表达差异化下降；YOCO 式"一层管缓存"是激进版）。

**B8 部署架构与延迟拆解**（仓1 02 题8/20；09 题1；仓2 题02/05/07）
 端到端三层：网络（网关/排队）→ 推理（prefill compute-bound 大 GEMM；decode bandwidth-bound，时间≈每 token 从 HBM 搬运权重+KV 的字节 / 带宽）→ 后处理（引用/审核/流式）。并行：TP 层内（NVLink 通信少、层内 communication 开销）、PP 流水（bubble 谨慎、吞吐取向）。框架一句话：vLLM 通用吞吐、SGLang 结构化输出+缓存调度、TRT-LLM 极致 kernel、Ollama 个人本地、DeepSpeed-Inference 生态融合。超长上下文（仓2 07 题）：先外置检索/滑窗/分层摘要，再谈"硬塞 1M"。
- 追问：TP 拆几卡怎么定？（层内通信 vs 权重切分粒度；从单卡放权重+KV 预算起推）。
- 追问：chunked prefill/prefill-decode 分离解决什么？（长 prompt 与短 decode 抢资源 →分别批量调度；2025 主流 serving 都在做）。

---

### 专题 C · PEFT 与对齐（RLHF → PPO → DPO → GRPO）

**C1 LoRA / QLoRA**（仓1 03 题1–4）
 冻结 W，学低秩增量 ΔW=BA（A∈R^{r×d} 高斯初始化、B 零初始化保证起步恒等），只训 ~0.5–1% 参数；合并进权重零推理延迟。r 选择：任务与目标模块复杂度定，常用 8–32，"宁小勿过拟合+多模块覆盖"优于单模块大 r。放哪些层：attention 投影（q,k,v,o）优先，效果饱和后再加 FFN。QLoRA：4bit NF4 冻结底座 + 低秩适配器反传 + 双量化（对量化常数再量化）+ paged optimizer，单卡微调 65B 的工程方案。
- 追问：LoRA 什么时候明显输给全参（复杂推理/分布外；可加大 r、加目标模块、lora_plus 分层 lr 缓解）。
- 易错：A/B 初始化写反（B 非零则第一步就扰动原模型）；说 LoRA"减少推理延迟"（合并后无延迟，未合并有 lossless 合并一说，注意口径）。

**C2 PEFT 对比**（仓1 03 题5；仓2 training-finetuning 07）
 Adapter 串行插入（推理有延迟）；Prefix/Prompt Tuning 把连续向量放输入端（占上下文、训练不稳但零改模型）；LoRA 等价于受约束的全参微调、性价比最高。LoRA 与 MLA 的低秩共享思想同构、与多头 W^O 的 rank-约束远亲——串场这条线是高分装饰。
- 追问：Prefix Tuning 的 token 预算要在哪个层面算（它占的是**上下文窗口**，不是权重——上下文工程视角的 PEFT）。

**C3 RLHF 三阶段 + PPO 四组件**（仓1 03 题6/7；仓2 rlhf 题6/8）
 三阶段：SFT 示范 → RM 偏好数据（Bradley-Terry 成对排序学标量奖励）→ PPO 优化。PPO 四组件：actor（策略）、critic（GAE 价值估计）、ref（冻结锚点算 KL）、RM（打分）。KL 常进 reward（−β·KL(π_θ‖π_ref)），critic 与 actor 同尺寸是显存主要负担。LM 状态空间是整段序列 → 价值回归天然难学、常震荡——这句直接引出 GRPO 动机。
- 追问：RM 训练损失长什么样（对 σ(r_w−r_l) 的负对数似然——与 DPO 损失同源，DPO 是把 RM"藏"进策略）。
- 易错：说"四个模型都训练"（实际 actor+critic 可训练、ref/RM 冻结）。

**C4 DPO 闭式推导**（见 1.2(3)；仓1 03 题8）
 关键三步不改：Z(x) 在 BT 差值中相消（配分函数定义在指数级序列空间上本不可算）；π* 重参数化为可训练 π_θ 的合法性（隐含奖励 r̂=β·log(π_θ/π_ref) 的序即真奖励的序）；β 的温度双重角色。追问预演：为什么无采样无 critic（把 RL 搜索变成对闭式解的回归）；离线局限（分布离开 π_ref 支撑即失效 → 在线迭代 DPO/KTO/SimPO 是缓解谱系）。
- 追问：DPO 的隐式难例加权是什么（梯度权重=σ(r̂_l−r̂_w)，错序越狠学得越狠→标注噪声也被"更用力学"，这是它对数据质量敏感的机制根源）。
- 易错：说 DPO 是"奖励模型损失的变体"——训练对象是策略本身。

**C5 GRPO 与 R1 谱系**（见 1.2(4)；仓1 03 题10/11/12）
 一句话："用同组采样算基线，critic 就失业了"。对比表速记：PPO=四模型+GAE+KL in reward；GRPO=actor+ref 两模型+组内 z-score+k3 独立 loss；DPO=两模型+纯监督 sigmoid。方差账：组均值误差 O(1/√G)，G 4→64 误差减 4 倍；RLOO 用 leave-one-out 消自包含偏差。DAPO 勘误（仓1 题12）：**Decoupled Clip and Dynamic sAmpling PO（ByteDance Seed, arXiv:2503.17002），属 GRPO/在线 RL 谱系**，原文将其写进 DPO 系是错的；它的 token-level 归一与动态采样分别治长回答梯度稀释与全同零方差组。
- 追问：GRPO 省的是什么资源（省训练侧一个每 token 反传的价值网；采样侧反而 ×G，rollout 占总时间 70–80%）。
- 追问：为什么 R1 后期敢 β=0（长 CoT 需要大规模偏离 π_ref，KL 反而束缚探索）。
- 易错：把 GRPO 写成"无 KL 的 REINFORCE"（原始 DeepSeekMath 明确带独立 k3 项，版本口径要讲清）。

**C6 对齐选型与微调常识**（仓1 03 题14–18；仓2 rlhf 题1/3/7）
 选型三角：有过程奖励/有 RM/算力充裕 → PPO；长 CoT+可验证规则奖励+中小集群 → GRPO(+DAPO)；只有成对偏好数据/不想搭采样管道 → DPO 系。on-policy vs off-policy：DPO 是离线（数据定死上限），在线迭代 DPO 用自家策略采样重标注。SFT vs 预训练：损失同（next-token）但数据分布异——指令负载、SFT 后置信度变化（温度与分布收窄）。数据质量>数量：清洗去重多样性比堆条数重要；灾难性遗忘对策=混入通用数据/低学习率/LoRA 天然隔离。
- 追问："微调能不能解决知识过时"？（不能——外部知识交给 RAG/工具；微调教**格式与行为**）。
- 易错：把"奖励塑形"答成"改 R 网络输出"（稀疏/延迟奖励的塑形=势函数/过程奖励/curriculum，仓2 rlhf 题05）。

**C7 特殊场景：多轮对话与 FC 的对齐设计**（仓1 03 题13/16/17）
 ChatML 三角色（system/user/assistant，tool 结果进 user 位），多轮拼接训练时 **loss 只在 assistant 段反向传播**（mask 掉 user/tool token）。Function Calling 的奖励函数设计（题13）：结构合法率（schema 通过即给分）→ 参数正确性（与 ground-truth 工具意图对齐）→ 任务完成（多轮终态）→ 效率惩罚（步数/token），逐层加权；格式奖励是 GRPO 里的可分解分项（正确性+格式）。
- 追问：多轮里中间轮次该不该学（常只学终轮或全轮按衰减——各家口径不同，答"声明选择+理由"即加分）。
- 易错：把 FC 对齐说成"教模型写 JSON 就行"（执行成功与否才是奖励主体）。

---

### 专题 D · RAG 全链路

**D1 三阶段与失败分布**（仓1 04 题1；仓2 rag 题03/05/06）
 阶段一索引（解析→切块→嵌入→入库）、阶段二检索（改写→召回→融合→粗排→精排）、阶段三生成（注入→引用→生成）。RAG 解决的三个原始问题：知识截止、私有知识、幻觉溯源。失败主要分布在**检索与生成接口处**（仓2 引 Gao et al. 2023 综述结论）——诊断要按 query/chunk/模型三层分开归因，这是评测专题 D6 的伏笔。
- 追问：RAG vs 长25上下文直塞（成本/失焦 context rot/可更新性三段论——1M 窗口也没让 RAG 失业）。
- 追问：RAG 意图识别为什么转向小 LLM（成本+低延迟，路由是分类不是生成——2026 口径，仓2 纠偏）。
- 易错：把 RAG 说成"仅向量检索"（漏后处理、生成约束与评估闭环）。

**D2 Chunking 与 Contextual Retrieval**（仓1 04 题2/3；仓2 rag 文档处理 题01）
 策略：固定窗口、递归分隔、语义切分、滑动窗口重叠。权衡：块大→召回粗、上下文预算压力；块小→上下文碎、语义断。升级路径（2026 口径，实测数字以 Anthropic 官方为准）：**Contextual Retrieval**——离线用 LLM 给每块生成 1–2 句"它属于全文哪里"的位置语境再检索：top-20 失败率 5.7% → 3.7%（仅上下文嵌入，−35%）→ 2.9%（+BM25，−49%）→ 1.9%（再加重排，−67%）；prompt caching 把逐块成本压到 ~$0.1/1M-token 级。注意"检索质量提升 5 法"实为**分桶后的 6 手段**（仓2 纠偏口径）。
- 追问：块 512 还是 1024 起？（中文 512–1024 token 常见起点，用 NDCG/Recall@k 网格调，别背结论）；语义切分的真实收益场景（结构散乱的会议/法务长文）。
- 易错：引 Contextual Retrieval 不说"相对降幅"口径（漏掉 5.7% 绝对基线）。

**D3 混合检索与索引**（仓1 04 题4/5；仓2 vector 9 题）
 BM25 = IDF × tf 饱和(k1) × 长度归一(b)，逐项与 TF-IDF 的区别（饱和与长度惩罚）；dense 管语义、BM25 管字面精确（型号/编号/缩写），错误模式正交 → RRF 融合（公式见 1.2(5)）。ANN：HNSW 分层跳表图贪心导航 O(logN)，内存贵；IVF-PQ 用量化压向量（400GB→6.4GB 量级）牺牲精度换内存；删除在 HNSW 是逻辑标记+压实（仓2 纠偏）。Embedding 升级后旧索引向量与新查询向量不同空间 → 全量重建或双索引过渡（仓2 题07）。
- 追问：一亿向量选索引（内存富余高 QPS → HNSW；内存敏感 → IVF-PQ/HNSW+PQ 混合）；增量索引怎么更新（upsert+周期压实+墓碑标记）。
- 追问：归一化为什么是 dense 检索的生命线（cos 的分母假设单位球；RRF 免疫但分数级融合不免疫）。
- 易错：rank 从 0 数起；把 RRF 分数当"语义相关度"解释。

**D4 重排与分层排序**（仓1 04 题7；仓2 rerank 5 题）
 召回求全（recall）、粗排求快（轻量打分削候选）、精排求准（cross-encoder 交互算 q·d 相关性）。**为什么不能跳过粗排**：精排是 O(候选数×交互计算)，百级候选可以、万级不行——召回/粗排是给精排"省算力"的前置漏斗。reranker 训练数据：hard negatives 挖掘（同主题不相关样本）比随机负例有效一个量级。多信号直接加权的陷阱：分数量纲不可比必须先归一化/学习排序（仓2 纠偏："直接加权"需同尺度化）。
- 追问：cross-encoder 为什么比 bi-encoder 准？（交互在 token 级可见，不必把全部语义压进单向量）；代价呢（不可预计算、必须在线打分）。
- 易错：跳粗排的"省钱"论证（省了精排算力，赔了排序质量与规模上限）。

**D5 查询改写与 Agentic RAG、GraphRAG**（仓1 04 题6/8/9；仓2 vector KG 2 题）
 查询改写族：同义扩写、HyDE（先造假设文档再检索）、多路拆解、意图路由（2026 口径：路由已转向小 LLM 分流）。Agentic RAG：检索成为 agent 的可决策工具——模型决定"再搜什么、搜几轮、信不信"，多跳问题（HotpotQA 式）是主战场。GraphRAG：实体→关系→社区摘要三级图， 回答全局性问题优于纯向量；构建挑战=实体消歧、关系抽取质量、更新成本；Zep/Graphiti 的时序知识图谱给事实加了有效期双时间线。
- 追问：Agentic RAG 的护栏（最多 N 跳、每跳预算、自评阈值、超越则降级普通 RAG）。
- 易错：把 GraphRAG 说成"更大的向量库"（拓扑+社区摘要+全局问答才是本质差）。

**D6 RAG 评估与错误定位**（仓1 04 题10；仓2 eval 题1/5/8）
 指标族：检索层 Recall@k / MRR / NDCG（公式与算例见 1.2(7)；交叉口径：文献 log2(i+1)/log2(i+2) 并存）；生成层 Faithfulness（答案是否被检索支持）与 Answer Relevance。错误定位三层递进：先看检索召回有没有 gold（没有 → 改 chunk/embedding/改写），有 gold 但排序低（→ 重排/融合），检索对了答案错（→ 生成端 prompt/上下文裁剪）。一页图：NDCG 评排序、MRR 评首位、Faithfulness 评生成——三把尺各管一段。
- 追问：知识型 vs 行为型错误的诊断岔路（仓2 eval 题06：知识错→检索/知识库层；行为错→指令与权限层）。
- 易错：只评检索不评生成——Faithfulness 缺位时幻觉漏网。

**D7 生成端约束与 RAG 监控闭环**（调研第五章；仓2 eval 题02/07）
 生成端三约束：只用检索支持的内容（"检索到的当数据不当指令"）、无证据即拒答/降级、引用可回指段落级。在线监控四件套：Faithfulness 抽检、引用覆盖率、用户反馈回灌成 hard negative 库、语义缓存（同义请求复用，注意阈值治理防错误缓存放大）。
- 追问："宁可不答不可错答"怎么制度化（拒答阈值+兜底话术+人工升级通道——评估函数里把不答错答分开计费）。
- 易错：把 RAG 监控做成"只有编辑率"（缺错误归因维度，改进无从下手）。

---

### 专题 E · Agent 六件套（loop · 工具 · 记忆 · 子代理 · MCP · 评测）

**E1 Agent Loop**（仓1 05 题1；仓3 s01）
 最小实现：`while stop_reason=="tool_use"`；四要素=模型决策/客户端执行/观察回填/终止三分类（模型自停、预算步数、人工中断）。工程必须项=max_turns 与预算（纯模型自停教学可以、生产不行）。链式衰减算术（0.95^20≈36%）是我方在 loop 题上的"必带数字"。追问"为什么 tool_result 回 user 角色"：API 角色约束 + 语义上观察=环境给用户世界的新输入（ReAct 的 Observation 槽位）。

**E2 工具系统（Function Calling 工程面）**（仓1 05 题3；仓3 s02）
 全链路：schema 声明 → 模型产出 name+args → harness dispatch 执行 → 结果回对话。三条铁律：①执行永远在客户端；②约束解码保证格式；③**错误是给模型的输入不是给用户的崩溃**（未知工具回 "Tool unknown + 可用清单" 让模型自纠，handler 异常原地捕获转文本）。工具描述=API 文档，描述质量直接决定调用准确率——它的权重高于换模型。路径类工具必须有 resolve+is_relative_to 白名单，并"在每个产生路径的入口各校验一次"。

**E3 记忆系统**（仓1 05 题6；仓2 agent-prompt 题02；仓3 s09）
 短期=上下文窗口+scratchpad；长期=外存三方案：mem0（抽取式四操作 ADD/UPDATE/DELETE/NOOP，Apache-2.0）、Letta/MemGPT（核心+归档分层自编辑，.af 格式）、Zep/Graphiti（时序知识图谱双时间线，事实有有效期）。工程四件套 store/recall/consolidate/forget；隐藏考点=遗忘与冲突合并。行内实现要点（s09）：持久性守门（必须显式 scope=persistent + 临时词表拒收）、索引常驻正文按需 + 字符预算、"召回内容与当前请求冲突时当前请求优先"（防记忆侧注入）。记忆是选择性存储，不是 transcript 备份，也不取代压缩。

**E4 子代理与多智能体**（仓1 05 题7/8；仓2 agent-prompt 题05；仓3 s06/s13）
 决策框架（考点 9 原文口径）：并行探索、上下文隔离、异构分工 → 上多 agent；代价三件套=成本倍增、失败级联、handoff 信息丢失。第一档是 subagent（一次调用即焚：全新 messages、只回传最终文本、工具表剔除 task 杜绝递归委派、hooks 照常生效）；第二档是持久团队（s13）：文件信箱总线+破坏性读+Condition 唤醒、原子认领（检查-修改-落盘一锁全包）、plan gate 必须落在**工具分发层**（"协议要生效靠运行时，不靠系统提示词"）、worktree 是目录隔离不是沙箱（网络/资源另算）。冲突解决=任务状态板只传状态、每队友独立上下文。

**E5 MCP**（仓1 05 题4/5；仓3 s14）
 2024-11-25 Anthropic 开源，一年内成跨厂商事实标准（"Agent 生态的 USB-C"）。三层：host（agent 应用）/client（连接代理）/server（工具提供方）；三类原语：tools/resources/prompts。解决 M×N：每个应用对每个工具两两写适配 → 各自实现一次 MCP 即互认。与 FC 互补线：MCP 是**分发标准**（工具上架），FC 仍是**模型层调用机制**；MCP 工具最终照样走 tool_use/tool_result。host 四步（s14 面试主答案）：发现 → `mcp__server__tool` 前缀命名空间+归一化冲突检测（64 字符上限）→ **授权只来自宿主配置**（readOnlyHint 等声明只是提示，deny-by-default）→ 并入统一工具池每轮重建。追问"两个 server 同名工具怎么办"：前缀隔离+冲突检测，宁可拒绝连接不可静默错配。
- 追问：传输层教学版为什么可信（进程内 mock 保留 inputSchema/annotations 真实字段，host 职责与传输解耦——换 stdio/SSE 不改答案）。

**E6 Agent 评测**（仓1 06 题5/6；仓2 eval 全模块；调研考点 11）
 基准分层选型：编码 SWE-bench（Verified=500 题人工校验；轨迹必背：2023-10 原始集 GPT-4 ~2% → 2024-08 Verified 发布时 33–49% → 2025 中 ~70% → 2025-11 **Gemini 3 Pro 76.2 / Codex-Max 77.9 / Opus 4.5 80.9**）；通用助理 GAIA；OS GUI OSWorld；网页 WebArena；工具+策略 τ²-bench；注入攻击 AgentDojo。自有评测三条：任务集带自动验证信号、回归集回放、LLM-as-judge 人工校准（自我偏好/位置偏差/一致性三病，缓解=pairwise+交换一致+BT 强度拟合）。分数只作初筛——这是 2026 区分度语句。
- 追问：agent 轨迹能交给 LLM-judge 逐步打分吗？（易被"看起来努力"带偏——用终态 outcome+过程抽查双层）。
- 易错：拿基准分外推私域表现（仓库清洁度/私有依赖会让分数优势消失）。

**E7 框架对比的答法**（仓1 05 题9；调研表六）
 LangGraph=图/状态机编排，checkpoint 与 interrupt 人审是一等公民（生产可控性最强、学习曲线陡）；AutoGen=actor 事件驱动多代理会话（研究范式广、生产要补基建）；CrewAI=角色化 Crew+Flow 双模式（上手最快、深控弱）；OpenAI Agents SDK=Swarm 正式化：handoffs/guardrails/session/tracing（轻盈但绑生态）。共识句收束：多智能体框架以"**编排确定性+模型智能**"为设计共识；多 agent 的成本与失败级联显著高于单 agent。
- 追问：什么情况写死 if/else 反而是正解（路径已知+可靠性要求高=workflow；考点 1 的工程面）。
- 易错：只背功能表格、说不出取舍因果链。

**E8 可靠 Agent 执行器（代码面收口）**（仓1 09 题5；仓3 s15/s17）
 最小可靠执行引擎四件：重试与退避（429 指数+抖动）、预算（步数/token 双上限）、人审卡点（高危分级审批）、checkpoint（崩溃后读 ckpt.json 续跑）。s15 的分层恢复链是其工程完整版：429 → 指数退避；529 连续 2 次 → 切 FALLBACK_MODEL（顺手完成模型路由）；max_tokens 超限 → 8000→16000 升档 + continuation prompt；prompt too long → reactive compact 重试一次（限一次，估算误差用一次重试兜底）。
- 追问：checkpoint 里存什么（消息列表截断点、任务状态、未完成 tool_use 清单——恢复后要能补发 tool_result）。
- 易错：把可靠性全押"换模型"（同模型下恢复设计差异远大于模型差异）。
- 追问：崩溃恢复后第一条消息发什么（重放最后一条 user + 补齐悬空 tool_result——API 校验配对完整性）。

---

### 专题 F · 安全与评测

**F1 风险面与越狱分类**（仓1 06 题1/2）
 风险四类：有害内容、幻觉与错误信息、隐私泄露、（agent 特有）越权执行。越狱手法谱系：角色扮演与情景包装、编码绕过（base64/拼接）、同形字混淆、梯度对抗后缀（GCG, Zou et al. 2023——防御侧只用于红队与对抗训练）。幻觉二分：知识型（事实错）与行为型（该做不做/做错），探测最小原型=SelfCheckGPT 式采样一致性（多次采样比对分歧，仓1 06 题4 有可跑 sketch）。

**F2 防御纵深（分层答案）**（仓1 06 题3；调研第五章；仓3 s03/s15）
 高分结构="没有根治 + 分层"：①输入过滤——注入检测、homoglyph 归一化、（结构化）策略查表；②权限隔离——默认只读、写与网络按需临时授予、代码执行进容器/worktree 默认禁网、凭证分权与轮换；③输出检查——PostToolUse 观察与告警、敏感操作人审；④系统性——红队例行、审计日志不含明文凭证。Agent 特有强化（s15）："压缩后的历史整体降级为不可信参考，唯一可授权的是当前用户请求"；记忆侧用确定性守门 + 临时词表挡会话级指令永久化。上线九条（调研报告第五章清单）可整段背：模型零代码更换、凭证可轮换、高危人审、三层注入防线、预算与降级、审计无明文、沙箱禁网、红队闭环、一键停机回滚。
- 追问：安全钩子异常时 fail-open 还是 fail-closed（安全=fail-closed 当拦截；日志类可 fail-open——按钩子声明容错策略）。
- 易错：声称"有根治方案"——2026 高分姿态是"无银弹+分层+红队闭环"。

**F3 评测方法与 LLM-as-judge**（仓1 06 题5/6/7；仓2 eval 题2）
 评估维度：知识/推理/指令遵循/安全/agent 行为；方法二分自动（基准+规则）vs 人工（打分+pairwise）。judge 三病与对策：自我偏好（偏好自家模型输出 → 去自我偏好的 pairwise 过滤）、位置偏差（交换顺序一致化）、一致性漂移（多 judge 校准）。Bradley-Terry（仓1 06 题6 可跑代码）：把成对比较拟合为强度分数 σ(r_w−r_l)，SGD O(E·I)——LLM 评测榜（Chatbot Arena 类）的数学底座。口径注记：MT-Bench 类 2023 年的分数线只当历史锚点，不当现状。

**F4 Agent 安全与过程分**（调研考点 12/20；仓3 s17）
 Agent 更危险的根源复述一遍（考点 12）：有工具执行权 + 上下文中指令与数据不可分辩。评测转向：从"能力分"到"安全与过程分"——过程分=合规使用工具/最小权限/及时求证的轨迹质量（s16 的每条 finding 独立对抗验证是工程示例）。长时程可靠性配方（考点 20）：compaction + checkpoint + 自验证（goal gate：独立无工具判断器只读证据、no-tool-use 不等于完成）——安全答到"过程可审计"这一层即达到 2026 口径。

**F5 生产监控与 ROI 论证**（仓2 eval 题02–08；调研第六章）
 在线四指标：任务成功率、步数与 token 分布（成本画像）、人审介入率、错误根因分布（意图/检索/工具/生成四桶）。方法论收口（仓2 题07 的判断矩阵）：知识缺失→RAG；格式/行为不合→SFT/对齐；路径固定→workflow；不可验证+低频任务→不上 agent。反直觉护城河数字：METR 2025 对照研究发现有经验开源开发者用 AI **主观提速、实测慢约 19%**——收益依赖任务可验证性与样板率（样板代码/新功能收益大，熟悉域微调收益小），这是讨论"要不要上 agent"的高分区素材。
- 追问：领导问"为什么这个场景不上 agent"（不可验证且无自动信号＝失败无法翻案；基建成本＞收益）。
- 易错：只报正面数字、无成本与失败口径——追问一层即穿。

---

## 四、Harness 专题（仓3 特色：learn-claude-code）

### 4.1 论点先行："Agency 来自学，不是编"

仓3 README 的总论点，也是四份详解的叙事地基：**感知-推理-行动的能力来自模型训练，不来自外部代码编排**。历史证据链：DQN（2013，仅凭像素与分数学会 7 个 Atari 游戏→2015 扩到 49 游戏职业水平）→ OpenAI Five（2019，自我对弈 4.5 万年 Dota 2，2:0 击败 Ti8 冠军 OG）→ AlphaStar（2019，星际 II 天梯宗师段位）→ 腾讯绝悟（2019，KPL 职业选手 1v1 十五连败仅此一胜）→ 2024–2025 的 LLM 编码 Agent（同一架构：训练出的模型 + 容其行动的环境）。

四份详解的分工：s01–s05 打地基（循环/工具/权限/钩子/计划），s06–s09 解决上下文稀缺，s10–s13 升级为持久运行时，s14–s17 收官能力接入与验证——4.2 的地图与 4.3 的考法全部由此四文件支撑。

由此得出两个工作定义："构建 agent"只有两种含义——**训练模型**（RL/微调/RLHF 改权重）或**构建 harness**（给模型一个可操作环境）。Harness = 工具 + 知识 + 观察 + 行动接口 + 权限；原文金句：**模型是司机，harness 是车；模型决定，harness 执行**。反例判词：把 if-else 分支、节点图、提示词瀑布串起来"造 agent"，造出的是鲁布·戈德堡机器（Rube Goldberg machine）——process 规则流水线 + 一个装成智能节点的补全 API；**你无法用工程把智能编造出来（agency is learned, not coded）**。

与调研报告考点 19 互证：核心循环只有 30 行（s01），价值全在周边工程——上下文管理、权限、错误恢复、验证回路。大厂面试里此论点的最佳表达是三层收束："换个模型抬地板；把编排做对才抬上限；没有验证层哪儿也进不了生产。"

### 4.2 十七 stage 设计地图（一张表）

| Stage | 主题 | 一句话机制 | 命中考点 |
|---|---|---|---|
| s01 | Agent Loop | while(tool_use){执行→回填观察}；终止=模型自停/人工；50000 字符截断=上下文工程雏形 | 考点 1/2/3/19 |
| s02 | Tool Use | dispatch map 让循环零改动生长 5 个工具；safe_path 路径白名单；未知工具回文本自纠 | 考点 4/14/19 |
| s03 | Permission | 执行前三闸门：硬 deny → 规则匹配 → 人审（默认 N）；拒绝也回填 tool_result 保持循环活性 | 考点 12/13/18 |
| s04 | Hooks | 四个生命周期事件（UserPromptSubmit/PreToolUse/PostToolUse/Stop）；钩子在主路径=模型不可绕过；Stop 钩子可强制续跑 | 考点 12/18/19 |
| s05 | TodoWrite | 三态单焦点状态机（≤20 条、同时仅 1 个 in_progress）；3 轮未更就注入 reminder（nudge 不硬路由） | 考点 2/17/9 |
| s06 | Subagent | task 工具=新 messages 的同步子循环（30 轮上限、子工具表无 task、只回传最终文本）——上下文隔离的实现面 | 考点 9/6/19 |
| s07 | Skill Loading | 两级渐进披露：目录进 system、SKILL.md 全文按需进 tool_result（append-only 保 prompt cache 前缀） | 考点 6/15 |
| s08 | Context Compact | 四步管线按"损失×成本"升序：大结果转存 → 中段归档 → 旧结果缩为路径 → 才轮到模型摘要；unseen 结果保护 + 路径白名单 | 考点 6/10/12/14/20 |
| s09 | Memory | store/recall/consolidate/forget 四件套 + 确定性守门（scope 表态、临时词表、三重去重）+ 索引常驻正文预算 | 考点 7/12/6 |
| s10 | Task System | 任务外置为 JSON 文件（id/status/owner/blockedBy）；DFS 拒绝依赖环；claim 前做依赖闸门；两阶段建边契约 | 考点 7/17/10 |
| s11 | Background Tasks | run_in_background 起后台线程立刻回 bg_id 占位；完成以 <task_notification> 注入尾部 user 消息；进程组级清理 | 考点 6/2/14 |
| s12 | Cron | 表达式+prompt 注册成 job：分钟标记去重、先持久化再入队、ack 后才删——at-least-once；定时轮禁止交互审批 | 考点 1/18/10 |
| s13 | Agent Teams | Lead+持久队友：文件信箱总线（破坏性读）、原子认领、plan gate 落在工具分发层、worktree 目录隔离 | 考点 9/13/10 |
| s14 | MCP Plugin | host 四步：发现→mcp__server__tool 命名空间+冲突检测→宿主侧授权（不信 server 自述）→并入池每轮重建 | 考点 5/4/13/12 |
| s15 | Integrated Harness | 13 个机制挂同一个 while True：事件进位图 + 四级压缩闸门 + 分层错误恢复（429 退避→529 换模型→超长缩扩→reactive compact） | 考点 19 全章证据 |
| s16 | Workflow Runtime | 一次 tool_use 启动脚本化编排（agent/parallel/pipeline/phase）；每步 agent() 结果按**语义 key**记 journal → resume 全缓存命中 | 考点 1/9/10/11/14 |
| s17 | Goal Loop | return 边界挂 Stop-hook 门：独立无工具判断器读 transcript 证据判 ok/impossible → 不满足写回缺失理由继续 | 考点 20/2/17/10/18 |

### 4.3 核心机制的面试考法（按机制组收束）

**答题总模板（背这一段就够起手）**：①一句话机制＋它改变了什么；②源码级实证（一处变量名/行为级细节："sXX 的 code.py 里……"）；③裁撤实验（"删掉它会怎样"+观察过的后果）。仓3 四份详解里每个 stage 的"【3.设计权衡】"与"【5.动手实验】"就是按这个模板组织的现成弹药；面试官任追一层，退解释放深度、进解释放经验。

**组一 · 骨架（s01/s02/s04）——"控制流由谁决定"**
考法：给一段"agent 框架"，问哪一行是 agent 的本质。答法：找 `while` + 判"下一个 tool_use 是否存在"那一行；模型输出决定走不走、往哪走。扩展：s04 的不可绕过性靠 `trigger_hooks("PreToolUse", block)` 写在主路径——不是模型可选服务；Stop 钩子返回字符串即把 user 消息塞回、`continue` 强制续跑。
- 追问：为什么观察回填 user 角色（API 角色约束+Observation 语义）；未知工具怎么处理（回"Unknown+可用清单"让模型自纠，-handler 异常必须原地捕获转文本否则击穿 REPL）。
- 追问：钩子异常放行还是拦（fail-closed/open 按钩子性质声明）；Stop 强制续跑怎么防死循环（force 计数预算）。
- 串答钩子：s02 dispatch map 的"加 5 个工具循环零改动"可以直接作为"harness 质量>模型更换"的实例证据。

**组二 · 上下文四解法（s06-s08 主线）——"上下文不够用怎么办"**
必背连答链：**隔离（躲）→ 按需（等）→ 压缩（挤）→ 外存（搬）**。s06 用另一段上下文躲开污染探索；s07 用目录+按需加载延迟注入（为什么进 tool_result 不改 system：append-only、缓存前缀、可审计）；s08 四步压缩（先确定性整理、摘要只做最后手段；切点避让 tool_use/tool_result 配对、unseen 保护防占位符幻觉）；s09 搬进外存活过压缩与重启。追问"子代理 vs 压缩二选一"——可再生细节宜隔离、不可再生推理脉络留主循环。
- 加分细节：s06 子工具表剔除 task 防递归委派（多层嵌套=无界成本嫁接）；s08 的字符估算+reactive retry-once 是"精度换吞吐"的标准取舍；两级披露中 description=路由广告（load 错的根因十有八九在 description）。

**组三 · 状态与外部事件（s05/s09-s12）——"状态怎么活过上下文"**
TodoWrite 证明"计划归模型、催更归 harness"（三态单焦点+3 轮 nudge）；任务系统证明"状态出对话才可恢复"（compact 吃不掉文件，两阶段建边契约应对"同批 tool_use 拿不到同伴 ID"）；s11 的隐藏主角是 **tool_use:tool_result 1:1 不变式**——后台结果不能复用原 tool_use_id 二次回填（会 400），所以走独立通知事件并合并进尾部 user 消息（还天然命中 prompt 缓存前缀）；进程组清理（start_new_session+killpg+atexit）防孤儿；s12 证明"编排确定性与模型智能的分界线"（cron 匹配/去重/持久化是确定性代码，进 loop 后才是模型的事）+ 自动触发轮**只许拒绝危险命令、不许替用户盲批**；at-least-once 交付=先持久化 pending_delivery 再入队、ack 才删、API 失败整段回滚。
- 追问：cron 用进程内线程还是系统 crontab（进程死=调度死 vs 零依赖+同进程锁模型；跨天补跑交给上层编排——LangGraph/CI 定时器/IM 触发）。
- 追问：claim 为什么原子（乐观发现+悲观提交；两步 scan→claim 把竞争点收敛进锁区）。

**组四 · 协作与编排（s13/s16）——"为什么要编排确定性"**
s16 是考点 1 金句的机制化：模型只提交 name/args/resume_from_run_id 三参数、不提交代码（最小攻击面）；parallel 等齐屏障 vs pipeline 无屏障；journal 语义 key=`sha256(kind|label|prompt|schema)`——**跨进程稳定**，换成内建加盐 hash() 则 resume 全 miss（仓3 有可跑反例实验）。与 LangGraph 对照：journal≈checkpoint、resume≈时间旅行、Budget≈guardrails。
- s13 三把锁补充：文件信箱总线（破坏性读+Condition 唤醒不空转）、原子认领（检查-修改-落盘一锁全包）、plan gate 落在工具分发层（system prompt 只是沟通不是约束——被注释掉的 plan gate 实验里队友第一个 bash 就真实执行）。
- 追问：worktree 里还会冲突吗（目录隔离≠资源隔离：共享任务板/端口/公共文件；完整答案=目录隔离+资源配额+共享文档锁）。

**组五 · 验证层（s17）——"怎么判定干完了"**
s17 主张：**模型提出停止，harness 决定是否允许**。机制：no-tool-use 只证明"本轮想停"；独立判断器**无工具**（不给旁路执行面、与 worker 自利倾向隔离）读 transcript 中的证据（`exit_code=1` 前缀是机器可读证据——删掉它判断就退化到猜文本，仓3 有实验）；六分支状态机 allow/defer/error/achieved/failed/block，连续 block 8 次上限交还用户且不伪装成功。证据契约挑明：判断器可靠性=对话证据密度，worker 侧 system 明确要求把具体结果写进对话。
- 追问：为什么判断器无工具（分层——真验证由 worker 执行，gate 只判"验证结果是否已写进工作记录"；安全——不再开旁路执行面与注入面）。
- 追问：为什么不定死最大轮数（完成性每轮刷新；固定 N 轮既杀合法长任务又不能证明完成；预算只做防失控出口）。
- 串答钩子：这是考点 20"长时程=压缩+checkpoint+自验证"第三件的落地——s08 是压缩、s10/s12 是持久化、s17 是自验证，三件套在 4.2 表格里互指。

**组六 · 集成（s15）——"3330 行里没有新东西，全是共存问题"**
三级追问模型：每类事件从 while True 的哪个位置进来（cron/后台/reminders/todo 固定次序物化为 user 消息）→ 失败谁兜底（429 指数退避→529 两次切 FALLBACK_MODEL→max_tokens 升档 continuation→prompt-too-long reactive compact 重试一次，重试限一次用估算误差兜底）→ 权限 hooks 与 MCP 策略统一卡点（异步轮交互审批不可用即拒绝）。收束语：换模型抬地板，把事件编排、压缩、恢复做对才抬上限。
- 追问：为什么不 at-most-once（丢提醒是产品级缺陷，重复提醒可容忍——at-least-once 的幂等由任务 ID+`[Scheduled]` 前缀双保险）。
- 追问：subagent 为什么敢丢中间消息（隔离本来就是目的；同时省主窗口重复读文件——成本与上下文双重收益）。

### 4.4 三仓串题示例（一份答案里三仓各给一句的排法）

- **"设计一个能跑 8 小时的 agent"**：仓1 09 题5 的四件（重试/预算/人审/checkpoint）开题 → 仓2 口径补"评测与监控闭环四指标" → 仓3 收尾：s08 压缩 + s10/s12 持久化 + s17 goal gate（长时程三件套点名第 20 考点）。
- **"MCP 为什么会赢"**：仓1 05 题4 的协议三层与 M×N → 仓3 s14 host 四步（授权只来自宿主配置）→ 调研报告结论（一年内成事实标准、生态 USB-C）。
- **"RAG 怎么继续优化"**：仓1 04 的改写/混合/重排清单 → 仓2 Contextual Retrieval 35/49/67 官方数字 → 仓3 s07 观点（索引与正文分离是按需召回的前提）。
- **"多智能体翻车在哪"**：仓1 05 题8 冲突解决 → 调研表六共识句 → 仓3 s13 的三把锁（信箱/原子认领/plan gate 落工具分发层）。
- 排法要诀：仓1 给骨架（不背长文）、仓2 给数字与新口径（区分度）、仓3 给"我读过源码/我跑过实验"（可信度）——三句换届，不抢戏。

---

## 五、30 天冲刺计划表（概念周 → 代码周 → 真题周）

> 用法：每天 2–4 小时；"读"=精读对应章节并复述公式，"练"=白板/IDE 落地，"测"=三十秒框架录音自测。绝对路径见第六章附录。
> 概念周 D1–D14（模型层+RAG）、代码周 D15–D20（仓3 动手+手撕代码）、真题/冲刺周 D21–D30（系统设计+安全评测+面经+模考）。

| 天 | 周 | 当日文件与任务 | 交付物 |
|---|---|---|---|
| D1 | 概念 | 仓1 01-Foundation/04 题1–7（架构/√dk/多头/MQA-GQA-MLA）+ 本总纲 1.2(1)(2)(6) | KV 公式三锚点能默写 |
| D2 | 概念 | 仓1 01 题8–17（RoPE/Norm/FFN/Decoder-only/BPE 前半） | Pre-Norm 一句话原理 |
| D3 | 概念 | 仓1 01 题18–32（BPE 全流程/损失/AdamW/warmup/混合精度）+ 08 题4 对照 | BPE 手推样例 |
| D4 | 概念 | 仓1 02 题1–3（KV 原理/显存/减少手段）+ 复算 1.2(1) 全表 | 容量规划 30 路并发推演 |
| D5 | 概念 | 仓1 02 题4–8 + 仓2 detailed-inference-optimization 题01/03 | Flash vs Paged 对照表 |
| D6 | 概念 | 仓1 02 题9–20（解码/量化/部署）+ 仓2 inference 题04/06 | INT4 尺寸估算 |
| D7 | 概念 | 仓1 03 题1–5（LoRA/QLoRA/PEFT 对比）+ 仓2 training-finetuning 题07 | LoRA 参数量手算 |
| D8 | 概念 | 仓1 03 题6–13（RLHF→PPO→DPO→GRPO→DAPO 勘误） | DPO 七步推导脱稿 |
| D9 | 概念 | 仓2 detailed-rlhf-alignment 题01–08 + 仓1 03 题14–18 | 选型三角三句话 |
| D10 | 概念 | 仓1 07-HotTopics 全 9 题（DeepSeek-V3/R1/MoE/tTC/长上下文/VLM）+ 调研表五 | 2025-11 模型口径一张卡 |
| D11 | 概念 | 仓1 04 题1–3 + 仓2 detailed-rag-doc-rerank RAG 基础 6 题 | RAG 失败三层归因图 |
| D12 | 概念 | 仓1 04 题4–7（embedding/混合检索/改写/重排）+ 仓2 vector 题01–09 | RRF 手算例 |
| D13 | 概念 | 仓1 04 题8–10 + 仓2 rerank 题（5法纠偏/多信号陷阱）+ Contextual Retrieval 数字 | chunking 权衡表 |
| D14 | 概念 | 仓2 detailed-multimodal 8 题 + 仓2 model-architecture 外围题自选 | VLM 结构口述 |
| D15 | 代码 | 仓3 docs/zh/detailed-s01-s05.md 精读；跑通 s01/s02，做截断与唯一性实验 | 最小 loop 亲手运行记录 |
| D16 | 代码 | 仓3 docs/zh/detailed-s06-s09.md 精读；做"unseen 保护拆除"与"记忆守门撕护栏"实验 | 四解法连答链录音 |
| D17 | 代码 | 仓3 docs/zh/detailed-s10-s13.md 精读；做环检测短路与 plan gate 移除实验 | 通知注入不变式说明 |
| D18 | 代码 | 仓3 docs/zh/detailed-s14-s17.md 精读；做 deny-default 翻转与 hash() 反例实验 | goal gate 六分支默写 |
| D19 | 代码 | 仓1 08-Coding 题1–7 白板复现（Attention/MHA/FLOPs/BPE/Beam/KV/LoRA） | 手撕代码集一 |
| D20 | 代码 | 仓1 08-Coding 题8–11（检索/RoPE/GQA）+ 09 题1 推理服务延迟拆解 | 手撕代码集二 |
| D21 | 真题 | 仓1 09-SystemDesign 题2–5（企业 RAG/并行策略/ZeRO/可靠 Agent） | 两张设计图 |
| D22 | 真题 | 仓1 06 题1–3 + 调研第五章上线九条清单 | 注入防御口径卡 |
| D23 | 真题 | 仓1 06 题4–7 + 仓2 detailed-evaluation-monitoring 全 8 题 | NDCG 手算 |
| D24 | 真题 | 仓1 05-Agent 全 9 题 × 仓2 detailed-agent-prompt 11 题交叉复盘 | Agent 六件套闭环 |
| D25 | 真题 | 仓1 10-RealQuestions 面经 1–4（豆包/AI 应用/搜推/Agent 平台） | 错题本一 |
| D26 | 真题 | 仓1 10-RealQuestions 面经 5–11 按章回扫 | 错题本二 |
| D27 | 冲刺 | 调研 Top20 逐题 30 秒自测（本总纲第二章矩阵） | 弱点清单 |
| D28 | 冲刺 | 白板模考：DPO 推导 + KV 三连算例 + RRF+NDCG 手算（限时 40 分钟） | 板书照片 |
| D29 | 冲刺 | 系统设计模考：可靠 Agent 系统（09 题5）+ 企业 RAG（09 题2）走查 | 设计答复稿 |
| D30 | 冲刺 | 收敛：错题本全扫 + 五个 2026 区分度考点终检（MCP/上下文工程/多智能体取舍/可靠性/注入防御） | 准考证式一页纸 |

### 5.1 每周验收标准（达不到就顺延，不硬赶）

- **概念周（D1–D14）**：闭卷复述 1.2 公式速查全部 8 组；DPO 七步推导与 GRPO 目标默写无误；RAG 组合链（2.1 第 3 条）3 分钟讲完；仓2 的 6 处纠偏口径能说出"原文错在哪、正确口径是什么"。
- **代码周（D15–D20）**：仓3 至少完成 6 个 stage 的动手实验并留运行记录；手撕 11 题里 Attention/MHA/KV-Attn/LoRA/RoPE/GQA 六个白板一次通过；能现场解释自己实验改动的那几行为什么有效。
- **真题/冲刺周（D21–D30）**：10-RealQuestions 97 题过两遍，错题只扫标记题；模拟面（搭伴或自录）连续 25 分钟走完任意一条 2.1 组合链不卡壳；上线九条与 Top20 三十秒框架全部录音自查。
- **兜底原则**：进度落后时砍 D14/D23 的外围题（multimodal/外围架构题），**不砍** D18/D25/D28——仓3 实验、面经回扫、白板模考是不可替代的三件套。

---

## 六、附录：全部生成文件索引

### 6.1 调研报告（口径基准）
- `/Users/hyx/agent-personal/personal-agent-research.md` — 250 行；第八章=三层知识树与考点 Top20，第五章=安全/测试/上线，附事实核查摘要（修正 4 处、存疑 5 处）。

### 6.2 仓1 LLM-Agent-Interview-Guide（10 个 Detailed-Answers，共 218 题）

| 文件 | 题数 | 覆盖 |
|---|---|---|
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/01-Foundation/04-Detailed-Answers.md` | 32 | Transformer/注意力/分词/损失与优化器 |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/02-Inference/04-Detailed-Answers.md` | 20 | KV Cache/加速/解码/量化/部署 |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/03-FineTuning/04-Detailed-Answers.md` | 18 | LoRA 族/RLHF-PPO-DPO-GRPO/SFT |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/04-RAG/02-Detailed-Answers.md` | 10 | 全链路：chunk/检索/RRF/改写/Agentic/GraphRAG/评估 |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/05-Agent/02-Detailed-Answers.md` | 9 | Agent 定义/ReAct/FC/MCP/记忆/多智能体/框架 |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/06-Safety-Evaluation/02-Detailed-Answers.md` | 7 | 风险/越狱/防御/幻觉/评测/judge |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/07-HotTopics/02-Detailed-Answers.md` | 9 | DeepSeek-V3/R1/MoE/tTC/长上下文/VLM/趋势 |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/08-Coding/02-Detailed-Answers.md` | 11 | 手撕：Attention/MHA/FLOPs/BPE/Beam/KV/LoRA/RoPE/GQA |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/09-SystemDesign/02-Detailed-Answers.md` | 5 | 推理服务/企业 RAG/并行策略/ZeRO/可靠 Agent |
| `/Users/hyx/agent-personal/LLM-Agent-Interview-Guide/10-RealQuestions/02-Detailed-Answers.md` | 97（11 场） | 字节面经逐题详解 |

### 6.3 仓2 llm-interview-questions/modules（9 个 detailed 文件，共 100 题）

| 文件 | 题数 | 覆盖 |
|---|---|---|
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-model-architecture.md` | 20 | 架构 20 题（MLA/RMSNorm/VQ-VAE/MoE 等） |
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-inference-optimization.md` | 7 | KV 显存/DeepSeek/Flash/显存优化/量化/超长上下文 |
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-model-training-finetuning.md` | 15 | KL/Flow-DDPM/自动微调/损失/训练曲线/SFT |
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-rlhf-alignment.md` | 8 | on/off-policy/PPO-GRPO-DPO/奖励塑形/RM |
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-rag-doc-rerank.md` | 12 | RAG 基础 6 + 文档处理 1 + 重排 5（含 Contextual Retrieval 核数） |
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-vector-search-kg.md` | 11 | 向量 9（BM25/HNSW/FAISS）+ 知识图谱 2 |
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-agent-prompt.md` | 11 | Agent 8 + Prompt 3（ReAct/记忆/多智能体/工具） |
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-evaluation-monitoring.md` | 8 | NDCG/人工-vs-自动/幻觉/错误定位/MRR |
| `/Users/hyx/agent-personal/llm-interview-questions/modules/detailed-multimodal.md` | 8 | VLM/扩散 SNR/关键帧/MoE 路由/Multi-modal RAG |

### 6.4 仓3 learn-claude-code/docs/zh（4 个 detailed 文件，覆盖 17 stage）

| 文件 | Stage 数 | 覆盖 |
|---|---|---|
| `/Users/hyx/agent-personal/learn-claude-code/docs/zh/detailed-s01-s05.md` | 5（s01–s05） | Loop/工具/权限/Hooks/TodoWrite |
| `/Users/hyx/agent-personal/learn-claude-code/docs/zh/detailed-s06-s09.md` | 4（s06–s09） | Subagent/Skill/压缩/记忆 |
| `/Users/hyx/agent-personal/learn-claude-code/docs/zh/detailed-s10-s13.md` | 4（s10–s13） | 任务系统/后台/Cron/团队 |
| `/Users/hyx/agent-personal/learn-claude-code/docs/zh/detailed-s14-s17.md` | 4（s14–s17） | MCP/集成宿主/Workflow/Goal Loop |

> 索引注记：仓1/仓2/仓3 各仓库内另有原题目文件与课程正文（`doc/0X-*.md`、`modules/0X-*.md`、`docs/zh/sXX-*.md` 旧版 12 课），本总纲与五章计划只引 detailed 详解为主线；仓3 旧版 `docs/zh/sXX` 编号与 17-stage 代码不一致，仅作术语参考。

### 6.5 配套实验清单（代码周要留痕的 8 个最小改动）

1. s01 步数上限：加 `turn` 计数器观察链式衰减（detailed-s01-s05 实验 1）。
2. s02 edit 唯一性：`count != 1` 返回提示，复刻真实 Edit 语义。
3. s03 会话级批准缓存：`APPROVED` 集合 → 讨论"持久白名单=四档权限的成型"。
4. s08 unseen 保护拆除：`consumed = list(results)` → 观察占位符幻觉。
5. s09 守门撕裂：删临时词表 → 会话级指令被永久化（记忆投毒实证）。
6. s11 协议伪装：通知改走原 tool_use_id → API 400（1:1 不变式实证）。
7. s14 deny 翻转：策略缺省 allow → 安全反模式现场演示。
8. s16 加盐哈希：`_stable_hash` 换内建 hash() → resume 全 miss（语义键必须跨进程稳定）。

### 6.6 术语速查（面试前最后一遍过）

| 术语 | 一句话 |
|---|---|
| context rot | 上下文越长、模型利用效率越低——上下文工程的总敌人 |
| compaction / microcompact | 历史压缩总管线 / 只清旧工具结果的轻量版 |
| harness | 工具+知识+观察+行动接口+权限：让模型在特定环境干活的全部外围工程 |
| handoff 有损 | 子代理只回传结论，中间推理回不来——多智能体三代价之一 |
| at-least-once | 事件投递语义：宁可重复不丢（cron/后台通知的选择） |
| 语义 key | 按输入内容哈希的 journal 键，跨进程稳定才可续跑 |
| goal gate | return 边界的独立判断器：决定"模型想停"是否等于"干完了" |
| RLVR | 可验证规则奖励（数学/代码对错即奖励），GRPO/R1 的搭档 |
| k3 估计器 | π_ref/π_θ − log(π_ref/π_θ) − 1：KL 的逐 token 无偏非负估计 |
| Contextual Retrieval | 给 chunk 注入 LLM 生成的位置语境再检索（35%/49%/67% 相对降幅） |
| hard negative | 同域不相关的难负例，reranker 训练数据的质量核心 |
| deny-by-default | 未配置即人审/拒绝——宿主授权的唯一缺省 |
| unseen 保护 | 压缩管线不得裁剪模型尚未消费的工具结果 |
| prefill/decode | 并行建 KV（compute-bound）/逐 token 读 KV（bandwidth-bound） |

### 6.7 专项深挖文档（第四层，总纲不重讲的内容）

- `/Users/hyx/agent-personal/Agent搭建面试专项.md` — **Agent 搭建实战四专攻**：①六大场景设计题（P1 编码 / P2 深度研究 / P3 客服 / P4 数据分析 / P5 Computer Use / P6 长时程，每题"题面→3 分钟版→分层展开→追问树→陷阱"五段式）+ 0→1 十步搭建模板与六层架构图；②Agent 层手撕代码 6 题（T1 loop / T2 工具注册 / T3 分层恢复链 / T4 结构化输出 / T5 子代理 / T6 checkpoint，全部白板可默写骨架）；③可观测与调试专题（turn-span 七字段 / 四桶归因+轨迹二分+裁撤实验三步法 / 三层告警 / 成本单位经济学，补三仓最大空白）；④组件选型决策链（框架/模型路由/记忆/FC-MCP/向量库/结构化输出六链）+ 高频追问 Top30 速答卡 + 7 天专项微计划。与专题 E 的分工：**E 讲概念，专项讲现场设计题怎么答、代码怎么写、线上怎么调**。

### 6.8 学习路线图（第五层：从底层原理到 Agent 实现的学习顺序总控）

- `/Users/hyx/agent-personal/学习路线图-LLM到Agent.md` — **循序渐进 9 个 Stage（0–8）**：地基自检 → Transformer 底层 → 推理工程 → 训练对齐 → RAG → Agent 核心★ → 协作生态 → 生产化 → 毕业项目（三选一）。每 Stage 三栏验收："**弄懂**（材料坐标精确到章）/ **弄明白**（可打勾验收单）/ **实现好**（动手产出，如 numpy attention、KV 计算器、LoRA 层、DPO loss、最小 RAG、agent-mini 递增五步、MCP server 接入、span 落盘+评测集、毕业项目硬性要求清单）"。附：里程碑总表（贴墙版）、与全部材料的坐标映射表、三遍法+费曼节奏建议、上游三仓同步机制（2026-11 已同步 learn-claude-code 至 448c75b）。与总纲 30 天计划的分工：**路线图管"先学什么学到什么程度"（依赖序），30 天计划管"应试怎么排期"（时间表）**。

*（本总纲 609 行 · 完）*
