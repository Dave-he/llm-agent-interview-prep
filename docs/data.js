const TREE = [
  {name: "模型层 —— 能力地板（模型决定下限）", note: "仓1 01/02/03/07 章 · 仓2 architecture/inference/training/rlhf/multimodal", kids: [
    {t: "Transformer 与注意力", d: "softmax(QKᵀ/√d_k)V。√d_k 双解释：方差拉回单位方差防 softmax 饱和 + 数值稳定。多头参数恒等式：h 头合计仍 4d²（单大头 4d²、h 小头 3d² + W_O d²），参数与 FLOPs 均不变；收益=低秩子空间集成，代价=KV cache 随头数线性膨胀——MQA/GQA/MLA 的出发点。"},
    {t: "位置编码与结构件", d: "RoPE 复数旋转：绝对形式、相对语义（内积只依赖 m−n）；外推问题→位置插值/NTK/YaRN；ALiBi 线性 bias 加 logits。Pre-Norm 梯度主干无衰减（现代主流）；RMSNorm 只做方差缩放省 reduce；SwiGLU 门控 FFN，d_ff=8/3·d 等参。Decoder-only 胜出四点：稠密监督、结构与目标统一、KV cache 前缀复用、工程可 scale。"},
    {t: "分词", d: "BPE：训练=最高频相邻对贪心合并，推理=查表切分 O(长度)；WordPiece 似然增益+## 前缀；Unigram 从大词表概率剪枝（反向）；中文靠 byte-level BPE 兜底。词表 32k–128k 的权衡：大表省 token 但 embedding 占比升、稀有词欠训。中文 1 token≈1–1.5 字、英文≈0.75 词。"},
    {t: "KV Cache", d: "KV = 2(K+V)·b·n_kv·L·s·d_head·dtype（Q 不缓存；MLA 无 ×2）。合法性：因果注意力下 k_j/v_j 只依赖 x_j。省的是重复投影 O(td)，省不掉注意力读取 4td→decode 是带宽瓶颈。演进：MQA(n_kv=1 质量掉)→GQA(LLaMA-70B g=8，1/8，近无损，可 mean-pool uptraining)→MLA(低秩压缩 c_kv 512 + k_rope 64 = 576 元素/层，RoPE 不可压需单独存)。"},
    {t: "推理工程", d: "解码：temperature/top-k/top-p（保概率质量优于固定个数）、重复惩罚。量化：GPTQ=逐层二阶信息权重误差最小化；AWQ=激活 salient 通道 per-channel scaling；13B FP16≈26G→INT4≈7G。FlashAttention：IO 感知分块+online softmax，减显存访问不减计算量。PagedAttention：KV 分页消碎片+prefix 共享。投机解码：草稿+验证，分布不变。TP 层内/PP 流水（bubble）。prefill compute-bound / decode bandwidth-bound。"},
    {t: "PEFT 与对齐", d: "LoRA：ΔW=BA，A 高斯 B 零（起步恒等），r 常用 8–32，合并零推理延迟。QLoRA：NF4+双量化+paged optimizer。RLHF 三阶段：SFT→RM(Bradley-Terry 成对)→PPO(四组件 actor/critic/ref/RM)。DPO：闭式推导，Z(x) 相消，无 critic 无采样，隐式难例加权=噪声敏感根源。GRPO：组均值当基线 critic 消失，k3 无偏 KL 估计；G=64 常见。DAPO：token-level 归一+动态采样，治长回答梯度稀释与零方差组。R1=GRPO+RLVR+少量 SFT 冷启动。"},
    {t: "前沿架构", d: "MoE：FFN→N 专家+router top-k，容量与算力解耦；细粒度+共享专家+aux-loss-free 负载均衡（DeepSeek-V3）；挑战：负载不均/专家崩塌/all-to-all。Test-Time Compute：算力从预训练搬到推理采样与验证，长 CoT 是可见形态。VLM=ViT patch 嵌入+projector 对齐进 LLM 词嵌入空间。"}
  ]},
  {name: "编排层 —— 可靠性上限（harness 决定上限）", note: "仓1 04/05/09 章 · 仓2 rag/vector/agent-prompt · 仓3 s01–s16", kids: [
    {t: "Agent Loop", d: "while(stop_reason==tool_use){执行→观察回填}。终止三分类：模型自停/步数预算/人工中断；max_turns 是生产必须项。tool_result 回 user 角色（Anthropic Messages API 约定；OpenAI Chat Completions 为独立 tool 角色；Observation 语义）。链式衰减 0.95^20≈36% → 每步验证+缩粒度。"},
    {t: "上下文工程", d: "管\"什么信息以什么形式在什么时点进窗口\"。总敌人 context rot。四解法：隔离(躲,子代理)/按需(等,渐进披露)/压缩(挤,四步管线按损失×成本升序)/外存(搬)。压缩切点避让 tool_use/tool_result 配对；unseen 保护防占位符幻觉。"},
    {t: "工具与协议", d: "FC 全链路：JSON Schema 声明→模型产结构化调用→客户端执行→tool message 回填。三条铁律：执行在客户端/FC strict mode・structured outputs 模式走约束解码保格式/错误是给模型的输入。工具描述=API 文档，质量>换模型。MCP：host/client/server 三层+tools/resources/prompts 三原语，解 M×N→M+N；与 FC 互补：MCP 管工具从哪来（分发标准），FC 管模型怎么调（调用机制）。host 四步：发现→命名空间+冲突检测→授权只来自宿主配置(deny-by-default)→并入池每轮重建。"},
    {t: "状态与记忆", d: "短期=窗口+scratchpad；长期=外存四件套 store/recall/consolidate/forget（遗忘与冲突是隐藏考点）。方案谱系：mem0(抽取式 ADD/UPDATE/DELETE/NOOP)/Letta(核心+归档分层自编辑)/Zep(时序知识图谱双时间线)。工程要点：持久性守门+索引常驻正文按需+\"召回与当前请求冲突时当前请求优先\"。"},
    {t: "协作形态", d: "subagent 三铁律：全新 messages/只回传最终文本(handoff 有损)/子工具表剔除 task 防递归委派。持久团队：文件信箱总线(破坏性读)+原子认领+plan gate 落在工具分发层。workflow 运行时：模型只提交 name/args 不提交代码(最小攻击面)；journal 语义 key=sha256 需跨进程稳定。共识：编排确定性+模型智能。"},
    {t: "RAG 与检索", d: "三阶段：索引(解析切块嵌入)→检索(改写召回融合粗排精排)→生成(注入引用)。失败集中在检索与生成接口处。混合检索 BM25+dense+RRF(k=60)；Contextual Retrieval：top-20 失败率 5.7%→1.9%（−35/49/67% 相对降幅）。重排：召回求全/粗排求快/精排求准（cross-encoder），hard negatives 是训练数据核心。Agentic RAG=检索成为可决策工具；GraphRAG=实体/关系/社区摘要，全局问答占优。"}
  ]},
  {name: "验证层 —— 进生产门票（最被低估的一层）", note: "仓1 06/08/10 章 · 仓2 evaluation-monitoring · 仓3 s15/s16/s17", kids: [
    {t: "评测基准分层", d: "编码 SWE-bench(Verified 500 题人工校验；轨迹 2%→33–49%→~70%→76.2/77.9/80.9)/通用助理 GAIA/GUI OSWorld/网页 WebArena/工具+策略 τ²-bench/注入 AgentDojo。分数仅初筛，必须私域评测集复验。"},
    {t: "评测方法", d: "自动(基准+规则) vs 人工(打分+pairwise)。LLM-as-judge 三病：自我偏好/位置偏差/一致性漂移；缓解=去自我偏好 pairwise+交换一致+BT 拟合。检索层 NDCG/MRR/Recall@k；生成层 Faithfulness+Answer Relevance。"},
    {t: "安全", d: "注入：外部内容携带恶意指令劫持模型；agent 有工具权+上下文内指令与数据不可分辩故更险。答案=无银弹+纵深分层：输入过滤(homoglyph 归一/策略查表)→权限隔离(默认只读/沙箱禁网/凭证分权)→输出检查(PostToolUse 观察告警)→系统性(红队/审计无明文)。记忆投毒：确定性守门+临时词表。"},
    {t: "运行时验证", d: "goal gate：模型提出停止，harness 决定是否允许。独立无工具判断器读 transcript 证据（exit_code 前缀是机器可读证据）；六分支状态机；连续 block 8 次交还用户不伪装成功。对抗验证：每条 finding 一个验证者。CI 挂评测：模型/工具/知识库任一变更前后跑回归集。"},
    {t: "线上监控", d: "四指标：任务成功率/步数与 token 分布/人审介入率/错误根因四桶(意图/检索/工具/生成)。步数分布漂移=退化早信号（成功率还看不见时它先动）。ROI：METR 2025 对照实验——主观提速、实测慢约 19%，收益依赖可验证性与样板率。"}
  ]}
];

const EXAM20 = [
  {n: 1, t: "Agent vs Chatbot vs Workflow", hot: 1, s: "总纲考点1 · 仓1 05题1 · s01/s12/s16", p: "Chatbot 单轮无状态；Workflow 控制流写死在代码；Agent 控制流由模型输出决定。金句：Workflow=编排确定性，Agent=模型智能。"},
  {n: 2, t: "Agent Loop 完整循环", hot: 1, s: "总纲考点2 · s01/s15", p: "while(出现 tool_use){执行→观察回填}；终止三分类=模型自停/步数预算/人工中断；max_turns 与预算是工程必须项。"},
  {n: 3, t: "ReAct 为何有效", hot: 1, s: "仓1 05题2 · s01", p: "Reasoning 与 Acting 交错：思考轨迹引导行动，行动结果修正推理；grounding 到观察减少幻觉。"},
  {n: 4, t: "Function Calling", hot: 1, s: "仓1 05题3 · s02", p: "传 JSON Schema→模型产结构化调用→执行在客户端→tool message 回填；并行调用=一次响应多 tool_use 块。"},
  {n: 5, t: "MCP", hot: 1, s: "考点5 · 仓1 05题4/5 · s14", p: "解 M×N 集成为协议：host/client/server 三层+tools/resources/prompts 三原语；MCP 管工具从哪来、FC 管模型怎么调，互补不互替。2026 必考。"},
  {n: 6, t: "Context Engineering", hot: 1, s: "考点6 · s07/s08/s11", p: "管\"什么信息以什么形式在什么时点进入有限窗口\"；核心敌人 context rot；四解法：隔离/按需/压缩/外存。"},
  {n: 7, t: "记忆系统", hot: 0, s: "仓1 05题6 · s09/s10", p: "短期=窗口+scratchpad；长期=外存四件套 store/recall/consolidate/forget；谱系 mem0(抽取式)/Letta(分层自编辑)/Zep(时序图)。"},
  {n: 8, t: "RAG 与 Agentic RAG", hot: 0, s: "仓1 04题6/8", p: "RAG 是检索增强模式，Agent 是多步系统；Agentic RAG=检索成为可决策工具（多跳/改写/自评再检索）。"},
  {n: 9, t: "多智能体取舍", hot: 1, s: "考点9 · 仓1 05题7/8 · s06/s13/s16", p: "收益=并行+上下文隔离+异构分工；代价=成本倍增+失败级联+handoff 有损；默认先单 agent+子代理。"},
  {n: 10, t: "可靠性瓶颈", hot: 1, s: "仓1 09题5 · s10/s11/s15/s17", p: "链式衰减 0.95^20≈36%；错误自反思噪声滚雪球；对策四件：验证信号/checkpoint/人审/缩小任务粒度。"},
  {n: 11, t: "评测基准", hot: 0, s: "仓1 06题5/6 · 仓2 eval", p: "SWE-bench(编码)/GAIA(助理)/OSWorld(GUI)/WebArena(网页)/τ²-bench(策略)/AgentDojo(注入)；分数仅初筛，必须私域评测复验。"},
  {n: 12, t: "Prompt Injection", hot: 1, s: "仓1 06题1–3 · s03/s14/s15", p: "外部内容携带恶意指令劫持模型；agent 有工具权+指令与数据不可分辩故更险；答案=无银弹+纵深分层防御。"},
  {n: 13, t: "沙箱与权限", hot: 0, s: "仓1 09题5 · s03/s13/s14", p: "默认只读、写/网络按需；代码进容器/worktree 禁网；凭证分权；四档权限=同一条管线的配置差异。"},
  {n: 14, t: "成本优化", hot: 0, s: "仓1 02题20 · s15", p: "模型路由(per-step)+KV/前缀缓存+上下文压缩+子代理避免重复读+预算上限降级；口诀：缓存>裁剪>路由>换模型。"},
  {n: 15, t: "微调 vs 提示 vs RAG", hot: 0, s: "仓1 03题14–18", p: "默认顺序：prompt/context 工程→RAG/工具(外部知识可更新可审计)→微调(格式/风格/降本/私密标注)；微调不解决知识过时。"},
  {n: 16, t: "论文脉络", hot: 0, s: "仓1 05题2 · s05/s10", p: "ReAct 交错推理行动→Reflexion 语言化自我反思→Voyager 技能库+自动课程→Generative Agents 记忆流+反思+计划。"},
  {n: 17, t: "规划技术谱系", hot: 0, s: "仓1 05题1 · s05/s10/s17", p: "CoT 线性→ToT 树搜索→任务分解+反思迭代；工程主流=plan-first+执行中重规划（以工具形态出现，TodoWrite）。"},
  {n: 18, t: "HITL", hot: 0, s: "仓1 09题5 · s03/s12/s15", p: "分级审批(高危阻断)+checkpoint 后批量人审+interrupt-on-external-effect+可回滚是放权前提。"},
  {n: 19, t: "harness 天花板论", hot: 1, s: "考点19 · 仓1 05/09 · s15", p: "模型是地板、harness 是天花板；30 行核心循环之外全是增值工程；同等模型下工具描述/错误恢复/验证信号的差异远大于换模型。"},
  {n: 20, t: "前沿：CU/长时程/个人 Agent", hot: 0, s: "仓1 07章 · s17", p: "GUI 原生操作扩覆盖面；30h 级长时程=compaction+checkpoint+自验证；个人主线=端侧模型+本地记忆；评测从能力分转向安全与过程分。"}
];

const FORMULAS = [
  {id: "f1", name: "KV Cache 显存", f: "KV = 2(K+V) × b × n_kv × L × s × d_head × sizeof(dtype)", back: "Q 不缓存；MLA 无 ×2。每 token 每层 = 2·n_kv·d_head·dtype；×L 层 ×序列长 s ×batch。\n锚点：LLaMA-2-7B MHA(L=32,n_kv=32,d=128)@4k = 512 KiB/token → 2.0 GiB；LLaMA-3-70B GQA-8(L=80,n_kv=8)@4k = 1.25 GiB/序列；反事实 70B MHA=10 GiB→GQA 省 8 倍；MLA(576 元素/层/token)=0.268 GiB@4k=GQA-8 的 28%、7B-MHA 的 7%（均为每层每 token 元素比：576/2048、576/8192，非总显存比）。128k 上下文 70B 档：MHA≈320GiB→GQA≈40GiB→MLA≈8.6GiB（L=61 口径）。容量规划：A100-80G+70B INT4（权重约 36–40G，剩 ~40G÷1.25GiB/4k）≈ 30 路并发。报数先声明 GiB/GB 口径。"},
  {id: "f2", name: "FLOPs 估算", f: "前向 ≈ 2 × N × D；训练 ≈ 6 × N × D", back: "N=参数量，D=token 数（乘加记 2 FLOPs；前向 2N+反向 4N）。注意力核心项：prefill 全序列 ≈ 2·T²·d；decode 每步 = 当前 qkv 投影 6d + 读旧 KV 4td——缓存省投影、不省注意力读取（decode 带宽瓶颈根源）。例：7B×1T token 训练 = 4.2e22 FLOPs。"},
  {id: "f3", name: "DPO 目标", f: "L_DPO = −E[ log σ( β·log(π_θ(yw|x)/π_ref(yw|x)) − β·log(π_θ(yl|x)/π_ref(yl|x)) ) ]", back: "推导主线：KL 约束 RLHF 目标→拉格朗日→π* = π_ref·e^{r/β}/Z→反解 r→代入 Bradley-Terry 成对 sigmoid→Z(x) 在奖励差中相消→π* 换成 π_θ。\nβ 双重角色：约束端越大越贴 π_ref；损失端 sigmoid 越饱和更新越保守。常用 β∈[0.05,0.5]，0.1 最常见。\n隐式难例加权：梯度权重=σ(r̂_l−r̂_w)，错序越狠学得越狠→标注噪声也被更用力学（噪声敏感的机制根源）。"},
  {id: "f4", name: "GRPO 目标", f: "Â_i = (r_i − mean{r_1..r_G}) / std{r_1..r_G}；J = E[(1/G)Σ(1/|y_i|)Σ min(ρ·Â, clip(ρ,1±ε)·Â) − β·D̂_KL]", back: "组均值即 V(x) 的蒙特卡洛基线→critic 消失。KL 用 k3 无偏估计：π_ref/π_θ − log(π_ref/π_θ) − 1 ≥ 0（逐样本非负）。\n版本口径：DeepSeekMath 归一化分母 (1/G)(1/|y_i|)=原式；全局 token 归一 1/Σ|y_i|=DAPO 修正，勿混写。\n显存账：可训练量减半（7B 约 250GB→130GB）；G=64 常见。坑：全对/全错组 std→0 零梯度→DAPO 动态采样过滤；R1/DAPO 后期 β=0。R1 配方=GRPO+RLVR+少量 SFT 冷启动。"},
  {id: "f5", name: "RRF 融合", f: "RRF(d) = Σ_{r∈R} 1/(k + rank_r(d))，k=60，rank 从 1 起", back: "用 rank 不用 score：BM25 无上界、cos∈[−1,1]，量纲不可比；rank 免校准、抗异常。\n手算例：A 路 d1 第1、d2 第3；B 路 d2 第1、d1 第5：RRF(d1)=1/61+1/65≈0.03177；RRF(d2)=1/63+1/61≈0.03226→d2 胜（\"两路都靠前\"压过\"单路第一\"）。\nk→∞ 趋近投票制；k=0 倒数权重最陡、top1 优势最大（第 2 名仍得 1/2，非\"只看 top1\"）。"},
  {id: "f6", name: "MHA→MQA→GQA→MLA", f: "瓶颈在每 token 的 KV 元素数（decode 带宽）", back: "MQA：n_kv=1，cache 缩 h 倍但质量掉（Q 头被强制同态）。\nGQA：n_kv=g，LLaMA-70B g=8→1/8，质量近无损（可由 MHA 权重 mean-pool uptraining）。\nMLA：跳出多头框架做低秩压缩，只缓存 c_kv(512)+k_rope(64)=576 元素/层；难点=RoPE 与压缩耦合→k 拆 content（可压）+rope（单独存，跨头共享一条 64 维）；质量略优于 GQA（DeepSeek-V2 论文）。DSA 稀疏注意力是 2025 续章。"},
  {id: "f7", name: "检索评测 NDCG/MRR", f: "DCG@k = Σ_{i≤k} (2^rel_i −1)/log2(i+1)；NDCG@k = DCG@k / IDCG@k", back: "查询内归一到 [0,1]。算例：输出 [3,0,3] k=3→DCG=7+0+3.5=10.5；IDCG=7+7×0.631+0.5=11.92→NDCG≈0.88（设相关性多重集 {3,3,1,0,0}，IDCG 理想排列 [3,3,1]）。rel=3 掉到第 3 位→7.92→≈0.66（掉位惩罚快于乱序）。\nMRR：首个相关结果排名倒数的均值（只认第一命中）。口径注记：log2(i+1) 与 log2(i+2) 索引平移写法并存，跨系统比分先固定口径。"},
  {id: "f8", name: "Agent 可靠性算术", f: "单步成功率 p、串行 n 步 → P = p^n；0.95^20 ≈ 0.36", back: "链式衰减是 agent 可靠性题的必带数字。0.95^10≈0.60、0.95^20≈0.36。\n对策四件：①验证信号优先（可执行测试/退出码）②checkpoint 与回滚 ③人审卡点 ④缩小任务粒度（p 提升 + n 缩短双管齐下）。"}
];

const QA30 = [
  {c: "搭建", q: "从零搭 Agent 第一件事做什么？", a: "需求判断三问（可验证/可逆/频率），不满足就 workflow 或不上。"},
  {c: "搭建", q: "最小可行 Agent 需要哪些组件？", a: "loop+工具+观察回填+终止；30 行核心循环之外全是增值工程（考点 19）。"},
  {c: "搭建", q: "工具给多少合适？", a: "从任务动词推最小完备集；工具越多选择越难，description 即路由（s07）。"},
  {c: "搭建", q: "单还是多 agent？", a: "并行/隔离/异构三收益 vs 成本倍增/失败级联/handoff 有损三代价；默认单+子代理。"},
  {c: "搭建", q: "Agent 状态放哪？", a: "能外置就外置（文件/DB）；对话内状态活不过 compaction（s10）。"},
  {c: "搭建", q: "上下文不够第一招？", a: "按需加载（等）优先于压缩（挤），压缩优先于外存（搬），子代理隔离（躲）看场景。"},
  {c: "搭建", q: "终止怎么设计？", a: "三分类：模型自停+步数/token 预算+人工中断；生产必有 max_turns。"},
  {c: "搭建", q: "什么时候上 MCP？", a: "工具要跨应用/跨团队复用、接第三方生态；单应用原生 FC 最短路径。"},
  {c: "搭建", q: "agent 和 workflow 混合怎么做？", a: "外层确定性编排（谁先谁后、预算、重试），内层模型智能（怎么干）——框架共识句。"},
  {c: "搭建", q: "第一版上线范围？", a: "只读+窄任务域+人审全开；灰度曲线放权，人审介入率当监控变量（F5）。"},
  {c: "调试", q: "Agent 输出错了怎么查？", a: "先四桶归因（意图/检索/工具/生成），再轨迹二分找偏航 turn，最后裁撤实验证因果。"},
  {c: "调试", q: "怎么复盘一次执行？", a: "trace 回放：turn 级 span（模型/token/延迟/工具结果），子代理 trace 用 parent_id 串回。"},
  {c: "调试", q: "怎么判断是模型问题还是 harness 问题？", a: "同输入做裁撤实验（删压缩/换工具描述/换路由）；harness 因素排除后剩下的才是模型问题（模型是地板，harness 是天花板）。"},
  {c: "调试", q: "工具老被误选？", a: "改 description（十有八九在 description）、合并相似工具、控制工具数、必要时分组暴露。"},
  {c: "调试", q: "循环重复同一动作？", a: "重复检测注入提醒换路径（nudge 不硬路由）+预算熔断；根因常是观察没消化，查截断。"},
  {c: "调试", q: "延迟怎么降？", a: "流式返回+并行工具调用+per-step 小模型路由+前缀缓存；computer-use 类再看动作合并。"},
  {c: "调试", q: "成本怎么估？", a: "先测 100 任务 token 分布（中位/P90），×单价×量，再叠缓存命中率与外部工具成本。"},
  {c: "调试", q: "线上怎么发现\"变笨了\"？", a: "回归集例行评测+成功率 7 日漂移告警；步数分布悄悄上涨=退化早信号（成功率还看不见时它先动）。"},
  {c: "选型", q: "自研还是上框架？", a: "工具少/特殊恢复逻辑/吃透原理→自研（loop 就 30 行）；要 checkpoint+人审+tracing 生产件→LangGraph 类。"},
  {c: "选型", q: "主力模型怎么定？", a: "私域任务集盲测，基准分只初筛；per-step 路由：简单步小模型。"},
  {c: "选型", q: "记忆方案怎么选？", a: "先反问要不要跨会话；轻度事实 mem0、分层长期 Letta、时序事实 Zep、结构化账目自建表。"},
  {c: "选型", q: "JSON 输出不稳？", a: "有 FC/structured outputs 就用约束解码；没有就三层防线（声明→校验回填→兜底）。"},
  {c: "选型", q: "向量库三问？", a: "规模多大、QPS 多高、要不要混合检索与复杂过滤；百万级以下 pgvector 就够。"},
  {c: "选型", q: "评测体系怎么起步？", a: "先建带自动验证信号的任务集（20–50 题）+回归回放，再谈 LLM-as-judge 与基准。"},
  {c: "安全", q: "上线前安全清单？", a: "上线九条：零代码换模型、凭证轮换、高危人审、注入三层防线、预算降级、审计无明文、沙箱禁网、红队闭环、一键停机回滚。"},
  {c: "安全", q: "工具结果里藏指令怎么办？", a: "当数据不当指令：引用隔离、摘要降级、压缩后历史整体视为不可信、唯一可授权的是当前用户请求。"},
  {c: "安全", q: "记忆会被投毒吗？", a: "会；确定性守门（scope 表态+临时词表）+来源隔离+冲突时当前请求优先。"},
  {c: "成本", q: "一个任务多少钱？", a: "token×单价（缓存命中另算）+外部工具成本+人审工时；报数带分布（中位/P90），不报单点。"},
  {c: "成本", q: "省 token 三板斧？", a: "稳定前缀吃缓存、工具结果裁剪与压缩、子代理隔离省重复读；顺序：缓存>裁剪>路由>换模型。"},
  {c: "成本", q: "什么时候不值得优化？", a: "低频任务人审成本>token 成本时先放量跑通再优化；METR 反例提醒：主观提速不算数，实测口径才算。"}
];

const DESIGN = [
  {id: "P1", tag: "Coding Agent", t: "从零搭建一个 Coding Agent", brief: "现场设计一个能修 bug、写小功能的编码 Agent（对标 Claude Code / SWE agent）。",
    min3: ["判断：编码任务有天然验证信号（测试/编译/退出码）、worktree 里错误可逆——最适合上 agent 的场景；单 agent+子代理。",
      "工具面：read/write/edit/bash/grep/glob 六件套起步；edit 带唯一性校验（count!=1 拒绝）、前置 read、bash 输出截断。",
      "上下文：不做全库 RAG——grep/glob 是模型主动调的检索工具（agentic 检索），文件系统即索引。",
      "权限+验证：worktree 隔离+高危命令人审+路径白名单；跑测试拿退出码（机器可读证据）→goal gate 判断器读证据→人审 diff。"],
    drill: ["为什么不用 RAG 索引代码库？→ grep 精确/实时/零维护 vs embedding 召回噪声+索引漂移。",
      "怎么防删库？→ worktree+路径白名单（resolve+is_relative_to）+破坏性命令人审+git 一键回滚；worktree 是目录隔离不是沙箱。",
      "上下文快满？→ 四步压缩按损失×成本升序：大结果转存→中段归档→旧结果缩路径→才轮到模型摘要；unseen 保护。",
      "怎么评测？→ SWE-bench 轨迹 2%→80.9 证明抬的是 harness 不是模型；私域任务集复验。"],
    trap: "答成\"接 GPT+写 prompt\"缺三要素；说用框架却讲不出工具面与验证回路；全程不提上下文管理。"},
  {id: "P2", tag: "Deep Research", t: "深度研究 Agent（多跳检索+报告）", brief: "给一个开放问题，产出一篇带引用的研究报告。",
    min3: ["plan→parallel searches→cross-check→outline→draft→cite 四段式：计划先行（人可改），子问题并行子代理（隔离+并行）。",
      "检索面：混合检索（dense+BM25+RRF）+查询改写+每跳结果裁剪（只留引用相关段，防 context rot）。",
      "验证：来源交叉（关键事实≥2 独立来源）+来源分级+引用回指段落级；无证据标注\"未找到可靠来源\"，宁缺勿编。"],
    drill: ["多跳失控？→ 跳数上限+每跳预算+自评阈值，超过降级单轮 RAG。",
      "防单源污染？→ 交叉验证+摘要降级（检索内容当数据不当指令，防间接注入）。",
      "为什么子代理？→ 上下文隔离+并行；代价 handoff 有损——要求回传结构化摘要。",
      "长报告？→ 分节生成+提纲锚点；全文一次生成必失焦。"],
    trap: "答成\"RAG 三阶段\"缺计划/多跳决策/交叉验证；引用不能回指是面试官最爱的反例。"},
  {id: "P3", tag: "企业客服", t: "企业客服 Agent（知识库+工单+人审）", brief: "知识问答+工单操作，怎么保证不上线翻车？",
    min3: ["分流三层：FAQ（置信度阈值→RAG 直答带引用）/半开放（Agent 有限工具）/转人工（低置信/高危/情绪）；拒答制度化。",
      "工具面收窄：订单查询/退款申请/工单创建少数几个；工具内嵌风控（退款限额）；高危工具强制人审。",
      "安全：用户输入与知识库内容都可能带注入——检索内容当数据不当指令。",
      "运营：四指标（成功率/步数 token 分布/人审介入率/错误根因四桶）；人审介入率是放权曲线自变量。"],
    drill: ["怎么量化？→ 解决率+处理时长+抽检错误率+成本/会话；主观提速不算数要实测口径。",
      "用户诱导退款？→ 高危人审+话术守卫；退款类永远不可全自动。",
      "为什么不上多 agent？→ 客服链路短路径半固定；成本倍增+失败级联在延迟敏感场景是负资产。",
      "知识库更新？→ 增量 upsert+周期压实+过期下线。"],
    trap: "只讲 RAG 不讲分流+放权曲线；没有拒答与转人工设计——客服题考边界感不是检索精度。"},
  {id: "P4", tag: "数据分析", t: "数据分析 Agent（Text2SQL+图表）", brief: "自然语言提问，产出查询结果和图表。",
    min3: ["判断：SQL 可执行=天然验证信号（跑得通≠答得对但跑不通一定错）；写库不可逆→只读账号。",
      "工程链：schema 按需注入（相关表+注释+样例）→few-shot→生成后自校验→EXPLAIN 干跑错误回填重试→执行后 sanity check。",
      "安全三闸：只读权限（权限层根绝写库）→SQL 静态检查（黑名单语法）→运行限额（LIMIT 强制+超时 kill+行数截断）。图表 spec 走结构化输出。"],
    drill: ["表几万张？→ schema RAG（表名/描述向量索引，先检索相关表再注入）+业务域分组。",
      "查询拖挂库？→ 只从读副本查+强制 LIMIT+超时+EXPLAIN 预检扫全表告警。",
      "用户要改数？→ 只读设计根绝；写需求走变更单+人审，不进 agent 执行面。",
      "多轮下钻？→ 槽位外置（维度/时间窗结构化存储）+上轮 SQL 与结果摘要进上下文。"],
    trap: "只讲 prompt 技巧不讲校验回路与只读权限——安全分占一半；权限在数据库层兜底不是只靠 prompt。"},
  {id: "P5", tag: "Computer Use", t: "Computer Use / 浏览器 Agent", brief: "设计一个能操作电脑/浏览器完成任务的 Agent。",
    min3: ["循环：观察（截图 or DOM/无障碍树）→定位元素→动作（click/type/scroll/navigate）→等页面稳定→再观察；终止=目标判定/步数预算/卡死检测。",
      "两条路线：视觉（纯截图，泛化强但慢）vs 结构（DOM/无障碍树，精确但动态页脆）——主流混用；优先无障碍树 role/name，视觉坐标兜底。",
      "安全：高危动作（支付/发送/提交）人审；凭据不入上下文（密码管理器代填）；虚拟机隔离。",
      "延迟账：每步秒级×几十步=分钟级——不适合高频任务，适合跨系统长尾自动化。"],
    drill: ["为什么比 API agent 难？→ 观察像素级有噪声+动作不可精确撤销+无结构化错误码+环境漂移（页面改版即全废）。",
      "怎么降误点？→ 动作前元素回显+hover 预览+高危二次确认；wait-for-stable 防页面未加载完就点。",
      "和 RPA 区别？→ RPA 路径写死；computer use agent 控制流由模型对观察的反应决定（考点 1 金句 GUI 版）。",
      "评测？→ OSWorld（GUI 桌面）/WebArena（网页）分层选型，环境贵，私有任务集+录制回放。"],
    trap: "吹\"万能操作电脑\"不提速度与稳定性代价；不区分视觉/结构两条路线说明没实操。"},
  {id: "P6", tag: "长时程", t: "长时程自动化 Agent（8 小时/定时/后台）", brief: "设计一个能连续跑数小时的监控-分析-汇总类任务，不跑偏、不丢状态、能恢复。",
    min3: ["开口先报三件套：compaction（s08 压缩）+checkpoint（s10/s12 持久化）+自验证（s17 goal gate）——缺一不可。",
      "触发面：cron at-least-once（先持久化 pending 再入队、ack 才删）+后台任务（bg_id 占位、完成以通知事件注入尾部 user 消息）。",
      "无人值守约束：自动触发轮只许拒绝危险命令、不许替用户盲批——交互审批不可用即 deny-by-default 拒绝。",
      "恢复链：429 指数退避→529 两次切兜底模型→max_tokens 升档续写→超长 reactive compact 一次→预算熔断挂起待人工。"],
    drill: ["崩在中间一步？→ 读 ckpt（消息截断点+任务状态+悬空 tool_use 清单）→补发 tool_result（API 校验 1:1 配对）→续跑。",
      "怎么发现空转？→ 步数/token 分布监控+相同动作重复检测+连续 block 8 次上限交还且不伪装成功。",
      "为什么不定死最大轮数？→ 完成性每轮刷新，固定 N 轮既杀合法长任务又不能证明完成；预算只做防失控出口。",
      "多 agent 抢任务？→ 原子认领（检查-修改-落盘一锁全包）；信箱破坏性读防重复消费。"],
    trap: "只答\"加个 max_turns\"是把终止当兜底；三件套+恢复链+无人值守权限收紧才是完整答案。"}
];

const CODE = [
  {id: "T1", t: "最小 Agent Loop（白板默写版）", idea: "四要素：模型决策/客户端执行/观察回填/终止三分类；完整版 60 行见仓1 05题1，默 13 行骨架。",
    code: `def agent_loop(user_input, tools, max_turns=25):
    messages = [{"role": "user", "content": user_input}]
    for _ in range(max_turns):
        resp = llm_create(messages, tools=tools)
        if not resp.tool_calls:
            return resp.content
        for call in resp.tool_calls:
            try:
                result = dispatch(call, tools)
                messages.append(tool_msg(call.id, trunc(str(result))))
            except Exception as e:
                messages.append(tool_msg(call.id, "Error: " + str(e)))
    return escalate_to_human()`,
    pit: "易错自查：忘 max_turns；异常没原地捕获击穿循环；tool_msg 不带 call.id（API 1:1 配对校验 400）；大结果不截断。追问：tool_result 为什么回 user 角色（Anthropic Messages API 约定；OpenAI Chat Completions 为独立 tool 角色；Observation 语义）。"},
  {id: "T2", t: "工具注册表 + 未知工具自纠", idea: "dispatch map 让循环零改动生长工具（s02）；未知工具回\"清单\"让模型自纠；schema 校验失败也回文本。",
    code: `TOOLS = {}
def register(fn):
    TOOLS[fn.__name__] = {"fn": fn, "schema": fn.schema}
    return fn

def dispatch(call, tools):
    if (t := tools.get(call.name)) is None:
        return "Tool unknown: " + call.name + " Available: " + str(sorted(tools))
    args = validate_schema(call.arguments, t["schema"])
    return t["fn"](**args)`,
    pit: "易错：未知工具直接 raise 击穿 REPL，模型失去自纠机会。追问：错误回填 vs 抛异常（异常终止循环；文本回填让模型下一轮修正——错误处理三层：自纠>重试>人审）。"},
  {id: "T3", t: "分层恢复链（重试/退避/预算/降级）", idea: "按错误类型分层处置，对齐 s15：429 退避、529 两次切兜底模型、max_tokens 升档续写、超长 reactive compact 一次、预算熔断转人审。",
    code: `def call_llm_robust(messages, st):
    backoff, overloaded = 1, 0
    while True:
        if st.turns >= MAX_TURNS or st.tokens >= MAX_TOKENS:
            raise BudgetExceeded(st)
        try:
            return llm_create(messages, model=st.model)
        except RateLimit:
            time.sleep(backoff + random.random()); backoff *= 2
        except Overloaded:
            overloaded += 1
            if overloaded >= 2: st.model = FALLBACK_MODEL
        except LengthLimit:
            st.max_tokens = min(st.max_tokens * 2, 16000)
        except ContextOverflow:
            messages = reactive_compact(messages)`,
    pit: "易错：所有错误塞一个 try 统一重试——错误分类决定恢复动作。追问：reactive compact 为什么限一次（估算误差用一次重试兜底；连续 compact=估算系统性偏差要修配置）。"},
  {id: "T4", t: "结构化输出三层防线", idea: "①声明层（prompt+schema）②校验层（失败带错误回填重试）③兜底层（FC/约束解码）。生产用③，手写①②面试演示。",
    code: `def ask_json(prompt, schema, retries=2):
    msgs = [{"role": "user", "content": prompt}]
    for i in range(retries + 1):
        raw = llm_create(msgs)
        try:
            return jsonschema_validate(json.loads(raw), schema)
        except (json.JSONDecodeError, ValidationError) as e:
            msgs += [{"role": "assistant", "content": raw},
                     {"role": "user", "content": "不合 schema：" + str(e) + "，请修正重出"}]
    raise SchemaFailed(raw)`,
    pit: "易错：只 try 不重试；错误信息不回填（模型不知道错在哪无法自纠）。追问：FC 为什么比求 JSON 稳（约束解码，格式在解码层保证不是靠模型自觉）。"},
  {id: "T5", t: "子代理调度器（一次性隔离）", idea: "task 工具=新 messages 的同步子循环；三铁律：全新上下文/只回传最终文本/子工具表剔除 task 防递归委派（s06）。",
    code: `def task(description):
    sub_tools = {k: v for k, v in TOOLS.items() if k != "task"}
    text, _ = agent_loop(description, sub_tools, max_turns=30)
    return text`,
    pit: "易错：把主循环 messages 传给子代理——隔离本来就是目的。追问：什么不该给子代理（不可再生推理脉络/主任务状态留主循环；可再生细节才隔离）。为什么剔除 task（不剔除=无界成本嫁接，A 委 B、B 委 C）。"},
  {id: "T6", t: "Checkpoint 恢复", idea: "ckpt 三件套：消息列表截断点/任务状态/悬空 tool_use 清单；恢复先补发 tool_result 再续跑（API 校验 1:1 配对）。",
    code: `def save_ckpt(path, st):
    write_json(path, {"messages": st.messages, "tasks": st.tasks,
                      "pending_tool_uses": st.pending})

def resume(path):
    st = load_json(path)
    for call in st.pending_tool_uses:
        st.messages.append(tool_msg(call.id, "interrupted: result lost, re-check"))
    return agent_loop_continue(st)`,
    pit: "易错：只存 messages 忘存任务状态——重启后失忆重做。追问：悬空 tool_use 不补会怎样（API 400，每个 tool_use 必须有配对 tool_result——协议 1:1 不变式）。"}
];

const PLAN = [
  {d: "第 1 天", t: "背十步模板+六层图；P1 精读；跑通 s01/s02", o: "十步脱稿口述录音"},
  {d: "第 2 天", t: "P2+P3 精读；D 组对照补检索细节", o: "两题白板图各画一遍"},
  {d: "第 3 天", t: "P4+P5+P6 精读；s15 恢复链对照", o: "P6 三件套+恢复链默写"},
  {d: "第 4 天", t: "T1–T6 手敲一遍（不复制粘贴）；错题自查", o: "六段代码白板可默写"},
  {d: "第 5 天", t: "可观测章精读；给实验 agent 加 turn-span 落盘", o: "一份真实 trace JSON"},
  {d: "第 6 天", t: "选型决策树背诵；速答卡过第一遍", o: "决策树白板默写"},
  {d: "第 7 天", t: "模考：抽 3 道场景题限时 15 分钟走完三段式", o: "3 份录音+弱点清单"}
];

