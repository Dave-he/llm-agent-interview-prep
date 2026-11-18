# 字节跳动真实面经逐题详解（11 场 · 97 题）

> 本文是 `01-ByteDance-Questions.md` 的配套详解。每题给"精简详解"：**核心答案链 + 一句追问**，每题独立可读懂；同题多次出现的按面经内顺序给一致答案。涉及长推导/长代码（DPO 推导、手撕 attention、BPE、LRU 等）给要点版，注明"完整版见 xx 章节相应题"。题号在每场内连续编号（原文件一面/二面合并计数）。

---

## 本章考点分布

原文件频率表：GRPO vs PPO 7/11、Self-Attention 计算 6/11、KV Cache 6/11、手写 Attention 5/11、LoRA 5/11、DPO 推导 4/11、Flash Attention 4/11、RAG 优化 4/11、MCP/Agent 4/11、ReAct 3/11、BPE 3/11、MoE 3/11、系统设计 3/11。

按模块合并看，97 题的分布为：**架构与 Attention 细节约 18 题**（含 RoPE、mask、归一化、激活）、**训练与对齐约 17 题**（RLHF/PPO/GRPO/DPO/SFT/LoRA）、**推理优化约 15 题**（KV Cache/Flash/投机/PagedAttention/量化）、**Infra 与系统设计约 13 题**（并行策略/ZeRO/vLLM/CUDA/kernel）、**RAG 约 12 题、Agent 约 14 题、安全约 6 题、手撕代码约 11 题、教育与多模态/热点约 6 题**。结论清晰可背：**字节面试三大火力点是"对齐算法（尤其 GRPO）+ KV/Attention 机制 + 手撕代码"**，二面必考 1 道系统设计或 1 道长推导。

---

## 面经 1：豆包大模型团队（2026.01）—— 13 题

**本场火力分布**：一面主打基础机制（Attention/KV/GRPO/手撕）+ RAG 项目深挖；二面上强度：MLA/Flash/DPO 长推导 + Agent 记忆与多轮服务两道系统设计。对齐与推理各占 1/3。

### 题 1：Self-Attention 的计算过程和复杂度分析

- **核心答案链**：输入 X∈R^(n×d) 经三个投影得 Q,K,V；Attention=softmax(QK^T/√d_k)V。两步矩阵乘：QK^T 产生 n×n 相似度矩阵，O(n²·d) FLOPs 与 O(n²) 显存；·V 再 O(n²·d)。三投影 O(n·d²)。总时间 O(n²d+nd²)、空间 O(n²+nd)。n≫d 时平方项主导——这就是长上下文不可扩展的根因，也是 Flash Attention 与 GQA/MLA 都针对的对象。
- **一句追问**：反向传播复杂度？——同数量级（需保存/重算 Q,K,V 和 S 的梯度），Flash 反向用重算替代存储。

### 题 2：为什么要除以 √d_k？

- **核心答案链**：假设 q、k 各分量均值 0 方差 1 且独立，则 q·k=Σq_i k_i 方差为 d_k（每项方差 1 × d_k 项）。点积标准差 √d_k，d_k=128 时 logits 幅值 ~10+，softmax 进入饱和区：大 logit≈1、小≈0，梯度萎缩且近似 one-hot。除以√d_k 使方差回到 1，softmax 保持在有梯度的温度区间。等价于把 softmax 的温度固定为 1。
- **一句追问**：为什么是除而不是中心化？——点积本身均值仍近 0，问题只在方差，缩放即可；换 sigmoid/GELU 打分就不再需要此归一。

### 题 3：Multi-Head Attention 的作用，为什么不用单头？

- **核心答案链**：单头只有一组 Q,K,V 投影，一次 softmax 只能表达"一种"相关性（低秩、单一注意力分布）。多头把 d 维拆成 h 个 d_h=d/h 子空间，各自独立算 attention 后拼接再投影 W^O：等于 h 个不同表示子空间下的关系模式（句法、共指、位置等）并行集成，总计算量不变（d_h·h=d）。本质是"集成的注意力分布 + 低秩约束正则"。
- **一句追问**：头数怎么选？——过小退化为单头、过大单头维度太小表达不足；经验 d_head=64~128，Llama-32B 用 40 头（d_head=128）。

### 题 4：KV Cache 的原理，如何计算显存占用？

- **核心答案链**：自回归解码每步只新增 1 个 token，但注意力需要所有历史 token 的 K,V；把历史 K,V 存下来避免重算，即 KV Cache。显存公式：**KV bytes = 2(K,V) × L(层数) × n_kv(每层 KV 头数) × d_head × s(序列长) × batch × 每元素字节数**。例：7B 模型（L=32、n_kv=32、d_head=128），fp16、4096 上下文、batch=1：2×32×32×128×4096×2B=2GB。可见长序列×大 batch 下 KV 反超权重（70B 权重 fp16 才 140GB，KV 随 s·b 线性涨）。
- **一句追问**：完整代码级推导见 02-Inference/01-KVCache 章；追问方向：GQA 后公式里 n_kv=组数、MLA 压缩到 d_c 维（见本场题 8）。

### 题 5：GRPO 和 PPO 的区别，为什么 DeepSeek 用 GRPO？

- **核心答案链**：PPO 四模型（actor/critic/ref/rew），用学习到的 V(s) 做基线算 advantage。GRPO 删除 critic：对同一 prompt 采 G 个响应，用组内平均奖励当基线，A_i=(r_i−mean)/std。优点：省一个大模型（显存-约20%，训练流程简化）、组内相对比较天然适配"同一问题多个答案"的可验证奖励。代价：每步采样量×G、估计的是 response 级 advantage（粒度比 token 级 value 粗）。
- **一句追问**：std=0（组内全对/全错）怎么办？——DAPO 提"动态采样过滤零组"，DeepSeek 工程上按常数 EPS 裁剪防除零。

### 题 6：手写 Self-Attention（Python）

- **核心答案链**（要点版，完整实现与测试见 08-Coding/01 对应题）：
```python
import torch, torch.nn.functional as F
def self_attention(x, wq, wk, wv, mask=None):
    # x:(B,n,d); wq/wk/wv:(d,d)
    Q, K, V = x @ wq, x @ wk, x @ wv          # (B,n,d)
    s = Q @ K.transpose(-2, -1) / (Q.size(-1) ** 0.5)   # (B,n,n)
    if mask is not None: s = s.masked_fill(mask, float('-inf'))
    return F.softmax(s, dim=-1) @ V
```
复杂度 O(n²d+nd²)。关键点：scaled 除法、mask 用 -inf 再 softmax、可选 dropout 在 softmax 之后。
- **一句追问**：加多头就在attention 前把 x reshape 成 (B,h,n,d/h)；面试常接着问 causal mask 写法。

### 题 7：项目中 RAG 的检索效果如何优化？

- **核心答案链**：按 pipeline 分层报：①数据层——清洗/结构化切分（语义+重叠 chunk）；②召回层——混合检索（BM25+向量）+ 查询改写（HyDE/Multi-Query）+ 调 embedding；③精排——cross-encoder rerank；④生成侧——引用约束+拒答。量化口径必备：Recall@10/Hit Rate 定位召回瓶颈，rerank 后 nDCG/MRR 定位排序瓶颈；RAGAS 四指标评整体。
- **一句追问**：recall 高但答案错，查什么？——排序（rerank 位置）与生成阶段（上下文没被用/引用错）。

### 题 8（二面 1）：MLA 的原理，与 GQA 相比的优势？

- **核心答案链**：GQA 用"KV 头分组共享"省 cache（多少损失精度）。MLA 把 K,V 联合低秩投影到latent c_KV（维度 d_c≪d_model）存进 cache，注意力前用两个吸收矩阵上采回 q,k 空间；RoPE 部分单独走小维度，避免与旋转矩阵冲突。成绩：DeepSeek-V2 论文摘要口径——KV cache 压缩 **93.3%**（降至 MHA 的 1/15 量级，等效 GQA 只留 2.25 组）、最大生成吞吐 **5.76×**，效果不掉反升（低秩结构自带压缩正则）。相对 GQA：不靠"砍头"而是靠"低秩重构"，质量与压缩解耦。
- **一句追问**：为什么解耦 RoPE？——旋转矩阵与低秩矩阵不可交换，位置部分单独缓存 d_rope 维。

### 题 9（二面 2）：Flash Attention 如何实现的？为什么更快？

- **核心答案链**：①tiling：Q/K/V 分块进 SRAM，out-of-core 算 attention，不物化 n×n S 矩阵；②online softmax：块间增量维护 m=max、l=Σexp，O 乘 e^{m_old−m_new} 校正，数学与全局 softmax 等价；③反向用保存的 m,l 重算省显存。快的真正原因：HBM 读写是瓶颈（memory-bound），flash 把 IO 从 O(n²) 降到 O(nd)，计算用 tensor core 打满。典型加速 2–4×、序列越长越明显。
- **一句话追问**：显存为什么降到 O(n)？——不存 S 矩阵，只存 O 矩阵与 m/l 变量。

### 题 10（二面 3）：DPO 的数学推导

- **核心答案链**（要点版，四步完整推导见 04 章标注的 03-FineTuning/02-Alignment 对应题）：①RLHF 目标 max_π E[r]+β·KL(π‖π_ref)；②其闭式最优解 π*(y|x)∝π_ref(y|x)·exp(r(x,y)/β)；③反解 r=β log(π/π_ref)+Z(x)，配分函数 Z 只依赖 prompt；④代回 Bradley-Terry 偏好模型 p(y_w≻y_l)=σ(r_w−r_l)，Z 逐项相消 ⇒ **L_DPO=−E[log σ(β(log π_θ(y_w|x)/π_ref(y_w|x))−log π_θ(y_l|x)/π_ref(y_l|x))]**。全程无 reward model、无 RL。
- **一句追问**：β 的作用？——隐式奖励的温度：小 β 偏离 ref 更狠（更激进），大 β 更保守对齐 ref。

### 题 11（二面 4）：如何设计一个 Agent 的记忆系统？

- **核心答案链**：分层：①工作记忆=当前窗口+ scratchpad；②情景记忆=事件日志（工具调用与结果，向量化可回查）；③语义记忆=从交互固化的事实/画像；④程序记忆=技能库/失败教训。机制四件套：写入（口径与置信度）、检索（向量+关键词混合+时间衰减）、更新（冲突合并）、遗忘（TTL/重要性打分）。关键设计取舍：全量进窗 vs 检索注入的成本与准确性权衡。
- **一句追问**：记忆冲突怎么处理？——时间戳+来源可信度做版本合并，关键事实加确认门。

### 题 12（二面 5）：MCP 和 Function Calling 的区别

- **核心答案链**：Function Calling 是**模型能力**：把工具 schema 放进上下文，模型输出结构化调用参数，执行靠宿主应用自己写。MCP 是**开放协议**：client-server 架构，标准化 tools/resources/prompts 三类原语，stdio/Streamable HTTP 传输，一次接入多方复用、权限/生命周期有协议约定。类比：FC=「模型会开单」，MCP=「全城统一接单协议」。关系：MCP server 的工具最终仍以 FC 形式暴露给模型。
- **一句追问**：什么时候自己写 FC？——单应用少量工具、强内控；生态级集成、多客户端复用上 MCP。

### 题 13（二面 6）：设计一个支持多轮对话的大模型服务

- **核心答案链**：分层说：①接入/会话层——session 粘性与上下文存储（Redis/DB，消息+摘要双份）；②推理层——vLLM 引擎池 + **prefix caching 复用前缀 KV** + continuous batching；③上下文管理——超窗滚动摘要/向量检索历史；④可靠性——SLO 分级（TTFT/TPOT）、限流降级、重试幂等；⑤观测——逐 token 延迟分布、缓存命中率。亮点话术：把"多轮"=「存储 + KV 复用 + 上下文压缩」三件事讲清楚。
- **一句追问**：多轮历史全塞进去还是检索？——主题漂移用摘要、引用用检索注入，混合策略。

---

## 面经 2：AI 应用研发（2026.01）—— 10 题

**本场火力分布**：均衡型"应用研发卷"——架构 2 题训练 4 题 RAG 2 题、手撕 2 题（MHA+LRU）。侧重"讲清楚原理 + 工程落地感"。

### 题 1：Transformer 的整体架构，Encoder 和 Decoder 的区别

- **核心答案链**：架构=输入嵌入+位置编码×N 层（自注意力+FFN，残差+Norm）+输出头。区别三点：①mask——encoder 双向可见，decoder 因果 mask；②注意力——encoder 只有自注意力，decoder 多一层 cross-attention（Q 来自 decoder、K/V 来自 encoder 输出）；③用途——理解/表示 vs 自回归生成。
- **一句追问**：现在生成模型为什么用 decoder-only？——见面经 11 题 1（每 token 监督信号+scaling 友好）。

### 题 2：RoPE 旋转位置编码的原理

- **核心答案链**：把 q,k 每两维当作复平面上的点，按位置 m 旋转角度 mθ_j（θ_j=base^{−2j/d}）。点积 (R_m q)^T(R_n k) 只依赖相对位置 m−n（旋转可交换），天然编码相对位置。性质：绝对位置诱导、相对性保证、外推可用 NTK/插值调 base，是当前主流（LLaMA/Qwen/DeepSeek 全家）。
- **一句追问**：RoPE 为什么和低秩 KV（MLA）打架？——R_m 与低秩吸收矩阵不交换，故解耦出一小段 rope 键单独缓存。

### 题 3：LoRA 的原理，秩 r 怎么选？

- **核心答案链**：冻结 W，学增量 ΔW=BA（A∈R^{r×d}，B∈R^{d×r}），输出 hx+(B A x)·α/r。参数从 d·k 降到 r(d+k)；初始化 A 高斯、B 零保证起点=原模型。r 选择：权重更新是低秩这个假设下，任务越"垂直"r 要求越低；实践 4–64，指令跟随 8–32、领域/多语言迁移偏大；用验证集收益曲线选，r 加倍收益<1% 即停。QLoRA 再叠 4bit 基座。
- **一句追问**：哪些层适配？——默认 attention 的 q/v 投影起步，激进做法 all-linear。

### 题 4：RLHF 三个阶段分别是什么？

- **核心答案链**：①SFT——指令数据监督微调，让模型"会答题"；②RM——同 prompt 多答偏好标注，Bradley-Terry 损失训练打分模型；③RL 优化——PPO（或 DPO 直推）最大化奖励同时 KL 约束不漂离 SFT。三阶段是框架不是铁律：DPO 合并 2+3、R1-Zero 跳过 1。
- **一句追问**：哪一步最贵最不稳？——PPO（四模型在线）；工业界低成本路线=高质量 SFT+DPO/GRPO。

### 题 5：PPO 中 Critic 模型的作用

- **核心答案链**：Critic≈价值网络 V(s)，学"当前状态未来预期回报"；用途：①算 advantage δ=r+γV(s')−V(s) 做方差缩减的基线；②GAE(λ) 多步信用分配到 token 级。结果：梯度噪声显著降低、训练稳定。代价：与 policy 同规模显存+训练管线复杂。
- **一句追问**：Critic 和 RM 什么关系？——RM 打分给"最终奖励"，Critic 估"过程价值"；RM 常兼作 Critic 初始化但语义不同。

### 题 6：GRPO 怎么去掉 Critic 的？优势是什么？

- **核心答案链**：同 M1-Q5 一致答案：同 prompt 组采样 G 个，组内均值当基线、std 归一，A_i=(r_i−mean)/std；KL 仍对 ref 保。优势：省 critic（少一个 ~policy 同规模模型的训练与显存），采样天然并行；对可验证奖励任务（数学/代码）尤其匹配。显存账：70B 训练少一个 70B critic + 其梯度/优化器状态（约省 140GB+ fp16 级别的内存带宽压力）。
- **一句追问**：代价？——response 级粒度粗、每个 prompt 采样倍增；于是有 token 级修正版本（DAPO 等）。

### 题 7：RAG 中 embedding 模型怎么选？chunk 大小怎么定？

- **核心答案链**：embedding 选型四步：①榜单锚定（MTEB/C-MTEB）看领域子集；②维度/推理成本（越大越准越贵）；③长文本支持（8k 上下文 embedding 适合长文档）；④私有领域做对比评测再决定微调与否（contrastive，难负样本）。chunk：起点 256–512 token+10–15% overlap；原则是"语义完整>固定长度"，实验以 Recall@k 回填定标。
- **一句追问**：chunk 大小怎么验证最优？——网格扫 chunk×overlap×topk 三元组，看 end-to-end faithfulness+hit rate。

### 题 8：BM25 和向量检索的区别

- **核心答案链**：BM25=稀疏词频打分（TF-IDF 变体+长度归一），零训练、可解释、精确词/专名强、语料外鲁棒；向量=稠密语义泛化，同义改写/跨语言强，但依赖模型与切分，专业术语/新词易漂。生产答案基本都是 hybrid（RRF 融合）因为错误模式互补。
- **一句追问**：RRF 为什么常用而不用加权分数？——rank-based 融合免校准两路分数分布。

### 题 9：手写 Multi-Head Attention

- **核心答案链**（同 M1-Q6 一致答案，多一个分头步骤；完整带注释版见 08-Coding/01）：核心三步——投影后 `q.view(B,n,h,dh).transpose(1,2)`；scaled dot-product（含 causal mask）；`out.transpose(1,2).reshape(B,n,d) @ Wo`。复杂度不变 O(n²d+nd²)，多的是 h 次 d/h 维小矩阵乘。
- **一句追问**：手写之后常被问 KV-cache 版怎么改（只算新 token 的 q，K/V append）——见面经 7 题 7。

### 题 10：LRU Cache（LeetCode 146）

- **核心答案链**（要点版，完整 O(1) 实现见 08-Coding/01）：哈希表 key→Node + 双向链表；get：命中后把节点搬到头；put：存在则更新+前移，不存在则头插、超容删尾。全部 O(1)。可用 `OrderedDict(move_to_end/popitem)` 一行级实现，但要口头说明双向链表手动版以显功底。
- **一句追问**：工程变体？——分片锁防竞争、TTL-LRU、Redis LRU 近似淘汰（采样）。

---

## 面经 3：搜索与推荐（2025.12）—— 10 题

**本场火力分布**：训练细节与工程优化主导（分词/归一化/优化器/精度 5 题），推理优化 4 题（KV/MoE/量化/投机），手撕 BPE 1 题。典型"底层细节岗"卷。

### 题 1：BPE 分词算法的训练和推理过程

- **核心答案链**：训练：①初始词表=所有单字符（含字节级）；②统计语料中相邻符号对频次；③把频次最高的 pair 合并成新 token 加入词表，重复至目标词表大小。推理（encode）：按训练时常量词表**从长到短贪心匹配**（或按合并序），确保与训练分布一致。
- **一句追问**：为什么字节级 BPE（BBPE）解决了什么问题？——OOV 彻底消失：任何 UTF-8 都能拆到 256 字节兜底。

### 题 2：LayerNorm vs BatchNorm

- **核心答案链**：BN 沿 batch 维归一（每特征跨样本统计量），依赖 batch 大小、训练/推理统计不一致；LN 沿特征维对单样本归一，与 batch 无关。NLP 中变长序列、batch 内 padding、batch 小（大模型卡少）都让 BN 失效，LN 天然契合Transformer；CV 里 BN 有效是因为空间位置共享统计。
- **一句追问**：RMSNorm 把 LN 砍掉什么还成立？——去均值（见 M5-Q4，方向不变性影响小）。

### 题 3：Pre-Norm vs Post-Norm，为什么现在都用 Pre-Norm？

- **核心答案链**：Post-Norm（原版）残差出口归一化，深层时梯度要穿过每层的 norm，易发散需 warmup 精调；Pre-Norm 把 norm 放残差分支入口，主通路 x 无阻尼直达输出，梯度ijf稳定，深层可训（几十上百层常态）。代价：表示能力略受限（有研究称相同深度 post 更优），但稳定性收益完胜。
- **一句追问**：为什么 Post-Norm 深层训不动？——残差路径上的 LayerNorm 每层重缩放梯度，深网起爆点敏感。

### 题 4：Adam 和 AdamW 的区别

- **核心答案链**：Adam 自适应矩估计（一阶/二阶动量+偏差修正）；但把权重衰减混进梯度后，会被自适应分母重缩放，L2 正则失效。AdamW 解耦：衰减项直接作用参数 w←w−η(λw+m̂/(√v̂+ε))，正则不被自适应缩放。训大模型几乎只有 AdamW（wd≈0.1）。
- **一句追问**：还有哪些变体值得提？——8-bit Adam 省显存、Lion（符号更新）、Muon（近期流行）。

### 题 5：混合精度训练，BF16 vs FP16 的区别

- **核心答案链**：混合精度=前向/反向用半精度+权重 FP32 master copy+FP32 累加。FP16：尾数 10 位动态范围小（上限 65504），易上/下溢需 loss scaling；BF16：指数 8 位同 FP32、范围同量级，几乎不溢出可免 scaling，精度损失换稳定性。现代训练默认 BF16（Ampere+ 支持）。
- **一句追问**：哪些东西绝不能半精度？——优化器状态、loss/累加器、LayerNorm 统计量、softmax 内部。

### 题 6：KV Cache 优化方法有哪些？

- **核心答案链**：与 M1-Q4 一致（公式），再上手段五层：①结构——GQA/MLA/YOCO；②管理——PagedAttention 分页、prefix 复用、按需淘汰（HF sliding window/StreamingLLM 首锚窗）；③数值——KV 量化（int8/fp8 per-channel）；④工具——投机解码间接降低单位请求占用（短驻留）、闪电注意力（线性化）。补充：长上下文下 KV 显存可与权重同量级，管理=显存分配问题。
- **一句追问**：量化 KV 会不会影响长程注意力？——会，对离群 softmax 分数敏感，常按 head/channel 分组量化缓解。

### 题 7：DeepSeek-V3 的 MoE 架构设计

- **核心答案链**：要点：①671B 总参/37B 激活，稀疏度约 1/18；②细粒度专家（256 ranked experts，每 token 激活 8+1 共享专家）；③auxiliary-loss-free 负载均衡（bias 动态调节路由）避免主任务被均衡损失伤；④FP8 训练+MTP（multi-token prediction）提采样密度。叙事核心：细专家×共享专家=专业分工不牺牲通用能力。
- **一句追问**：MoE 的负载不均衡怎么办？——capacity factor+drop/rewrite、bias 调节；评估指标=专家利用率方差。

### 题 8：模型量化的方法，GPTQ 和 AWQ 的区别

- **核心答案链**：与 M10-Q5 联动。同点：训练后权重量化（PTQ），W4A16。GPTQ：基于 Hessian 的逐列误差补偿（量化一列、把误差转嫁给未量化列），显存/时间均衡，通用；AWQ：观察到"激活显著的权重通道更关键"，per-channel scale 重保护显著通道，无需反传、kernel 友好、对分布异常更稳。
- **一句追问**：还有别的路线？——SmoothQuant（激活也缩到 W8A8）、bitsandbytes Nf4、HQQ、SqueezeLLM。

### 题 9：投机解码的原理

- **核心答案链**：与 M10-Q3 一致：小模型草稿 k 个 → 大模型一次前向并行验证 → 按拒绝采样修正，输出分布=大模型。加速比≈(1−α^{k+1})/(1−α)·(k+1)/(k·τ+1)（α=接受率，τ=draft/target 时间比）；工程用 n-gram 匹配/EAGLE/draft 头提高 α。
- **一句追问**：批处理下还有效吗？——大 batch 下 bottle 是 compute，边际收益下降；对单流/低并发最有效。

### 题 10：手写 BPE 分词器

- **核心答案链**（要点版，完整可运行实现见 08-Coding/01）：两个函数——`train(corpus, vocab_size)`：dict 计数 pair→合并并记录 merge 顺序；`encode(text)`：pre-tokenize（正则切块）→逐块按 merge 优先级循环合并。核心陷阱：merge 必须按训练次序（保存 rank dict）。
- **一句追问**：与 WordPiece 区别？——WordPiece 按互信息/似然增益选合并，BPE 按频次；Unigram 反向从大词表剪枝。

---

## 面经 4：Agent 平台（2025.12）—— 10 题

**本场火力分布**：Agent 岗——概念+系统设计 8 题、手撕 2 题（余弦相似度+TopK）。重点考"组件拆解能力与错误恢复"这类工程判断。

### 题 1：什么是 LLM Agent？核心组件有哪些？

- **核心答案链**：定义=以 LLM 为推理核心、能感知环境/规划行动/使用工具/累积反馈以完成目标的系统。四件套：①Planning（任务分解/反思/自我批判）；②Memory（上下文内 scratchpad+外部存储）；③Tool Use（function calling/MCP、代码解释器、检索）；④Action 循环（观察→思考→行动→观察）。必备：错误恢复与终止条件。
- **一句追问**：与 chatbot 差别？——chatbot 只做一次映射，Agent 有多步交互与状态。

### 题 2：ReAct 框架的原理

- **核心答案链**：Thought(推理)→Action(工具调用)→Observation(结果)三步交替循环直到 Finish。作用：①CoT 的推理显式化；②行动把外部事实拉进来，减少幻觉；③轨迹可读可调试。工程要点：prompt 模板+停止条件+最大步数兜底。
- **一句追问**：ReAct 的失败模式？——观察噪声导致偏航、无限循环（重复同 action）、陷入局部计划；对策是反思与重规划。

### 题 3：Function Calling 的工作流程

- **核心答案链**：与 M1-Q12 衔接：①定义工具 JSON Schema；②inject 进系统上下文；③模型判定需要则输出 JSON 参数（structured output）；④宿主执行并回传 tool result；⑤模型基于结果续写或再调用；⑥多轮循环至完成。协议要点：并行调用、错误回填、schema 校验。
- **一句追问**：怎么防"模型瞎编参数"？——schema 验证+受限解码/约束采样。

### 题 4：MCP 协议的架构和通信方式

- **核心答案链**：同 M1-Q12 补足协议细节：Host(IDE 应用)、Client(每 server 一个连接)、Server(能力提供方)。传输 stdio（本地进程）/Streamable HTTP（远程，SSE 下行+POST 上行）。三类能力：tools（模型可调用）、resources（ App 数据）、prompts（模板）。生命周期：initialize 能力协商→调用→通知。
- **一句追问**：为什么要 client-server 隔离？——权限最小化、进程隔离、生态复用。

### 题 5：Agent 如何处理多步骤任务的错误恢复？

- **核心答案链**：四层防御：①预防——schema 校验/沙箱执行/白名单；②检测——输出验证器+断言；③恢复——把错误 Observation 回填让模型自修（reflexion 式）、局部重试、回退到检查点；④兜底——最大步数/预算熔断、人工移交。核心设计哲学：**把错误当 Observation，而不是异常**。
- **一句追问**：什么错误不该模型自修？——不可逆副作用（写库/支付）走人工确认。

### 题 6：多智能体系统的协作模式

- **核心答案链**：四种主流模式：①编排者-执行者（orchestrator 分派任务）；②流水线接力（生成→评审→修订，session handoff）；③辩论/投票（多 agent 独立作答+judge 聚合）；④社会模拟/角色群聊。关键取舍：通信成本 vs 精度；共享黑板（状态总线）避免上下文平移丢失。
- **一句追问**：如何防多 agent 互相污染？——隔离上下文+中间结论结构化+裁决者。

### 题 7：如何评估 Agent 的准确性和可靠性？

- **核心答案链**：三维度——①任务成功率（端到端 pass@1）、步效率（完成步数/预算）；②过程质量：轨迹 vs 参考轨迹（LLM-judge 评分）、工具调用正确率；③鲁棒性：扰动注入（工具失败/噪声观察/对抗输入）下的稳定性+成功率方差。框架：AgentBench/τ-bench。上线指标：成功率+成本+时延三线。
- **一句追问**：为什么 step 准确率高≠任务成功？——错误会传播，端到端才是金标准。

### 题 8：设计一个带记忆的多轮对话 Agent

- **核心答案链**：同 M1-Q11 一致答案，补充工程侧：存储分层（热：Redis 会话；冷：向量库+事实表）、写入时机（turn 结束异步抽取事实）、检索策略（混合+去重+时间衰减）、更新冲突处理、评测（记忆命中率/事实一致性）。重点叙事：**记忆是"写-读-改-忘"全链路，不只是个向量库**。
- **一句追问**：怎么防记忆被用户"投毒"？——来源隔离+敏感写入二次确认+置信度衰减。

### 题 9：手写余弦相似度搜索函数

- **核心答案链**（要点版，完整 numpy/torch 版与批量优化见 08-Coding/01）：
```python
import numpy as np
def cos_topk(q, X, k):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    qn = q / np.linalg.norm(q)
    sims = Xn @ qn                       # (N,)
    idx = np.argpartition(sims, -k)[-k:] # O(N) 选择
    return idx[np.argsort(sims[idx])[::-1]]
```
复杂度 O(N·d) 检索 + O(N) topk；海量数据换 FAISS/HNSW 近似。
- **一句追问**：为什么要归一化？——等价于把点积变成角度，检索时可用内积索引。

### 题 10：TopK 问题（堆排序）

- **核心答案链**（要点版，完整实现见 08-Coding/01）：维护小顶堆大小 k，扫一遍：元素>堆顶则替换；结束堆内即 TopK。时间 O(n log k)、空间 O(k)；海量/分布式可桶内 topk→归并。对比：全排序 O(n log n)、快选平均 O(n) 但不稳定。
- **一句追问**：流式数据怎么办？——堆天然支持，或 count-min/蓄水池变体。

---

## 面经 5：大模型算法（2025.11）—— 10 题

**本场火力分布**：算法功底卷——FLOPs/结构组件/长上下文 5 题、对齐 3 题、幻觉与手撕 2 题。计算量化题多，对数字敏感度要求高。

### 题 1：Transformer 的 FLOPs 如何估算？

- **核心答案链**：经验式：**训练 6·N·D FLOPs**（N 参数、D token；前向 2ND 反向 4ND）；推理前向 2·N·D。attention 额外项 12·L·s²·d（同上 M1-Q1）。例：7B、1T token：6×7e9×1e12=4.2e22 FLOPs；A100 312TFLOPs(BF16)、MFU≈40% ⇒ 单卡约 12e14/s ⇒ 显著多卡×多天。掌握6ND 即可现场心算训练成本。
- **一句追问**：MFU 怎么算？——实际吞吐/理论峰值；MFU 低通常是通信/IO bound。

### 题 2：GQA 的原理和实现

- **核心答案链**：把 KV 头数从 h 降到 g（h%g=0），每 g 个 Q 头共享一组 KV。实现上仅改 reshape：K:(B,g,s,dh)→expand 到 (B,h,s,dh)（`repeat_interleave` 或 tile）。MHA→GQA：KV cache 与带宽 ÷(h/g)，如 32 头→8 组省 4×；质量几乎无损（70B 级实测）。
- **一句追问**：MQA（g=1）什么时候掉点明显？——头间 KV 差异大的模型/细粒度分布任务；经验折中 g=4~8。

### 题 3：SwiGLU 激活函数

- **核心答案链**：FFN(x)=W_down·(Swish(W_gate x)⊙W_up x)，Swish(x)=x·σ(x)。GLU 思想：一路算内容一路算门，逐元素相乘实现"条件激活"。消融结论（Shazeer）：同参量下 SwiGLU 稳定优于 GELU-FFN。参数守恒技巧：隐层维度取 2/3·4d。Llama/Qwen 全系使用。
- **一句追问**：为什么比 ReLU 好？——平滑有负区+门控乘法交互，梯度表现好。

### 题 4：RMSNorm 是什么？和 LayerNorm 的区别

- **核心答案链**：RMSNorm(x)=x/√(mean(x²)+ε)·g：**去掉均值中心化**，只做尺度归一。收益：少一遍 mean/统计量，仅这一次 pass、速度更快（~10% 级归一化开销节省），对方向信息基本无损（证明方向不变性足够）。LLaMA 系与多数 SOTA 模型标配。
- **一句追问**：为什么可以不中心化？——Transformer 后接线性层+偏置可吸收 bias，缩放为主效应。

### 题 5：长上下文怎么做？NTK-aware RoPE 的原理

- **核心答案链**：长上下文三板斧：①直接训练（数据充足最优）；②RoPE 外推改造——线性插值（每维等比压缩，高频信息受损）→**NTK-aware：改 base** θ'=θ·s^{d/(d−2)}，等效把低频维度外推、高频维度近似不动（"频谱分段插值"思想）→ YaRN 再加 attention 温度修正。③层级/滑窗限制。NTK 本质=按频率差异化处理插值率。
- **一句追问**：为什么高频维度不宜插值？——高频编码局部相对位置，压缩会破坏近距 token 分辨。

### 题 6：DPO 和 RLHF 训练哪个更好？各自优缺点

- **核心答案链**：同 M1-Q10。简明结论：DPO 优点——离线、稳定、便宜（无 RM/无 rollout，显存约 2 模型 vs PPO 4 模型）；缺点——隐式奖励受限于数据分布（分布外过拟合）、无在线探索、对偏好噪声敏感、KL 控制只能靠 β。RLHF(PPO) 优点——在线探索、可与实时代码执行器/规则奖励接、上限高；缺点——超参敏感、管线复杂。选型：数据充足/预算紧=DPO；有验证器/需要上限=PPO/GRPO。以官方与综述口径为准。
- **一句追问**：混合折中方案？——iterative DPO（在线采样本再 DPO）、KTO（按二元反馈）。

### 题 7：GRPO 的优势函数是怎么计算的？

- **核心答案链**：同 M1-Q5/M2-Q6 一致：组内 G 个采样奖励 r_i；â_i=(r_i−mean(r))/std(r)（或带 β clip 防极端）。因 GRPO 是 response 级，同一响应所有 token 共享同一 advantage（RLOO 就是其去 mean 版）。公式背后的直觉：**组间对比替代学出来的 value 估计，是蒙特卡洛版 baseline**。
- **一句追问**：为什么除 std 有争议？——会造成难度自适应缩放（难题小 std 反被放大），DAPO 建议 clip。

### 题 8：大模型幻觉的原因和解决方案

- **核心答案链**：原因五层——①数据（错误/过时/长尾分布）；②目标错位（MLE 学的是"像"不是"对"，RLHF 偏向"自信回答"）；③解码随机性；④上下文压缩（attention 稀释）；⑤知识边界（模型不知道自己知道什么）。对策对应建：RAG/工具、对齐训练减过度自信、引用与拒答训练、自我一致性/校准、解码温度控制。评估：Faithfulness 指标+幻觉率基准。
- **一句追问**：为什么 RLHF 会加剧幻觉？——偏好数据偏好"确定性答案"，模型学会硬编看起来合理的回答。

### 题 9：手写 LoRA 层

- **核心答案链**（要点版，完整带训练开关版见 08-Coding/01）：
```python
class LoRA(nn.Module):
    def __init__(self, base, d_in, d_out, r=8, alpha=16):
        super().__init__()
        self.base = base; self.base.weight.requires_grad_(False)
        self.A = nn.Parameter(torch.randn(r, d_in) * 0.01)
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scaling = alpha / r
    def forward(self, x):
        return self.base(x) + (x @ self.A.T) @ self.B.T * self.scaling
```
参数量 r(d_in+d_out)；B 零初始化保证等价基座。复杂度前向多 O(nd·r)。
- **一句追问**：为什么 B 零 A 随机？——任意非零都会破坏起点；A 非零是为了对称破缺。

### 题 10：合并 K 个有序链表

- **核心答案链**（要点版，完整实现见 08-Coding/01）：小顶堆装 k 个链头，弹最小接结果、推 next。O(N log K) 时间 O(K) 空间；分治两两归并同复杂度常数更优。
- **一句追问**：Python heapq 不能比较自定义节点怎么办？——元组打包 (val, idx) 或实现 `__lt__`。

---

## 面经 6：国际化 TikTok（2025.11）—— 10 题

**本场火力分布**：架构与解码主导——Attention 变体 2 题、数据/对齐 4 题、推理引擎 2 题、手撕 2 题（beam+层序）。偏"通用算法+基础工程"。

### 题 1：Self-Attention 为什么用 Softmax？能换成别的吗？

- **核心答案链**：softmax 提供三点：①可微的概率归一（turtle 性质可做加权平均）；②指数放大对比度=锐利选择；③处处有梯度（尽管饱和）。可替换分析：sigmoid（各自独立门，Qwen 曾用）、稀疏 top-k（省算力但不可微需 straight-through）、线性 attention（softmax 的 exp 核换成恒等/其他核，O(n) 但表达降），纯线性（线性注意力家族）。
- **一句追问**：换 ReLU×L2 归一会怎样？——非负+对比度弱，性能大幅降；softmax 的 sharpness 是关键。

### 题 2：Causal Mask 是什么？为什么需要？

- **核心答案链**：训练时全序列一次并行前向；若无 mask，位置 t 能看到未来 token，训练与推理（逐步生成）分布不一致且口试泄答案。实现：上三角 −inf 加到 logits 再 softmax（允许看自身对角）。数学等价：把联合分布按条件链式分解 p(x_≤n)=∏p(x_t|x_<t)。
- **一句追问**： permitting "看到自己"为什么可以？——预测第 t 位时输入含 x_t，掩掉的是严格未来 j>t。

### 题 3：Tokenizer 对模型性能的影响

- **核心答案链**：影响四点：①序列形状（Fertility）：tokenizer 太碎→上下文显存算力×膨胀；②下游均匀性：多语言/代码覆盖差→相应语言推理差；③语义边界：子词切割破坏 word 边界→拼写类任务差；④词表大小：与 embedding 参数和 softmax 算力 trade-off（128k~256k 主流）。BBPE 解决越野车覆盖问题。
- **一句追问**：换 tokenizer 必须重训吗？——一般要；vocab 变化=embedding 尺寸变；有把旧词表对齐的热启动技巧但少用。

### 题 4：SFT 数据质量 vs 数量哪个更重要？

- **核心答案链**：LIMA 立的 flag：1 千精造样本可逼近大数 SFT 水平——**质量>数量**上限效应；但覆盖度（任务/格式/难度多样性）也基金：太少覆盖=能力窄。共识：质量管上限、多样性与数量管覆盖；先用质量筛（去重/评分/难度分层），再补覆盖。数量级：千级高质量指令即可对齐行为，万~百万级扩能力。
- **一句追问**：怎么量化"质量"？——人工评分+模型打分一致性+消融实验（数据量-效果曲线）。

### 题 5：PPO 中 KL 惩罚的作用

- **核心答案链**：与 M1-Q5/M10 无关但常连考：KL(π_θ‖π_ref) 序为"别离 SFT/人类分布太远"的信任域：①防 reward hacking 与模式塌缩；②稳定训练（策略漂移可控）；③是 RLHF 目标式中的显式项，per-token 加/减到 reward（k1/k2/k3 三种无偏估计器实践常用 k3）。
- **一句追问**：β 怎么调？——自适应（控制器式）增长换稳定性，目标 KL 定 λ。

### 题 6：GRPO 中 group 大小怎么选？

- **核心答案链**：与 M5-Q7 联动：G↑=基线估计更准（蒙特卡洛方差∝1/G）但算力线型×G。实践：G=4~16；难题（求探索）G 要大，简单任务小 G；显存上限= G×(推理显存)。采样效率策略：动态 G（简单题早停）。
- **一句追问**：G 与温度怎么搭配？——温度高配大 G 增多样性；低温度小 G 稳定 exp。

### 题 7：PagedAttention (vLLM) 的原理

- **核心答案链**：OS 分页思想搬显存：KV cache 切成固定大小 block（16 token/块），逻辑序列=块表；请求间共享物理块（前缀允许多请求复用）。收益：①消除内部碎片→显存利用率 从 ~20-40% 到 >90%（有效并发×2-4）；②动态增长不需预分配；③copy-on-write 支持投机/beam 分叉。vLLM 吞吐高 2-4× 的核心。
- **一句追问**：碎片化省下的显存去哪了？——更多并发序列同时驻留，batch 变大。

### 题 8：Continuous Batching 是什么？

- **核心答案链**：静态 batch 等最慢序列结束换批（GPU 大量空转）；continuous（iteration-level）batching 每一步 decode 迭代重组成 batch：完成的立即退出、排队的立即插入。收益：吞吐 2-3×+（对比静态）、延迟方差小。实现依赖：调度器+无状态 step 执行器+PagedAttention（无连续 KV 约束）。
- **一句追问**：与 in-flight batching同义吗？——基本同义，TensorRT-LLM 用语。

### 题 9：手写 Beam Search 解码

- **核心答案链**（要点版，完整实现见 08-Coding/01）：每步所有 beam 取 top k logprob 扩展，合并取累计分数前 k；**长度归一（/len 或 MT 端 length penalty α）**防偏爱短句；EOS beam 进 finished。复杂度 O(steps·k·V)。可选多样 beam penalty。
- **一句追问**：为什么开放域生成不用 beam？——多样化文本任务 beam 产出重复/模板化；温度采样更自然，beam 残留 MT/翻译。

### 题 10：二叉树层序遍历

- **核心答案链**（要点版，代码见 08-Coding/01）：队列 BFS；每层记录当层 size，逐个弹并 push 子节点；输出按层分组。O(n) 时间 O(w) 空间（w=最大宽度）。
- **一句追问**：Z 字形变体？——层间 flag 反转 insertion 方向或 deque 双端。

---

## 面经 7：飞书智能助手（2025.10)—— 8 题

**本场火力分布**：RAG 主场 5 题+Agent 1 题+手撕 2 题（KV-cache attention+LRU 复现）。典型"应用层岗位"，考察端到端产品化思维。

### 题 1：完整描述 RAG 系统的架构

- **核心答案链**（与 M1-Q7 一致展开）：离线管线：加载→清洗→分块（策略+overlap）→embedding→向量索引（混合索引留 BM25）→元数据表。在线管线：query 预处理（改写/拆解/路由）→多路召回（向量+关键词+结构化过滤）→融合去重→rerank→topK 拼上下文（含引用）→LLM 生成→后置校验（groundedness）→反馈采集。每层 1 个优化杠杆，报架构时按"离线/在线"两页讲。
- **一句追问**：多跳问题怎么扩展？——迭代检索（retrieve-reason-retrieve，Agentic RAG）。

### 题 2：向量数据库的选型（Milvus/Weaviate/Chroma）

- **核心答案链**：按场景矩阵：Chroma=轻量原型/单机（有几行代码集成）；Weaviate=内置混合检索+模块化（schema 灵活，中小规模 SaaS 部署）；Milvus=大规模分布式（亿级向量、存算分离、GPU 索引、云原生 K8s 部署）。PGVector=已有 PG 且规模不大，运维一体化。决策因子：规模/QPS/混合检索需求/运维团队。
- **一句追问**：为什么规模上去 Milvus 赢？——十亿级 ANN 索引分片+副本成熟。

### 题 3：查询改写的方法

- **核心答案链**：HyDE（生成的 hypothetical 答 doc 写文档检索）、Multi-Query（一问题→N 个改写视角并合并结果）、Query 分解（多跳拆子问题）、缩写/口语归一、历史对话重写（指代消解——多轮 RAG 必备）成本与收益：每步都调 LLM，延迟×2~3，按需开关。
- **一句追问**：HyDE 失效场景？——专有术语/ erected 知识生成失败反而污染检索。

### 题 4：Reranking 的作用和常用模型

- **核心答案链**：召回保查全、rerank 保查准。常用：cross-encoder（bge-reranker/ Cohere Rerank、jina-reranker）精度高逐对打分 O(K) 次前向成本高；ColBERT 风格 late-interaction 折中；LLM-as-reranker（listwise）最强但最贵。价值量化：top1→top10 recall 提升后，rerank 拼 nDCG@5 常见 +5%~15%（视数据集）。
- **一句追问**：rerank 慢怎么办？——缓存+两阶段（轻模型粗排→重模型精排）+topK 裁剪。

### 题 5：如何评估 RAG 系统？RAGAS 指标

- **核心答案链**：RAGAS 四件套：①Faithfulness（忠实度：答案是否可由上下文推出，防幻觉）；②Answer Relevancy（答与问相关）；③Context Precision（找回来的有多少有用）；④Context Recall（标准答案需要的信息召回了多少；需标答）。补充：引用命中率、端到端用户指标。分层报告很重要：**定位"检索错 or 生成错"**。
- **一句话追问**：没有 ground truth 怎么办？——LLM-as-judge 生成参考+人工抽检校准。

### 题 6：Agent 的 Planning 模块实现？

- **核心答案链**：与 M4-Q2 集成：技术选型四路：①prompt 内规划（ReAct/CoT，最简单）；②结构化 plan（先产出任务 DAG 再执行，LangGraph 式）；③反思式（Plan→执行→Reflexion 修正）；④搜索式（MCTS on action 空间，算力换最优）。工程依赖：工具接口稳定、子任务可验证、失败回退策略。
- **一句追问**：planning 过度会怎样？——计划与执行脱节（世界状态变化），执行时需 permit 修订（ replanning）。

### 题 7：手写 KV Cache Attention

- **核心答案链**（要点版，与 M2-Q9 联动；完整实现见 08-Coding/01）：只对最新 token 计算 q（或增量），K/V 拼 cache：
```python
k_new, v_new = x @ Wk, x @ Wv          # 只算新 token
K = torch.cat([cache_k, k_new], dim=1) # (B,s_new,dh)
V = torch.cat([cache_v, v_new], dim=1)
o = (q_new @ K.transpose(-2,-1)/dh**0.5).softmax(-1) @ V
cache_k, cache_v = K, V                # 就地更新
```
复杂度每步 O(s·dh) 而不是 O(n²)；显存即 M1-Q4 公式。
- **一句追问**：RoPE 在 cache 模式下怎么处理？——对新 token 用其绝对位置旋转，K/V cache 存"旋转前"值即可（等价）。

### 题 8：LRU Cache 设计

- **核心答案链**：同 M2-Q10 一致答案（哈希+双向链表 O(1)）。补充设计题视角：并发（分片锁）、分布式一致性失效、带 TTL 版本——工程加分点。
- **一句追问**：为什么不用单向链表？——删除需前驱，双向 O(1)。

---

## 面经 8：AI Infra（2025.10）—— 8 题

**本场火力分布**：纯 Infra 卷——并行策略/通信/训练技巧 5 题、推理引擎 2 题、系统设计 1 题。要求有数字、有取舍逻辑。

### 题 1：大模型分布式训练的并行策略

- **核心答案链**：四维正交：**DP**（数据并行，复制权重切数据）+ **TP**（层内切矩阵，每层两次 AllReduce）+ **PP**（层间分段，micro-batch 流水，气泡率 (p−1)/(m+p−1)）+**EP**（MoE 专家并行）+序列/上下文并行（长序列 attention 分块）。选定公式：单卡放不下权重→TP/PP；放得下但要吞吐→DP+Zero；通信看 NVLink 域。
- **一句追问**：为什么 TP 喜欢打满机内？——TP 通信量 O(层×激活) 极高，跨机就慢。

### 题 2：DeepSpeed ZeRO 三个阶段的区别

- **核心答案链**：切分优化器状态/梯度/参数：**ZeRO-1** 切优化器状态（Adam 占 12B/p，4×省显存）；**ZeRO-2** +切梯度（8×），反向叠加 Reduce-Scatter；**ZeRO-3** 再切参数（N_d 倍，前向/反向按需 all-gather）。实现难度与通信成本递增。选型：模型能进卡用 1-2，进不去了 3（与 PP/EP 组合）。
- **一句追问**：与 FSDP 什么关系？——FSDP≈ZeRO-3 思想的 PyTorch 原生实现。

### 题 3：张量并行和流水线并行怎么选？

- **核心答案链**：TP：通信极重（每层 forward/backward 各 1 AllReduce+AllGather），但气泡小、适合**机内 NVLink**；PP：层间只传 P2P activation（小），气泡 (p−1)/(m+p−1)，适合**跨机弱互联**但要求 micro-batch 数≫stage。经验法则：先 TP 填满节点（4/8 卡内），再 PP 跨节点，最后 DP 撑吞吐。
- **一句话追问**：TP 度数用 2 还是 4/8？——看模型层宽与带宽，TP=4/8 时 NCCL 带宽要求高。

### 题 4：All-Reduce 和 Ring All-Reduce 的区别

- **核心答案链**：朴素 AllReduce：参数 server/tree，中心带宽瓶颈 O(N)；Ring All-Reduce：N 卡成环，*(N−1)* 均分两阶段（reduce-scatter+all-gather），每卡通信 2(N−1)/N·S 字节，**与卡数无关、带宽最优**。TP 里的 AllGather/Reduce-Scatter 就是 ring 的半程。
- **一句追问**：小消息时哪个更快？——树形（延迟主导）；Ring 是带宽最优、树形是延迟较优，NCCL 自适应选。

### 题 5：梯度累积的作用和实现

- **核心答案链**：目的：小显存上等效大 batch（batch=per_gpu×accum×world）。实现：loss=(loss/accum).backward() 累 N 次后 optimizer.step()+zero_grad。两个坑：①loss 定要**先除 accum**（等价平均）；②BN 统计仍按 micro-batch（Transformer 无此虞）；③ 与 DDP：反同步（每 accum 步进 all-reduce）需要 no_sync()。
- **一句追问**：累积与 ZeRO 冲突吗？——不冲突，但 ZeRO-2/3 每 micro-step 需要梯度 communication 时机管理。

### 题 6：vLLM 的架构和核心优化技术

- **核心答案链**：与 M6-Q7/8 合并：架构=Frontend(OpenAI API)→Scheduler→Core Engine(PagedAttention+continuous batching)→worker(TP)。五技术：PagedAttention、continuous batching、prefix caching、量化、speculative decoding；特点：吞吐冠军（2-4× vs 静态）。SGLang 补 RadixAttention 前缀树复用。
- **一句话追问**：与时序起伏对抗设计？——抢占（preemption）+swapping 处理 KV 溢出。

### 题 7：CUDA kernel 优化的基本方法

- **核心答案链**：三板斧：①识别 bound（memory/compute）→roofline；②访存优化：合并读写（coalescing）、SRAM/寄存器 tiling、消除 bank conflict；③并行结构：block/warp/thread 网格覆盖、warp divergence 消除、占用率（occupancy）调优；④tensor core+FMA 混合精度；⑤算子融合（减 kernel launch 与中间显存）。
- **一句话追问**：测什么指标判断优化成功？——有效带宽 (GB/s) 达峰比例与 SM busy，而非单纯时间。

### 题 8：如何设计一个大模型推理服务？

- **核心答案链**：与 M1-Q13 一致答案再补 Infra 侧：引擎集群（vLLM/SGLang）+模型版本管理+灰度/回滚、KV 显存预算与抢占、异地多活、成本核算（token 成本/GPU 时）、SLO 分级队列、观测（TTFT 分布、缓存命中率、GPU MFU）。架构叙事按"接入-调度-引擎-存储-观测"五层。
- **一句追问**：扩容指标是什么？——KV cache 水位与排队延迟，不是 CPU。

---

## 面经 9：安全与治理（2026.02）—— 6 题

**本场火力分布**：纯安全治理：风险全景→越狱→注入→对齐→红队→偏见递进。考"体系化攻防框架"。

### 题 1：LLM 常见的安全风险

- **核心答案链**：五类——①内容：有害生成/幻觉误导/版权；②对抗：越狱/提示注入；③隐私：训练数据泄露/PII 提取/记忆；④系统：模型窃取、投毒、供应链；⑤Agent 特有：工具滥用、间接注入。治理=分类分级+红线测试+持续监控。
- **一句追问**：LLM 风险与传统应用安全最大不同？——自然语言边界模糊，无法静态规则枚举完。

### 题 2：越狱攻击的类型和防御

- **核心答案链**：类型六类：角色扮演、虚构场景包装、编码绕过（base64/低资源语言）、多轮温水煮蛙、前缀注入（"从现在起你..."）、优化对抗（suffix GCG/USB 搜索）。防御：训练侧（安全 SFT/RLHF+拒绝分布）、推理侧（ moderation 模型双层+输入改写归一）、系统侧（系统提示强化+输出拦截）。原则：**不完备，纵深防御+持续红队**。
- **一句追问**：为什么没有被绝对防住的越狱？——语言空间开放+目标防不全枚举，安全是概率性指标（ASR）。

### 题 3：提示注入如何防范？

- **核心答案链**：与 M11（教育）联动。要点：区分直接 vs 间接注入（工具内容携带指令）。防御：①指令/数据信道分离（结构与标记隔离）；②工具输出先送去污/摘要再入上下文；③权限最小化+高危动作人工确认；④不变性测试（固定指令重复校验）；⑤可检测（查询加自检）。核心心法：**把外部内容当不可信数据，而非指令**。
- **一句追问**：为什么"别听外部指令"的提示本身也软？——它只是又一道自然语言防线，必须叠加架构控制。

### 题 4：RLHF 对齐训练如何增强安全性？

- **核心答案链**：机制：安全偏好数据纳入 RM（拒绝有害+不过度拒绝），SFT 演示拒答风格，PPO/GRPO 优化安全 reward（含红线 clause）；进阶：安全关键场景 reward 加权/约束 RL（如 Safe RLHF 拉格朗日）。效果度量：有害率下降×有用率保持（**安全-有用 trade-off 曲线**）。
- **一句追问**：副作用？——过度拒绝（over-refusal），需良性请求正样本校准（OR-Bench）。

### 题 5：Red Teaming 的流程

- **核心答案链**：圈闭环：①范围与威胁画像（使用场景/用户/风险清单）→②攻击面梳理（提示/工具/多模态/多轮）→③攻击生成（人工+自动化 LLM 攻击器+遗传变体）→④评估（ASR/严重度分级）→⑤归因与修复（数据/护栏/策略）→⑥回归与持续。指标：攻击成功率<阈值+覆盖清单完备性。
- **一句追问**：自动化红队会不会自我作弊？——需要对攻击有效性做人工抽样确认。

### 题 6：大模型的偏见检测和缓解

- **核心答案链**：检测：基准（BBQ/StereoSet/WinoBias）+反事实测（换人口属性看输出漂移）+表示探针（embedding 关联性）。缓解三层：数据去偏（重采样/清洗）→训练（公平约束/RLHF 偏好校准）→推理（提示中立+输出平衡）+长期审计监控（线上分布漂移）。关键：**偏见定义需先约定（群体公平性的度量口径）**。
- **一句追问**：公平与准确冲突怎么办？——不可强行拉平概率，透明报告+场景化策略。

---

## 面经 10：推理优化（2026.01）—— 6 题

**本场火力分布**：专精推理：KV 公式三连算、Flash 内部、投机正确性证明、蒸馏/量化对比、系统级延迟。每题都要有数字。

### 题 1：KV Cache 显存计算公式（MHA/GQA/MLA 分别算）

- **核心答案链**（与 M1-Q4 一致升级版）：通式 **bytes = 2(K,V)·L·n_kv·d_head·s·b·bytes_per**。
- MHA（n_kv=h）：7B 例 2.1GB/4k 上下文（见 M1-Q4 数字）。
- GQA（n_kv=g）：÷(h/g)，如 h=32→g=8 为 1/4。
- MLA：cache 只存潜向量 512 维+解耦 RoPE 键 64 维**一条**（跨头共享），合计 576 元素/层/token（fp16 1152B≈1.1KB）；官方口径：较 MHA 压缩 93.3%、吞吐 5.76×（论文摘要），等效 GQA 只留 2.25 组；每层元素比 = GQA-8 的 28%、7B-MHA 的 7%（报数带口径：**每层每 token 元素比**，非总显存比）。 
- **一句追问**：decode 阶段 FLOPs 与显存带宽比？——decode=memory-bound，KV 读取带宽是 roofline；所以 MLA 压缩=直接提 decode 速度。

### 题 2：Flash Attention 的 tiling 策略和 online softmax

- **核心答案链**：与 M1-Q9/M10-Q2 三次出题统一口径：内循环 K/V 块 j、外循环 Q 块 i；局部 softmax 增量：
m_new=max(m_old, rowmax(S_j))；l_new=e^{m_old−m_new}·l_old+e^{m_new−rowmax}·rowsum_j；
O←e^{m_old−m_new}·O+e^{...}·P_j V_j。Tiling 块大小由 SRAM 决定（A100 192KB）。数学等价证明=把 max 提出来。
- **一个追问**：与 FA2 改进？——FA2 精确前后向重排（NON-MATMUL 换前移）、交换循环让 Q 在外（寄存器驻留 O）。

### 题 3：投机解码为什么能保证输出分布一致？

- **核心答案链**：与 M3-Q9 一致升级为证明：单 token 接受概率 α=min(1,p/q) 期望；拒绝采样理论保证 marginal=p。完整推导两行：E[accept]·q+(1−Σ)·residual=p(x)（代数展开即可复现）。
- **一句追问**：多 token 链式验证的分布也一致吗？——是，逐位独立成立，全局仍=target 分布（不是逐条件分布近似）。

### 题 4：模型蒸馏和量化的区别

- **核心答案链**：与 M10-Q4 同答案：维度不同——蒸馏改"参数量"（重训练，算力/显存全按规模缩小），量化改"数值位宽"（免训/轻训，显存÷4 但速度依赖内核）。可叠加：先蒸馏后量化为工业最佳实践。
- **一句追问**：蒸馏损失哪类能力最先掉？——长尾知识（稀疏记忆）；推理能力需 CoT/RL-SFT 蒸馏补。

### 题 5：INT4 量化后精度下降怎么办？

- **核心答案链**：与 M3-Q8 联动，给出对策五条（校准集/group size/AWQ scale/敏感层保高精度/QAT 微调回收）+量化验收门禁（MMLU 掉点<1%）。
- **一个追问**：只量化 weight 不量化 activation（W4A16）的好处？——无需激活校准风险、算子在反量化后 fp16 matmul，精度可控。

### 题 6：如何从系统层面优化大模型推理延迟？

- **核心答案链**：与 M8-Q8/M10-Q6 同源：延迟拆解 TTFT（prefill）+TPOT（decode）。手段六类：并行 prefill+chunked prefill（TTFT）、投机+ Medusa（TPOT）、prefix cache（多轮）、量化+flash kernel、SLO-aware 调度（分池）、网络/布局优化。方法论：**先 profile 定位主导项（prefill or decode or 调度 or 网络），再对症下药**。
- **一句追问**：优先级冲突（首 token 快 vs 整体快）怎么权衡？——业务分级：交互式保 TTFT、批处理保吞吐。

---

## 面经 11：教育 AI（2026.02）—— 6 题

**本场火力分布**：架构选择 2 题+前沿热点 3 题（test-time compute/R1/VLM）+应用设计 1 题。话题前沿感强，开放题为主。

### 题 1：为什么现在的 LLM 都是 Decoder-Only？

- **核心答案链**：统一口径（M11 专用版）：①训练效率：因果 LM 每 token 都出监督， utilization 100%（Bert 式只部分 token）；②架构与 scaling 简单统一（in-context learning 涌现）；③推理天然匹配（增量解码+KV cache）；④历史与生态实证（GPT 系良性循环）。结论要注意分寸：这是工程+数据综合选择，不是定理。
- **一句追问**：还有哪些架构在挑战？——Mamba/混合 SSM、扩散 LM、encoder 路线（BERT 系在 embedding 保留席位）。

### 题 2：Encoder-Decoder 架构适合什么任务？

- **核心答案链**：与 M11-Q1 衔接：适合"输入=复杂结构化理解+输出=受约束生成"：机器翻译、摘要、结构化抽取→生成；小模型时代 enc-dec（T5/UL2）参数效率高；当代多用 decoder-only+prompt 承接，且多模态里 vision encoder+text decoder 是 enc-dec 思想的延续。
- **一句追问**：为什么多模态常用 enc-dec 思想？——视觉端双向编码器提表示，语言端自回归生成。

### 题 3：test-time compute 是什么？

- **核心答案链**：推理时算力扩展范式：型式化为"在解码分布/搜索空间上花更多算力换性能"：自一致性投票、Best-of-N+RM、长 CoT、工具迭代、树搜索（MCTS/ToT）。 Scaling law 从"训练侧"转到"推理侧"（batch size 时代结束后的新杠杆）；关键洞察（Snell et al.）：最优 TTC 依赖模型大小与题目难度，小模型+大 TTC 可胜大模型。
- **一句追问**：什么时候 TTC 失效？——能力完全不在分布内（概率支撑外），再采样也造不出正解。

### 题 4：DeepSeek-R1 的训练方法

- **核心答案链**：与 M1/M3 的 GRPO 链成一整套：R1-Zero=base+GRPO+规则奖励（accuracy/format），涌现反思与长思维链（含语言混杂副作用）；R1 加 cold-start 数据（可读 CoT）反复 RL+SFT 混合管线：①冷启动 SFT→②推理 RL→③拒绝采样 SFT（含写作等非推理任务）→④全场景 RLHF。副产物：蒸馏到 1.5B~70B 小模型（SFT 蒸馏＞小模型自 RL）。
- **一句追问**：为什么不用过程奖励（PRM）？——自动标注难、易被 hack；R1 用规则结果奖励+GRPO 绕开。

### 题 5：多模态大模型（VLM）的架构

- **核心答案链**：三大块+训练三阶段：视觉编码器（ViT，CLIP/SigLIP 预训练）→投影层（MLP/Q-Former/cross-attention；主流回归简单 MLP）→LLM。训练：模态对齐预训练（可冻结视觉）→指令微调→偏好/RL。关键 trade-off：分辨率/token 预算（动态分辨率 Qwen2.5-VL）、图像 OCR 细粒度、视频时序采样。
- **一句追问**：幻觉在 VLM 有什么新形态？——"对图脑补"（对象不存在但语言上通顺），grounding 评测重要。

### 题 6：如何将 LLM 应用于教育场景？个性化学习路径设计

- **核心答案链**：五层方案：①学情建模：错题→知识点图谱标注+能力画像（知识追踪思想）；②路径规划：LLM+课程图谱（SFT+规则约束），按掌握度推荐下一目标——等级=习得度、难度模型、遗忘曲线；③交互：苏格拉底式提示、分步脚手架、即时反馈；④内容生成：RAG 教材（防幻觉）、题目变体生成、错因归因解释；⑤闭环评测：A/B 后测成绩/留存/掌握度提升。红线：安全合规+防代写（不直接给答案）。
- **一句追问**：最容易被挑战的问题？——"LLM 判错/判对错乱"，需要结构化评分器+人工复核抽检。

---

## 频率与难度对照速查（面试前最后 1 小时看这个）

| 高频度 | 题目对应 | 出现场次 |
|---|---|---|
| ★★★★★ | GRPO vs PPO（M1-5/M2-5/6/M5-7/M6-5/6 + M11-4 隐含） | 7/11 |
| ★★★★★ | Self-Attention 计算+√dk+多头（M1-1/2/3/6 等） | 6/11 |
| ★★★★★ | KV Cache 原理与显存（M1-4/M3-6/M7-7/M10-1） | 6/11 |
| ★★★★ | 手写 Attention 家族（M1-6/M2-9/M7-7） | 5/11 |
| ★★★★ | LoRA 原理/r 选择/手写（M2-3/M5-9） | 5/11 |
| ★★★★ | DPO 推导与对比（M1-10/M5-6/M10 隐含） | 4/11 |
| ★★★★ | Flash Attention（M1-9/M3 隐含/M10-2） | 4/11 |
| ★★★★ | RAG 优化链路（M1-7/M2-7/8/M7-1~5） | 4/11 |
| ★★★ | Agent/MCP（M1-11/12/M4 全/M7-6） | 4/11 |

---

## 本章高频追问 Top5

1. **GRPO 五连击**：为什么能去 critic？→怎么算 advantage？→group 大小？→std=0 怎么办？→与 DAPO/RLOO 差别？（深度解析见 M1-Q5、M5-Q7）
2. **KV Cache 三连击**：显存公式面板算（MHA/GQA/MLA 三口径）→为什么是 decode bottleneck？→有哪些压缩手段？（M1-Q4、M10-Q1）
3. **Flash Attention「复述并证明无损」**：online softmax 两步增量更新写出来，证明与全局 softmax 等价（M1-Q9、M10-Q2）
4. **DPO 现场推导**：四步骨架（闭式解→反解 reward→Z 相消→BT 模型）必须背熟，追问 β 含义（M1-Q10、M5-Q6）
5. **RAG 追问链**：评价指标→怎么定位瓶颈→chunk 怎么定→rerank 值不值这个延迟→怎么评测无标注数据（M1-Q7、M7-Q1~5）

---

## 按模块合并的 30 天刷题计划表

> 规则：每天 2–3 小时；"熟练度=能 30 秒口头作答"。D30 之前的每个模块结束做对应面经题自测。

| 阶段 | 天数 | 模块（合并来源章节） | 每日核心任务 | 对应本卷题目 |
|---|---|---|---|---|
| 一、基础机制 | D1–D4 | Transformer/Attention（01-Foundation/01） | 手推 attention 复杂度、√dk、多头、RoPE、mask、Flash Attention 复述 | M1-1/2/3/9、M2-1/2/9、M5-2、M6-1/2、M10-2、M11-1/2 |
| 一、基础机制 | D5–D7 | 分词/归一化/优化器（01-Foundation/02,03） | BPE 训练+手撕、Pre/Post-Norm、LN/RMSNorm、AdamW、BF16/FP16、梯度累积 | M2-3、M3-1~5、M6-3、M8-5 |
| 二、训练 | D8–D10 | PEFT（03-FineTuning/01） | LoRA/QLoRA 原理+r 选型+手写 LoRA 层 | M2-3、M5-9 |
| 二、训练 | D11–D14 | 对齐（03-FineTuning/02） | RLHF 三阶段、Critic、KL 惩罚、DPO 四步推导、GRPO 全家+组大小 | M1-5/10、M2-4/5/6、M5-6/7、M6-5/6、M11-4 |
| 二、训练 | D15 | SFT 数据工程（03-FineTuning/03） | 质量 vs 数量（LIMA）、蒸馏与量化对比 | M5-8、M6-4、M10-4 |
| 三、推理 | D16–D19 | KV/引擎（02-Inference/01） | KV 显存公式三口径、GQA/MLA、Flash tiling、PagedAttention、Continuous batching | M1-4/8、M3-6、M5-2、M6-7/8、M8-6、M10-1 |
| 三、推理 | D20–D21 | 解码与投机（02-Inference/02) | 投机解码分布一致性证明、beam search、解码策略 | M3-9、M6-9、M10-3 |
| 三、推理 | D22–D23 | 量化部署（02-Inference/03） | GPTQ/AWQ/SmoothQuant 对比、INT4 掉点对策、蒸馏选择 | M3-8、M10-4/5 |
| 四、Infra | D24–D25 | 并行与 kernel（09-SystemDesign/02-Infra） | 4 种并行+选型公式、ZeRO 三阶、Ring AllReduce、CUDA 优化三板斧 | M8-1~5/7 |
| 五、RAG | D26–D27 | RAG（04-RAG） | 架构全景、选型矩阵、查询改写、Rerank、RAGAS | M1-7、M2-7/8、M7-1~5 |
| 六、Agent | D28 | Agent（05-Agent） | 组件/ReAct/FC/MCP/记忆/多智能体/错误恢复/评估 | M1-11~13、M4-1~8、M7-6 |
| 六、安全 | D29 上午 | 安全（06-Safety-Evaluation） | 风险六类、越狱/注入、RLHF 安全、红队、偏见 | M9-1~6 |
| 七、热点 | D29 下午 | 热点（07-HotTopics/MoE、长上下文） | MoE-V3、NTK 长上下文、test-time compute、FLOPs 心算 | M3-7、M5-1/5、M11-3/5 |
| 八、冲刺 | D30 | 全真模拟 | 按场抽面经（1 场 1 小时×3 轮），错题回表定位；最后 1 小时只看本章【高频追问 Top5】和速查表 | 97 题全部 |

（上表与原章节文件对齐：对应分章详解位于 `01-Foundation`、`02-Inference`、`03-FineTuning`、`04-RAG`、`05-Agent`、`06-Safety-Evaluation`、`07-HotTopics`、`08-Coding`、`09-SystemDesign` 各章。）
