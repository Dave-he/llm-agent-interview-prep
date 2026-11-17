# 04-RAG 章节题目详解

> 配套原章节：`01-RAG-Complete-Guide.md`（只读）。本文件为每道题的五段式详解：考察意图 / 深度解析 / 30秒答题框架 / 追问与陷阱 / 知识串联。研究背景：`/Users/hyx/agent-personal/personal-agent-research.md`（下称【研究文档】，冲突处以官方论文/公式为准）。

## 本章考点分布

| 题号 | 题目 | 考频 | 核心考点 |
|------|------|------|----------|
| 1 | 什么是 RAG？⭐⭐⭐⭐ | 必考开场题 | 端到端全流程（离线/在线）、RAG vs 微调 vs 长上下文 |
| 2 | 文档处理流程 | 高频 | 解析/清洗/去重/元数据/增量更新的工程细节 |
| 3 | Chunking 策略 | 高频 | 粒度 tradeoff、递归/语义/父子块、Recall@k 调参 |
| 4 | 常用 Embedding 模型 | 高频 | 对比学习原理、指令前缀、选型决策表 |
| 5 | 向量 vs 关键词 vs 混合检索 | 字节二面高频 | BM25 公式、HNSW/IVF-PQ、RRF 融合推导 |
| 6 | 查询改写 | 中高频 | HyDE 分布对齐原理、Multi-Query、Step-Back、子查询 |
| 7 | 为什么需要重排序 | 高频 | bi-encoder vs cross-encoder、K 值设计、延迟量化 |
| 8 | Agentic RAG | 2026 必考新点 | Self-RAG/CRAG/多跳、RAG-长上下文-Agentic 三角取舍 |
| 9 | Graph RAG | 进阶区分题 | 全局/多跳动机、Leiden 社区、local/global search、成本 |
| 10 | RAG 评估指标 | 高频收尾题 | RAGAS 五指标公式、badcase 归因（检索段 vs 生成段） |

分布规律：开场概念题（1）→ 离线索引（2-4）→ 在线检索（5-7）→ 进阶架构（8-9）→ 评估闭环（10）。2026 年趋势：面试官已不满足于"讲流程"，必追"归因定位"（题 10）与"Agentic 化"（题 8）；RAG vs 长上下文的取舍（1M 窗口时代）常作为压轴组合题。

---

### 题 1：什么是 RAG？⭐⭐⭐⭐

**【考察意图】**

- 验证你能否把 RAG 从一句口号展开成**端到端系统**：离线索引线 + 在线服务线，每段能说出做什么、用什么、延迟量级。
- 验证你理解 RAG 存在的**根因**：LLM 参数化知识的结构性缺陷，而不是"因为流行所以用 RAG"。
- 验证技术选型判断力：RAG / 微调 / 长上下文三选一（或组合）的决策逻辑，能否给量化成本对比。

**【深度解析】**

**一、根因：参数化知识的四大局限**

LLM 把知识压进权重，推理时不可增删，带来四个结构性问题：

1. **知识截止**：训练数据有时间封顶，之后的世界变化（新政策、新版本号）一无所知。
2. **私有/长尾缺失**：企业内部文档、个人笔记从未进过训练集——没见过的东西，模型只能"编"。
3. **幻觉**：自回归模型逐 token 采样最大似然续写，机制上"说错话"和"说对话"是同一条生成路径；没有检索兜底时，流畅性 ≠ 正确性。
4. **不可溯源**：知识藏在参数里，答案无法指回证据，合规/审计场景直接不可用。

RAG 的解法一句话：**把"知识存取"从参数空间搬回外部非参数库，把模型职责收敛为"读材料、写作答"**。学术表述即 Gao et al. RAG Survey 里的 non-parametric memory（类比【研究文档】中"外部知识优于参数化知识，可更新可审计"的选型判断）。

**二、端到端两条线（必须能默写）**

```
【离线索引线】
原始文档(PDF/HTML/DB) → 解析(OCR/表格还原) → 清洗去重(MinHash/SimHash)
→ 分块 Chunking(256-1024 token, overlap 10-20%) → Embedding 编码
→ 双索引入库：向量 ANN(HNSW/IVF) + 稀疏倒排(BM25) + 元数据(来源/时间/权限/标题路径)

【在线服务线】
Query 理解与改写(指代消解/HyDE) → 混合召回(dense + BM25, 各 top 50-100)
→ RRF 融合 → 过滤(权限/去重/时间) → Cross-Encoder 重排(top 50→3-10)
→ Prompt 组装(引文编号 [1][2]、相关性排序) → LLM 生成(带引用+拒答门控) → 上线后分层监控
```

**各段延迟量级**（工程量感，字面二面常追问）：

| 阶段 | 典型延迟 | 说明 |
|------|---------|------|
| query embedding | ~5-20ms | 单句 GPU encode |
| ANN 检索（HNSW） | ~1-10ms | 近似 O(log N) |
| rerank 50 条（512 token/条） | ~50-300ms | 50 次 cross-encoder 前向 |
| LLM prefill + 生成 | 首token 0.3-2s 起 | **延迟大头**，流式输出改善体感 |

结论：检索段合计 <100ms 量级，**生成段才是延迟与成本大头**——优化优先级应据此排。

**三、RAG vs 微调 vs 长上下文（对比原表扩充）**

| 维度 | RAG | 微调 | 长上下文直塞 |
|------|-----|------|-------------|
| 知识更新 | 改索引即可（分钟级增量 upsert） | 重新训练（LoRA 小时级、全量天级） | 每次请求重付 token 成本 |
| 可溯源 | 强（引用编号可回指原文） | 弱（知识熔进参数） | 中（来源在 prompt 里但易"视而不见"） |
| 幻觉抑制 | 好（有据可依+拒答门控） | 一般（不消除机制性幻觉） | 差（长文注意力衰减，见 lost in the middle） |
| 风格/格式控制 | 弱（只能 prompt 约束） | 强（行为内化） | 弱 |
| 成本结构 | 在线多检索段延迟 + prompt token 费 | 离线训练费 + 省在线 prompt | KV cache 显存与 token 费随长度线性涨 |
| 多用户共享知识 | 天然适合（一套索引服务所有人） | 不适合（知识要用参数承载） | 每个会话都塞一遍，浪费 |

**选型决策树**（面试可直接画）：

```
知识/信息频繁变化 or 需溯源 or 多用户私有库？
  → 是：RAG（底座）
要改的是"行为/风格/格式/领域语言"而非知识？
  → 是：微调（LoRA 优先，见 03-FineTuning）
语料 ≤ 上下文窗口 且 一次性单任务分析？
  → 是：长上下文直塞最省事
三者可叠加：RAG 供知识 + 微调教模型"读材料写作答"的纪律
```

【研究文档】考点 15 给出同构的默认顺序：**先 prompt/context 工程 → 工具与 RAG → 最后才是微调**。

**四、范式谱系（加分项）**

Naive RAG（检索-拼接-生成一杆子）→ Advanced RAG（检索前优化：改写/路由；检索后优化：重排/压缩）→ Modular/Agentic RAG（检索变成可插拔、可循环的模块，检索决策交给 agent）。这条谱系正好是本章题 6/7/8 的展开顺序。

**【30秒答题框架】**

"RAG 是检索增强生成：用外部知识库弥补 LLM 参数化知识的三大缺陷——知识过时、私有数据缺失、幻觉且不可溯源。它分两条线：离线把文档解析清洗、分块成 256-1024 token，向量化后连 BM25 倒排一起入库，元数据带来源和权限；在线把 query 改写后双路召回、RRF 融合、cross-encoder 重排出 top 3-10 拼进 prompt，带引用生成并做拒答门控。和微调的分工：知识要频繁更新、要溯源选 RAG；要改风格格式选微调，两者可以组合。延迟上检索段仅百毫秒级，生成段才是大头。"

**【追问与陷阱】**

- 追问：**RAG 能根治幻觉吗？** 不能。三类残余幻觉：① 检索空转（库里根本没有）→ 需拒答门控；② 检索到错误/过时材料 → 垃圾进垃圾出；③ 生成段无视材料自由发挥 → 需忠实性约束。答案要点是"分层归因 + 分层治理"（引题 10）。
- 追问：**既然有 1M 上下文为什么还要 RAG？** 三点：成本随长度线性涨（token 费 + KV 显存）；注意力质量随长度衰减（context rot，【研究文档】考点 6）；语料超过窗口或多用户共享时塞不下。
- 追问：**知识截止后的增量更新怎么做？** 增量 upsert + 版本/时间元数据过滤 + 删除级联（详见题 2）。
- 陷阱：只背"向量数据库"四个字，不提 BM25、不提重排（立刻露怯）；把 RAG 说成"替代微调"（混淆非参数知识/参数能力的边界）；全程无一个数字（延迟、chunk 大小、K 值）。

**【知识串联】**

- `05-Agent/01-Agent-Complete-Guide.md`：Agentic RAG 之于 RAG，正如 agent 之于 chatbot（研究文档考点 8，检索成为 agent 的可决策工具）。
- `02-Inference/01-KVCache-And-Acceleration.md`：长上下文成本论证依赖 KV cache 显存公式（7B fp16 约 0.5MB/token，128K 约 64GB）。
- `03-FineTuning/01-PEFT.md`：RAG vs 微调表中的训练成本量级（LoRA 小时级 vs 索引分钟级）。
- `06-Safety-Evaluation`：检索内容是不可信外部输入，prompt injection 防御在 RAG 场景同样适用。
- `09-SystemDesign`：RAG 在线链路就是一个标准检索系统设计题的特化。

---

### 题 2：文档处理流程

**【考察意图】**

- 是否**亲手搭过**离线 pipeline：能说出解析的真实痛点（PDF 双栏、扫描件、表格），而不是"调个 loader 就行"。
- 元数据与权限意识：过滤、引用、多租户隔离的物理基础都在离线阶段打好。
- 数据新鲜度与成本意识：去重、diff 检测、增量 upsert 这些"不性感但省钱"的环节。

**【深度解析】**

**一、六段流水线全景**

```
解析 → 清洗 → 去重 → 分块 → 编码 → 入库(+元数据)
```

**① 解析**（难点最密集的一段）：

| 格式 | 坑 |
|------|-----|
| PDF | 双栏乱序、页眉页脚噪声、扫描件需 OCR（OCR 错字率直接污染下游 embedding）、公式/上标丢失 |
| 表格 | 直接线性化会丢行列对应关系——需结构化（Markdown/HTML 表）与"行→语义句"两种粒度双索引 |
| HTML | 去 nav/footer/广告噪声，保留 DOM 结构作为分块边界线索 |
| 图片/PPT | 布局感知解析或 VLM captioning 后当文本索引 |

**② 清洗**：unicode 归一化（NFKC）、空白与水印、目录页剔除；格式统一为纯文本或 markdown。

**③ 去重**（直接影响检索质量）：

- **SimHash**：指纹逐位比较 Hamming 距离，O(1) 判近重复，适合 URL/正文级。
- **MinHash + LSH**：近似 Jaccard 相似度，按 minhash 签名分桶 O(n·k)，桶内精确比对。作用机制不只是省存储：**重复 chunk 会挤占 top-k 名额**，同一内容多份变体让结果 diversity 崩塌，且 RRF 融合后同源条目重复计票——必须按 doc 级去重。

**④ 分块**：详见题 3。

**⑤ 编码**：批式 GPU encode（batch 拉满吞吐）；同批内 in-batch negatives 是对比训练思想（呼应题 4）；chunk 计数用**目标 embedding 模型的 tokenizer**（不同 tokenizer 长度不同，呼应 `02-Tokenization`）。

**⑥ 入库与元数据**：

- 双索引：向量 ANN（HNSW/IVF-PQ）+ BM25 倒排（Elasticsearch/OpenSearch/内置 sparse）。
- 元数据 schema：`doc_id / chunk_id / title_path(层级标题) / 时间戳+版本 / 权限 acl / 语言 / 页码`——检索期过滤、引用回指、TTL 失效全靠它。

**二、增量化与成本（工程分水岭）**

- 文档内容哈希 → 变更检测 → **只重嵌入变更 chunk**：embedding 是离线最大算力项，O(N·L·d) 一次全量 + 后续按 diff 增量。
- 删除要级联：文档删除后其 chunk 若留在索引里，会出现"源头没了还能检索到"的合规事故。
- 全量重建走"双库灰度"：新索引建好 → 切流量 → 下旧库。

**三、存储量化（可复现口径）**

- 1 个 1024 维 fp32 向量 = 1024 × 4B = **4KB**；1 亿 chunk 原始 ≈ **400GB**。
- fp16 减半至 200GB；IVF-PQ（m=64 子量化器、8-bit 码本）压到约 **64B/向量 ≈ 6.4GB**（压缩比 64×，换召回率损失，见题 5）。
- HNSW 邻接表额外开销：每向量 O(M) 个指针（M=16 时约 32-64B/向量），1 亿向量光图指针就数 GB——内存要与向量预算一起算。

**【30秒答题框架】**

"离线六步：解析、清洗、去重、分块、批量 embedding、双索引入库。解析重点打 PDF 的双栏、扫描件 OCR 和表格结构化；去重用 SimHash 或 MinHash+LSH，因为重复 chunk 会挤占 top-k 名额；入库要向量 HNSW 加 BM25 倒排双索引，元数据记来源、时间、版本、权限——过滤和引用全靠它。工程上两条命门：元数据设计决定过滤/引用/多租户能不能做；哈希 diff + 增量 upsert 决定新鲜度和索引成本。"

**【追问与陷阱】**

- 追问：**表格怎么检索？** 结构化保留 + 行转语义句两种粒度都索引；QA 别指望模型自己还原行列对应。
- 追问：**为什么去重能提升检索质量？** top-k 名额是有限资源，重复变体挤占名额降低多样性（MMR 相关），还会让答案被同一重复源反复加权。
- 追问：**1 亿向量放内存放得下吗？** fp32 400GB 放不下 → PQ 压缩或磁盘型索引（IVF-PQ/DiskANN），给数字。
- 陷阱：以为 PDF 拿默认 loader 抽出来就能用（版面解析是最大隐性工时）；忘权限元数据导致多租户 A 能搜到 B 的内容；不做版本管理导致"删了还在搜"。
- 追问：**多模态（图片/截图）怎么进索引？** VLM caption 转 text 入向量索引 + 保留原图链接供引用，或用多模态 embedding 一把梭（成本高）。

**【知识串联】**

- 题 3（分块是它的下游）、题 4（编码的下游约束 chunk 长度）。
- `02-Tokenization`：chunk 长度按目标 embedding 的 tokenizer 数，不同模型 tokenizer 差异大。
- `09-SystemDesign`：索引分片、双库灰度属通用分布式设计。
- 题 10 / `06-Safety-Evaluation`：权限元数据缺位 = 多租户数据泄漏事故的直接来源。

---

### 题 3：Chunking 策略

**【考察意图】**

- 能否论证"**分块粒度是检索质量的第一杠杆**"——比换 embedding 模型影响往往更大。
- 两端 tradeoff 的机制级理解：太碎→语义断裂；太大→池化稀释 + 上下文预算浪费。
- 是否知道生产最佳实践（父子块）与调参方法论（用 Recall@k 曲线，不是拍脑袋）。

**【深度解析】**

**一、为什么 chunk 决定上限（机制）**

设 chunk c 由 n 个 token 经池化（mean/CLS）得到向量 E(c)。相似度 cos(E(q), E(c)) 中：

- **n 太小**：事实所需证据被切断（"上下文 A 段说条件、B 段说结论"被切开），任一半块都不足以支撑答案 → 检索命中了"半个答案"。
- **n 太大**：多个主题被平均进一个向量，主题分布变宽，与具体 query 的**峰值相似度被池化稀释**（信噪比下降）；同时 top-k 中每个大块挤占 LLM 上下文预算，触发 lost in the middle（题 7）。

所以存在一个"召回信息量 vs 信噪比"的 U 形 tradeoff，最优区间由**评测集**定，经验起点 256-1024 token、overlap 10-20%。

**二、策略对比（原表扩充）**

| 策略 | 机制 | 复杂度 | 适用 |
|------|------|--------|------|
| 固定大小 | 滑窗按 token 切 + overlap | O(n) | 简单基线 |
| 递归分割 | 按分隔符层级（段落→句子→词）递归降级，保自然边界 | O(n) | **通用默认**（LangChain RecursiveCharacterTextSplitter） |
| 语义切分 | 相邻句 embedding 相似度跌破阈值处切 | O(n) 次 encode，成本高 | 高质量语料 |
| 结构感知 | 按 markdown 标题/代码块/表格边界切 | O(n) | 技术文档 |
| **父子块** | 子块（1-2 句，高精度召回）命中后返回父块（512-1024 token，完整上下文） | 双套索引 | **生产推荐** |
| 句子窗口 | 命中句 → 扩窗 ±n 句 | 类似父子 | 类似 |

**父子块（small-to-big）是本题主拿分点**：检索用"小"（embedding 峰值尖锐、信噪比高），生成用"大"（上下文完整）。两套索引：sentence-level 向量 + parent-level 存储，命中子块回父块并去重。

**三、参考实现（可运行）**

```python
# 递归分割 + 语义切分最小实现
from typing import List

SEPARATORS = ["\n\n", "\n", "。", "!", "?", " ", ""]  # 递归降级的切分符层级

def recursive_split(text: str, max_len: int, separators=SEPARATORS) -> List[str]:
    """递归字符分割。复杂度 O(n·L)：每层最多切一遍全文，递归深度 ≤ len(separators)。"""
    if len(text) <= max_len:
        return [text]
    for sep in separators:                       # 从最强结构分隔符开始尝试
        parts = text.split(sep)
        if len(parts) == 1:
            continue                              # 该分隔符切不开（如全是英文哈希串）
        chunks, buf = [], ""
        for p in parts:                           # 贪心拼接：不超过 max_len 就并入当前块
            cand = (buf + sep + p) if buf else p
            if len(cand) <= max_len:
                buf = cand
            else:
                if buf: chunks.append(buf)
                buf = p if len(p) <= max_len else ""   # 超长片段下降一级继续处理
                if len(p) > max_len:
                    chunks.extend(recursive_split(p, max_len, separators))
        if buf: chunks.append(buf)
        return chunks
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]  # 兜底硬切

def semantic_split(sents: List[str], embedder, tau: float) -> List[List[str]]:
    """语义切分：相邻句相似度跌破均值减 tau 时断开。O(n) 次 encode + O(n) 相似度。"""
    import numpy as np
    E = np.asarray(embedder.encode(sents))                 # O(n) 编码
    E = E / np.linalg.norm(E, axis=1, keepdims=True)       # 归一化 → 点积=余弦
    sims = (E[:-1] * E[1:]).sum(-1)                        # O(n) 相邻相似度
    thresh = sims.mean() - tau                             # 动态阈值(常用 mean - std*)
    cuts, group = [0], [sents[0]]
    for i, s in enumerate(sims):
        if s < thresh:                                     # 相似度骤降 = 主题切换点
            cuts.append(i + 1)
    cuts.append(len(sents))
    return [sents[cuts[i]:cuts[i+1]] for i in range(len(cuts)-1)]
```

复杂度标注：递归分割 O(n·L)（n 为长度字符数）；语义切分多付一次全文 embedding（离线可承受，在线重嵌入时贵）。

**【30秒答题框架】**

"分块是检索上限的第一杠杆。太小语义断裂、答案被切两半；太大池化稀释、向量信噪比下降还占上下文预算。默认用递归分割——按段落句号空格逐级降级，保自然边界，256-1024 token 加 10-20% overlap；高质量语料用语义切分，生产最推荐父子块：子块保证召回锋利度，父块保证喂给 LLM 的上下文完整。所有大小最终用评测集跑 Recall@k 曲线定，不拍脑袋。"

**【追问与陷阱】**

- 追问：**overlap 的作用和代价？** 防边界切断关键表达；代价是索引膨胀（膨胀率 1/(1-overlap 比例)）和重复命中，需在融合后按 doc 去重。
- 追问：**chunk 大小和模型有关吗？** 两处相关：embedding 模型窗口上限（传统 BGE 512 token、bge-m3 8192），LLM 侧上下文预算与注意力质量。
- 追问：**怎么验证你的分块设置更好？** 固定其余环节，只改 chunking，对比 Recall@k / Context Recall（题 10）——单变量对照实验。
- 陷阱：按固定字数硬切（丢自然边界）；用字符数不用 token 数（tokenizer 差异）；overlap 设太大导致索引膨胀 + 同源重复计票。
- 追问：**代码文档怎么切？** 按函数/类 AST 边界切，保持定义-注释-实现一体。

**【知识串联】**

- 题 2：title_path 前缀拼进 chunk 补偿切断的语境（chunk 检索的是局部，元数据还原文档结构）。
- 题 7：大 chunk 直连 lost in the middle 问题。
- `01-Foundation/01-Transformer`：池化机制与注意力预算。
- 题 10/`06-Safety-Evaluation`：chunking 调参必须挂评测集（评测驱动开发）。

---

### 题 4：常用 Embedding 模型

**【考察意图】**

- 考**选型方法论**而非背模型名：给一个业务场景能否落地成结构化决策。
- 机制理解：对比学习如何训练出可检索向量；为什么 query/passage 要不同指令前缀。
- 成本与工程意识：维度-存储-延迟的量化换算；榜单只能初筛、域内评测才算数。

**【深度解析】**

**一、原理底座：对比学习（必须会写损失函数）**

语义检索向量由 InfoNCE 类对比损失训练：

$$\mathcal{L} = -\log \frac{\exp(\cos(q, d^+)/\tau)}{\sum_{i \in \{d^+, d^-_1, \dots, d^-_n\}} \exp(\cos(q, d_i)/\tau)}$$

要点（面试逐条挤出）：sim 用 **L2 归一化后的点积 = 余弦**；正例 = 相关段落，负例 = in-batch negatives（大批次多负例）+ hard negatives（挖掘的易混难例）；温度 τ 典型 0.02-0.05，越小对难例越敏感。这使得"query 语义"与"相关文档语义"在球面上对齐——整个向量检索的理论基础。

**二、选型决策表（对着表答）**

| 决策维度 | 选项 A | 选项 B | 判断要点 |
|----------|--------|--------|----------|
| 质量 | MTEB/C-MTEB 榜单 | **自建领域评测集** | 榜单只做初筛；自家语料的 Recall@k 定生死 |
| 维度 | 512-1024 | 3072（OpenAI 3-large） | 维度↑ 精度/存储/延迟全变差；MRL（Matryoshka）支持任意截断降维 |
| 最大长度 | 512 token（传统 BGE） | 8192（bge-m3、jina-v3） | 必须 ≥ 你的 chunk 长度，否则截断丢信息 |
| 语言/领域 | 单语专精 | 多语（bge-m3 100+ 语） | 混合语料选多语；专名词多的领域考虑微调 |
| 稀疏/多向量支持 | dense only | BGE-M3 三合一 | 一套模型同时供 dense/sparse/ColBERT 式多向量 |
| 部署 | API（简单、按次付费、数据出域） | 自托管（可控、可微调、数据内网） | 合规与成本的双向决策 |

**代表模型点评**（原表 4 款 + 当前生态）：

- **OpenAI text-embedding-3-large（3072 维）**：商业第一梯队；支持 MRL 降维到任意目标维度省存费。
- **BGE-M3（1024 维）**：中文生态事实标准；dense + sparse + multi-vector 三模式，最大 8192 token；开源可自托管可微调。
- **E5-Mistral-7B-Instruct（4096 维）**：LLM-based encoder，MTEB 强；推理显存/延迟高，在线成本敏感场景慎用。
- **GTE-Qwen2 家族**：基于 Qwen2 的 embedding；官方为 1.5B→1536 维、7B→3584 维（原表写 1024 与模型卡不一致，以 HF 模型卡为准）；中英表现强。
- 补充：**Qwen3-Embedding**（2025 年发布，多尺寸、强中文）——2026 面试提一句显得跟得上生态（【研究文档】表五同源 Qwen 生态线）。

**三、指令前缀与相似度度量（高频事故点）**

- 非对称模型对 query 和 doc 用**不同前缀**：如 BGE 系列 query 侧加 `"为这个句子生成表示以用于检索相关文章："`，passage 侧不加。原理：指令把"检索意图"注入 query 向量，使 query 分布与训练时的分布一致——**漏加前缀属于线上常见事故，召回直接掉**。
- 相似度：归一化后点积 = 余弦；**未归一化模型别用内积**（长度主导相似度）；余弦 vs 欧氏在归一化下单调等价。
- fusion 用 RRF 时不受此影响（rank-only），但 dense 单路内部要对（题 5）。

**四、量化成本换算（数字必须会说）**

- 存储：1024 维 fp32 = 4KB/向量；千万级 = 40GB，亿级 = 400GB → PQ 压缩可达 64B/向量级（题 5 详）。
- 在线延迟量级：轻量 BERT 级模型 GPU encode 单句 ~几 ms；7B 级 LLM-encoder 数十至上百 ms——**在线改写流程里的 embedding 延迟要计入链路预算**（题 1 的分段延迟表）。
- 微调选项：领域词表丰富/同义映射怪时，用对比损失 + 领域 hard negatives 微调（成本远低于微调 LLM，参照 `03-FineTuning/01-PEFT` 思路），往往能把领域 Recall@k 拉一截。

**【30秒答题框架】**

"选型看五维：质量（MTEB 初筛加自建评测定夺）、最大长度要覆盖我的 chunk、维度权衡精度和存储（1024 维 fp32 一个向量 4KB）、中英文与领域、API 还是自托管。中文生产默认 BGE-M3：1024 维、8K 上下文、dense 加 sparse 加 ColBERT 多向量三合一，开源可微调。两个必踩坑：query 和 passage 的指令前缀要按模型卡加，漏了直接掉召回；相似度用归一化点积即余弦，未归一化模型别用内积。榜单只能初筛，最后用自己的评测集说话。"

**【追问与陷阱】**

- 追问：**为什么 query/doc 前缀不同？** 训练分布使然：指令引导 query 映射到检索语义子空间（E5/BGE 的 instruction-tuned embedding）；漏加 = 分布漂移 = 掉分。
- 追问：**要不要在私有语料上微调 embedding？** 徽章判断：领域专有名词密集、通用模型同义映射差时值得；用对比学习 + 领域负例，成本比微调生成模型低一个量级。
- 追问：**MRL 是什么？** Matryoshka Representation Learning：训练时让前 k 维分级承载语义，可推理后任意截断降维（成本-精度开关）。
- 追问：**in-batch negatives 为什么 batch 越大越好？** 批内负例数 = batch_size - 1，更多负例 → 更硬的判别任务 → 分离度更好；受显存/GPU 数限制（题 3 的对比学习基础）。
- 陷阱：误用内积配非归一化向量；漏指令前缀；把榜单分数当最终依据；忽略 embedding 窗口与 chunk 长度不匹配。

**【知识串联】**

- `01-Foundation/01-Transformer`：池化与编码器结构；`03-Loss-And-Optimization`：InfoNCE/温度参数的通用损失视角。
- `03-Finetuning/03-Instruction-Tuning`：指令前缀与指令对齐思想同源。
- `02-Inference/03-Quantization-And-Deployment`：embedding 模型的量化部署与 LLM 量化同构思想。
- 题 5（向量路相似度是混合检索一翼）、题 10（评测闭环验证选型）。

---

### 题 5：向量检索 vs 关键词检索 vs 混合检索

**【考察意图】**

- **字节二面标志题**：三种检索的机理差异要到"为什么"层面；BM25 公式要能默写并逐项解释；RRF 要能推导 + 实现。
- ANN 索引工程深度：HNSW/IVF-PQ 的复杂度与内存量化。
- 边界感：什么场景纯 BM25 反而更好——考察是否唯 dense 论。

**【深度解析】**

**一、BM25 完整公式（默写级）**

$$\text{score}(q,d) = \sum_{t \in q} \underbrace{\ln\left(\frac{N - df_t + 0.5}{df_t + 0.5} + 1\right)}_{\text{IDF: 稀有词加权}} \cdot \frac{tf_{t,d}\,(k_1+1)}{tf_{t,d} + k_1\left(1 - b + b\,\dfrac{|d|}{\overline{dl}}\right)}$$

逐项拆解：

- **IDF**：文档频率 df 越低、词提供的信息量越大；*+1 平滑*防负值。
- **tf 饱和（k1 ≈ 1.2-2.0）**：与 TF-IDF 的线性 tf 不同——分子分母都含 tf，词频增益递减：出现 10 次 ≈ 2 次的含义，防关键词堆砌作弊。k1=0 时完全不看 tf，k1→∞ 趋向线性 tf。
- **长度归一（b ≈ 0.75）**：|d| 超出平均长度 $\overline{dl}$ 就惩罚；b=1 完全按长度惩罚、b=0 关闭归一。防止长文档靠"塞词"吃分。

BM25 弱点：不知道同义词（"笔记本"≠"laptop"）、不理解语义改写——这正是向量检索的领地，反之亦然。

**二、向量检索与 ANN 索引（工程量化）**

相似度：$\cos(E(q), E(d))$（归一化点积）。全库暴力扫是 O(N·d)，生产用**近似最近邻（ANN）**：

| 索引 | 原理 | 查询复杂度 | 代价 |
|------|------|-----------|------|
| **HNSW** | 分层可导航小世界图：顶层稀疏长边负责跳跃，底层稠密负责精确，贪心逐层下降 | 近似 O(log N) | 建索引慢（增量插入友好但删除靠墓碑）；**内存贵**：邻接表 ~M×指针/向量（M=16，每向量 32-64B 指针开销，1 亿向量数 GB） |
| **IVF-PQ** | 粗量化分桶（nlist ~ √N 量级）缩小搜索范围 + 乘积量化：d 维劈 m 份各聚 256 类，存 8-bit 子码 | O(nprobe·N/nlist)（nprobe 为探查桶数） | 内存极省：1024 维 fp32 4KB/向量 → m=64 时约 64B/向量（**64×压缩**，1 亿向量 400GB→6.4GB）；召回率典型 0.90-0.97@R@10（调 nprobe 换） |

一句话数值结论：**内存充足、QPS 高 → HNSW；内存敏感 → IVF-PQ 系列**。三个轴权衡：recall / latency / memory，此消彼长。

**三、为什么必须混合**

错误模式正交：

- dense 失效场景：精确字符串（型号 "RTX-4090"、编号、人名生僻字、缩写 "API"、代码函数名）——embedding 拿这些"语义模糊但字面唯一"的 token 没办法。
- BM25 失效场景：同义改写（"报销"↔"花销"）、跨语言、长表述。

BEIR 上混合检索普遍优于任一单路——**正交错误模式互补**是理论依据。

**四、RRF 融合（原章节公式，要学会推导与实现）**

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

- rank 从 **1** 开始（第一名得 1/61）；k=60（Cormack et al. 2009 经典缺省）。
- **为什么用 rank 不用 score**：BM25 分数无上界、cos ∈[-1,1]，**两路分数量纲不可比**；要做分数级融合必须先校准（min-max/z-score），而校准依赖分布稳定性。rank 是纯序数、天然免校准、对单路分数异常鲁棒。
- **k 的作用**：压低第 1 名和第 61 名之间的票差（1/61 vs 1/121，差距仅 1.97 倍）——k 越大权差越平（趋近"只数出现路数"），k=0 退化为"纯 top1 抢票"。60 是论文验证的鲁棒缺省。

**手算示例**（面试现场能算 = 加分）：

文档 d1、d2，两路检索排名——A 路：d1 第 1、d2 第 3；B 路：d2 第 1、d1 第 5。k=60：

- RRF(d1) = 1/61 + 1/65 ≈ 0.01639 + 0.01538 = **0.03177**
- RRF(d2) = 1/63 + 1/61 ≈ 0.01587 + 0.01639 = **0.03226**

d2 胜——"两路都靠前"压过"单路第一"，这正是融合要的效果。

**五、参考实现（可运行，全纯 Python）**

```python
import math
from collections import defaultdict

def bm25_score(query_tokens, doc_tokens, docs_lens, avgdl, df, N, k1=1.2, b=0.75):
    """单文档 BM25 打分。tf 用 Counter 预存则单次打分 O(|q|)。
    全库打分 O(Σ|d|)——生产走倒排索引，只算 postings 命中项。"""
    tf = defaultdict(int, {t: doc_tokens.count(t) for t in set(doc_tokens)})  # 演示用
    score = 0.0
    for t in set(query_tokens):
        if t not in tf or t not in df: continue
        idf = math.log((N - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
        denom = tf[t] + k1 * (1 - b + b * len(doc_tokens) / avgdl)
        score += idf * tf[t] * (k1 + 1) / denom
    return score

def rrf_fuse(rankings: list[list], k: int = 60, topn: int = 50):
    """多路结果 RRF 融合。rankings[i] = 按名次排好的 doc_id 列表(rank 从 1 计)。
    复杂度 O(Σ|ranking_i|)，用 dict 累加。"""
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):   # 从 1 开始！
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)[:topn]

# demo：两路结果 → RRF（复现正文手算例）
rank_a = ["d1", "d1x", "d2", "d3", "dy"]        # A 路（d1 第1，d2 第3）
rank_b = ["d2", "z1", "z2", "z3", "d1"]         # B 路（d2 第1，d1 第5）
print(rrf_fuse([rank_a, rank_b])[:2])            # → d2 排在 d1 前
```

复杂度：BM25 经倒排后打分 O(Σ postings 长度 · 平均|q|)；RRF 融合 O(路数 × 每路 K)；排序 O(M log M)。

**【30秒答题框架】**

"三种方法错误模式正交：dense 管语义泛化和同义改写，BM25 管字面精确匹配——型号、编号、缩写这些 embedding 的死穴。所以各召回 top 50-100 用 RRF 融合：score 等于 Σ 1/(k+rank)，k=60，rank 从 1 起。用 rank 不用分数，因为 BM25 和余弦分数量纲不可比，需要校准；rank 序数天然免校准。BM25 公式能默写：IDF 乘 tf 饱和项（k1 控制词频增益递减）乘长度归一项（b=0.75）。向量侧 ANN 用 HNSW，查询近似 O(logN)，内存敏感就上 IVF-PQ，能把 4KB 一个向量压到 64B。"

**【追问与陷阱】**

- 追问：**RRF 的 k 调大调小什么效果？** k→∞：所有名次权重趋同（近似投票制）；k=0：只有各路 top1 有效。60 鲁棒；可用小调优，但收益边际小。
- 追问：**什么场景纯 BM25 更好？** 法律/医疗法条（术语必须精确）、代码/函数名检索、无 GPU 环境；dense 反而在这些场景产噪。
- 追问：**分数级融合什么时候优于 RRF？** 同构分数（两路都是 dense）且校准数据充足时；RRF 丢掉了"分数差距大小"信息。
- 追问：**一亿向量怎么选索引？** 内存放得下且追求高 QPS → HNSW；内存敏感 → IVF-PQ（400GB→6.4GB）或 HNSW + PQ 混合向量。
- 陷阱：rank 从 0 数起（名次偏移错误）；把 RRF 分数当"语义相关度"解释（它是启发式融合分）；两路 K 太不对称导致融合偏置；忘记去重（同 doc 多 chunk 重复计票）。
- 追问：**HNSW 为什么快？** 分层跳表式图 + 贪心导航，期望 O(logN)；请求延迟稳定但**建索引与内存成本高**。

**【知识串联】**

- 题 7：RRF 之后的 cross-encoder 重排——融合解决"排序质量"，重排解决"最后一段精排"。
- 题 4：归一化是 dense 路相似度正确性的前提（RRF 免疫这个问题）。
- `02-Inference/03-Quantization-And-Deployment`：PQ 量化与模型权重量化同构思想（用精度换资源）。
- `08-Coding` / `09-SystemDesign`：本题代码与索引选型是最常出现在手撕与系统设计中的 RAG 素材。
- `06-Safety-Evaluation`：权限过滤在检索层前置。

---

### 题 6：查询改写

**【考察意图】**

- 四种方法的**机理**而非名字：HyDE 为什么有效（分布对齐论证）是本题主考分点。
- 成本效益意识：改写写在检索之前，每次调用都有真金白银——能否按"先便宜后昂贵"排序工具箱。
- 多轮对话特有问题的处理（指代消解）——生产线的第一道改写。

**【深度解析】**

**一、总纲：检索的第一瓶颈是 query-doc 语义鸿沟**

用户 query（短、口语化、含指代）与文档（长、陈述式、术语化）在词汇与文体上都不同分布。改写 = **在检索前用 LLM/规则把 query 变换到更容易检索的形态**，用小额检索前成本买召回率。

**成本/收益排序**（按部署优先级）：

```
规则层(零成本)：指代消解/纠错/同义扩展/缩写展开
→ Multi-Query(1 次 LLM 调用)：多视角改写
→ Step-Back(1 次)：抽象化背景检索
→ HyDE(1 次生成)：假设文档检索
→ Sub-Query(按需)：复杂问题分解
```

**二、HyDE 的机制级解释（主拿分点）**

流程：q → LLM 零样本生成假设性**答案**文档 $\hat{d}$ → 用 $E(\hat{d})$ 去检索真实文档。

**为什么假设答案比原 query 更好检索**：

- 对比学习的训练目标是把"相关文档对"拉近（d↔d' 同为陈述式长文本），**文档-文档子空间的相似性结构比 query-文档更良**；$\hat{d}$ 与真实文档同文体、同信息密度，落在同一分布内，$E(\hat{d})$ 与 falcon 真实 d 的距离期望小于 $E(q)$ 与 d 的距离。
- 反直觉点（HyDE 原论文自认）：**收益来自文体对齐与邻域结构，而非假设答案是事实正确的**——$\hat{d}$ 可以有幻觉，只要它"长得像答案"就能指导检索方向。
- 风险与工程对策：幻觉可能把检索带偏到错误主题 → **必须与原 query 并联检索（union/RRF 两路合并）**，保留原始语义锚点。
- 失效场景：query 本身已是标准文档式短语（改写空转）；领域词表 LLM 不熟悉时编造噪声实体（如行业黑话）。

**三、Multi-Query / Step-Back / Sub-Query**

| 方法 | 机制 | 收益 | 代价 | 适用 |
|------|------|------|------|------|
| **Multi-Query** | LLM 把 q 改写成 n 个不同视角变体（同义 + 角度切换），并行检索后 union/RRF 合并 | 召回覆盖方差下降（多角度捞） | 1 次 LLM + n 倍检索量 | 泛问、模糊表达 |
| **Step-Back** | 先生成"退一步"的抽象问题（背景/概念层），原始与退化双检索 | 先背景后细节，答案推理材料更完整 | 1 次调用 | 具体细节的机理类问题 |
| **Sub-Query** | 复合问题拆成多个原子子问题分检索再汇合 | 破解"一次检索装不下多信息源" | 多次检索 | 对比类/多跳问题 |
| **指代消解** | 多轮对话把"它/那个"绑定实体改写为独立 query | 生产必做，不做 = 后续全错 | 规则或 1 次调用 | **多轮对话场景的第一优先级** |

**数学化理解（召回覆盖）**：设每次检索命中 gold 概率 p，n 个独立视角全 miss 概率 (1-p)^n → 并联改写把单路漏检率指数级压低（假设视角相关独立；实际相关时收益打折，用 CC 系数修正——面试口头带一句"多个变体相关性越低收益越大"即可）。

**四、参考实现（HyDE + Multi-Query 骨架）**

```python
def hyde(query: str, llm, retriever, hyde_topk=10):
    """HyDE：生成假设答案 → 向量检索 → 与原 query 结果 union。
    成本：1 次 LLM 生成 + hyde_topk 检索。"""
    hypo = llm(f"请直接写出下列问题的回答（无需事实核查，用于检索）：{query}")
    hypo_hits = retriever.search(hypo, topk=hyde_topk)      # 假设文档路
    orig_hits = retriever.search(query, topk=hyde_topk)     # 原始锚点路(防带偏)
    return rrf_fuse([[h for h, _ in hypo_hits],
                     [h for h, _ in orig_hits]])            # 复用题5 RRF

def multi_query(query: str, n: int, llm, retriever, topk=20):
    """Multi-Query：n 个视角并行检索 → RRF 合并。成本：1 次 LLM + n 倍检索。"""
    variants = llm(f"把问题改写成 {n} 个不同视角的检索式，每行一个：{query}").split("\n")
    rankings = [retriever.search(v, topk=topk) for v in variants[:n] if v.strip()]
    rankings.append(retriever.search(query, topk=topk))      # 原始 query 恒在场
    return rrf_fuse(rankings)
```

**五、评估绑定**：任何改写是否值得，必须端到端验证 Recall@k / Context Recall 的提升 ≥ 成本（延迟+费用）；"看起来检索结果变丰富"不算数。

**【30秒答题框架】**

"改写的目的：弥合 query 和文档之间的语义与文体鸿沟。按成本从低到高：先做零成本规则——尤其是多轮对话的指代消解，这是生产第一优先级；然后 Multi-Query 让 LLM 把 query 变 3-5 个视角并行检索，n 路 RRF 合并，miss 概率从 1-p 变 (1-p)^n；Step-Back 先检索抽象的背景问题再检索细节；HyDE 让 LLM 生成一个假设答案直接拿去检索——原理是假设答案和真实文档文体同分布，向量空间更近，但它靠邻域结构而非事实正确性，所以必须和原 query 并联防幻觉带偏。复杂多跳问题拆子查询逐个检索。所有改写用端到端 Recall@k 验证。"

**【追问与陷阱】**

- 追问：**HyDE 假设答案是错的为什么还能检对？** 核心逻辑：检索是相似度任务，不需要事实对，只要文体/主题分布与文档一致；该为训练分布论证 + 承认极端情况下噪声主题会带偏（并联 union 兜底）。
- 追问：**多轮对话的改写为什么是第一优先级？** "它多少钱"不消解指代，任何检索方法都失效；且成本最低。
- 追问：**改写能不能叠加？** 技术上能，但每一路都是一次 LLM 调用+检索的预算；用评测决定组合，别叠成"全家桶"。
- 陷阱：改写后丢弃原 query（单一 variant 是 HyDE 的致命陷阱）；用同义改写不设防幻觉实体；改写不加缓存；没归因就上链路（应先做题 10 归因确认瓶颈在检索前段）。
- 追问：**改写延迟怎么控？** 改写模型用小模型路由（【研究文档】考点 14）、高频 query 缓存改写结果、多路并行发起检索。

**【知识串联】**

- 题 5：改写出的多路结果天然进 RRF。
- 题 8：Agentic RAG 把改写从"pipeline 定死的一步"内生化成 agent 自主决策。
- `05-Agent/01-Agent-Complete-Guide.md`：ReAct 中每次工具调用的 query 生成本质上就是运行时改写。
- 题目【研究文档】考点 6（上下文工程的"什么信息进窗口"——改写是输入侧的上下文工程）。

---

### 题 7：为什么需要重排序？

**【考察意图】**

- 机制层面讲透 bi-encoder vs cross-encoder：为什么快慢与精度互换——**token 级 cross-attention** 是根源。
- pipeline 定位与 K 值设计、延迟预算的量化能力。
- 承上启下：重排解决"精排"，但**不能突破召回天花板**（与题 8/13 的联动源于此）。

**【深度解析】**

**一、双塔 vs 交叉编码（对比核心）**

| | Bi-Encoder（双塔） | Cross-Encoder（交叉编码） |
|---|---|---|
| 计算图 | E(q)、E(d) 各自独立编码，score=cos(E(q),E(d)) | [CLS]q[SEP]d[SEP] 一次前向，query token 与 doc token 逐层 cross-attention，[CLS] 出相关性 logit |
| 交互粒度 | 仅最终池化向量（"先各自压缩再见"） | token 级细交互——词对齐、短语匹配、否定词敏感都能捕捉 |
| 能否预计算 | **能**：doc 向量离线算好建 ANN | **不能**：分数依赖 q-d 联合分布，无法离线展开 |
| 在线成本 | 1 次 query encode（~几 ms）+ ANN O(log N) | **每个候选一次完整前向**：50 条 × 512 token 序列 ≈ 50 次 Transformer 前向 |
| 精度 | 中（MS MARCO 上普遍弱于 cross，NDCG@10 差距数个百分点量级，具体看 baseline） | 高 |
| 角色 | 召回层（大集合、快） | 重排层（小集合、精） |

一句话锚点："**双塔是先压缩后匹配，单点粗匹配；cross 是先交互再压缩，token 级细匹配**"。ColBERT 是中间形态：doc 的 token 级多向量离线存，在线做 MaxSim（late interaction）——近 cross 的精度、近 bi 的速度；BGE-M3 的 multi-vector 模式即此。

**二、在 pipeline 中的定位与量化**

```
混合召回 top 50-200 → Cross-Encoder 重排 → top 3-10 → Prompt 组装
```

- **K 值设计**：召回阶段 K 由 Recall@k 曲线决定（K 太小丢真、太大重排爆炸）；重排后 3-10 条是 LLM 上下文预算的甜点位——多了触发 lost-in-the-middle 与成本。决策表：预算 100ms 重排窗口 ~ 50 条（中等模型）/ 200 条（蒸馏小模型）。
- 延迟预算（对话可用）：检索段合计 ~50ms 不含重排；重排 50-300ms（看 K 与模型规模）；LLM 段 0.3s-数秒——**重排是花百毫秒买显著 NDCG 提升的划算交易**，但高 QPS 时需要蒸馏/量化/层剪枝（bge-reranker-v2 有 layerwise 剪分支）。
- 常用法：bge-reranker-v2-m3（多语、8K）、Cohere Rerank 3（API）、Jina-reranker、英文经典 cross-encoder/ms-marco。

**三、进阶与消融意识**

- LLM 直接当 reranker：listwise——把 10 条候选一次给 LLM 排序，可依据复杂理由判断；效果常优于 pointwise，但贵且慢；MonoT5 是传统 pointwise。
- **重排天花板论**：重排是精排不是召回——gold 根本没进 top50，重排再强也没用（Recall 上限钉死）。这一笔把话题引向：重排前保留查询的结果和宽度（题 5 融合 K 要足够）、改写补召回（题 6）、失败兜底 agentic（题 8）、以及归因确认瓶颈（题 10）。
- 重排后的**上下文排序**：top 结果进 prompt 要防 lost in the middle（Liu et al. 2023，U 形注意力曲线：头尾好、中部差）——最重要材料放开头，次重要放结尾。

**【30秒答题框架】**

"重排解决双塔模型的精度不足：bi-encoder 各自编码、最终只在池化向量上做粗匹配，token 级交互全丢了；cross-encoder 把 query 和 doc 拼一起进 Transformer，做 token 级 cross-attention 深交互，精度高但每对候选都要完整前向、没法预计算，所以只能放在召回之后的小集合。流水线是召回 50-200 → 重排 → 3-10 进 prompt，重排花几十到几百毫秒换显著 NDCG 提升，划算。两条底线：重排不能突破召回的天花板——没检到的排不出来；重排结果进 prompt 要防 lost in the middle，重要材料放开头。"

**【追问与陷阱】**

- 追问：**cross-encoder 为什么不能建 ANN 索引？** 分数需要 q-d 联合前向，doc 无法独立成向量预存——病根在计算图结构。
- 追问：**ColBERT 是什么、为什么算折中？** late interaction：token 级向量离线存 + MaxSim 在线近似，精度/速度都在两者之间。
- 追问：**重排 K 怎么定？** 评测集上扫 K ∈ {50,100,200} 的 Recall@k 增益 vs 延迟曲线，卡延迟预算定。
- 追问：**重排分数能当绝对阈值用吗？** 不能跨 query 直接比（logit 未校准）；要阈值需用评测集标定。
- 陷阱：**顺序做反**（先重排后融合——融合必须在重排之前，否则多路无法合成一个统一候选池）；top-k 塞给 LLM 无脑 20 条不控成本；把重排当成"必须更贵的模型才有效"（蒸馏版常够用）。

**【知识串联】**

- 题 5：RRF 融合在前、重排在后的顺序是硬约束。
- 题 10：Context Precision 是重排质量的直接度量指标。
- `01-Foundation/01-Transformer`：cross-attention 机制根源（这里就是你判"query 对 doc 每个 token 注意"的结构基础）。
- `02-Inference/03-Quantization-And-Deployment`：重排器量化/蒸馏部署。
- 题 8：重排天花板 → agentic 补召回。研究文档"context engineering"（prompt 预算有限 → 重排本质是预算分配）。

---

### 题 8：Agentic RAG

**【考察意图】**

- 范式转变的理解：检索从 **pipeline 的固定一步** 变为 **agent 可决策、可循环的工具**。
- 具体机制：Self-RAG 的反思 token 设计、CRAG 的失败恢复分支、多跳迭代（IRCoT/Self-Ask）。
- 三角取舍判断力：RAG vs 长上下文 vs Agentic RAG，每种选择的成本收益与组合策略——2026 年面试的实质区分题。

**【深度解析】**

**一、范式光谱与三个自主性**

```
Naive RAG:  检索(一次) → 拼接 → 生成
Advanced RAG: 检索前改写 + 检索后重排（仍是固定管路）
Agentic RAG: LLM 循环决策——何时检索、检索什么、结果够不够？
```

三个自主决策权：**When**（要不要检索——闲聊/常识问题免检索）、**What**（query 怎么生成/分解——题 6 的改写从 pipeline 配置变成 agent 的工具参数）、**Enough**（结果质量自评——不充分就改写重来）。

**二、代表机制（细节须准确）**

**Self-RAG（Asai et al., 2023-10）**：训练模型输出四类反思 token：

| 反思 token | 判断什么 |
|-----------|---------|
| Retrieve | 当前是否需要检索 |
| IsRel | 检回的片段相关性 |
| Issup | 答案关于片段的支撑性 |
| Isuse | 回答的实用性 |

推理时按 token 分支：决定检索时机，并在多候选中按反思标记挑选。实现来源：**用监督数据微调**（GPT-4 造数据 + 判别标签；教学解释见【研究文档】的微调心理学类比——反思能力是数据教出来的，不是天然 prompt 出来的）。工程常用折衷：不微调，用 prompt 让现成模型模拟这一流程。

**CRAG（Corrective RAG, 2024-01）**：轻量检索评估器（T5 级）给片段打置信 → **Correct**（精炼后采用）/ **Incorrect**（丢弃，触发 **web 搜索兜底**+query 重写）/ **Ambiguous**（两者结合）。价值：把"检索失败可恢复"变成显式分支，切断错误检索向生成段的传导——直接回应题 13 的 badcase：检索不对时的自动挽救。

**多跳检索**：IRCoT（检索与 CoT 交错：每步推理结果决定下一次检索词）、Self-Ask（问题分解成子问题链依次检索）。对比：Naive RAG 一次检索的天花板在"证据分散于多文档"时必破。

**三、三角取舍（RAG vs 长上下文 vs Agentic，压轴权重）**

| 决策因子 | 长上下文直塞（Gemini 3 Pro 1M / GPT-5.1 400K，【研究文档】表五） | 标准 RAG | Agentic RAG |
|----------|---------------------------------|----------|-------------|
| 语料规模 | ≤ 窗口（1M token 顶天） | 任意（索引无限扩展） | 任意 + 可跨源 |
| 成本 | **最高**：token 费 + KV 显存随长度线性涨（7B fp16 ≈ 0.5MB/token，128K ≈ 64GB；prefill 注意力 O(L²)，FlashAttention 优化后计算仍二次方） | 低（一次检索 + 固定 prompt） | **LLM 调用 ×2-4**（决策循环 + 自评 + 兜底），延迟 2-5× |
| 注意力质量 | 长文衰减（context rot；lost in the middle） | 上下文短而精 | 好但依赖自评器不漂移 |
| 检索失配风险 | 无（全部可见） | 每步存在（需题 6 改写 + 题 7 重排缓解） | **有自愈机制**（改写重试、CRAG 兜底） |
| 溯源与更新 | 中/版本不友好 | 强（索引即事实源） | 强 |
| 失败模式 | 长文中段"读不到"（位置偏差） | 检索空转/字段错 | 循环不收敛（需停机条件） |
| 可靠性风险 | 低（链路短） | 低 | **链式衰减**：多步串行成功率 0.95^n（n=20 步 ≈ 36%，【研究文档】考点 10） |

**组合策略（一句话收尾）**：生产主流 = **RAG 底座 + 按需 agentic**——用题 13 的归因识别失败模式；单跳事实问答走标准 RAG；多跳/检索不稳/复杂推理才路由到 agentic 通道（快慢分层，当费用敏感时小模型做预判——【研究文档】考点 14 模型路由）。

**四、工程护栏**

- 最大迭代上限（3-5 步）+ 停机条件（重复 query 停/自评置信停/预算停）。
- 每轮检索结果去重与合并（防止循环自我喂养、重复结果反复进入上下文）。
- 全链路 tracing（每步 query、命中、判分落日志）——否则线上不可调试。

**【30秒答题框架】**

"Agentic RAG 把检索从写死的 pipeline 一步变成 agent 可决策的工具，三个自主性：要不要检索、用什么 query、结果够不够自己判。代表机制：Self-RAG 用反思 token 教模型决定检索时机与自评；CRAG 用轻量评估器给检索结果打折，不合格就重写 query 或走 web 兜底；多跳用 IRCoT/Self-Ask 迭代。和长上下文的取舍：语料小于窗口且单次任务，直接塞最省事；海量、高频更新、多用户必 RAG；多跳和检索不稳才上 agentic——代价是 LLM 调用翻两到四倍、多步链式可靠性衰减。生产主流是 RAG 底座加按复杂度路由的 agentic 通道。"

**【追问与陷阱】**

- 追问：**Self-RAG 反思 token 从哪来？** 监督数据微调（GPT-4 生成 + 人工判标）；不微调的 prompt 模拟版是工程折衷，效果打折。
- 追问：**CRAG 引 web 搜索安全吗？** 引入不可信外部源与新攻击面（检索内容注入，见 06-Safety）——兜底源也要过滤与溯源。
- 追问：**怎么决定要不要上 agentic？** 用题 13 归因：失败集中在检索不齐/多跳信息分散时值得；单跳检索稳定别过度工程。
- 陷阱：把它说成"加了个 while 循环"（必须有决策依据与自评信号）；不做迭代上限与预算上限；忽视多步链式衰减（0.95^n）；全上 agentic 不分情况。
- 追问：**循环几次合适？** 经验 3-5 步内；停机条件三件套：重复检测、置信阈值、预算上限。

**【知识串联】**

- `05-Agent/01-Agent-Complete-Guide.md`：Agentic RAG 是 ReAct/反思（Reflexion）范式在检索域的特化（研究文档考点 3/8/16）。
- 题 6：查询改写在 agentic 框架里从"pipeline 配置"升格为"工具参数"。
- 题 13：归因结果决定 agentic 的投入产出判断。
- `02-Inference/01-KVCache-And-Acceleration`：长上下文成本的显存公式支撑。
- 【研究文档】考点 8/9/10/14（RAG 与 agent 的关系、多智能体慎用、可靠性链式衰减、成本路由）。

---

### 题 9：Graph RAG

**【考察意图】**

- 讲清 GraphRAG 解决的**独有动机**：全局归纳性问题与多跳关系问题——不是"向量检索加花活"。
- 机制细节诚实度：实体抽取 → 图 → **Leiden 社区检测** → 分层摘要 → local/global 双搜索模式。
- 成本觉悟与适用边界：离线建图 LLM 成本数倍以上，何时该上、何时不该上。

**【深度解析】**

**一、动机：向量检索的两个死穴**

1. **全局/归纳性问题**："这批文档的三大主题是什么""给这个项目写一份风险综述"——证据散落全部语料，top-k 相似片段**凑不出全景**；向量检索天生是"局部匹配"机制。
2. **多跳关系查询**："A 的合作方的投资标的有哪些"——两跳关系在向量空间里**结构信息丢失**（E(A合作B) 与 E(B投资C) 不相邻）。

**二、GraphRAG 机制（Microsoft, Edge et al. 2024-04《From Local to Global》）**

**离线索引（成本大头）**：

```
语料 chunk 化 → LLM 抽取(实体 + 关系 + 声明, 每 chunk 一次抽取调用)
→ 知识图谱(节点=实体, 边=关系带描述)
→ Leiden 社区检测(把稠密连通的实体聚成社区)
→ 逐社区 LLM 生成摘要，社区→父社区逐级向上(分层摘要树)
```

- **Leiden 而非 Louvain**：Leiden 修复 Louvain 的缺陷——可能产生"内部不连通的社区"；Leiden 保证社区连通、且层次结构稳定。
- 抽取成本：每 chunk 一次 LLM 调用 → **离线索引成本是纯向量 RAG 的数倍以上**（微软官方 FAQ 即建议用便宜模型如 gpt-4o-mini 级做抽取来控制成本；具体金额随模型价变动，回答时说量级并声明随价变动，不编造确定数）。

**在线查询两模式**：

- **Local search**（细节/实体问题）：从 query 抽实体 → 图中定位节点 → 拉取邻域（相关实体、边描述、关联 chunk、社区摘要）→ LLM 综合。
- **Global search**（总结性问题）：query 与**社区摘要**比相似度选 top-K 社区 → map（每社区若干局部答案）→ reduce（汇总全局答案）——map-reduce 风格，只对摘要树操作，**不扫原图文**，因此能回答"全局性"问题。

**三、生态类比（加分）**

- 传统 KGQA 时代：人工 schema + 模板/子图匹配，重而窄。
- **Zep/Graphiti 时序图**（【研究文档】表四）：双时间线事实有效期——"事件 A 何时成立、何时失效"建模，适合动态事实与合规审计；mem0 的图记忆是同一思想在个人记忆域的落地（记忆系统图化是同构趋势）。
- 轻量替代品：LightRAG 等 —— 说明"是否需要完整 GraphRAG"是个光谱决策。

**四、适用边界与决策流程（面试收束）**

```
先用向量 RAG 打底（成本低、覆盖广）
→ 跑题 13 归因：badcase 是否集中在 多跳关系 / 全局总结 / 实体关系型?
→ 集中 → 引入图（先局部搜索最稳）；全局总结量大才上 global search
→ 持续评估建图质量（实体消歧错误的连锁污染比向量检索错误更难修）
```

**【30秒答题框架】**

"GraphRAG 解决向量检索的两个死穴：全局归纳性问题——总结类 query 靠 top-k 相似片段凑不出全景；以及多跳关系问题——结构信息在向量空间丢失。做法：离线用 LLM 抽实体建图，Leiden 社区检测分组，逐社区生成分层摘要；在线 local search 走实体邻域答细节，global search 对社区摘要做 map-reduce 答总结。代价要讲清楚：离线 LLM 抽取让索引成本贵数倍以上，图抽取自身也有错误，实体消歧错了会连锁污染。工程建议先用向量 RAG，归因证明 badcase 集中在多跳和全局题再上图。"

**【追问与陷阱】**

- 追问：**和传统知识图谱问答的区别？** KGQA 依赖人工 schema 建模；GraphRAG 让 LLM 自动抽取 + 社区摘要，免 schema 但图有噪声——换来的是大规模开放语料的可行性。
- 追问：**为什么用 Leiden 不用 Louvain？** Leiden 保证社区内部连通、拓扑更稳——回答出"连不通的社区会破坏摘要有效性"是加分。
- 追问：**global search 为什么必须用社区摘要？** 直接对全部原文做 map-reduce 成本不可控；摘要树的层次化把 O(全库) 压到 O(社区数 × 每社区一次)。
- 陷阱：把 GraphRAG 当"万能升级"（多半场景钱花在不需要结构化关系的语料上）；忽略实体消歧错误（同名实体合并错 → 关系全错）；忘了它解决的是**特定问题类型**而非整体准确率。
- 追问：**图和向量怎么配合？** 双路：图邻路扩展 + 向量语义检索合并进重排；互相补结构信息与语义信息。

**【知识串联】**

- 题 1（检索段定位）、题 5（双路融合）、题 13（从 badcase 归因驱动是否引入图）。
- 题 8：多跳问题也可用 agentic 迭代解——与图的两条路线做成本对比，是高级追问素材。
- `05-Agent/01-Agent-Complete-Guide.md`：图作为 agent 的另一类记忆/工具。
- 【研究文档】表四（Zep/Graphiti 时序图、mem0 图记忆——图谱化记忆的生态）。

---

### 题 10：RAG 评估指标

**【考察意图】**

- RAGAS 指标的**计算原理**（LLM-as-judge 流程 + 公式形态），不是背名称表。
- 归因方法论：**幻觉到底出在检索段还是生成段**——本场面试的"落点题"，检验端到端诊断能力。
- LLM-as-judge 的已知坑与校准手段——评测素养（研究文档考点 11）。

**【深度解析】**

**一、指标全景：两条轴 + 五指标（RAGAS）**

评估轴：检索质量（有 gold）与生成质量（LLM-as-judge）；RAGAS 核心五指标：

| 指标 | 侧 | 衡量 | 计算（流程级） |
|------|-----|------|----------------|
| **Faithfulness** | 生成 | 答案是否忠于 context | 答案拆原子陈述 → 逐句 LLM 判"可由 context 推出？"，忠实度 = 可支撑数/总陈述数 |
| **Answer Relevancy** | 生成 | 答案是否切题 | 从答案**反向生成** n 个问题 → cos(E(q_i), E(q)) 均值 |
| **Context Precision** | 检索 | 相关块是否排名靠前 | 逐位次判断相关性 v_k → 平均精度式加权：CP = (1/Σv_k)Σ_k v_k·precision@k |
| **Context Recall** | 检索 | 答案所需信息是否都检到 | GT 答案拆句 → 逐句"能否归因于 context"，可归因句占比 |
| **Answer Correctness** | 端到端 | 与标准答案一致 | 事实重叠（陈述级 F1）与语义相似度加权：w·F1 + (1-w)·sim |

公式形态（面试直接可写）：

- $\text{Faithfulness} = \dfrac{\#\{\text{v}_i=1\}}{|\text{陈述集}|}$
- $\text{AnswerRelevancy} = \dfrac{1}{n}\sum_i \cos(E(q_i), E(q))$
- $\text{ContextPrecision} = \dfrac{1}{\sum_k v_k}\sum_{k=1}^{K} v_k \cdot \text{precision@}k$
- $\text{ContextRecall} = \dfrac{\#\{\text{可归因 GT 句}\}}{\#\{\text{GT 句}\}}$
- $\text{AnswerCorrectness} = w\cdot F1_{\text{fact}} + (1-w)\cdot sim_{\text{semantic}}$

传统 IR 指标（有标注时）：$\text{Recall@K} = \frac{|\text{gold}\cap\text{topK}|}{|\text{gold}|}$；$\text{MRR} = \frac{1}{|Q|}\sum_q \frac{1}{\text{rank}_q}$；$\text{NDCG@K} = \frac{DCG}{IDCG}$，$DCG = \sum_{i=1}^{K}\frac{2^{rel_i}-1}{\log_2(i+1)}$（注意对数底 2 与 i+1 的标准形式）。

**指标-问题矩阵（把指标变成"改哪里"的处方）**：Context Recall 低 → 分块/改写/召回问题；Context Precision 低 → 重排顺序问题；Faithfulness 低 → 生成段不守规矩；Answer Relevancy 低 → 生成段偏题。**这一矩阵是题 13 归因的数学化。**

**二、badcase 归因：检索段还是生成段（主拿分流程）**

```
Step 0 知识在库里吗？——不在 → 知识库缺内容（更新文档），别调算法
Step 1 gold chunk 进 top-k 了吗？（用 Context Recall / 自建标注）
   ├─ 没进 → 检索段问题，再细分：
   │    分块切断 / query-doc 词汇鸿沟(该改写) / embedding 领域弱
   │    / 缺 BM25 路 / K 太小 / 重排把对的挤掉
   └─ 进了但答案没用 → 生成段问题，再细分：
        位置效应(lost in the middle, 挪到头尾) / 材料冲突未消解
        / prompt 缺"只依据材料"约束 / 缺拒答门控 / 模型能力不足
```

**定夺实验（counterfactual swap，加分点）**：

- 固定生成段（同 prompt、同模型），只换检索输入 → 指标大幅波动 ⇒ 检索占主导。
- 固定检索输入，只换模型/prompt → 波动 ⇒ 生成占主导。
- 该对照实验把"责任占比"量化，而不是吵谁的锅。

**线上评估**：目的指标（解决率/满意度/badcase 率人工分级）+ 代理指标（引用点击率、答案-检索相似度分布漂移），分层抽样人工标注；与研究文档"评测挂 CI、改动带对比 signed-off"对齐。

**三、参考实现（RAGAS 风格最小实现）**

```python
import numpy as np

def faithfulness(answer: str, contexts: list[str], judge) -> float:
    """忠实度：拆原子句 → 逐句判支撑。复杂度 O(|claims|) 次 judge 调用。"""
    claims = judge(f"把回答拆成原子事实句，每行一个：{answer}").splitlines()
    ctx = "\n".join(contexts)
    supported = sum(
        1 if judge(f"该陈述能否仅由下面材料推出？只答 是/否\n材料：{ctx}\n陈述：{c}") == "是"
        else 0
        for c in claims
    )
    return supported / max(len(claims), 1)

def answer_relevancy(answer: str, query: str, llm, embedder, n=3) -> float:
    """切题度：从答案反推 n 个问题 → 与原 query 余弦均值。O(n) embed + O(n·d) 相识。"""
    gens = [s.strip() for s in
            llm(f"根据这个回答，反生成 {n} 条它可能在回答的问题，每行一条：{answer}").splitlines()][:n]
    E = lambda ts: (lambda m: m / np.linalg.norm(m, axis=1, keepdims=True))(np.asarray(embedder(ts)))
    Qo, Qg = E([query]), E(gens)                          # 归一化: 点积=余弦
    return float((Qg @ Qo.T).mean())

def context_recall(gt_answer: str, contexts: list[str], judge) -> float:
    """召回度：GT 答案逐句判能否归因到 context。O(|gt 句|) 次 judge。"""
    sents = [s for s in gt_answer.split("。") if s.strip()]
    ctx = "\n".join(contexts)
    hit = sum(1 if judge(f"该句信息能否由材料归因？ 是/否\n材料：{ctx}\n句：{s}") == "是" else 0
              for s in sents)
    return hit / max(len(sents), 1)
```

实现注意真实版要点：judge 温度 0、固定版本；embedding 用同一检索也用的编码模型保持一致口径。

**四、LLM-as-judge 的坑（必答）**

- judge 偏好：长度偏好、自负偏好、position 偏置——**双 judge（不同厂商）交叉 + 固定比例人工抽检校准**（研究文档考点 11：LLM-as-judge 要人工校准）。
- 评测纪律：评测集与调参/超参选择版本隔离（否则 judge prompt 过拟合到调参集）；**评测集必须含拒答样本与检索失败样本**；线上 A/B 要显著性检验。

**【30秒答题框架】**

"评估分检索侧和生成侧。检索侧：Context Recall 看 GT 答案句都能不能归因到检索结果——测漏检；Context Precision 用逐位次相关性的平均精度式——测排序，直接衡量重排效果。生成侧：Faithfulness 把答案拆原子句逐句判 context 能否推出，比率即忠实度；Answer Relevancy 从答案反推问题算与原 query 的余弦均值；加 Answer Correctness 补端到端正确性。归因口诀：先确认库里有知识，再看 gold chunk 进没进 top-k——没进是检索问题（分块/改写/召回/重排细分），进了没用是生成问题（位置/指令/模型）；用 counterfactual swap——固定一段换另一段——量化责任占比。judge 要双模型交叉加人工抽检校准。"

**【追问与陷阱】**

- 追问：**Faithfulness 高但答案错了，可能吗？** 可能——模型忠实复述了一个错误 context（库内容过时/错误）。faithfulness 只测"不编造"，Correctness 才测"对"。
- 追问：**没有标准答案时 Context Recall 怎么算？** 用 LLM 从 query+理想回答生成假设 GT（HyDE 式反向），但必须人工抽检校验。
- 追问：**judge 用什么模型/温度？** 固定版本 + 低温 + 双厂商交叉 + 人工抽检一致性；同一 judge 内嵌自一致性（重复判)。
- 追问：**NDCG 与 MRR 差别？** NDCG 允许多级相关且位置折减 log2，衡量整页排序；MRR 只看第一个相关位置，用户"首条即中"体验指标。
- 陷阱：只报一个总指标不分层（检索与生成指标混着报 = 归因失语）；评测样本全是"有效问题"（缺拒答/失败样本，误导上线判断）；judge prompt 调参集中泄露（评测过拟合）；线上无显著性检验。
- 追问：**"看起来变好了"怎么反驳？** 单变量对照、固定其他段、指标差值加置信区间；归因因此才可能。

**【知识串联】**

- 题 7：Context Precision 是 rerank 的直接考卷；题 3/4：Recall 曲线是 chunk 与 embedding 调参依据。
- 题 8：归因结论决定是否要 agentic 补救——评估是架构决策的输入。
- `06-Safety-Evaluation`：评测纪律与 LLM-as-judge 校准的通用规范。
- 【研究文档】考点 11（评测素养：基准只初筛，自家任务评测集才算数；评测挂 CI）、考点 6（context rot 同样约束评估侧的 context 长度设定）。

---

## 本章高频追问 Top5

1. **"RAG、微调、长上下文到底怎么选？"**（题 1/题 8）——决策树作答：知识与溯源需求 → RAG；行为与风格 → 微调；单次小语料 → 长上下文；三者可叠，且默认顺序是 prompt/context 工程 → RAG → 微调（研究文档考点 15）。
2. **"RRF 为什么用名次不用分数？k=60 起什么作用？"**（题 5）——两路分数量纲不可比、校准依赖分布；rank 是序数免校准；k 压制名次间权差，60 是论文鲁棒缺省；能手算两路示例为最强证明。
3. **"幻觉出在检索段还是生成段？怎么做归因？"**（题 10）——三步：库里有知识吗（不是算法问题）→ gold chunk 进 top-k 吗（漏检 = 检索段）→ 进了没用（生成段：位置/指令/模型）；加分项 counterfactual swap 量化责任占比。
4. **"cross-encoder 为什么不能建索引？rerank 的 K 怎么定？"**（题 7）——分数依赖 q-d 联合前向无法离线展开；K 用评测集 Recall@k 增益 vs 延迟曲线卡预算；补充 ColBERT 的 late interaction 是折中。
5. **"什么情况值得上 Agentic RAG / GraphRAG？成本会不会失控？"**（题 8/9）——归因先行：失败集中在多跳/检索不稳/全局总结才上；Agentic = LLM 调用 ×2-4 延迟 ×2-5 且多步链式衰减（0.95ⁿ），GraphRAG = 离线索引 LLM 成本数倍以上；生产主流是 RAG 底座 + 按需路由与预算上限护栏。
