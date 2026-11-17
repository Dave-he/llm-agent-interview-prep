# 系统设计 · 逐题详解（09-SystemDesign）

> 依据同目录 `01-System-Design.md`（原章节，只读）逐题展开；研究背景：仓库根目录 `personal-agent-research.md`（个人 Agent 系统调研报告 v1.1，2026-11-13）。所有外部数字标注来源（论文/官方/调研报告），容量估算中标注"假设"的为推导用工程假设，非官方数据。与文档冲突时以论文官方口径为准。

## 本章考点分布

| 原顺序 | 题目 | 类型 | 核心考点 |
|---|---|---|---|
| 1 | 设计一个大模型推理服务 | 服务综合设计四步法 | QPS→并发→卡数换算、双阶段 workload、KV 显存公式、prefix cache 会话复用、TTFT 分解、成本治理四杠杆 |
| 2 | 设计一个企业级 RAG 系统 | 服务综合设计四步法 | 离线/在线双管道、混合检索+RRF、重排、增量索引、评估监控闭环、ACL |
| 3 | 大模型分布式训练的并行策略 | 机制+选型（训练平台视角） | 五种并行原理、通信量与 bubble 公式、3D 并行决策树、数据/DPO/评估闭环、6ΨT 吞吐算例 |
| 4 | DeepSpeed ZeRO 三个阶段 | 显存机制题 | 16Ψ 公式、三阶段切分与渐近 4×/8× 语义、通信量 1.5×、7B/13B/65B 显存算例 |
| 5 | 设计一个可靠的 Agent 系统 | 平台综合设计四步法 | 沙箱分级、MCP 网关（M×N→M+N）、人审队列与 fail-closed、链式衰减 0.95^20≈36%、预算熔断 |

考察维度权重：容量估算与显存公式（题 1/3/4，字节二面最爱默写）、可靠性工程量化（题 5）、成本治理算例（题 1/5）、2026 新考点 MCP/上下文压缩（题 5、题 1）。

---

### 题 1：设计一个大模型推理服务

#### 【考察意图】

验证四件事：(1) 能否把"模型"当"系统组件"来做标准容量换算（DAU→QPS→并发→卡数）；(2) 是否理解 LLM 推理与传统 RPC 的本质差异——prefill/decode 双阶段、KV Cache 是"有状态资源"、流式输出、瓶颈从算力转向显存带宽；(3) 多轮对话特有问题：session 状态管理、prefix cache、KV 跨请求复用；(4) 成本意识与预算熔断设计。LLM 服务"单个请求很轻、单张卡片很贵"，能否算清"一张卡养多少用户"是二面分水岭。

#### 【深度解析】——四步法

**Step 1 需求澄清（先问，再假设）**

面试官没给指标时，必须先要四类数字，没给就给假设：

- **流量**：10 万 DAU；假设人均日均 10 次 LLM 调用 → 100 万次/日；平均 QPS = 10⁶/86400 ≈ 11.6；典型互联网对话产品早晚高峰集中 20% 时段，峰值系数取 5 → **峰值 ~60 QPS**。
- **延迟**：P99 首 token（TTFT）< 3s（原题给定）；P50 端到端 < 10s、吞吐（TPOT，decode 组内每 token 延迟）< 50ms——这两个要主动问。
- **会话形态**：多轮对话，平均单次请求输入（系统提示+历史）~6K token、输出 ~200 token（假设）。
- **成本**：月 token 预算（见 Step 4 成本治理）。

**并发换算（核心算式）**：平均输出 200 token ÷ 单流 decode ~50 token/s（持续批处理下 P50 口径，假设）≈ 4s/请求；在途并发流 ≈ 60 QPS × 4s = **~240 并发流**。

**Step 2 顶层架构图（在原章节图上加两块关键组件）**

```
                    ┌──────────────┐
 用户 SSE 长连接 ──▶ │ Load Balancer │  (一致性哈希: conversation_id 粘性路由)
                    └──────┬───────┘
        ┌──────────────────┼─────────────────┐
        ▼                                    ▼
┌─── API Gateway ───┐              ┌── Session Store ──┐
│ 认证/配额/限流/审计 │              │ Redis: 历史消息+   │
└───────┬───────────┘              │ 摘要/最近20轮缓存   │
        ▼                          └────────┬──────────┘
┌── Scheduler (每引擎内置) ─────────────────▼──┐
│ 等待队列(优先级:付费>免费) → Prefix Cache 索引 │
│ → Continuous Batching 组批 → 抢占/换入换出     │
└───────┬──────────────────────────────────────┘
        ▼
┌── Inference Engine vLLM(单卡7B:TP=2) ×N ──┐   ┌── 模型路由 ──┐
│ PagedAttention · Prefix Caching(APC)      │◀──│ 简单意图→小模型│
│ GPU KV Pool(≈54GB) · 分块哈希复用         │   │ 复杂→旗舰     │
└───────────────────────────────────────────┘   └─────────────┘
        │ 流式吐 token                          ┌── 观测 ──────┐
        ▼                                       │ TTFT/TPOT/并发│
 返回用户；轮次落库 →(超窗)→ Context Compaction  │ /命中率/成本  │
```

**Step 3 关键模块深挖**

**(a) 双阶段 workload——一切调度设计的根源。** Prefill（吃整段 prompt，计算密集，FLOPs ≈ 2·Ψ·T，Ψ=参数量、T=输入 token 数，另有 attention 二次项 ≈ 4·L·d·T²）；Decode（一步一步吐 token，每步都要读完整份权重与该序列全部 KV——**访存密集**）。二者混跑会互相劣化（长 prefill 卡住别人 decode）→ 生产用 **chunked prefill** 切片注入，兼顾 TTFT 与 TPOT。

**(b) KV Cache 显存公式（必默写）**：

每 token KV 字节 = `2 × n_layers × n_kv_heads × head_dim × dtype_bytes`。以 7B（L=32, d_model=4096）fp16 为例：

- MHA（n_kv=32, d_h=128）：2×32×4096×2B = **512KB/token**
- GQA（n_kv=8）：2×32×1024×2B = **128KB/token**（省 4×，并发同增 4×）

单卡 A100-80GB：权重 14GB（bf16）+ 引擎开销 ~4GB，`gpu_memory_utilization=0.9` → KV 池 ≈ 70−18 ≈ 54GB → GQA 下 ≈ 54×10⁷⁰/128×10³ ≈ 42 万 token。平均在途 6K 上下文 → **约 70 并发流/卡**（若 max_len 8K 全满跑则 53，取余量后按 50-60 计）。

**(c) PagedAttention**：传统按 max_len 预分配连续 KV，实测碎片浪费 60–80%；PagedAttention 以 16-token 物理块分页 + 逻辑块表，浪费 <4%（vLLM 论文口径）；同显存下并发提升 → 系统吞吐相对 Orca/FasterTransformer 2–4×、相对 HF Transformers 最高 24×。

**(d) 多轮对话 × prefix cache（本题灵魂）**：

- **原理**：多轮的第 n+1 轮 prompt = 第 n 轮 prompt + 新增轮次，前缀完全一致。vLLM APC 把 KV 池按块哈希（前缀哈希匹配），命中块直接复用，不重算。
- **命中率推导**：设每次新增 Δ token、当前上下文 E[T]，则命中率 ≈ 1 − Δ/E[T]。取 Δ=1K、T 均值 6K → ~83%；叠加系统提示跨用户全局复用（1-3K 前缀全员命中）。
- **TTFT 量化**：4K 全量 prefill FLOPs = 2×7×10⁹×4096 ≈ 5.7×10¹³；A100 bf16 峰值 312 TFLOPS × MFU 45% ≈ 140 TFLOPS 有效 → **~0.41s**；命中 80%（只算 ~800 新 token）→ **~82ms**。TTFT 从 0.4s 降到 0.1s 以下。
- **两个必答边界**：① 命中复用**只省 prefill**，decode 每步仍要全量重读 KV（6K×128KB=768MB/步 máximo），长会话 decode 时间仍随轮次线性涨——这才引出上下文压缩；② 缓存 KV 仍占显存，若被 LRU 淘汰需重算（幂等，只是变慢）。

**(e) Session 管理**：网关无状态（JWT 鉴权），会话历史存 Redis+落库；LB 按 conversation_id 一致性哈希粘性路由到同一实例以提高 KV 显存命中率；实例故障/扩缩容后亲和失效 → 直接重新 prefill（幂等，降级为慢而非错）。超窗治理：滑动窗口 + 摘要压缩（context compaction，调研报告 §七·考点 6：上下文越长利用率越低 context rot，压缩同时降 KV 显存与 decode 重读量）。

**(f) Continuous Batching 与调度**：静态 batch 要等该批最长序列结束，GPU 空转；iteration-level 调度（Orca）每步重组 batch。调度器要实现：SLA 分层优先级队列、per-user token bucket 防单会话独占（max parallel seqs per user）、抢占策略（swap out KV vs 重计算；上下文短则重算，长则换出到 CPU）、**过载保护**：队列深度上限，超限 503 + Retry-After（宁可快速失败不可雪崩排队——排队等待计入 TTFT，P99 就没了）。

**(g) 低延迟补件**：投机解码，接受率 α、草稿 β 时的期望产出 E = (1−α^{β+1})/(1−α)（α=0.7, β=4 → ≈2.8 token/步）；W4A16 权重拿出来给 KV 腾地方；高频工具/模板响应走 deterministic 缓存。详见 `02-Inference/` 三份文件。

**Step 4 容量估算与瓶颈（算例）**

| 项 | 数值 | 公式 |
|---|---|---|
| 峰值 QPS | 60 | 10⁶ req/日 × 峰值系数 5 ÷ 有效秒数（假设） |
| 在途并发 | ~240 流 | 60 QPS × 4 s/请求 |
| 每卡并发上限 | ~70 流 | 54GB ÷ (6K×128KB=768MB) |
| 卡数（显存主导） | 240/70 ≈ 3.5 → **6–8 张 A100**（含故障余量） | max(并发式, 吞吐式) |
| 校验（吞吐式） | 需 12,000 tok/s ÷ ~3,000 tok/s/卡 ≈ 4 卡 | 60 QPS×200 tok；每卡=B×1/步长≈70×50 |
| TTFT 预算分配 P99 | 队列 1s + prefill 0.1s（命中后）+ 首 token ~50ms + 链路 ~0.2s < 3s ✓ | 分项记账 |

关键瓶颈迁移：① 长上下文冷启动 prefill（chunked prefill + 路由粘性）；② decode 显存带宽（并发与时延此消彼长，admin 用"并发×TPOT<500ms"做门禁）；③ 网关 SSE 长连接数（keepalive、连接池）；④ 会话热点 Redis。显存不是堆参数，是**并发的货币**。

**成本治理算例（假设月 500 亿 token，输入 350 亿/输出 150 亿；官方价引自调研报告表五）**

直接调用对比（单请求 6K 入/200 出，混合价）：
- 旗舰 API（Opus 4.5 $5/$25 每 M token）：混合 ≈ (6000×5+200×25)/6200 ≈ **$5.65/M**。
- 开源 API（DeepSeek-V3.2 $0.28/$0.42）：混合 ≈ **$0.28/M**；若输入 50% 命中缓存（命中价 $0.028，同表）→ ≈ **$0.16/M**。
- 自建 8×A100（云时价假设 $1.5/h/卡 ≈ $8.8K/月；月产出 6.2×10⁹ token）→ ≈ **$1.4/M**——注意利用率仅 ~17%（2385/4/3500 平均口径），闲时必须塞离线任务（embedding/评测/预跑）自建才划算。

降本四杠杆 waterfall（月 500 亿 token，目标 −50%）：

```python
# 可运行：成本 waterfall 估算（假设值均显式标注）
def cost_waterfall(routes=0.8, cache_hit=0.50, cache_ratio=0.10,
                   compact=0.50, out_cut=0.15, in_share=0.70):
    c1 = (1-routes) + routes/10          # 假设:小模型≈旗舰价1/10
    c2 = 1 - in_share*(cache_hit*(1-cache_ratio))  # 缓存命中角按1折(调研报告:DeepSeek命中$0.028/未命中$0.28)
    c3 = 1 - in_share*compact            # 假设:历史压缩-50%输入
    c4 = 1 - out_cut*(1-in_share)        # 假设:输出-15%
    mult = c1*c2*c3*c4
    return mult                          # 0.28*0.685*0.65*0.85 ≈ 0.106
print(f"降本 {1-cost_waterfall():.1%}")   # ≈ 89%（>50% 目标；复杂度 O(1)）
```

配套预算熔断三级：单请求 max_tokens/花费上限 → per-tenant 日预算 → 全站阈值触发降级路由（切小模型/排队）。每级改动过质量门禁回归（见 06 章），路由误判率回升可自动回切。

#### 【30秒答题框架】

"先要指标：DAU、每用户调用量、TTFT 门槛。10 万 DAU、日均 10 次调用、峰值 5 倍 → 60 QPS、并发 ~240 流。分层四段：LB 粘性路由 → 无状态网关限流 → 调度器持续批处理+优先级队列 → vLLM 引擎组。三张底牌：PagedAttention 把 KV 碎片从 60–80% 压到 4%；GQA 把每 token KV 从 512KB 降到 128KB；前缀缓存把多轮 TTFT 从 0.4s 砍到 0.1s。算显存：54GB 池 ÷ 768MB/流 ≈ 70 并发/卡 → 6–8 张 A100。decode 是带宽瓶颈且每步重读全量 KV，所以要配上下文压缩。成本上模型路由+输入缓存+压缩三杠杆能做到降 80%+，配预算熔断降级。"

#### 【追问与陷阱】

1. **"前缀缓存能救多轮的 decode 延迟吗？"** 不能。prefill 可跳过已算 token，但 decode 每步必须重读该序列全部 KV（O(T)，访存量随轮次涨）。正解是上下文压缩/摘要，见 05-Agent 记忆与 06 章评估。
2. **TTFT vs TPOT 混淆**：一字（每 token）延迟与首字延迟是两个 SLO，优化手段不同（特性：prefill 侧=缓存/切片；decode 侧=批处理/投机/量化）。
3. **10 万 DAU≠1 万 QPS**：DAU→QPS 必须给"人均调用量+峰值系数"两个假设再换算，直接给"万级 QPS"会被抓估算素养。
4. 只优化输入 token 忘了**输出贵 3–10 倍**且 decode 恰是带宽瓶颈；max_tokens/风格控制是免费降本。
5. 实例间 KV 池不共享 → 粘性路由的代价是负载倾斜；要不要上共享 KV 存储（Mooncake 类 KVCache 分离架构）要能说出权衡（一致性/复杂度 vs 命中率）。
6. 抢占选 swap 还是 recompute：上下文短→重算（省 H2D 传输）；长→swap；两者与 TTFT SLO 的取舍要讲。
7. 引用数字要给出处：vLLM 2–4×/<4% 是论文口径，别拍脑袋。
8. 水平扩展时假设"多买卡就行"——TP 混合要重算每卡 KV 容量（KV 按 TP 维数分摊、每卡并发式变化）。

#### 【知识串联】

机制底层在 `02-Inference/01-KVCache-And-Acceleration.md`（PagedAttention/Continuous Batching/投机解码）、`03-Quantization-And-Deployment.md`（W4A16/部署）、`02-Decoding-Strategies.md`（采样与延迟）；Comprehension 压缩与 prompt 工程对接 `01-Foundation/02-Tokenization.md`（token 计费口径）与 `05-Agent/01`（记忆系统、上下文工程）；成本治理与题 5 预算熔断共用一套三级阈值设计；字节原题复现见 `10-RealQuestions/01-ByteDance-Questions.md` 面经 8（AI Infra）、面经 10（推理优化）。

---

### 题 2：设计一个企业级 RAG 系统

#### 【考察意图】

验证：(1) 是否区分离线/在线两条管道的 SLO 与失效模式（离线要吞吐与一致性，在线要延迟与权限）；(2) 混合检索与重排的数学与量化延迟；(3) 企业场景独有项——ACL 权限、增量更新、可观测（召回质量/幻觉率/引用正确率三指标）；(4) 容量估算（语料→chunk→显存/内存）与瓶颈识别。答成"向量库+GPT"是明显挂点。

#### 【深度解析】——四步法

**Step 1 需求澄清**

先要五组假设（未答则自设）：语料 100 万文档（ mishmash 3000 token/篇 ≈ 30 亿 token）；在线检索 10 QPS、RAG 端到端 P99 < 3s；时效——新文档 5 分钟内可检索；质量门禁——召回质量 recall@10 ≥ 0.9（离线评测）、引用正确率、幻觉率抽检 < 1%；权限——按部门/角色文档隔离。每个都对应后面一个模块。

**Step 2 顶层架构图（在原文件双管道上细化）**

```
┌──────────────────── 离线索引管道（批处理 SLO:吞吐+分钟级新鲜度）────────────────────┐
│ 数据源(Confluence/GitHa飞书/OSS) → Connector+CDC(增量:文件指纹)                    │
│  → 解析(文本/表格OCR/代码) → 清洗(去重:MinHash-LSH;去噪)                          │
│  → 分块(语义块512tok/overlap64; 保留父子块) → 向量化(embedding服务,GPU批量)         │
│  → 双路入库: 向量库(HNSW) + 全文索引(BM25,ES) → 权限ACL随索引入库 → 版本/回滚       │
└──────────────────────────────────────────────────────────────────────────────────┘
┌──────────────────── 在线查询管道（SLO:P99<3s, 权限优先）─────────────────────────────┐
│ 查询 → 意图/改写(HyDE/多查询) → 规则:ACL预过滤                                     │
│  → 混合检索: 向量top50 + BM25 top50 → RRF 融合 → cross-encoder 重排 top10          │
│  → Prompt组装(引用块id) → LLM生成(流式) → 后处理(引用对齐/事实核查/拒答检查)        │
│  → 监控: recall代理(点击位置)/幻觉抽检(LLM-judge+人工)/时效/成本                    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**Step 3 关键模块深挖**

**(a) 离线索引。** 解析最脏：表格/扫描件走 OCR+版面模型；代码按结构切。去重用 MinHash-LSH（近线性 O(n)，超参 b/r 控制召回-精度）。分块：固定 512 token+64 重叠是基线，但标题/语义边界切块显著降噪；**父子块**——检索用小块（精准命中），送 LLM 用父块（上下文完整），是 recall 与 context 预算双赢的标准解。增量索引走 CDC（文件指纹/updated_at），向量库 upsert，删除先墓碑标记再物理清理；embedding 模型升级 = 新版本集合双写 + 灰度切流 + 旧版本 T+7 下线（版本化，可回滚）。

**(b) 在线混合检索。** 向量（语义泛化）+BM25（专名/精确串）互补；融合用 RRF：`score(d) = Σ_i 1/(k + rank_i(d))`，k=60（Cormack 2009 经典口径）——只看排名不看原始分，免去两路分数归一化的坑。可选 RRF 15 行可运行核心：

```python
def rrf(rank_list, k=60):                       # rank_list: [(doc_id, rank), ...] 每路
    scores = {}
    for doc_id, rank in rank_list:
        scores[doc_id] = scores.get(doc_id, 0) + 1.0/(k+rank)
    return sorted(scores.items(), key=lambda x:-x[1])   # O(Σ路数×候选数)
```

重排：cross-encoder 对 (query,doc) 全量前向，O(候选数×(q+d) 长度)——top50 候选 × 512 token 对，A100 上单请求 ~20–40ms（假设），是精度最贵的一跳，只热重排 top50 而不是全库。

**(c) 权限（企业独有）**：ACL 必须做**预过滤**（查询期把文档 id 白名单下推到检索层）；后过滤会拿掉命中后 top-k 变稀疏、召回雪崩。敏感库的 embedding 本身可能泄露内容（向量反推），doc 权限一并进索引元数据。检索进来的**网页/文档内容一律视为不可信输入**（prompt injection 防线之一，与 06 章三层纵深衔接）。

**(d) 评估与监控闭环**（本题主判点）。三层：
- 离线回归集：标注问答对 ≥500 条，指标 recall@k / MRR / nDCG@10；每次换 embedding/切块参数跑 diff，防"改进检索、毁掉另一类 query"。
- 在线代理指标：首答引用点击率、答案无助率（用户追问比例）、超时率。
- 质量抽检：LLM-as-judge 打 groundedness/citation accuracy，judge 必须定期与人工标注对齐（kappa 校准）；输出幻觉率分位数监控、告警阈值。评测基准选型参考调研报告 §五：能力分仅初筛，必须自家任务集复验。

**Step 4 容量估算与瓶颈（算例）**

- 100 万文档×3000 token = 30 亿 token ÷ 512 ≈ **600 万块**。
- 向量内存：1024 维 fp16 = 600 万×1024×2B ≈ **12.3GB**；HNSW 图（M=16，每点 32 边×4B ≈ 128B/点）≈ **0.8GB**；原文 ~6GB → **单机 32–64GB 可扛主集**，生产 3 副本。
- 离线首建：A100 上批量embedding ~5000 块/s（假设）→ 600 万块 ≈ **20 分钟**；CDC 增量分钟级满足 5 分钟时效。
- 在线延迟账：HNSW top50 百万级库 ~5–10ms + BM25 ~10ms + 重排 20–40ms + LLM（prefill 3K ≈ 100ms + decode 200 tok ≈ 4s 流式感受延迟）→ 检索链路本身 P99 <100ms，**大头在 LLM**，直接复用题 1 的推理服务容量结论（RAG 内部 LLM 峰值 ~5 QPS）。
- 瓶颈清单：长尾脏文档解析、重排吞吐（扩 batch 或蒸馏 reranker）、LLM 成本（父子块控上下文长度）、向量库单点（分片+副本）。

#### 【30秒答题框架】

"两条管道：离线索引与在线查询，SLO 完全不同要分开设计。离线：连接器增量采集→元数据解析清洗→MinHash 去重→512 语义块分 KB→embedding→向量库加 BM25 双路落库，版本化可回滚。在线：查询理解改写→ACL 预过滤→向量+BM25 混合→RRF 融合（k=60 免分数归一）→cross-encoder 重排 top50→生成→引用对齐。容量算例：100 万文档 30 亿 token≈600 万块，fp16 向量 12GB+HNSW 图 0.8GB，单机可扛 3 副本；检索链路 <100ms，延迟大头在 LLM。成败在监控：离线 recall@k 回归集+在线幻觉抽检+LLM-judge 人工校准，引用正确率是唯一客户买的指标。"

#### 【追问与陷阱】

1. **"Gemini 1M 上下文了还要 RAG？"** 算账：把 30 亿 token 全塞单次上下文不经济不合规；但小语料（<50K token）直接长上下文免检索组件（调研报告表五：Gemini 3 Pro 1M）。回答框架：语料规模×更新频率×权限×成本四维。
2. **RRF 为什么 k=60**：压制"第一名压倒一切"，让多路共识稳定贡献；换 k 只在分布极端倾斜时做。
3. **分块越大越好？** recall@ 提升但噪声淹没答案、上下文预算爆炸；父子块/目标句检索是正解。
4. **向量库删除**：墓碑+定期 rebuild；纯物理删除在 HNSW 里代价大且影响图拓扑。
5. **embedding 模型换版**：不能原地混跑——新旧向量空间不可比；双集合+灰度。
6. **幻觉治理只答"prompt 里写不要编"是 0 分**：工程手段=引用强约束（答案句子绑定 chunk id）+ 生成后事实核查 + 抽检闭环。
7. **ACL 后过滤陷阱**：top-k 先取后过滤，权限密集部门瞬间召回枯竭——必须预过滤。
8. 估算不给出假设（块大小/维度/文档平均长度）直接报"50GB"是硬伤。

#### 【知识串联】

检索细节与评测全套在 `04-RAG/01-RAG-Complete-Guide.md`；生成侧推迟到题 1 推理服务容量；文档内容注入防线与 `06-Safety-Evaluation/01-Safety-And-Evaluation.md`（三层纵深）；"检索成为 Agent 工具"即 Agentic RAG 演进（`05-Agent/01`、调研报告 §七·考点 8）；真题引用见 `10-RealQuestions/01-ByteDance-Questions.md` 面经 3（搜索）、面经 7（飞书智能助手）。

---

### 题 3：大模型分布式训练的并行策略

#### 【考察意图】

验证分布式训练全景：能否按约束（模型规模/卡数/拓扑）推导并行组合，而不是背五名词表；两个"系统味"公式（AllReduce 环通信 2(N−1)/N、流水线 bubble (p−1)/(m+p−1)）与 6ΨT 吞吐估算是否真会算；能否站在"训练平台"视角补齐数据管理→DPO 对齐→评估闭环的工程链路。默写不出通信量/bubble 的，通常没在真实集群训过。

#### 【深度解析】——四步法（平台视角）

**Step 1 需求澄清**：模型多大？（决定底线并行度）卡多少什么互联？（NVLink 600GB/s 单机 vs InfiniBand ~50GB/s 跨机，数量级差 10×）训练类型？（预训练/指令微调/DPO——决定有无优化器分片必要）序列多长？（决定 SP/CP）预算与工期？（决定 LoRA vs 全参）。

**Step 2 顶层图（并行组合决策树）**

```
单卡放得下? ──是──▶ DDP(/ZeRO-1,2)          # 最快最简单
   │否
   ├─ 层数多/断点可分 ─▶ +PP(1F1B)  →  跨机优先PP(通信少)
   ├─ 单层权重太大     ─▶ +TP       →  单机内TP(↓通信密集,须NVLink)
   ├─ 序列超长         ─▶ +SP/CP    →  序列/上下文维切
   └─ MoE              ─▶ +EP       →  专家并行
显存仍紧? ─▶ ZeRO-3/FSDP(+激活检查点+offload)  ;  推荐组合: DP(ZeRO-1/2)+PP+TP(3D并行)
```

**Step 3 关键模块深挖（五种并行+公式）**

| 并行 | 切法 | 通信原语与量 | 关键公式/数字 |
|---|---|---|---|
| 数据并行 DP | 全模型复制，批切分 | 环 AllReduce：每步 ~2·(N−1)/N·S | S=梯度字节（bf16 2Ψ） |
| 张量并行 TP | 权重矩阵行列切 | 每层 fwd×2 + bwd×2 次 AllReduce，每次 ~b·s·d·2B | 算例：b=1,s=4096,d=4096 → **128MB/次** → 必须 NVLink 600GB/s，跨机 50GB/s 直接卡死 |
| 流水线并行 PP | 按层分段 | 只传段边界激活 b·s·d | bubble 占比 = **(p−1)/(m+p−1)**（1F1B，m=micro-batch） |
| ZeRO/FSDP | 优化器/梯度/参数分片 | ZeRO-1/2 通信量=普通DP；ZeRO-3 多 fwd+bwd all-gather → **1.5×** | 见题 4 |
| 序列并行 SP/CP | 序列维切 attention/LN | Ring/AllGather 注意力块 | 长上下文专用，与 TP 正交互补 |

**吞吐估算（必会算例）**：训练每 token FLOPs ≈ **6·Ψ·T**（前向 2ΨT + 反向 4ΨT）。8×A100 做 7B：有效算力 = 8×312 TFLOPS×MFU 40% ≈ 998 TFLOPS → 998×10¹²/(6×7×10⁹) ≈ **2.4 万 token/s**。1B token 全参一轮 ≈ 11.6 小时——面试报这个数足以自证"算过账"。

**训练平台闭环（原章节之外的平台补充，调研报告 §五/§六）**：

```
数据: 接入→去敏→清洗→MinHash去重(近O(n))→配比采样→数据集版本化(可追溯)
训练: LoRA/QLoRA 起步(单卡即训)→全参(DPO)→checkpoint+断点续训
对齐: DPO(需 policy+ref 双模型; LoRA DPO 单卡可跑显存算例见下)
评估: 领域自动评测集+LLM-as-judge(需人工校准)→回归门禁→线上A/B
回灌: 线上badcase→入回归集→下一轮训练目标
```

**DPO 显存算例（7B）**：全参 DPO = policy(参数2Ψ+梯度2Ψ+优化器12Ψ=16Ψ) + ref 前向 2Ψ + 双序列(chosen/rejected)激活 → 一体卡放不进 80GB，走 ZeRO-2/3 才可行；**LoRA DPO**：基模 bf16 14GB + ref bf16 14GB + LoRA 参数与优化器 <0.5GB + 激活 → **~40GB，单 A100-48G 即训**——3 人小团队第一优先选型。LoRA r=16 全线性层可训参数 ≈ 2×r×Σ(in+out)×L ≈ 2×16×78080×32 ≈ **80M（约 1.1%）**，可独立验证。警惕主观评估陷阱：调研报告 §六——Copilot 官方称专项任务提速 55%，但 METR 2025 对照实验发现熟手开发者实测反而慢 ~19%；**自评满意度≠客观收益**，评估闭环必须要有客观验证信号（编译/测试/对比 signed-off）。

**Step 4 容量与瓶颈**：MFU 是第一 KPI（预训练 <45% 常态，要能说出提升抓手：重叠通信/计算、激活检查点密度、kernel 融合）；通信瓶颈：TP 卡在机内带宽（128MB/次×每层×每微步），PP 卡在 bubble（p=8,m=32 → 7/39≈18% 空转）与段间带宽，DP 卡在梯度同步（80GB 卡 2TB/s HBM 与 NVLink 赛跑）；掉点检查（checkpoint 恢复时间=x）与坏卡重算是平台必须项。

#### 【30秒答题框架】

"按约束推导而不是背表：单卡放得下就 DDP；放不下先上 ZeRO-1/2 省优化器和梯度；单层还大再 TP——但 TP 每层两次 AllReduce、单层单本 128MB 级流量，只能锁在 NVLink 机内；层数多跨机用 PP，bubble 是 (p−1)/(m+p−1)，用 micro-batch 摊薄；长序列加 SP，MoE 加 EP。吞吐用 6ΨT 算：8×A100 训 7B，MFU 40% ≈ 2.4 万 token/s。平台侧三件事不能少：数据版本化与去重、DPO 对齐（LoRA DPO 单卡 40GB 即可）、评估回归门禁——主观提速可能是负 19%，客观验证信号才是锚。"

#### 【追问与陷阱】

1. **bubble 公式把 p 与 m 搞反**是最高频翻车；要能推 1F1B 时序图。
2. **TP 上跨机**：不看通信量直接"3D 并行都开"会被追问"每次 AllReduce 多大"卡死。
3. ZeRO 与 DP 关系说不清：ZeRO 是 DP 之上的显存优化维度，不是第五种并行。
4. 6ΨT 里 forgetting 6 的来历（fwd 2 + bwd 4）说不清。
5. MoE 显存估算忘专家参数总量（总参≠激活参）。
6. 只背 LoRA 不算参数量："r=16 省多少"要给出 ~80M/7B（全线性层）的推导。
7. 数据并行批次与 TP 各自的 per-device batch 混淆（global_batch = dp×m×micro_bs）。
8. DPO 为什么需要 ref：KL 锚点防 reward hacking，忘提 ref 显存翻倍。

#### 【知识串联】

PEFT 全景与显存推导见 `03-FineTuning/01-PEFT.md`；DPO 对齐数学在 `03-FineTuning/02-Alignment.md`；优化器/梯度基础在 `01-Foundation/03-Loss-And-Optimization.md`；训练↔推理显存公式互为镜像（题 1 KV 公式）；评估闭环方法学在 `06-Safety-Evaluation/01`；调研报告 §七·考点 15（微调/RAG/工具三者选型顺序）是本题的选型总纲；真题见 `10-RealQuestions/01` 面经 5、8。

---

### 题 4：DeepSpeed ZeRO 三个阶段

#### 【考察意图】

验证显存问题能不能"公式级"回答：16Ψ 基线怎么来的、三阶段各切什么、原文件表里"4×/8×"到底是按 N 线性还是渐近上限（80% 的人背错语义）、ZeRO-3 通信代价为何 1.5×。这是 LLM Infra 岗的"九九乘法表"，必须脱口而出（对应 `10-RealQuestions/01` 面经 8 AI Infra 原题）。

#### 【深度解析】——四步法（机制版）

**Step 1 需求澄清（问题定义）**：混合精度训练单卡显存 = 参数(fp16 2Ψ) + 梯度(fp16 2Ψ) + 优化器状态(fp32 权重副本 4Ψ + Adam 一阶矩 4Ψ + 二阶矩 4Ψ = **12Ψ**) → **基线 16Ψ 字节**。ZeRO 的本质想法：三块状态里有大量**冗余**（每张卡都存全量），按需分片。

**Step 2 三阶段（原表扩展：加入精确每卡公式）**

| 阶段 | 切分内容 | 每卡显存 | 渐近节省(N→∞) | 通信量变化 |
|---|---|---|---|---|
| ZeRO-1 | 优化器状态 | 12Ψ/N + 4Ψ | 4× | 与普通 DP 相同 |
| ZeRO-2 | +梯度 | 14Ψ/N + 2Ψ | 8× | 相同 |
| ZeRO-3 | +参数 | **16Ψ/N** | N× | **1.5×**（fwd all-gather 2Ψ + bwd all-gather 2Ψ + grad reduce-scatter 2Ψ，对比标准 4Ψ） |

**关键纠偏（原表"4x/8x"的语义）**：4×/8× 是 N→∞ 的**渐近上限**，不是固定收益；真实节省是 (12Ψ/N+4Ψ)/16Ψ 这类连续值。ZeRO-2 不切参数的原因：前向需要的参数反正已在卡上（all-gather 之后人人都有一份），唯一能省的只有优化器与梯度冗余。

**Step 3 显存算例（8×A100-80GB，仅放"参数+梯度+优化器"，不含激活）**

| 模型 | 基线 16Ψ | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| 7B | 112GB ✗ | 38.5GB | 26.25GB | **14GB** |
| 13B | 208GB ✗ | 71.5GB | 48.75GB | **26GB** |
| 65B | 1040GB ✗✗ | 209GB ✗ | 316GB ✗ | **130GB（8 卡仍不够 → 16 卡 / offload / QLoRA）** |

叠加激活：激活 ≈ 每层 ~b·s·(数十个 d 级张量)×2B，开 Flash-Attention+激活检查点后 per-layer 代价再降一个量级，训练型显存余量要按"重放一段"（checkpoint+recompute）设计。65B 行正是 QLoRA（NF4 65B→35GB 级，LoRA 可训）存活的战场。

**Step 4 工程要点与瓶颈**：ZeRO-3 的 1.5× 通信发生在**每层每微步**的专业 all-gather 上，跨机 50GB/s 会被放大；对策：offload 优化器/参数到 CPU 内存（带宽假设 ~1–2×10² GB/s，慢但省）或 NVMe，justify 训练时间换显存；**ZeRO-3 与 TP 同用的代价**（Ref：PyTorch FSDP 也称"ZeRO-3 等价实现"）；ZeRO++（量化 all-gather/双倍切分 reduce）一提即可。

#### 【30秒答题框架】

"基线 16Ψ 字节：参数 2Ψ+梯度 2Ψ+优化器 12Ψ（fp32 副本+动量+方差各 4Ψ）。ZeRO-1 切优化器：12Ψ/N+4Ψ，渐近省 4×，通信不变；ZeRO-2 加切梯度：14Ψ/N+2Ψ，上限 8×；ZeRO-3 连参数也切：16Ψ/N，省 N×，但前向反向都要重新 all-gather，通信变 1.5×。算例：7B 八卡，ZeRO-2 26GB、ZeRO-3 14GB；65B 必须 16 卡或配 offload——这就是 QLoRA 起源的显存约束。注意 4×/8× 是 N 趋无穷的渐近上限，不是每卡都按这个省。"

#### 【追问与陷阱】

1. **把 4×/8× 当"分片越多线性越省"**：ZeRO-1 实际节省 = 16Ψ/(12Ψ/N+4Ψ)，依赖 N，须能现场推导。
2. **ZeRO-3 = FSDP**？等价表述要主动说（PyTorch FSDP 官方即声明与 ZeRO-3 同思路），说不出会被判"只在文档层面"。
3. **ZeRO-1/2 通信量与 DP 相同**的机制：梯度 reduce-scatter 本来就是分片后的——答"会多通信"是错的。
4. ZeRO-3 与 TP 的兼容成本：两层切分互相放大 all-gather，一般 ZeRO-3 配 PP/DP 而不是再叠 TP。
5. offload 不是免费：CPU 内存带宽（几十到几百 GB/s，远低于 HBM 2TB/s）→ 训练速度换显存，要能算 trade。
6. 激活值常被漏算：显存=bw×(16Ψ/N+激活)，只报状态显存是半答。
7. 优化器为什么必须 fp32：bf16 加法精度不足（损耗累积），答"习惯"不行——回到 `01-Foundation/03-Loss-And-Optimization.md`。

#### 【知识串联】

优化器与混合精度基础在 `01-Foundation/03-Loss-And-Optimization.md`；显存公式的推理侧孪生（KV per-token 公式）在题 1 与 `02-Inference/01`；QLoRA 极限压缩对应 `03-FineTuning/01-PEFT.md`；平台化落地（checkpoint/断点续训/评估闭环）在题 3；真题见 `10-RealQuestions/01-ByteDance-Questions.md` 面经 8（AI Infra）原题。

---

### 题 5：设计一个可靠的 Agent 系统

#### 【考察意图】

验证 2026 年最高频的新考点群：Agent 平台不是"给模型加个工具循环"，而是**可靠性与安全系统工程**——能否给链式衰减做量化并给出对策收益；能否把 MCP 当成"协议层集成设计"（M×N→M+N、三层角色、三原语）落成数据面/控制面；人审队列与沙箱的 fail-closed 设计；成本与预算熔断三级阈值。调研报告 §七点名：可靠性、MCP、预算上限是区分 2026 与传统八股的考点。

#### 【深度解析】——四步法

**Step 1 需求澄清（约束四件套）**：任务类型（工具调用/代码/浏览器操作——决定沙箱形态）；规模（假设 1 万任务/日，均值 ~20 步/任务）；可靠性（高危外部动作 100% 人审拦截；整体任务成功率目标——见 Step 3 数学）；成本（per-task token 预算假设 $0.5 封顶）。

**Step 2 顶层架构图（在原文件图上补齐人审/状态/观测三个原文件缺失的平面）**

```
用户请求 → API 网关 → 任务队列
   │
   ▼
┌──── Agent Runtime（有状态,LangGraph/checkpoint）────────────────────────┐
│ 意图路由 → [直接答 | 走 agent loop]                                      │
│ Loop: LLM(经模型网关: 模型路由/限流/预算) → 结构化 tool call             │
│   → MCP 网关(鉴权/限流/schema校验/审计) → 执行平面                      │
│ 执行平面: 沙箱池(microVM默认禁网) | code runner(容器/worktree) | API 调 │
│   → 结果裁剪(不全文回填) → 追加上下文 → 重规划                          │
│ 高危动作 → interrupt → 人审队列(分级) → 批准/拒绝/修改 → fail-closed    │
│ 每步 checkpoint 持久化(可断点恢复/时间旅行调试)                          │
└────────────────────────────────────────────────────────────────────────┘
   │ 全链路 trace
   ▼
观测: trace(LLM调用/工具调用/成本) → 回放/红队闭环 → 评测集回归(SWE-bench Verified体系+自有任务集)
```

**Step 3 关键模块深挖**

**(a) 意图路由与模型路由**：先小模型分类（简单寒暄/FAQ 直接答， 甚至用检索模板），复杂任务进 agent loop；loop 内 per-step 路由——工具选择/摘要用小模型，规划/复杂推理用旗舰（Factory Droid 的 per-step 路由口径，调研报告表二）。触发词：约 80% 流量是低复杂度，一档路由就省 7×+（算例见 (g)）。

**(b) 执行引擎与 checkpoint**：LangGraph 式图态机 + 每工具调用后落 checkpoint → 断点恢复、时间旅行调试、事后回放归因，互为表里（LangGraph checkpoint/interrupt 一等公民，调研报告表六）。最大步数/预算/超时三闸门是**工程必须项**（无界 loop = 停不下来的吞金兽）。

**(c) MCP 网关（控制面）**：MCP（2024-11-25 Anthropic 开源，调研报告 §一）= host/client/server 三层，统一后集成从 **M×N 降到 M+N**，提供 tools/resources/prompts 三原语。自建"MCP 网关"承载：工具注册与发现、per-tool OAuth 凭证保管与轮换、调用量限流、JSON Schema 校验、审计日志、**危险操作打标**（外部效应类路由到人审队列）。执行放 client/server 数据面，控制面集中——这是"网关"存在的意义。与 function calling 的关系：MCP 分发工具，模型层仍靠 function calling 结构化调用，互补不是替代（调研报告 §七·考点 5）。

**(d) 沙箱（执行平面）**：隔离强度阶梯 进程级（macOS Seatbelt/bubblewrap）→ 容器（gVisor 拦截系统调用，性能折损 ~2× 假设）→ microVM（Firecracker：每实例开销 <5MB、启动 ~125ms，AWS 官方口径）。默认策略（调研报告 §五）：**只读起步、文件系统白名单（worktree）、网络默认禁、凭证不落沙箱（经网关注入单工具最小权限）**；代码执行一律容器+禁网；浏览器操作给独立画像与登录态隔离。预热池维持 N 个热 microVM 吞掉 125ms 冷启动毛刺。

**(e) 人审队列（HITL）**：四档权限模型（参照 Claude Code 四档：只读/自动编辑/全自动+白名单/人审全批，调研报告 §七·考点 13）。原则三条：**interrupt-on-external-effect**（只拦对外副作用动作，如发邮件/转账/删除）、checkpoint 之后**批量**人审而非逐步（点效率）、**fail-closed**（超时未批 = 拒绝，绝不默认放行）。审批吞吐算例：审批人 ~15 单/小时（假设）×8h ≈ 120 单/人/日；若全局 1 万任务×20% 高危 = 2000 单/日需 17 人——**通过白名单+分级把高危占比压到 2%** → 200 单/日 → 2 人，HITL 才不打断价值（调研报告 §七·考点 18）。

**(f) 错误恢复与可靠性数学（必算）**：

- 链式衰减：单步成功率 s、n 步串行 → sⁿ。**0.95²⁰ ≈ 0.358 ≈ 36%**（调研报告 §七·考点 10 原口径）。
- 对策量化：
  - 重试一步（失效瞬态）：p_eff = 1−0.05² = 0.9975 → 0.9975²⁰ ≈ **95.1%**；
  - 校验信号（测试/编译/回放）把单步抬到 0.99 → 0.99²⁰ ≈ **81.8%**；
  - 组合并缩短链长（子任务分解+人审卡点）→ 逼近生产可用上限。
- 因此可靠性的第一杠杆不是"更强模型"而是**验证信号密度**（调研报告 §八：成熟度取决于验证信号密度）与步骤结构。

**(g) 成本治理（预算熔断）**：三级预算（单请求 → per-task → per-tenant 日预算 → 全站），红线触发降级路由（超大任务切小模型/并行子流程→拒绝）。算例：复杂任务 ~20 步 × (输入 6K+输出 0.2K) ≈ 12.4 万 token/任务；旗舰混合价 ~$0.7/任务（Opus 档，与调研报告 §六"单任务数美元"一致）→ 1 万任务/日 ≈ $7K/日；路由后 80% 走小模型（$1/10 假设）→ **~$1K/日**，7× 降幅。上下文压缩（12K 历史→摘要 3K）输入再砍 75%，工具输出裁剪是 loop 内 token 爆炸的第一止血点。

**Step 4 容量与瓶颈**：人审队列审批吞吐是隐藏瓶颈（见 (e) 算例）；沙箱池冷启动与峰值并发（预热+优先级抢占）；MCP 网关单点（双活+无状态）；观测面 token 计量必须到 step 粒度否则成本不可归因。最痛的失败级联：子代理并行任务一个失败触发连锁重试风暴 → 需要全站重试预算（end-to-end token/spend cap）。

**可运行参考实现（最小可靠执行引擎骨架，纯标准库）**：

```python
# agent_runtime.py — 最小可靠 Agent 执行引擎（重试/退避/预算/人审/checkpoint）
import json, time, random

class BudgetExceeded(Exception): pass
class ToolDenied(Exception): pass

class AgentRuntime:
    def __init__(self, tools, llm_call, max_steps=50, max_tokens=200_000):
        self.tools, self.llm = tools, llm_call          # tools: name -> fn(args)->str
        self.max_steps, self.max_tokens = max_steps, max_tokens

    def run(self, goal, state=None, approve=None):
        """approve: 危险工具审批回调(None=全自动); state: checkpoint dict 或 None"""
        state = state or {"steps": [], "tokens": 0}
        outcome = None
        while outcome is None:                          # O(max_steps × 工具成本)
            state["tokens"] += len(goal) // 4           # 粗粒度计token(生产用tokenizer)
            if state["tokens"] > self.max_tokens:
                raise BudgetExceeded(state["tokens"])   # 预算熔断(三级之一)
            if len(state["steps"]) >= self.max_steps:
                return {"status": "max_steps", "state": state}
            decision = self.llm(goal, state)            # {"tool","args"} 结构化输出
            if decision["tool"] == "finish":
                outcome = {"status": "done", "state": state}; break
            fn, name = self.tools[decision["tool"]], decision["tool"]
            if fn.risky and approve is not None:       # interrupt-on-external-effect
                if not approve(name, decision["args"]):
                    state["steps"].append({"tool": name, "result": "DENIED"}); continue
            ok, result, attempts = False, None, 0
            while not ok and attempts < 3:              # 重试: 指数退避+抖动
                try:
                    result = fn(decision["args"]); ok = True
                except Exception as e:
                    attempts += 1
                    if attempts == 3: result = f"FAILED: {e}"  # fail-closed
                    time.sleep(min(2**attempts * 0.1, 5) + random.random()*0.1)
            state["steps"].append({"tool": name, "result": str(result)[:2000]})  # 裁剪
            self.checkpoint(state)                      # 每步持久化→可断点恢复
        return outcome

    def checkpoint(self, state):
        with open("ckpt.json", "w") as f:               # 生产换 LangGraph/DB
            json.dump(state, f)

# 复杂度: 时间 O(步数×工具延迟×重试), 空间 O(步数×结果长度截断值); 崩溃恢复: 读取ckpt.json续跑
```

#### 【30秒答题框架】

"先分层：意图路由把 80% 简单流量挡在 agent 之外；真 agent 走有状态执行引擎（LangGraph 式 checkpoint+三闸门），工具面用 MCP 网关做控制面——鉴权/限流/审计/凭证托管，把 M×N 集成压成 M+N；执行平面分级沙箱，microVM 开销 <5MB、125ms 冷启动，默认禁网只读起步；高危外部副作用被 interrupt 进人审队列，四档权限+fail-closed 超时拒绝。可靠性讲数学：0.95²⁰≈36%，重试抬到 95%、校验信号抬到 82%+，所以第一杠杆是验证信号与 checkpoint。成本三级预算熔断：per-request/per-task/per-tenant，红线降级路由——路由把日成本从 $7K 压到 $1K（7×）。全程 trace + 自有评测集回归，SWE-bench 类基准只做初筛。"

#### 【追问与陷阱】

1. **多智能体 vs 单 Agent**判断题：并行探索/上下文隔离/异构分工才上多智能体；代价=成本倍增+失败级联+handoff 信息损耗（调研报告 §七·考点 9）。默认先"单 agent+子代理"。
2. **"加模型能力就能解决可靠性"**——错。0.95→0.99 的责任主要在 harness（工具描述/错误处理/验证回路），调研报告考点 19：模型是地板，harness 是天花板。
3. **fail-open 陷阱**：审批超时默认放行、重试无限——任何"默认成功"路径都是事故源。
4. **MCP≈function calling / MCPワー abstraction**两种极端都错：MCP 是分发生态层协议、function calling 是模型层机制（互补）；同时 MCP server 文件就是"检索内容"级别的不可信数据。
5. **工具输出全文回填上下文**：token 指数膨胀+context rot，必须截断/结构化、蒸馏；反映成本曲线的主要是回填策略。
6. **token 计费不埋点**：成本治理无从谈；trace 必须覆盖 LLM span+工具 span。
7. 主观评估陷阱：用户"感觉快"（Copilot 口径 55%）与实测（METR −19%）要分离，评测集必须带客观验证信号。
8. **沙箱只放代码执行**：浏览器/API 调用同样是攻击面（prompt injection→数据外发），最小权限按任务临时授出。

#### 【知识串联】

范式（ReAct/反射）与 function calling/MCP/记忆/多智能体原理在 `05-Agent/01-Agent-Complete-Guide.md`（本题为该章的系统级落位）；注入防御三层纵深与评测基准集群在 `06-Safety-Evaluation/01-Safety-And-Evaluation.md`；Agentic RAG 与检索工具化在题 2；模型路由与预算熔断与题 1 成本治理共用参数（waterfall 函数）；上下文工程（compaction/context rot）与 2026 新考点在 `07-HotTopics/01-Hot-Topics-2025-2026.md`；Agent 平台真题氛围参考 `10-RealQuestions/01-ByteDance-Questions.md` 面经 4（Agent 平台）；调研报告 §五"上线清单"是本题的 checklist 收口（可回滚开关/一键停机/日志不含明文凭证）。

---

## 本章高频追问 Top5

1. **多轮对话里 prefix cache 为什么救不了 decode 延迟，正解是什么？** —— prefill 可跳过命中块（TTFT 0.4s→0.1s 算例），但 decode 每步必须全量重读 KV（O(T) 访存）；正解是上下文压缩/摘要控制 T，而非只堆缓存。
2. **两个必默写数据公式？** —— 推理侧 KV/token = 2·L·n_kv·d_h·bytes（7B fp16：MHA 512KB、GQA 128KB）；训练侧混合精度基线 16Ψ 字节与 ZeRO 三阶段每卡公式（12Ψ/N+4Ψ → 14Ψ/N+2Ψ → 16Ψ/N）。
3. **Agent 可靠性 0.95²⁰≈36% 怎么救？** —— 重试（0.9975²⁰≈95%）、校验信号抬单步（0.99²⁰≈82%）、分解链长+checkpoint+人审 fail-closed；验证信号密度是第一杠杆。
4. **降本 50% 的杠杆组合？** —— 模型路由（×0.28）+ 输入缓存（×0.685，DeepSeek 命中价 1/10 口径）+ 上下文压缩（×0.65）+ 输出控制（×0.85）→ 叠乘 ≈0.11；伴随三级预算熔断与质量回归门禁。
5. **MCP 在 Agent 平台里到底管什么？** —— 工具分发与集成的协议层（host/client/server，tools/resources/prompts，M×N→M+N），function calling 仍是模型层调用机制；网关负责鉴权/限流/schema 校验/审计/危险操作路由人审。
