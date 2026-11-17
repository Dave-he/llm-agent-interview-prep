# 02-Inference 逐题详解（KV Cache 与加速 / 解码策略 / 量化与部署）

> 题源：`01-KVCache-And-Acceleration.md`（题 1–8）、`02-Decoding-Strategies.md`（题 9–13）、`03-Quantization-And-Deployment.md`（题 14–20），共 **20 题**，覆盖率 100%；编号沿用原文件出现顺序，表格/清单也按题处理（题 3、8、12、14、15、19、20 含原表）。原章节文件只读未改。
> 本轮未提供附加研究文档；一切口径以原文件公式与官方论文为准（FlashAttention arXiv:2205.14135、Orca OSDI'22、vLLM SOSP'23、Levy/Leviathan 投机解码 arXiv:2211.17192 与 Chen arXiv:2302.01318、GPTQ arXiv:2210.17323、AWQ arXiv:2306.00978、QLoRA arXiv:2305.14314、DeepSeek-V2/V3 arXiv:2405.04434/2412.19437），不生造数字；可推导的数字（显存、roofline、误差方差）全部给出推导链。
> 硬件基准（下文所有延迟算例共用）：A100-80G SXM——HBM 带宽 2039 GB/s、BF16 dense 312 TFLOPS、SRAM 约 20 MB。

## 本章考点分布

| 主题 | 题号 | 原文热度 | 面试定位（结合 10-RealQuestions 统计：KV Cache 6/11 次、Flash Attention 4/11 次） |
|---|---|---|---|
| KV Cache 原理 / 显存 / 压缩 | 1–3 | ⭐⭐⭐⭐⭐ | 一面必问；显存三算例（MHA/GQA/MLA）是区分点 |
| Flash Attention | 4 | ⭐⭐⭐⭐ | 二面深挖：tiling + online softmax 归纳推导、IO 复杂度 |
| PagedAttention / Continuous Batching | 5–6 | ⭐⭐⭐⭐ | serving 岗必考：碎片分析 + 迭代级调度 |
| 投机解码 | 7 | ⭐⭐⭐ | 必须会证“分布一致”（真题原句：投机解码为什么能保证输出分布一致） |
| 推理框架对比 | 8 | ⭐⭐ | 选型口试题 |
| 解码策略（贪心/Beam/温度/Top-K/Top-P/重复惩罚） | 9–13 | ⭐⭐⭐ | 热身题，常与幻觉/安全联动 |
| 量化基础 / PTQ vs QAT | 14–15 | ⭐⭐⭐ | 仿射量化公式 + Δ²/12 误差学 |
| GPTQ / AWQ / GGUF / INT4 补偿 | 16–19 | ⭐⭐⭐ | 真题原句：GPTQ 和 AWQ 的区别；INT4 量化后精度下降怎么办 |
| 部署架构 / TP vs PP / 端到端延迟 | 20 | ⭐⭐⭐⭐ | 系统设计题的微观底座 |

面试分布规律：一面问“是什么、多大”（题 1/2/9–12），二面三道硬门是 **Flash 推导、投机解码无损证明、量化数值方法**（题 4/7/16–17），serving 岗再加一组**碎片+调度+延迟拆解**（题 5/6/20）。

---

### 题 1：KV Cache 的原理 ⭐⭐⭐⭐⭐

**1.【考察意图】** 验证三件事：①是否理解自回归生成的**因果结构**（历史 token 的 K/V 与“当前步”无关，这是缓存合法性的唯一依据）；②能否分清 prefill 与 decode 两个阶段的计算形态差异；③能否自己推出“缓存到底省了哪一项、没省哪一项”——这一句决定后面显存、带宽、长上下文所有追问的成败。

**2.【深度解析】**
自回归分解 $p(x_{1:T})=\prod_{t} p(x_t|x_{<t})$。因果注意力下第 $t$ 步：

$$\text{Attn}(q_t)=\sum_{j\le t}\frac{e^{q_t\cdot k_j/\sqrt d}}{\sum_{j'\le t}e^{q_t\cdot k_{j'}/\sqrt d}}\,v_j,\qquad q_t=W_Qx_t,\ k_j=W_Kx_j,\ v_j=W_Vx_j$$

**合法性关键**：$k_j, v_j$ 只依赖 $x_j$ 本身，与谁在 query 它无关。所以生成第 $t$ 步时 $k_{1:t-1}, v_{1:t-1}$ 与上一步一模一样——缓存它们零误差。这就是因果模型独有的性质（双向编码器不存在）。

**省了什么、没省什么**（单层单头，历史长 $t$，乘加记 2 FLOPs）：
- 无 cache：每步重算历史 K/V $4td$（K、V 两个投影）+ 当前 qkv $6d$ + 注意力打分加权 $4td$；
- 有 cache：只算当前 qkv $6d$ + 注意力 $4td$；
- 结论：**消掉的是 $O(td)$ 的重复投影**；$4td$ 的注意力读旧 KV **仍然在**，而且是对显存里越攒越多的 KV 流式读取——这解释了 decode 随上下文变长变慢（memory-bound，题 20 定量），也解释 KV 显存成为独立瓶颈（题 2 数值）。

**两阶段形态**：
- **prefill**：prompt 全量一次并行前向，产出 KV cache + 首 token（算力密集，GEMM 大矩阵）；
- **decode**：每步 1 个 token，串行 $(\text{max\_new}-1)$ 次（带宽密集，权重+KV 反复读而复用率低）。

**3.【30秒答题框架】** “自回归每步条件在历史上；因果注意力里历史 token 的 K/V 只由它自己的隐状态决定，跨步不变，所以每层缓存 K/V，每步只算新 token 的 q/k/v 并追加进缓存。省掉的是重复的 K/V 投影，省不掉的是对历史 KV 的注意力读取——所以 KV 显存随上下文线性涨、decode 是带宽瓶颈。prefill 并行建缓存，decode 逐 token 读缓存，这是所有推理引擎优化的出发点。”

**4.【追问与陷阱】**
- *训练时能用吗？* 不需要：训练一次并行吃全序列（teacher forcing），K/V 只算一遍，没有“重复”可省；KV cache 是**推理态**概念。
- *Q 为什么不缓存？* 历史 token 的 Q 已经被消费完，不会再被读。
- *缓存的是 RoPE 前还是后？* 常规实现在投影后打分前加 RoPE，缓存的是**已旋转**的 K；MLA 是例外（内容/位置解耦，题 3）。
- 陷阱一：说 KV cache “加速了注意力计算”——注意力 FLOPs 没减，减的是投影重复。陷阱二：说 “prefill 也靠它加速”——prefill 本来就无重复投影，它的加速点在 GEMM 并行度（题 20）。

**5.【知识串联】** 01-Foundation/04 题 3/4（注意力逐元展开与 √d_k）、题 6/7（MHA→GQA→MLA 与本文件题 2 公式同一套）；08-Coding/02 题 6（手写 KV Cache Attention，本题的可执行版）；题 2（显存量化）、题 20（decode 带宽瓶颈推导）；10-RealQuestions 面经 1 一面第 4 题（KV Cache 原理+显存连问）。

---

### 题 2：KV Cache 的显存计算 ⭐⭐⭐⭐⭐

**1.【考察意图】** 真题原句“KV Cache 显存计算公式（MHA/GQA/MLA 分别算）”。考察：①公式每个因子能否**从张量形状推导**（不是背）；②三种架构的代入与数量级感；③能否用公式反推**容量规划**（一张卡塞多少并发）——serving/部署岗二面必落到的落点。

**2.【深度解析】**
公式（原文件）：$\text{KV} = 2 \times b \times n_{kv} \times L \times s \times d_{head} \times \text{sizeof(dtype)}$。
因子溯源：**2** = K 和 V 各一份；$b$、$s$、$n_{kv}$、$d_{head}$ = 缓存张量 `[B, n_kv, s, d_head]` ×2 份；乘 $L$ = 每层各存一份；bytes = FP16=2 / INT8=1。**Q 不在其中**。

**算例 1（MHA，LLaMA-2-7B）**：L=32，n_kv=32，d_head=128，s=4096，b=1，FP16。
逐层逐步：
- 每 token 每层 = 2 × 32 × 128 × 2B = **16,384 B = 16 KiB**；
- 每 token 全模型 = 16 KiB × 32 = **512 KiB**；
- ×4096 token = 2048 MiB = **2.0 GiB**（二进制口径；十进制 ≈ 2.15 GB——与原文件“2 GB”一致。面试报数说明口径，加分）。

**算例 2（GQA-8）**：LLaMA-3-70B：L=80，n_kv=8，d_head=128。
- 每 token 每层 = 2×8×128×2B = **4096 B**；全模型 = 320 KiB/token；s=4096 → **1.25 GiB/序列**。
- 反事实：若 70B 用 MHA（h=64）：2×64×128×2B = 32768 B/层/token → ×80 = 2.5 MiB/token → 4k = **10.0 GiB**。GQA-8 省 **8 倍**（= KV 头数之比 64/8）。8B 版同参数（L=32）→ 0.5 GiB，可作小模型锚点。

**算例 3（MLA，DeepSeek-V3）**：不缓存逐头 K/V，缓存低秩潜向量 $c_{KV}\in\mathbb R^{512}$ + 解耦 RoPE 键 $k_R\in\mathbb R^{64}$（题 3 给出为什么必须解耦）：
- 每 token 每层 = (512+64) 元素 = **1152 B**（FP16）；61 层 → **68.6 KiB/token**；s=4096 → **0.268 GiB**。
- 横向比（每 token 每层存储量）：7B MHA = 8192 元素 vs MLA 576 元素 → **7.0%**；GQA-8 = 2048 元素 → **28%**；相对 7B-MHA 基线节省 1−576/8192 = **92.97%**（DeepSeek-V2 论文口径“KV cache 减少 93.3%”即此量级）。原文件表格 "~6.7%" 是论文“降低 93.3%”的补数一致口径；面试时引用 **576 元素/层 + 93% 量级** 两个可复现数字最稳。

**容量规划示范（全部可推）**：A100-80G 跑 LLaMA-3-70B INT4：权重 ≈ 70e9 × 0.5B = 35 GB + 分组 scale（group=128 时每权重复用一个 FP16 scale ≈ 0.125 bpw，约 1 GB 级）→ 占 ~36–40 GB；剩 ~40 GB 给 KV；÷1.25 GiB/4k 序列 → **约 30 路并发**；假想 MHA 版只剩 40/10 = **4 路**——KV 头数从 64 到 8，直接把并发上限放大 8 倍，这就是 GQA 对吞吐的“隐藏贡献”。

**参考实现（可运行，实测输出如下）**：
```python
def kv_cache_gib(layers, n_kv, d_head, seq, batch=1, dtype_bytes=2, project_kv=True):
    """KV 显存 = 2 × batch × n_kv × L × s × d_head × bytes
    project_kv=True 表示 K、V 各存一份（MHA/GQA）；MLA 只存一个 576 维向量 → 传 False 并令 n_kv=1。
    复杂度 O(1)"""
    factor = 2 if project_kv else 1
    total = factor * batch * n_kv * layers * seq * d_head * dtype_bytes
    b_tok_layer = factor * n_kv * d_head * dtype_bytes
    return {"total_gib": round(total / 2**30, 3),
            "B/token/layer": b_tok_layer,
            "KB/token/all_layers": round(b_tok_layer * layers / 1024, 1)}

print(kv_cache_gib(32, 32, 128, 4096))                 # MHA-7B  → total 2.0, 512 KiB/token
print(kv_cache_gib(80,  8, 128, 4096))                 # GQA8-70B → 1.25, 320 KiB/token
print(kv_cache_gib(80, 64, 128, 4096))                 # 反事实 MHA-70B → 10.0
print(kv_cache_gib(61,  1, 576, 4096, project_kv=False))  # MLA-V3 → 0.268, 68.6 KiB/token
```
实测输出：`2.0 / 1.25 / 10.0 / 0.268`，与手算一致。

**3.【30秒答题框架】** “公式 = 2×batch×KV头数×层数×seq×每头维度×字节数，2 是 K+V。口径分三层：每 token 每层、每 token 全模型、每序列。三个锚点数：7B MHA@4k = 2 GiB（512 KiB/token）；70B GQA-8@4k = 1.25 GiB，假想 MHA 是 10 GiB，差 8 倍；DeepSeek MLA 只存 512+64=576 维，是 7B-MHA 的 7%，4k 才 0.27 GiB。最后落到容量：70B INT4 在 80G 上权重约 36–40 GB，剩 40 GB 除 1.25 得约 30 路并发。”

**4.【追问与陷阱】**
- *为什么有个 2？* K 与 V 各一份。答“QKV 三份”直接出局——Q 不缓存（题 1）。
- *INT8 KV 能省一半吗？* 显存能，但需要 per-head/per-channel scale 与配套 kernel；KV 数值范围随位置漂移，动态量化是难点（题 19 的 KV-Q 小节）。
- *batch 与它的关系？* 线性。所以 batching 的上限由 KV 预算决定，与题 5（分页）、题 6（调度）构成一体两面。
- 陷阱：GiB/GB 混报；把 d_model 当 n_kv·d_head（MHA 时二者相等 = 4096，GQA 时不等）；忘记 70B 的层数是 80 而非 32；把 MLA 的 576 说成“每头 576”——它是**每层每 token 共 576 个数**。

**5.【知识串联】** 题 1（缓存合法性）、题 3（每个压缩手段都砍公式里的一个因子）、题 5（同一预算下靠分页榨利用率）、题 20（roofline 延迟公式用的正是这份数）；01-Foundation/04 题 7（同款公式的训练侧视角）；10-RealQuestions 二面清单第 1 题（MHA/GQA/MLA 分别算）。

---

### 题 3：如何减少 KV Cache？⭐⭐⭐⭐

**1.【考察意图】** 原表题。考察：①是否系统掌握全部手段并知道**每一刀砍在公式哪个因子上**；②高频深挖是“MLA 为什么必须解耦 RoPE”与“GQA 怎么从 MHA 来”——能不能把结构设计与数值设计串成一条线。

**2.【深度解析】**
把题 2 公式当靶子：$M=2\cdot b\cdot n_{kv}\cdot L\cdot s\cdot\text{bytes}$，逐方法对应砍因子：

| 方法 | 砍的因子 | 节省比（原表口径/可推导值） | 代价与条件 |
|---|---|---|---|
| MQA | $n_{kv}\to 1$ | 1/h（7B→1/32≈3%） | 所有 Q 头共享 1 组 KV，质量掉明显（Shazeer 2019） |
| GQA | $n_{kv}\to g$ | g/h（8/32=25% 存量，省 8/9... 以“比 MHA 省 1−g/h”计：8 头组=省 87.5%） | 质量接近 MHA；可由 MHA 权重均值池化转换（uptraining，Ainslie 2023） |
| MLA | $n_{kv}d_{head}\to (d_c+d_R)=576$ | 每层每 token 576 vs MHA-7B 8192 → **7.0%**；vs GQA-8 → 28%（论文口径：降 93.3%） | 必须重训；RoPE 解耦设计（见下） |
| KV 量化 | bytes 2→1/0.5 | 50% / 75% | 需 scale 方案与 kernel；动态范围漂移是难点 |
| 滑窗注意力 | $s\to w$ | w/s | 远程召回受损；主流是“滑窗+少数 global 层”（Mistral 风格）与 attention-sink |
| 跨请求共享 | 复数序列共块 | 公共前缀只存一份 | 需 PagedAttention/RadixAttention 机制（题 5、题 8），不缩单序列 |

**MLA 与 RoPE 不相容（本题核心推导）**：MLA 的省法是让 $W_{UK}$（把 $c_{KV}$ 变回逐头 K 的矩阵）被“吸收”进 query 侧算子，从而缓存只留 $c_{KV}$。但 RoPE 是**位置相关的旋转算子**：$\langle R_m q,\ R_n k\rangle = q^\top R_{n-m} k$，$R$ 依赖相对位置 $(m,n)$，不能表示成任何与位置无关的矩阵乘积——吸收失效。DeepSeek 的解法把每头的 K 拆成两部分：**内容部分**（从 $c_{KV}$ 低秩恢复，无位置算子，可吸收）与**位置部分** $k_R\in\mathbb R^{64}$（单独过 RoPE、单独与 query 的 RoPE 分量做点积、单独缓存）。这就是缓存 = 512+64 = 576 维的由来：既保住位置建模，又保住吸收优化。

**滑窗的工程补丁**：纯滑窗丢远处信息。常见修复：局部滑窗 + 少量全注意力层（Mistral 交错设计、Gemini 的 local/global 混合）；StreamingLLM 用保留最早几个 sink token 维持流式稳定。

**KV 量化补充**：KV 对 outlier 比 weight 更敏感（动态范围随序列漂移），落地多是 per-head scale 的 INT8 或 FP8，INT4 KV 少见——与题 19 的权重量化是**两套**问题。

**3.【30秒答题框架】** “五条主线各砍公式一个因子：MQA 砍 KV 头到 1（省 1/h，质量伤）；GQA 分组共享（主流，质量近 MHA，能从 MHA 池化转换）；MLA 低秩压缩到 512+64 维（相对 MHA 约 7%，代价是 RoPE 必须解耦——位置旋转算子不可吸收）；KV 量化砍 dtype（减半或 3/4）；滑窗砍 seq（配 global 层或 sink 兜底）。优先级：先结构（GQA/MLA），再数值（量化），最后才牺牲长度。”

**4.【追问与陷阱】**
- *“g/h 还是 1−g/h？”* 原表写 g/h 是“剩余比例”；描述节省时要说“省 1−g/h=87.5%（8 组 32 头）”——比例方向说反是高频失误。
- *MLA 为什么 $k_R$ 只有 64 维够用？* RoPE 本质是低维旋转频率参数化，位置信息容量需求远小于内容通道；这是设计取舍（论文口径）。
- *GQA 从 MHA 怎么来？* KV 头均值池化分组 + 少量恢复步（uptraining）；MP 风格的搭桥题。
- 陷阱：说“GQA 减少头数”——**Q 头数不变**，减的是 KV 头数；把 MQA 和 GQA 的比例写反。
- *prefix caching 算不算压缩？* 不压缩单序列，但跨请求去重省总显存（接题 5/8），一道说出是亮点。

**5.【知识串联】** 题 2（公式与算例）、题 5（共享的落地机制）、07-HotTopics §1.1（MLA/DeepSeek 架构条目）、09-SystemDesign（容量规划）、10-RealQuestions 面经 3 第 6 题（KV Cache 优化方法有哪些）与二面清单第 1 题。

---

### 题 4：Flash Attention 的原理 ⭐⭐⭐⭐

**1.【考察意图】** 二面核心真题（原句：“Flash Attention 的 tiling 策略和 online softmax”）。考察三层：①能否定位真瓶颈——“慢在 HBM 往返而不在 FLOPs”（IO-bound 视角）；②能否现场完成 **online softmax 的归纳推导**（max/sum 修正项），这是“背过”与“会证”的分水岭；③说清它是**精确等价**而非近似（与 linear/sparse attention 划清界限）。

**2.【深度解析】**

**瓶颈定位。** 朴素实现：算 $S=QK^\top/\sqrt d$（n×n）写回 HBM → 读回做 softmax 写回 → 读回来乘 V。对 n² 级中间矩阵做了**三次 HBM 旅行**。GPU 内存层次（原文件口径，A100 量级）：SRAM ~20 MB / ~19 TB/s（片上）；HBM ~40 GB / ~1.5–2 TB/s（片外，差 ~10×）。计算量固定 $\Theta(n^2 d)$，慢全慢在搬运 $\Theta(n^2)$ 中间量。

**核心机制 = tiling + online softmax**。把 K/V 沿序列维切成块逐块喂入 SRAM；对每行维护三件套：running max $m$、running 指数和 $\ell$、running 未归一化输出 $\tilde O$：

$$m_{t+1}=\max(m_t,\ m_{\text{block}}),\qquad
\alpha=e^{m_t-m_{t+1}},\qquad
\ell_{t+1}=\alpha\,\ell_t+\textstyle\sum_{j\in blk}e^{s_{ij}-m_{t+1}},\qquad
\tilde O_{t+1}=\alpha\,\tilde O_t+\textstyle\sum_{j\in blk}e^{s_{ij}-m_{t+1}}v_j$$

**归纳证明（为什么与精确 softmax 逐位一致）**：
- **不变量（归纳假设）**：处理完前 $t$ 个块后 $\tilde O_t=\sum_{j\le J_t}e^{s_{ij}-m_t}v_j$，$\ell_t=\sum_{j\le J_t}e^{s_{ij}-m_t}$。
- **基例**：初始 $m_0=-\infty,\ \ell_0=0,\ \tilde O_0=0$；首个块后 $\tilde O_1=\sum_{j\in b_1}e^{s_j-m_1}v_j$，成立。（数值上 $\alpha_1=e^{-\infty}=0$，不会把垃圾读进来。）
- **归纳步**：块内最大值 $m_b$。则每个旧项换基准：$\ e^{s_j-m_t}=e^{s_j-m_{t+1}}\cdot e^{m_t-m_{t+1}}=e^{s_j-m_{t+1}}\cdot\alpha$，故 $\tilde O_t=\alpha\tilde O_t$ 的旧存量被整体折算，加上新块贡献即得 $\tilde O_{t+1}$ 满足不变量；$\ell$ 同理（把 $v_j$ 去掉）。$\alpha$ 就是题目说的 **max 修正项**：新基准变大时旧累计整体缩小。
- **收敛**：结束时 $m_T$ = 全局最大（max 单调不减且吸收所有块），故 $\tilde O_T=\sum_j e^{s_{ij}-m_T}v_j,\ \ell_T=\sum_j e^{s_{ij}-m_T}$，分子分母**同基底相除，平移项消掉**：

$$O_i=\tilde O_T/\ell_T=\frac{\sum_j e^{s_{ij}}v_j}{\sum_j e^{s_{ij}}}=\text{softmax}(QK^\top/\sqrt d)\,V\quad\text{精确成立，非近似}$$

数值副产品：所有指数项 $\le e^0=1$，天然防溢出——正是一般 softmax“减 max”的在线化（与 01-Foundation 题 4 的 fp16 溢出讨论同源）。

**逐块演化实录（8×4 玩具例，块大小 2，第 0 行实测输出）**：

| 处理到块 | $m$ | $\ell$ | $\tilde O[0]$ | 修正项 $\alpha$ |
|---|---|---|---|---|
| 块 1 | 0.449 | 1.284 | −0.269 | 0.0（基例：$e^{-\infty}=0$） |
| 块 2 | 0.449 | 2.911 | −1.163 | 1.0（max 未刷新，无修正） |
| 块 3 | 1.055 | 3.335 | 0.802 | 0.545（新 max 到来，旧累计整体折算） |
| 块 4 | 1.055 | 4.427 | 0.917 | 1.0 |

修正项 $\alpha$ 的两种典型值都能看到：max 未更新 → $\alpha=1$（无操作）；新 max 到来 → $\alpha<1$（旧累计按 $e^{m_t-m_{t+1}}$ 折算）。

**代价与收益**：
- 显存：$O(n^2)\to O(n)$（只留 O 与三件套；S、P 从不落 HBM）：
- HBM 流量：朴素 $\Theta(nd+n^2)$ → FA $\Theta(n^2 d^2/M)$（M=SRAM 容量；d=128 时 $d^2\ll M$）；
- 端到端实测 2–4× 加速（原文件口径）；反向不存 P，用重算换显存。
- 三括号口径：**计算 $\Theta(n^2d)$ 不变；显存 $\Theta(n)$；IO $\Theta(n^2d^2/M)$**——报错任何一个都会露馅。
- FA-2（Dao 2023）：减少非 matmul 指令、按输出块重排循环更利于并行；FA-3：用 Hopper 异步（TMA 等）较 FA-2 再提约 1.5–2×（论文口径）。

**参考实现（numpy，与精确 softmax 误差实测 4.4e−16）**：
```python
import numpy as np

def flash_attn(Q, K, V, scale=None, Bc=32):
    """tiling + online softmax 单遍实现。
    计算复杂度 O(n^2·d)；额外显存 O(n)（S/P 只在块内活动，从不落 HBM）"""
    n, d = Q.shape
    scale = 1 / np.sqrt(d) if scale is None else scale
    O = np.zeros((n, d))
    m = np.full(n, -np.inf)      # running max（基例：−∞）
    l = np.zeros(n)              # running exp-sum（平移后）
    for j0 in range(0, n, Bc):   # 真实 kernel 是 Q 行块 × K/V 列块的双层 tile
        S = scale * (Q @ K[j0:j0+Bc].T)          # (n, Bc)
        m_new = np.maximum(m, S.max(axis=1))
        alpha = np.exp(m - m_new)                # max 修正项：旧累计从基准 m 换到 m_new
        P = np.exp(S - m_new[:, None])
        O = alpha[:, None] * O + P @ V[j0:j0+Bc] # Õ = α·Õ + 块内加权和
        l = alpha * l + P.sum(axis=1)
        m = m_new
    return O / l[:, None]        # Õ / ℓ = 精确 softmax 输出

def ref_attn(Q, K, V, scale=None):
    d = Q.shape[1]; scale = 1 / np.sqrt(d) if scale is None else scale
    S = scale * (Q @ K.T)
    P = np.exp(S - S.max(axis=1, keepdims=True))
    return (P / P.sum(axis=1, keepdims=True)) @ V

rng = np.random.default_rng(0)
Q, K, V = rng.normal(size=(256,128)), rng.normal(size=(256,128)), rng.normal(size=(256,128))
print(np.abs(flash_attn(Q, K, V) - ref_attn(Q, K, V)).max())   # 实测 4.4e-16 → 数学等价
```

**3.【30秒答题框架】** “朴素 attention 慢在把 n×n 的 S 和 P 在 HBM 来回倒三趟，是 IO-bound 不是算不动。FA 把 K/V 切块喂 SRAM，块内 softmax 不完整，靠 online softmax 补：维护 running max 和 running 指数和，新块来了先把旧累计乘 e^{旧max−新max} 修正基准再加新块贡献。归纳证明：结束时分母分子同基底相除，平移项消掉，与精确 softmax 完全一致，还顺带防 exp 溢出。代价：计算 O(n²d) 不变，显存 O(n²)→O(n)，HBM 流量 O(n²d²/M)，实测 2–4×；反向靠重算不存 P。”

**4.【追问与陷阱】**
- *“为什么不用两遍法（先全局扫 max 再扫 sum）？”* 也精确，但 K/V 要从 HBM 读两遍；online 单遍是 IO 层面的关键选择。
- *decode 时还快吗？* decode 只有 1 个 query，没有 n×n 中间矩阵可省；FA 收益在 prefill/训练。decode 瓶颈在权重+KV 读取（题 20），靠 batching 与分页解决（题 5/6）。
- *什么时候不显著？* n 很小、d 很大塞不进 SRAM、或瓶颈已在带宽。
- 陷阱：说成“近似/稀疏注意力”（错：精确等价）；把 IO 复杂度当计算复杂度；把显存 O(n) 说成“注意力复杂度 O(n)”；说“需要先扫一遍全局 max”——那正是 online 要避免的两遍法。

**5.【知识串联】** 01-Foundation 题 4（√d_k 与数值稳定，减 max 是本题退化单块版）；08-Coding（手写 attention 后验证 online softmax 等价是经典白板题）；题 1/2（长上下文显存压力是 FA 的动机）；题 20（prefill 段 kernel 效率来源）；07-HotTopics（长上下文/新硬件 kernel 叙事）。

---

### 题 5：PagedAttention 解决什么问题？⭐⭐⭐⭐

**1.【考察意图】** vLLM 核心真题。考察：①能否精确定名**三类浪费**（预留浪费、内部碎片、外部碎片）并给量级；②能否描述块表/共享/写时复制并映射 OS 类比；③收益数字（论文口径）与残余碎片来源。

**2.【深度解析】**

**浪费解剖**。常规实现按 `max_seq_len` 预分配一段**连续** KV：
- **预留浪费**：实际长度 ≪ max_len，多买的空间整段闲置（vLLM 论文统计平均 **60–80%**）。自查推导例：max=2048、实际 200 → 仅 ~10% 被真正占用；
- **内部碎片**：即使按估计长度分配，增长不可预知，仍需留余量；
- **外部碎片**：反复分配/释放后连续大块越来越难找。

**方案 = 操作系统分页**：KV 切成固定块（vLLM 默认 **16 token/块**），逻辑序列经 **block table（页表）** 映射到非连续物理块：
- 按需追加物理块 → 内部碎片只剩**最后一块的未用槽位**（16 块时上界 15 个无效槽位），残余浪费 < **4%**（论文口径）；块不要求连续 → 外部碎片消失；
- **跨序列共享**：物理块带引用计数。并行采样（同 prompt n 个样本）、beam search 前缀、后来的 prefix caching 都只读共享，**分叉处 copy-on-write** 才复制块。

**收益与成本**：论文口径吞吐 2–4×（对比 FasterTransformer/Orca），显存利用率接近按需满载。成本：attention kernel 必须改为 **paged/gather 式**（按 block table 逐块取 K/V）——这是 vLLM 定制 CUDA kernel 的价值，不是“换个索引”的 Python 技巧。

**3.【30秒答题框架】** “问题是按最大长度预分配连续 KV，浪费 60–80%。PagedAttention 借 OS 分页：KV 切 16-token 页，块表做逻辑→物理映射，按需分配；内部碎片只剩尾块（<4%），外部碎片消失。块引用计数支持跨序列共享、分叉写时复制——beam/并行采样公共前缀只存一份。等效给 KV 装了个虚拟内存系统，配合连续批处理吞吐 2–4×。”

**4.【追问与陷阱】**
- *块大小怎么选？* 越小内部碎片越低但页表/gather 开销越高；16 是折中（工程默认）。
- *抢占时 KV 怎么办？* swap 到 CPU 或 recompute（丢块重 prefill）：长 prompt 适合 swap、短 prompt recompute 常更快——按 TTFT 预算选。
- 陷阱：“内部碎片 <4%” ≠ “利用率 100%”（还有元数据与策略保留）；prefix 共享不是零成本（分叉要 CoW）；逻辑块 ≠ 物理块。
- *与 continuous batching 的关系？* 分页提供了“新请求插入不要求连续大块”的前提，没有分页，迭代级调度在显存上寸步难行（题 6）。

**5.【知识串联】** 题 2（KV 显存是被管理对象）、题 6（调度依赖分页弹性）、题 8（vLLM 技术标签；SGLang RadixAttention 是同思想的树化）、09-SystemDesign（资源虚拟化是通用范式）；OS 三件套：page table / CoW / swap。

---

### 题 6：静态 Batching vs 连续 Batching ⭐⭐⭐⭐

**1.【考察意图】** Orca 论文的 iteration-level scheduling 是 serving 必考。考察：①一句话讲清调度粒度从“请求”降到“一次前向”；②现场量化静态批的 padding 浪费；③抢占、chunked prefill 等细节——“用过 vLLM” vs “读过 vLLM”的分水岭。

**2.【深度解析】**

**静态批浪费量化（现场可推）**：批内长度 [256, 512, 1024, 2048]，整批配到 max=2048：
有效算力占比 = (256+512+1024+2048)/(4×2048) = 3840/8192 = **46.9%**。且短请求结束后槽位冻结，等最长者完成才释放——延迟与吞吐双输。

**连续批（Orca OSDI'22：iteration-level scheduling；vLLM 工程化）**：调度粒度 = 一次 forward。每步自由进出：完成的 seq 交还槽位与 KV 预算，新请求（含 prefill 段）即时插入。
```
while True:                                  # vLLM 调度循环（简化）
    在 KV 块预算内选本轮可运行序列（prefill 段 + decode 混排）
    预算不足 → 抢占：swap 出 CPU 或丢弃重算（recompute）
    执行一次混合 forward；长 prompt 可切片（chunked prefill）
    KV 追加新块（分叉处 CoW）；新 token 流式下发
```
配合题 5 分页，插队不再需要连续显存——静态批的两头浪费（padding + 槽位等待）双双归零。原表口径吞吐 **2–3×**（vLLM 总体 2–4× 含分页贡献）。

**关键工程细节**：
- **chunked prefill**：长 prompt 切片混入 decode 步，避免一次大 prefill 卡死全场 decode（ITL 尖刺）；代价 TTFT 略升、尾延迟大降——TTFT vs ITL 的经典权衡。
- **抢占策略**：swap vs recompute 的选择取决于重算成本与显存水位。
- **指标口径**：TTFT / ITL(TPOT) / throughput。连续批的本质是允许调度器在三指标间主动取舍。

**3.【30秒答题框架】** “静态批等最长的人、pad 全员，批内有效算力可能只有 47%（256–2048 算例）；continuous batching 把调度粒度降到一次 forward，完成即退、新请求即进，配合分页没有显存级浪费，吞吐 2–3×。落地三件事：抢占（swap 或重算）、chunked prefill 防 decode 卡顿、TTFT/ITL/吞吐取舍——batch 不是越大越好：decode 是带宽瓶颈，KV 预算和 SLO 定上限。”

**4.【追问与陷阱】**
- *padding 不能解决吗？* padding 只省算力浪费，省不了“整批等待释放”的槽位冻结。
- *批越大越好？* decode memory-bound：权重读取被摊薄（好），KV 读取随 batch 线性涨（坏），最终卡 HBM 带宽与 KV 容量；过大批导致 ITL 抖动与抢占频繁——答案是 SLO 指导的准入控制。
- 陷阱：把请求级 dynamic batching（批内不变）当成 continuous batching——关键词 **iteration-level**。
- 追问：投机解码在批内的调度坑（一次 forward 验证多 token 改变步内形状，题 7）。

**5.【知识串联】** 题 5（没有分页就没有低摩擦插入）、题 7（批内验证形状）、题 20（decode 带宽公式解释为什么批好做大）、09-SystemDesign（“算力-显存-SLO 三角”的推理侧实例）。

---

### 题 7：投机解码的原理 ⭐⭐⭐

**1.【考察意图】** 真题原句（面经 3 第 9 题 + 二面清单第 3 题“为什么能保证输出分布一致”）。三层：①**拒绝采样**与无损性证明（多數人止步于流程复述）；②期望产出 $(1-\alpha^{\gamma+1})/(1-\alpha)$ 的推导与边界；③变体（MTP/Medusa/EAGLE）与负加速条件。

**2.【深度解析】**

**流程（Leviathan et al. 2023 / Chen et al. 2023，采样式）**：草稿模型 $q(\cdot|x)$ 自回归生成 γ 个候选；target $p(\cdot|x)$ **一次前向**给出这 γ 个位置的全部条件分布；从左到右验证；若 γ 个全接受，再从 $p$ 的最后一个位置额外采一个 **bonus token**。

**单点拒绝采样**：草稿提议 $x\sim q$，接受概率
$$A(x)=\min\!\big(1,\ p(x)/q(x)\big)$$
拒绝则从**规范化残差**重采样：$r(x)\ \propto\ (p(x)-q(x))_+$。

**无损性证明（输出分布严格 = p）**：
1. 接受路径贡献：$q(x)\cdot\min(1,\,p(x)/q(x))=\min(q(x),p(x))$；
2. 恒等式一（残差归一常数）：$\sum_x(p-q)_+=\sum_x(q-p)_+$——逐项比较：$(p-q)_+-$(q-p)_+$ = p−q，对 x 求和两侧同归零。因此在任意 x 处 $(p-q)_+(x)\,/\,\text{norm}$ 的 norm = $P(\text{reject})$；
3. 恒等式二（拒绝率）：$\sum_x\min(p,q)=\sum q-\sum(q-p)_+=1-\sum(p-q)_+$，故 $P(\text{reject})=1-\sum_x\min(p,q)=\sum_x(p-q)_+$；
4. 拒绝路径贡献：$P(\text{reject})\cdot r(x)=(p-q)_+(x)$；
5. 合并：$\min(p,q)+(p-q)_+\equiv p$（分情形验证：$q<p$：$q+0=p$ ✓；$q\ge p$：$p+(p-q)=p$ ✓）。∎

即**无论 p、q 差多少**，单 token 边际分布恒等于 target；再按 target 自回归链组合，全局分布也一致——“无损”是严格数学性质，不是工程近似。

**期望产出与加速**：设单 token 平均接受率 $\alpha=\mathbb E[\sum_x\min(p,q)]$（p、q 重叠面积）。分解：首个拒绝发生在第 k+1 位（概率 $\alpha^k(1-\alpha)$，产出 k+1 个）；γ 个全接受（概率 $\alpha^\gamma$，产出 γ+1 含 bonus）：
$$\mathbb E[\text{tokens/block}]=\sum_{k=0}^{\gamma}\alpha^k=\frac{1-\alpha^{\gamma+1}}{1-\alpha}\quad(\alpha=1\to\gamma+1;\ \alpha=0\to1)$$
时间账：target 一次 $t_t$ + draft γ 次（$t_d$）。设 $c=t_d/t_t\approx0.1$，$\text{Speedup}\approx E/(1+\gamma c)$。代入自算网格：

| α | E（token/块） | 加速比（γ=4, c=0.1） |
|---|---|---|
| 0.8 | 3.36 | **2.40×** |
| 0.7 | 2.77 | 1.98× |
| 0.4 | 1.65 | 1.18× |
| 0.2 | 1.25 | 0.89×（负优化） |

两个必要条件缺一不可：draft **够像**（α 高）且**够小**（c 低）；论文口径实测：T5 系 2–2.5×、Chinchilla 70B 2–3×。

**参考实现（拒绝采样 + 分布一致性实测验证）**：
```python
import numpy as np
p = np.array([0.35, 0.25, 0.25, 0.15])   # target p（大模型）
q = np.array([0.10, 0.40, 0.40, 0.10])   # draft  q（小模型）
rng = np.random.default_rng(0)

def spec_sample(N=500_000):
    x = rng.choice(len(q), size=N, p=q)                   # ① 提议 x~q
    acc = rng.random(N) < np.minimum(1.0, p[x] / q[x])    # ② min(1, p/q) 接受
    r = np.maximum(p - q, 0.0); r /= r.sum()              # ③ 残差 ∝ (p−q)₊
    x[~acc] = rng.choice(len(p), size=int((~acc).sum()), p=r)
    return x

emp = np.bincount(spec_sample(), minlength=4) / 500_000
print(np.minimum(p, q).sum())          # 实测 0.7  = α = Σmin(p,q)
print(np.maximum(p - q, 0).sum())      # 实测 0.3  = 1−α = Σ(p−q)₊
print(emp)                             # 实测 ≈ [0.350 0.249 0.250 0.150] = p ✓
```

**变体**：自草稿家族——Medusa（多解码头）、EAGLE（复用特征层）、DeepSeek-V3 的 MTP（原生多 token 训练头，见 07-HotTopics §1.4）；树状候选+树形验证；一次前向的 KV 管理带“回滚到接受位”的 undo（题 5）。

**3.【30秒答题框架】** “draft 自采 γ 个，target 一次前向并行验证。单点拒绝采样：x~q，以 min(1,p/q) 接受，拒绝从 (p−q)₊ 重采。无损两步：接受贡献 = min(p,q)，拒绝贡献 = (p−q)₊，两者恒等于 p——输出分布与只用 target 完全一致，bonus 也严格来自 p。期望产出 E=(1−α^{γ+1})/(1−α)：α=0.8、γ=4、draft 0.1 代价时 2.4×；α=0.4 → 1.2×；α=0.2 → 负优化。draft 要既像又小，缺一变负收益。变体：Medusa/EAGLE/MTP 自草稿、树形验证。”

**4.【追问与陷阱】**
- *为什么 bonus 不破坏无损？* bonus 本就从 target 采样，天然是 p 的合法样本。
- *γ 怎么选？* 由 α 与 c 定；γ 尾部概率按 α^γ 衰减，经验 3–6，可动态调。
- *KV 怎么处理？* 拒绝时回滚到最后接受位，target 补一次前向；工程在 PagedAttention 的块管理上做 undo（题 5）。
- 陷阱：把接受率说成“KL 距离”（α 是重叠度量 Σmin(p,q)，不是散度）；把 beam 验证当标准做法（beam 会改变边际分布）；漏掉残差是**规范化**的 (p−q)₊ 而非 (q−p)；忘算 draft 时间的加速比全是空头支票。
- *任务差异：* 代码/结构化文本 α 高；自由文本 α 低易负优化。

**5.【知识串联】** 题 11/12（温度改变 p 锐度 → 改变 α；采样流水线顺序）、题 4（一次前向的并行=短 prefill，直接吃 FA 红利）、题 20（TPOT 视角），07-HotTopics §1.4（MTP）、10-RealQuestions（面经 3 第 9 题 + 二面清单第 3 题）。

---

### 题 8：推理框架对比（vLLM / TensorRT-LLM / SGLang / DeepSpeed-Inference / Ollama）⭐⭐

**1.【考察意图】** 原表清单题。考察选型能力：不记 API，而是“什么场景选谁 + 技术标签卡是否准确”。加分项：每家**关键技术词**与适配边界。

**2.【深度解析】**

| 框架 | 核心技术 | 一句话标签 | 适用 / 不适用 |
|---|---|---|---|
| vLLM | PagedAttention（题 5）+ 迭代级调度（题 6）+ prefix caching | 开源 OG，显存效率+高吞吐默认选 | 通用在线/离线服务、研究原型 |
| TensorRT-LLM | kernel 融合 + 静态图 engine + in-flight batching + 量化生态 | 卡上性能天花板（NVIDIA 官方） | 单机/节点极致 SLO；换模型/shape 重 build engine |
| SGLang | RadixAttention（前缀基数树 KV 共享）+ 结构化输出前端 | agent/多轮/高前缀复用回本快 | 复杂控制流与多轮对话 |
| DeepSpeed-Inference | 张量并行 + 注入式量化 kernel | 与 DeepSpeed 训练栈一体 | 已有 DS 生态 |
| Ollama | GGUF + llama.cpp（题 18） | 本地/边缘一键跑 | 个人桌面；**不适合**生产高并发 |

不冲突的定位：RadixAttention 管**跨请求**前缀共享（树结构），PagedAttention 管**请求内** KV 虚拟化（块管理）——vLLM 的 prefix caching 与 SGLang radix tree 是同一问题的两种实现取径。

**3.【30秒答题框架】** “五张牌：vLLM 万金油开源默认；TRT-LLM 性能上限最高但 engine 静态编译、换模型重 build；SGLang 吃 agent/多轮的 RadixAttention 前缀红利；DeepSpeed 吃自家训练栈；Ollama 是个人本地工具不是服务。回答永远落回场景：并发、SLO、前缀复用率、卡型与生态。”

**4.【追问与陷阱】**
- *TRT-LLM 为什么快？* build 期算子融合/布局/量化，runtime 无图构造开销；代价是灵活性（每个新 shape 重编译）。
- *Ollama 为什么不能进生产？* 无迭代级调度、无多实例管理，吞吐与 SLO 都不是设计目标。
- 陷阱：把 RadixAttention 当 PagedAttention 的竞品（正交互补）；Ollama 报进生产推荐。

**5.【知识串联】** 全章（题 1–7 的技术点都映射到这些框架的模块）、题 18（GGUF/llama.cpp 生态）、题 20（部署形态）、09-SystemDesign（框架是组件不是答案本体）。

---

# 解码策略详解（题 9–13）

### 题 9：贪心解码 (Greedy Decoding) ⭐⭐⭐

**1.【考察意图】** 热身题里的陷阱题。考察：①局部 ≠ 全局最优（能否给数值反例）；②重复退化（neural text degeneration）；③与“确定性”的微妙关系（T→0 等价但工程上 batch 会破坏 bit 级一致）。

**2.【深度解析】**
$\hat x_t=\arg\max_x p(x|x_{<t})$。

**局部 vs 全局（构造反例）**：第一步 token0 概率 0.6 > token1 的 0.4；但 token0 之后最高只有 0.501，token1 之后最高 0.9。贪心路径 0.6×0.501 ≈ 0.30 < 替代路径 0.4×0.9 = 0.36——**每步都拿最大，总概率反而小**（这正是 题 10 beam toy 实验里 B=1 与 B=2 的差异）。贪心的补救就是 beam search。

**重复退化**：重复 n-gram 后上下文同样，argmax 永远同一个 → 陷入短循环（“很好很好很好…”；Holtzman 2020 的 degeneration 现象在贪心/低温下最重）。工程解：看到 题 13 的惩罚族，或 no_repeat_ngram 硬约束。

**复杂度**：每步 O(V) argmax + 一次前向；总 O(T) 次前向，无额外开销。

**确定性陷阱（工程易翻车）**：T=0 语义上是贪心，但服务端同一问题两次请求结果可能不同——batch 组合改变 GEMM 归约顺序，浮点非结合律导致 logit 平级别翻转。要做“严格可复现”，需固定 batch 形状/装卡方式，不是设 temperature=0 就完事。

**3.【30秒答题框架】** “每步取 argmax。三个要点：①局部最优不保全局最优（0.6×0.501 < 0.4×0.9 的两步反例）；②开放式文本会重复退化，所以产文任务几乎不用，靠 top-p/温度/惩罚；③T→0 与贪心等价，但严格确定性要固定 batch 形状——浮点非结合会翻转并 argmax。适合分类、抽取、严格 JSON 这类‘唯一正确答案’的任务。”

**4.【追问与陷阱】**
- *T=0 和 greedy 一样吗？* 语义等价（softmax 的 argmax 极限）；实现上服务端等价走 argmax 分支。
- *贪心能减少幻觉吗？* 不能——低温度只是更自信地编造（与 06-Safety 的幻觉机制讨论联动）。
- 陷阱：把贪心说成“全局最优的近似”（它连局部最优都不保证总概率最大）；把 beam search 说成“贪心的 top-k 每步拼接”（见 题 10 陷阱）。

**5.【知识串联】** 题 10（反例同源、beam 补救）、题 11（T→0 边界）、题 13（重复治理）、06-Safety（贪心与幻觉的关系）；01-Foundation（argmax 与采样管线的概率语义）。

---

### 题 10：Beam Search ⭐⭐⭐

**1.【考察意图】** ①写出长度惩罚并解释**为什么必须除 $|y|^\alpha$**（对数似然随长度单调变负 → 不归一会偏短）；②复杂度 O(T·B) 次前向与 KV 共享；③知道它何时该用（翻译/语音等封闭任务）、何时不该（开放式生成）。

**2.【深度解析】**
$\mathrm{score}(y)=\frac{1}{|y|^\alpha}\sum_{t=1}^{|y|}\log P(y_t|y_{<t})$。

**为什么必须长度归一**：log 概率之和是负数累加，序列越长亏得越多；不归一全局偏向**最短**句。GNMT 用平滑版 $(5+|y|)^\alpha/6^\alpha$，避免超短句畸形占优——能补这句是加分点。α=0 退回 sum（最偏短），α=1 即平均 logprob；翻译任务常用 0.6–0.7。

**何时用/不用**：
- 适用：分布尖锐、有客观好坏的封闭文本（翻译、语音识别、受限码）；
- 不适用：开放式生成——beam 输出高频安全词、多样性塌缩（Holtzman 2020：top-k/beam 的退化证据）；Stahlberg & Byrne 2019 甚至显示大 beam 的精确搜索常收敛到空串/短串（“好内容不在最高概率路径上”）。对话/写作任务用 **采样 + top-p** 取代 beam。

**与 KV cache 的配合**：B 条 beam 的公共前缀 KV 只存一份，分叉处 copy-on-write——vLLM 的块共享正好吃这口（题 5）。

**参考实现（含长度惩罚与全局最优对照，可运行）**：
```python
import numpy as np

def beam_search(next_lp_fn, start, T, B=2, alpha=0.0):
    """next_lp_fn(prefix)->下一 token 的 log 分布。复杂度 O(T·B) 次前向（可共享前缀 KV）"""
    beams = [(0.0, tuple(start))]
    for _ in range(T):
        cand = []
        for sc, seq in beams:
            lp = next_lp_fn(seq)                    # 每条 beam 一次前向
            for t in np.argsort(-lp)[:B]:           # 各扩 top-B → 共 B×B 候选
                cand.append((sc + float(lp[t]), seq + (int(t),)))
        cand.sort(key=lambda c: c[0] / (len(c[1]) ** alpha), reverse=True)
        beams = cand[:B]                            # 长度归一后全局排序取 top-B
    return max(beams, key=lambda c: c[0] / (len(c[1]) ** alpha))[1]

def oracle(prefix):                                 # 玩具两步问题
    return np.log([0.6, 0.4]) if len(prefix) == 1 else \
           (np.log([0.501, 0.499]) if prefix[-1] == 0 else np.log([0.9, 0.1]))

print(beam_search(oracle, [0], 2, B=1))   # 实测 (0, 0, 0)：0.6×0.501≈0.30，局部最优
print(beam_search(oracle, [0], 2, B=2))   # 实测 (0, 1, 0)：0.4×0.9 =0.36，全局最优
```
与 题 9 的反例同源：B=1 就是贪心（拿到次优解），B=2 靠第二条路径找回全局最优。

**3.【30秒答题框架】** “B 条路径并行扩展，每步把全部 B×B 候选按累计 logprob 排序留 top-B；核心细节是长度惩罚——log 和随长度单调变负，不归一会偏短句，GNMT 用 (5+L)^α 平滑、MT 常用 α≈0.6–0.7。复杂度 O(T·B) 次前向，前缀 KV 可共享。翻译/语音好用；开放式生成因多样性塌缩被采样+top-p 取代（Holtzman 2020、Stahlberg 2019 的证据）。”

**4.【追问与陷阱】**
- *为什么开放式任务不用？* mode-mismatch：beam 只在分布的尖峰处取样，而好文本常在锋面之外；多样性与事实性都不由高概率保证。
- *B 越大越好？* BLEU 收益迅速边际化，成本线性；开放任务反而恶化。
- 陷阱：与 top-k 采样混淆（beam 是确定性多路径**序列级**搜索；top-k 是单步随机截断）；把长度惩罚说成“鼓励长句”（它是**抵消短句偏置**）；epilogue：beam 结束条件（EOS 聚合后的打分比较）也常被追问。

**5.【知识串联】** 题 9（反例同源）、题 12（采样侧替代方案）、08-Coding（手写 beam 是常考白板题）、10-RealQuestions（翻译/语音岗常考）。

---

### 题 11：Temperature Sampling ⭐⭐⭐

**1.【考察意图】** 考数学底（为什么除以 T 会锐化/压平，作用在什么域）与调参直觉（用在哪、错在哪）；能区分这里的采样温度与校准/蒸馏温标的“同名不同义”。

**2.【深度解析】**
$$P(x_i)=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

**本质一句话**：温度改写**logit 差的刻度**。任意两 token 的概率比固定为
$$\frac{P(x_i)}{P(x_j)}=e^{(z_i-z_j)/T}$$
T↓ → 同样的 logit 差被指数放大，分布尖峰化（更确定）；T↑ → 比值趋 1，分布平坦化（更随机）；T→0 是贪心极限。**熵随 T 单调变化**。

**实测（令 z=[3.0, 2.0, 1.0, 0.2]）**：
- T=0.3 → [0.964, 0.034, 0.001, 0.000]（≈贪心）
- T=1.0 → [0.639, 0.235, 0.087, 0.039]（原分布）
- T=2.0 → [0.450, 0.273, 0.166, 0.111]（先摊平）

```python
import numpy as np
def temperature_sample(logits, T=0.7, rng=None):
    """O(V)；数值稳定写法：先除 T，再统一减 max 的 softmax"""
    z = np.asarray(logits, float) / max(T, 1e-6)
    z = z - z.max()
    p = np.exp(z); p /= p.sum()
    return (rng or np.random.default_rng()).choice(len(p), p=p)
```

**组合顺序（易考）**：惩罚 → 温度 → top-k → top-p → 采样。温度作用于 logits（截断之前）；顺序颠倒会改变 top-p 候选集——“先截断再降温”是另一套分布。

**调参直觉**：代码/抽取/分类 0.1–0.3；通用对话 0.7–1.0；创意 >1（多样性换一致性）；过高（>1.5–2）长尾噪声头（乱码/幻觉 token 概率起爆——接 06-Safety）。

**同名注意**：校准用的 temperature scaling（压过自信 logits，T 常 >1）、蒸馏温标（蒸出暗知识）与采样温度机制同为"logits/T"，**目的完全不同**——别串。

**3.【30秒答题框架】** “温度是 logits 除以 T 再 softmax；本质改写 logit 差的刻度（概率比 = e^{Δz/T}）：T<1 锐化、T>1 压平、T→0 贪心。实测锚点：同一组 logits，T=0.3 时 top1 已 0.96，T=2 时摊到 0.45。落地：代码/抽取 0.2 级，对话 0.7+，创意可 >1 但配 top-p 防长尾。流水线上顺序是先温度后截断，跟校准/蒸馏的同名温标不是一回事。”

**4.【追问与陷阱】**
- *温度影响 top-k 吗？* top-k 的**成员**不受影响（单调保序），但 **top-p 的集合大小**受影响（摊平 → 累积概率更晚达标 → 集合更大）。这是组合深度题的判据。
- *T=0 怎么实现？* 特判走 argmax；数值上除 1e-6 不可靠（会溢出）。
- 陷阱：“温度放大 logits”——它只**缩放**不整体上加；说“温度和 top-k 二选一”（可以叠加，top-k 兜底截断）。
- *高频追问：为什么训练完要重新调 T？* 训练目标不管采样语义；T 是部署面参数，与训练分布脱钩。

**5.【知识串联】** 题 12（组合参数）、题 15（校准温标对齐讨论）、03-FineTuning（蒸馏 T）、06-Safety（高 T 与毒性/幻觉长尾）；题 4（softmax 减 max 的数值稳定实现同样应用于此处）。

---

### 题 12：Top-K vs Top-P 的区别？⭐⭐⭐

**1.【考察意图】** 原表题。核心判据是“**候选集是否随分布自适应**”：能不能在两个极端（尖锐/平坦）下说清两者行为差异、现代组合管线（惩罚→T→K→P→min-p）、以及实现细节（argpartition O(V) vs sort O(V log V)）。

**2.【深度解析】**
- **Top-K**：固定 K 个候选（与分布无关）。尖锐分布引入长尾噪声（K=50 但只有 2 个 token 概率 >0.05）；平坦分布截掉合理长尾。
- **Top-P (nucleus)**：候选集 = 使累积概率达到 p 的**最小**集合（$\sum_{x\in V_p} p(x) \ge p$）。候选数**自适应**：尖锐时自动收窄到 1 个，平坦时自动扩大到上千——按“概率质量”截尾，所以两个极端都稳。

**进阶 min-p**：候选条件 $p_i \ge p_{\min}\cdot p_{\max}$——相对阈值锚定当前 top-1；在高温（top-p 集合爆大）时再兜一层底，是最近服务端常见的第三截断（同类还有 η 截断；有 min-p 语义即可讲清，不必报库）。

**现代组合管线（HF/服务端口径）**：`repetition penalty → temperature → top_k → top_p (→ min_p) → sample`。每层作用维度不同：惩罚与 T 改 logits 内容与刻度，K 定**数量**、P 定**质量**，min-p 定**相对幅度**。

**参考实现（一体化采样器，覆盖 题 11/12/13 的 pipeline）**：
```python
import numpy as np
rng = np.random.default_rng(0)

def sample(logits, temperature=1.0, top_k=0, top_p=1.0, min_p=0.0,
           repetition_penalty=1.0, generated=None):
    """HF 顺序：repetition penalty → temperature → top_k → min_p → top_p → 采样。
    复杂度：top-k 用 argpartition O(V)；top-p 需 sort O(V log V)"""
    logits = np.asarray(logits, float).copy()
    if repetition_penalty > 1.0 and generated:                       # ① 乘性惩罚（logit 域）
        idx = np.unique(generated)
        logits[idx] = np.where(logits[idx] > 0, logits[idx] / repetition_penalty,
                                               logits[idx] * repetition_penalty)
    if temperature != 1.0:                                           # ② 温度（截断之前）
        logits = logits / max(temperature, 1e-6)
    k = int(top_k) if top_k and top_k > 0 else logits.size
    keep = np.argpartition(-logits, k - 1)[:k]                       # ③ top-k，O(V)
    z = logits[keep]
    if min_p > 0:                                                    # ④ min-p：p_i ≥ min_p·p_max
        z = z[np.exp(z - z.max()) >= min_p]
    p = np.exp(z - z.max()); p /= p.sum()
    order = np.argsort(-p)
    cut = int(np.searchsorted(np.cumsum(p[order]), top_p)) + 1       # ⑤ ≥p 的最小前缀
    idx = order[:cut]; p = p[idx] / p[idx].sum()
    return keep[idx][rng.choice(len(idx), p=p)]

z = np.array([3.0, 2.0, 1.0, 0.2, -1.0])
print(sample(z, temperature=0.7, top_k=3, top_p=0.85, min_p=0.1,
             repetition_penalty=1.3, generated=[0, 2]))    # 实测只会在 top-k∩min-p 集合内采样
```

**3.【30秒答题框架】** “top-k 截**数量**、top-p 截**质量**。差异在候选集是否自适应：分布尖时 K 会放进一排 1e-3 的噪声 token，分布平时 K 又截掉合理长尾；nucleus 两个极端都稳，所以生产默认温度≈0.7 + top_p≈0.9。进阶：min-p 用相对阈值 p_i≥p_min·p_max 再兜一层。要能报实现：惩罚与温度动 logits，top-k 用 argpartition、top-p 用 sort 取累积 ≥p 的最小前缀（边界是 ≥ 不是 >）。”

**4.【追问与陷阱】**
- *顺序为什么不能换？* top-p 的集合必须在**温度调整后**的分布上确定；调换则 temperature 对已剔除概率的截断整套失效。
- *要不要 K 与 P 同时开？* 可以：K 是硬上限、P 是软质量，很多服务把 K 设 50–100 兜底。
- 陷阱：把 top-p 写成"cumsum > p"（是**≥ p 的最小集合**）；k=0 语义（不启用/全量表）在一库一变化，混库提参数时会踩坑。
- *跟幻觉的关系：* 低 top-p 降低长尾噪声——但这只压“胡说噪声”，不治“自信编造”（06-Safety）。

**5.【知识串联】** 题 9/10（同一解码家族）、题 11（顺序与温度联动）、题 13（同一条流水线的 logit 前置层）、06-Safety（截断与毒性）、08-Coding（采样器实现题）。

---

### 题 13：重复惩罚 ⭐⭐⭐

**1.【考察意图】** 考工程细节：三种惩罚（repetition / frequency / presence）的**作用域（logit 域）**与**量纲差异**；分辨 HF 的**符号分裂**实现（正 logit 除 α、负 logit 乘 α）为什么是对的——这是最能拉开候选人差距的口试题。

**2.【深度解析】**
原表公式是概率域 $p'(x)=p(x)/\alpha$（对已生成 token）；**落地实现在 logit 域**（HF 口径）：
$$z(x)\leftarrow z(x)/\alpha\ \ (z>0),\qquad z(x)\leftarrow z(x)\cdot\alpha\ \ (z<0)$$
**为什么必须分裂符号**：概率域除法方向永远唯一；但 logit 的符号“两次估义”——z>0 表示模型已经支持这个 token（p>0.5），除以 α>1 变小 → 概率下降 ✓；z<0 是“不支持”，若也除以 α（−2 → −1.54）**概率反而回升**——惩罚变成了奖励；乘以 α（−2 → −2.6）才继续压低。所以 HF 按符号分裂方向。硬背“除 α”在面试里是经典暴露点。

- **Frequency penalty**（OpenAI 风格，线性）：$z(x)\leftarrow z(x)-c\cdot \mathrm{count}(x)$，出现越多罚越重；抑制复读但保持模板顺畅。
- **Presence penalty**（OpenAI 风格，布尔）：出现过就减一个固定值；泛用防“再次提同一个话题”。
- **no_repeat_ngram**：硬约束——拥有 n 个已生成 n-gram 的后继 token，直接把 logit 拉到 −∞。最干脆，适合事实问答的短文。
- **参数量级**：repetition_penalty ≈ 1.1–1.3；frequency/presence ≈ 0.1–2（OpenAI API 范围）。**下限**：全部叠加过度惩罚会把 token 卡死在 EOS/换行走不出。
- **陷阱**：对代码 / 数学 / 命名等**合法重复**（列名、常量、三反引号围栏）是误伤 — 好工程纪律是只对重复性敏感的文本类使用，或用停止条件 / 结构化解码替代。

**复杂度**：每步 O(|已生成 token 池|)，web-scale 无忧。

```python
import numpy as np
def repetition_penalty(logits, generated, a=1.2):
    """logit>0 ÷ α；logit<0 × α（HF 口径）。O(|generated|)"""
    idx = np.unique(generated)
    logits[idx] = np.where(logits[idx] > 0, logits[idx]/a, logits[idx]*a)
    return logits

def frequency_penalty(logits, generated, c=1.0):
    idx, cnt = np.unique(generated, return_counts=True)
    logits[idx] -= c * cnt                # 线性按次数
    return logits

def presence_penalty(logits, generated, c=1.0):
    logits[np.unique(generated)] -= c     # 出现即固定值
    return logits
```

**3.【30秒答题框架】** “三兄弟：repetition 是**乘性**——注意在 logit 域要按符号分裂（正除 α、负乘 α，都朝‘降概率’的方向；不分符号负 logit 会越罚越热）；frequency 线性按出现次数罚；presence 出现一次罚个固定值。都在温度与截断**之前**、logit 域。参数量级：α 1.1–1.3、f/p 0.1–2。工程提醒：这是局部启发式，会误伤代码/数学的合法重复——配上硬 ban（no_repeat_ngram）、停止条件或结构化解码一起用。”

**4.【追问与陷阱】**
- *乘 α vs 除 α？* 见符号分裂推导——这题是区分度最高的追问。
- *penalty 在哪个域？* logit 域，softmax 前；在温度与 top-k/p 之前。
- *frequency 与 presence 区别？* 次数线性 vs 出现即罚（一个是“密度惩罚”，一个是“话题惩罚”）。
- 陷阱：f 和 p 的量纲混用；把“重复惩罚”当“解决 hallucination”（它只管字面重复）。

**5.【知识串联】** 题 9（贪心的重复退化是需求来源）、题 12（同条解码流水线，且其参考实现包含 repetition penalty）、06-Safety（惩罚 ≠ 事实性修复）、03-FineTuning（RLHF 侧对重复的系统性处理）。

---

# 模型量化与部署详解（题 14–20）

### 题 14：什么是模型量化？⭐⭐⭐

**1.【考察意图】** 地基题。考察：①仿射量化的**完整公式**（scale / zero-point / clip）与对称/非对称、粒度三级（per-tensor / per-channel / per-group）；②量化误差数学（舍入噪声方差 $\Delta^2/12$，为什么“bit 降一半、误差翻四倍”）；③收益账的三条（显存/带宽/计算——第三条 W4A16 并不赚）。

**2.【深度解析】**
**仿射映射**：$x_q=\mathrm{clip}(\mathrm{round}(x/s)+z,\ 0,\ 2^b-1)$，反量化 $\hat x=s(x_q-z)$。
- **对称量化**（权重量化默认）：$z=0$，$s=\max|x|/(2^{b-1}-1)$，网格 $[-(2^{b-1}),\,2^{b-1}-1]$（4bit: [-8,7]，**不是 [-8,8]**，边界常查）；
- **非对称**：zero-point $z\in\mathbb Z$ 对齐偏斜分布（如 ReLU 后全正激活）；
- **粒度三级**：per-tensor（一个 s，快而糙）→ per-channel（每输出通道一个 s）→ per-group（每 g=128 个权重一个 s，4-bit 标配）。

**误差数学（4-bit 一切的根）**：均匀舍入噪声方差 $=\Delta^2/12$；Δ 由组内极值定 → **outlier 决定一切**。正弦满幅的 SQNR≈6.02b+1.76 dB，每降 1 bit −6 dB——INT8→INT4 掉 36 dB 量级，这就是“4-bit 必须配补偿”的理论原因，也是 题 16/17/19 存在的理由。
**实测锚点（构造数据：Gaussian 权重 + 两列 outlier ×20，可复现）**：rel 误差 per-tensor INT8 5.7% → per-group(32) **1.1%**；INT4 per-tensor 52.4% → per-group(32) **16.4%**——分组量化是最便宜的一刀：

```python
import numpy as np
def quant(W, bits, group=0):
    """对称量化。group=0→per-tensor；group>0→按输入维分组。O(W.size)"""
    if group:
        B = W.shape[1] // group
        Wh = W.reshape(W.shape[0], B, group)
        s = np.abs(Wh).max(axis=2, keepdims=True) / (2**(bits-1) - 1)
        return (np.clip(np.round(Wh/s), -(2**(bits-1)), 2**(bits-1)-1)*s).reshape(W.shape)
    s = np.abs(W).max() / (2**(bits-1) - 1)
    return np.clip(np.round(W/s), -(2**(bits-1)), 2**(bits-1)-1) * s

rng = np.random.default_rng(0)
W = rng.normal(size=(64, 256)); W[:, [7, 128]] *= 20     # 人造 outlier
rel = lambda A: np.linalg.norm(W-A)/np.linalg.norm(W)
print(rel(quant(W, 8)), rel(quant(W, 8, 32)))   # 实测 0.057 → 0.011
print(rel(quant(W, 4)), rel(quant(W, 4, 32)))   # 实测 0.524 → 0.164
```

**收益三账**（面试要能分开报）：
- **显存账**：FP16→INT8 半价、→INT4 四分之一价（70B：140 → 70 → ~35 GB＋分组 scale 约 1 GB 级）；
- **带宽账**：decode 是 bandwidth-bound（题 20 的 roofline：每 token 必读全权重），权重的字节线性减少 → 单流 decode 近线性提速（理论 4×，kernel 不打满）；
- **计算账不变**：W4A16 的 kernel 实际是“反量化到 FP16 再做 GEMM”——**FLOPs 一点没少**；想要算子级加速要 W8A8/FP8。
**分类轴**：weight-only（W4A16：GPTQ/AWQ/GGUF/QLoRA-NF4）vs 全量化（W8A8：SmoothQuant 处理激活 outlier）vs KV cache 量化（题 3/19）vs 低比特浮点（FP8 E4M3/E5M2、NF4 非均匀码本）。

**3.【30秒答题框架】** “把 32/16bit 压到 8/4bit：公式 x̂=s(x_q−z)，对称无 z、非对称有；粒度 tensor→channel→group，4-bit 必须 group。误差是 Δ²/12，Δ 由组内极值定——outlier 决定精度，所以分组是最便宜的一刀（实测：带 outlier 的权重 INT4 误差从 52% 到 16%）。收益两改：显存线性省、decode 带宽近线性提速；但 W4A16 **不省计算**，算子级加速要 W8A8/FP8。补偿手段（GPTQ/AWQ）全从‘Δ 由极值定’这一句推出来。”

**4.【追问与陷阱】**
- *对称 4-bit 网格边界？* [-8, 7]（补码对称—负多一个格点）；
- *zero-point 为什么存在？* 对齐偏斜分布（激活全正）；
- 陷阱：“量化必然损精度”（INT8 per-channel 常近无损）；“W4A16 加速 prefill”（没有——算力项不变）；把激活量化说简单了——激活动态范围由输入决定，不能像权重预先标定，这是 W8A8 难的根因（SmoothQuant 的移动 scale 恒等式是预告）。

**5.【知识串联】** 题 15/16/17/19（量化大厦）、题 18（CPU 侧 K-quant）、03-FineTuning QLoRA（NF4/双量化）、题 20（带宽公式）、10-RealQuestions 二面清单第 4 题（蒸馏和量化的区别——一个是训练侧压缩、一个是数值表示压缩）。

---

### 题 15：PTQ vs QAT ⭐⭐⭐

**1.【考察意图】** 原表题。考察：①两流程差异（离线校准 vs 插 fake-quant 重训）；②STE 为什么“能训”；③LLM 时代**为什么全用 PTQ**（算力经济学账），QLoRA 作为“折衷桥梁”。

**2.【深度解析】**
- **PTQ**：训练完离线量化；只需少量校准数据。方法阶梯从 RTN（朴素舍入）→ 二阶补偿（GPTQ 题 16）→ 激活感知缩放（AWQ 题 17）→ 混合精度分解（LLM.int8：outlier 通道留 FP16）。成本锚点：GPTQ 论文口径 175B 级模型单卡 A100 约 4 GPU 小时。
- **QAT**：训练时插 fake-quant：前向模拟 `clamp(round(w/s)+z)`，反向用 **STE**（直通估计器：round 的导数数学上是 Dirac 脉冲全零，STE 近似为 clip 区间内恒 1，区间外乘指示 0），scale 也可以学（LSQ）。
- **成本账**：QAT 恢复质量上限高（<4-bit 尤其），但引入**完整训练循环**——百亿参数下基本不可行。所以 LLM 时代的工程答案是：**部署走 PTQ，微调补偿走 QLoRA**（4-bit 基座 + LoRA），QAT 留给端侧小模型。
- **W8A8 的特殊一环**：激活 outlier 是 W8A8 的拦路虎 → SmoothQuant 用恒等缩放迁移 $W'=W\odot \mathrm{diag}(s),\ x'=x/\mathrm{diag}(s)$（数学不变，把激活的难度搬到权重）；LLM.int8 则干脆 outlier 通道留 FP16。

**3.【30秒答题框架】** “PTQ：训练后离线，几百条校准数据，方法从 RTN 到 GPTQ/AWQ；INT8 近无损、4-bit 靠补偿，175B 约 4 GPU 小时。QAT：训练时插 fake-quant，STE 直通梯度，质量上限高但要整个训练循环，百亿级不现实。QLoRA 是折衷桥梁：4-bit（NF4+双量化） PTQ 基座上做 LoRA。所以 LLM 时代答案是 PTQ 为主、QLoRA 补偿、QAT 留端侧小模型。”

**4.【追问与陷阱】**
- *STE 推导一遍？* round 的导数除 Dirac 脉冲外处处 0 → 换成区间指示函数（clip 内 1）；LSQ 把 s 当参数并给出带 α 的显式梯度。
- *PTQ 误差会跨层累积吗？* 会——GPTQ 的“逐层 + 层内补偿”就是控制策略。
- 勘误口径：原表把 QLoRA 标为 QAT——严谨地说 QLoRA 是 **PTQ 基座 + LoRA 微调**（论文：NF4 + double quantization + paged optimizer），面试按此口径讲并主动说明。
- *为什么激活量化更难？* 动态范围不能预标定（题 14 末段）——SmoothQuant/LLM.int8 的存在理由。

**5.【知识串联】** 题 14（误差学）、题 16/17（两大 PTQ 方法）、03-FineTuning/01-PEFT（QLoRA 原理节）、题 18（GGUF 给 CPU 的另一叙事）、10-RealQuestions 二面清单第 4/5 题。

---

### 题 16：GPTQ ⭐⭐⭐

**1.【考察意图】** 真·量化二面硬核（真题原句“GPTQ 和 AWQ 的区别”的前半）。考察：①层内目标 $\min\|WX-\hat W X\|^2$ 与 **Hessian 为什么是 $2XX^\top$**；②逐列量化 + 误差回传的闭式（白板级）；③Cholesky / 分组 / act-order 工程件；④它 **省带宽不省算力** 的 W4A16 定位。

**2.【深度解析】**

**层内目标（一步都不跳）**：权重只对“喂进它的校准输入 X”负责 →
$$\arg\min_{\hat W}\ \|WX-\hat WX\|_F^2$$
对 $\hat W$ 这是**二次型**，Hessian 精确为 $2XX^\top$（每层、每组织一次；常数 2 常省略）——这是 GPTQ 名字里“Hessian”的全部来源。逐层独立求解，跨层误差实测可控。

**逐列补偿闭式（OBQ/GPTQ 核心）**：量化第 q 列 $\hat w_q=\mathrm{quant}(w_q)$ 后，把误差 $
\delta_q=w_q-\hat w_q$ 通过其最优分配传播到**未量化列**（一阶最优化解）：
$$\Delta W_F=-\,\frac{w_q-\hat w_q}{\big[H^{-1}\big]_{qq}}\ \big[H^{-1}\big]_{:,\,q}\ \Big(\text{F 的含义：按 }H^{-1}\text{ 第 q 列把误差“倒”给其它列}\Big)$$
物理含义直观：$[H^{-1}]_{qq}$ 大 → 该列自己“吸收误差能力强”，别人替它补得更多；如此往复逐列扫完全层，每一步都是给定“前面已定”下的全局最优步。

**工程三件**：
- **Cholesky 加速**：朴素每列要重算 $H^{-1}$ 子矩阵；GPTQ 用 $H^{-1}$ 的 Cholesky 分解 $H^{-1}=LL^\top$ 逐列预展开，整层降到 O(n³)（175B ≈ 4 GPU 小时，单 A100，论文口径）；
- **damping**：$H+\lambda I$ 防奇异（校准数据少、输入相关）；
- **act-order（desc_act）**：按 $\mathrm{diag}(H)$（= 输入平方范数）降序处理列——先处理“激活大”的高/难列，把最后的补偿交给误差预算大的后段；group-size=128 独立 scale。

**结果定位（W4A16）**：权重存 INT4 + per-group scale，kernel 运行时反量化成 FP16 做 GEMM。**省的是权重带宽**（decode 近线性提速，题 20 roofline）；**算力没省**（对 prefill 无收益）——这是它和 AWQ 与一切 W4 的共同定位。

**参考实现（核心逐列补偿，小矩阵可跑，实测输出）**：
```python
import numpy as np
rng = np.random.default_rng(0)
n, m = 32, 64
W = rng.normal(size=(m, n))          # W: (d_out, d_in)
X = rng.normal(size=(200, n)).T      # 校准输入 (d_in, s)
H = X @ X.T                          # 经验 Hessian < XXᵀ >（常数 2 省略）
H[np.arange(n), np.arange(n)] += 0.01 * np.trace(H) / n   # damping
Hinv = np.linalg.inv(H)

def q4(w, bits=4):                   # 对称逐列量化
    s = np.abs(w).max() / (2**(bits-1) - 1)
    return np.round(w/s).clip(-(2**(bits-1)), 2**(bits-1)-1) * s

order = np.argsort(-np.diag(H))      # act-order: 对角大的列先处理
Wq = W.copy()
for r, i in enumerate(order):
    w = Wq[:, i]
    qw = q4(w)
    err = (w - qw) / Hinv[i, i]       # 误差按 H⁻¹ 对角元归一
    Wq[:, order[r+1:]] -= np.outer(err, Hinv[i, order[r+1:]])   # 回传：ΔF = −err·H⁻¹_{i,rest}
    Wq[:, i] = qw

obj = lambda A: np.linalg.norm((W - A) @ X, 'fro')**2
rtn = np.apply_along_axis(q4, 0, W)
print(round(obj(rtn), 2), round(obj(Wq), 2))          # 实测 5102.60 → 4782.78（目标 −6.3%）
print(round(np.linalg.norm(W-rtn)/np.linalg.norm(W), 4),
      round(np.linalg.norm(W-Wq)/np.linalg.norm(W), 4))  # 实测 0.111 vs 0.118
```
**最有教学价值的现象（有实测）**：GPTQ 的**权重自身误差反而略大**（0.111→0.118），但**输出误差更小**（obj −6.3%）——因为它优化的就是 $\|\Delta W\cdot X\|$ 而非 $\|\Delta W\|$：误差被搬到 X 不密集的方向。这一句话能证明你真懂 GPTQ。

**3.【30秒答题框架】** “GPTQ 是 weight-only PTQ：层内目标 ‖WX−ŴX‖²，Hessian 精确等于 2XXᵀ。逐列量化，每量一列按 −(w−q)/H⁻¹·H⁻¹ 列切片的闭式把误差回传给未量化列，让剩余权重吸收；前面发现权重误差可能略增但输出误差下降——它优化的是输出面误差。Cholesky 一次分解把整层复杂度降到 O(n³)，175B 单 A100 约 4 小时。附加项：damping 防奇异、group 128、act-order。运行时 W4A16：省带宽不省算力。与 AWQ 一句话分工：GPTQ 事后补偿，AWQ 事前缩放。”

**4.【追问与陷阱】**
- *为什么逐层不端到端？* 逐层解析闭式 + 实测误差可控；跨层补偿没有好的显式解。
- *Hessian 是谁的？* 是**量化目标**对 W 的 Hessian = 2XXᵀ（输入协方差二阶矩），不是模型参数的 Hessian——最常见的含糊表述。
- *3-bit 行吗？* 论文支持更极端设定但要配整套工程；常规部署 3-bit 是质量临界（题 19）。
- 陷阱：说“GPTQ 需要反向传播/重训”（不需要，闭式）；把 AWQ 的缩放混进 GPTQ（GPTQ 也可以叠加 act-order 之外，不缩放）。

**5.【知识串联】** 题 14（Δ²/12 误差学）、题 17（数值路线对照表）、题 15（PTQ 大类）；03-FineTuning QLoRA（4-bit 基座上的微调补偿）；10-RealQuestions 二面清单第 5 题（INT4 掉点怎么办——按本题补偿瀑布处理）。

---

### 题 17：AWQ (Activation-aware Weight Quantization) ⭐⭐⭐

**1.【考察意图】** 真题原句（GPTQ vs AWQ）的另一半。考察：①核心洞察——**权重的重要性看它乘到的激活，不看权重自身幅值**；②等价缩放的恒等性 + 如何降低显著通道误差（不靠反传）；③与 GPTQ 的**数值路线分工**（事前缩放 vs 事后补偿）。

**2.【深度解析】**

**观察（论文口径）**：按激活幅度选出的 ~1% "salient" 通道承载了模型大部分能力；对应权重直接舍入损伤最大。混合精度保留它们（少数 FP16）的朴素方案在权重存储粒度上不划算（离群访存打碎 kernel）——更快路线是**等价缩放**。

**等价缩放（核心推导）**：插入恒等恒等式，缩放的 y 不变：
$$y=Wx=\big(W\,\mathrm{diag}(s)\big)\big(\mathrm{diag}(s)^{-1}x\big),\qquad \widetilde W=W\,\mathrm{diag}(s),\ \tilde x=x/\mathrm{diag}(s)$$
——给**显著通道**的权重放大 s、激活缩小 1/s：FP 下输出一模一样；量化时若按**新权重分布**分配 scale，显著通道的**相对舍入误差更小**（等效把误差从“重要通道”转移到“不重要通道”）。

**scale 怎么找（无反传）**：$s=s_x^{\alpha}/s_w^{1-\alpha}$，$s_x$ = 校准激活的通道均幅、$s_w$ = 权重通道幅值；α∈[0,1] 网格搜索（步长 0.05）——目标直接放在**量化域**上：
$$\min_\alpha\ \big\|Q(W\mathrm{diag}(s))\,(\mathrm{diag}(s)^{-1}X)-WX\big\|^2$$
不碰梯度、不用反传、几百条校准样本；再配一个 per-channel **clip 搜索**压制大权重 outlier。kernel 则将 1/s 吸收进前一层 / Norm 的权重（数学恒等链），几乎零开销。

**实测小例（构造激活重尾，可跑）**：
```python
import numpy as np
rng = np.random.default_rng(0)
d_out, d_in = 32, 64
W = rng.normal(size=(d_out, d_in))
act = np.exp(rng.normal(size=d_in)); act[[7, 19, 41]] *= 40   # 人造重尾激活
Xc = rng.normal(size=(400, d_in)) * act                        # 校准激活

def quantize(w, bits=4, group=16):                             # 沿输入维分组对称量化
    q = np.empty_like(w)
    for c in range(0, w.shape[1], group):
        blk = w[:, c:c+group]
        s = np.abs(blk).max() / (2**(bits-1) - 1)
        q[:, c:c+group] = np.clip(np.round(blk/s), -(2**(bits-1)), 2**(bits-1)-1) * s
    return q

def awq_search(W, Xc, alphas=None):
    alphas = alphas if alphas is not None else np.arange(0, 1.01, 0.05)
    sx = np.sqrt((Xc**2).mean(0)); sw = W.std(0) + 1e-9
    best = (np.inf, None)
    for a in alphas:
        s = sx**a / sw**(1-a)
        e = np.mean((quantize(W*s) @ (Xc/s).T - W @ Xc.T)**2)   # 量化域目标
        if e < best[0]: best = (e, a)
    return best

e_rtn = np.mean((quantize(W) @ Xc.T - W @ Xc.T)**2)
e_awq, a_star = awq_search(W, Xc)
print(round(e_rtn, 2), round(e_awq, 2), round(a_star, 2))   # 实测 432.30 → 205.80，α*=0.15（−52.4%）
```

**与 GPTQ 对照（背下这张表，真题常直接问区别）**：

| 维度 | GPTQ | AWQ |
|---|---|---|
| 数值思想 | 舍入后**误差再补偿**（Hessian=2XXᵀ 逐列回传） | 舍入**前缩放重分配**误差（激活感知） |
| 二阶/校准信息 | 输入二阶矩阵 XXᵀ | 激活**一阶**均幅统计 |
| 求解方式 | 闭式逐列更新（O(n³)/层） | α 网格搜索（无梯度） |
| 部署目标 | W4A16 同样 | W4A16 同样（scale 吸收，无额外开销） |
| 质量 | 强基线，“先验误差吸收” | 通常在 4-bit 更稳（原表口径；多篇对比 AWQ 总体更准，GPTQ 也在持续更新——工程做法是两个都试） |

两者**可叠加**（AWQ 建立缩放 → GPTQ 在缩放后的权重上再补偿）。

**3.【30秒答题框架】** “AWQ 一句话：重要的不是权重幅值，是它**乘到的激活**——1% 显著通道承载主要能力。做法是等价缩放：W·diag(s)·(x/diag(s))，FP 下恒等；显著通道权重放大后，同 group 的相对舍入误差变小，误差被挤到不重要通道。s=s_x^α/s_w^{1−α} 在量化域网格搜，不反传、几百条样本搞定（实测小例 MSE 从 432 降到 206）。与 GPTQ 分工：GPTQ 事后补偿（用 XXᵀ 二阶），AWQ 事前缩放（激活一阶），两者可以叠。”

**4.【追问与陷阱】**
- *“等价缩放何时不成立？”* 恒等只在 FP 下；量化后误差重分配的**有效性**依赖校准分布 = 部署分布（AWQ 与 GPTQ 同病同防）。
- *scale 怎么进 runtime？* 吸收进上一层的输出通道 scale / 前层权重 / Norm 参数（chain 恒等），kernel 几乎零开销。
- 陷阱：AWQ ≠"混合精度"（论文明确不用）；AWQ ≠ QAT（无训练、无反传）；把 $s_w$ 说成不重要——$s_w^{1-\alpha}$ 的分母选择防止放大巨幅权重挤爆格点（也是 clip 搜索的动机）。
- *3-bit？* 低位下“一阶缩放”不够，需要码本讨论（NF4/GGUF K-quant，题 18/19）。

**5.【知识串联】** 题 14（Δ 由极值理论）、题 16（对照表和可叠加性）、题 15（PTQ 桥位）、03-FineTuning QLoRA（另一条 4-bit 思路：NF4 = 信息论最优正态码本）、10-RealQuestions 面经 3 第 8 题。

---

### 题 18：GGUF (llama.cpp) ⭐⭐

**1.【考察意图】** CPU/端侧生态定位：①GGUF 是**格式**而不是一种量化算法（含 header/metadata/张量/vocab 的单文件容器）；②block-wise 量化的谱系思想（Q4_0 → K-quant 超块 → IQ 码本）；③imatrix 与 AWQ 的思想对应。

**2.【深度解析】**
- **容器**：单文件（header + KV 元数据 + 张量表 + 分词器），mmap 懒加载；Ollama/llama.cpp 消费的就是它。
- **逐块量化（legacy）**：每 **32 个权重一个块、共用一个 FP16 scale**：
  - Q4_0：32×0.5B + 2B = 18B/块 → **4.5 bpw**；7B 模型权重 ≈ 7e9×4.5/8 ≈ **3.9 GB**（可推导）；
  - Q8_0：32×1B + 2B = 34B/块 → **8.5 bpw**（近 FP16 质量）。
- **K-quant（Q4_K 系）**：超块套块——256 权重的 super-block 内再分 sub-block 携带**次级 scale**（两阶量化），有效位宽 ≈4.5–4.9 bpw，同位宽下精度显著优于 legacy；`_S/_M/_L` 是各层混用位宽的“配比档”。**IQ 系**（IQ2/IQ3/IQ4_XS）引入码本，把 2–3 bit 做到可用。
- **imatrix（重要性矩阵）**：用校准数据统计每通道激活强度做误差加权——**思想上即 AWQ 的 CPU 侧实现**（点出来是亮点）。
- **kernel 现实**：CPU SIMD 逐块解包 → dequant → 点积；难度在内存布局与 cache 行对齐。同样吃 roofline：显存省 3–4×，速度受 DDR 带宽限制。
- **数量级锚点（全部可推导）**：7B FP16 ≈ 14 GB；Q4_0 ≈ 3.9 GB；Q4_K_M ≈ 4.2 GB（≈4.8 bpw）；Q8_0 ≈ 7.4 GB（8.5 bpw）。

**3.【30秒答题框架】** “GGUF 是 llama.cpp 的单文件量化容器；核心是 CPU 友好的 block-wise 量化：legacy 每 32 权重一个 FP16 scale（Q4_0=4.5 bpw），K-quant 用超块分层 scale（Q4_K 系 ≈4.8 bpw，精度更好），IQ 系用码本下探 2–3 bit。imatrix 用激活统计加权误差——AWQ 思想的 CPU 版。桌面默认 Q4_K_M（7B 约 4.2 GB，FP16 的 30%）；Q8_0 近无损（8.5 bpw）。”

**4.【追问与陷阱】**
- *Q4_K_M vs Q8_0？* 默认 Q4_K_M（尺寸/质量甜点）；求近无损 Q6_K/Q8_0；IQ2 只在极端约束用。
- 陷阱：把 GGUF 当“一种量化算法”（先是个格式，量化是张量编码方式）；GGUF 权重与 GPU serving 框架（vLLM/TRT-LLM）不互通（生态隔离）；perplexity 相近 ≠ 下游质量无损（低位宽先垮长文本一致性）。
- *为什么 CPU 侧谱系这么密？* 内存带宽是 CPU 推理主瓶颈，每 0.3 bpw 就是几个 GB 的差——“用精度换字节”的设计分层。

**5.【知识串联】** 题 14（误差学与码本思想）、题 19（对比表 CPU 行）、题 8（Ollama 的底层生态）、03-FineTuning（NF4 码本同思想）；CPU roofline 与 题 20 同构（HBM 换 DDR 带宽）。

---

### 题 19：量化对比表（FP16/INT8/GPTQ/AWQ/GGUF/QLoRA）+ INT4 精度补偿手段 ⭐⭐⭐

**1.【考察意图】** 原表题 + 真题原句“INT4 量化后精度下降怎么办”。两层：①六方法一张定位图（位宽-类型-场景）；②INT4 掉点的**系统性排查瀑布**——考链路思维，不是名词罗列。

**2.【深度解析】**

| 方法 | 位宽 | 类型 | 适用场景 | 决策问句 |
|---|---|---|---|---|
| FP16 | 16 | 基线 | GPU 充足、SLO 苛刻 | 卡能不能装下？ |
| INT8 (W8A8) | 8 | PTQ | 轻度压缩 + 算子级提速 | 想带宽**和**计算双收益？ |
| GPTQ | 3/4 | PTQ | GPU、weight-only | 省 2–4× 显存、接受二阶流程？ |
| AWQ | 4 | PTQ | GPU，4-bit 质量更稳（原表口径） | 校准集可靠、求稳定质量基线？ |
| GGUF | 2–8 | PTQ | CPU/端侧/单机 | 没 GPU 或本地？ |
| QLoRA (NF4) | 4 | PTQ 基座 + LoRA 训练 | **微调**场景 | 要在 4-bit 上继续训吗？ |

（勘误口径：原表把 QLoRA 标为 QAT——严谨说它是 **PTQ 基座 + LoRA 补偿训练**。QLoRA 三件套：NF4（正态分布信息论最优码本）、Double Quantization（量化常数 0.5 → 0.127 bit/param，65B 约省 3 GB）、Paged Optimizers。）

**INT4 补偿手段全家桶（按“最便宜优先”）**：
1. **细粒度分组** group=64/128——直接压 Δ，最便宜一刀（题 14 实测：误差 52% → 16%）；
2. **校准集治理**：覆盖真实分布与任务（最被低估的一刀；数据偏后面全白搭）；
3. **AWQ 激活感知缩放**（题 17）：把误差挤去不重要通道；
4. **GPTQ 逐列补偿**（题 16）：把挤剩的误差让未量化列吸收；
5. **混合精度残留**：极敏感层（首行/语言头/离谱 outlier 层）保 FP16（LLM.int8 思路）；
6. **act-order / clip 搜索**：题 16/17 的附属开关（成本/收益再看）；
7. **QLoRA 微调补偿**：最后一手，总有训练余力时用。
**排查瀑布**：分层定位（逐层 probe 量化前后输出）→ 校准/group → AWQ scale → GPTQ 补偿 → 混合精度 → QLoRA 恢复。
**KV cache 是另一套账**：KV 动态、范围随位置漂移，用 per-head scale / FP8，不要照搬权重方案（题 3/14）。

**3.【30秒答题框架】** “定位图：8-bit 近无损且能拿算子收益；4-bit 是性价比临界，也是补偿技术的战场；2–3 bit 必须配码本（K/IQ、NF4）。INT4 掉点按瀑布走：分层定位 → 校准集与 group 细化 → AWQ 缩放 → GPTQ 补偿 → 混合精度兜底 → QLoRA 恢复。别忘 KV 与权重是两套量化问题。”

**4.【追问与陷阱】**
- *INT4 vs INT8 带宽收益？* 带宽/显存线性减半；kernel dequant 开销使实际吃不满 2×（roofline 精神，题 20 验证）。
- *蒸馏 vs 量化？*（真题）：蒸馏 = 知识迁移（小模型重训，改参数量），量化 = 数值表示压缩（同模型少 bit，不改参数量）。
- 陷阱：比较 GPTQ/AWQ 不提“可叠加”与“同吃校准分布”；把 NF4 说成 GPTQ 的变体（码本路线，独立方法）。

**5.【知识串联】** 题 14（误差学）→ 16/17（两大方法）→ 18（CPU 码本分支）；03-FineTuning QLoRA 全套；题 20（TPOT 视角）；10-RealQuestions 二面清单第 4/5 题（蒸馏 vs 量化、INT4 补偿）。

---

### 题 20：部署架构（单机/分布式）+ TP vs PP + 从请求到 GPU 的端到端延迟拆解 ⭐⭐⭐⭐

**1.【考察意图】** 系统题锚。考察：①端到端延迟拆解（TTFT / TPOT 分段归因：prefill compute-bound、decode bandwidth-bound——要能拿公式讲）；②TP/PP 通信特征与量化（AllReduce 频率、bubble 公式）；③容量与并发的 roofline 账。

**2.【深度解析】**

**① 请求 → 响应全链路（六段）**：
1. **网络/网关/限流** ~1–10 ms；
2. **tokenizer** ~1 ms 量级（随 prompt 长度线性）；
3. **排队/调度**（waiting→running，受 KV 预算与抢占策略支配，题 5/6）——TTFT 里“纯策略可调”的一段；
4. **prefill**（TTFT 主体，**compute-bound**）：
$$\text{FLOPs}\approx 2Ns + 4Ls^2 d\quad(\text{线性层 }2Ns\text{；注意力 QK 与 attn·V 因果减半 }4Ls^2d)$$
实测锚（7B；N≈6.7e9、L=32、d=4096、s=1024）：2×6.7e9×1024 + 4×32×1024²×4096 ≈ 1.37e13 + 0.27e13 = **14.9 TFLOPs**；A100 BF16 312 TFLOPS × 50% MFU ≈ 156 TFLOPS → **约 95 ms**（纯 roofline 推导，非基准数据）；
5. **decode**（TPOT 主体，**bandwidth-bound**）——每步无论 batch 大小都要读一遍全权重**加**全部 KV：
$$\text{TPOT}\approx\frac{\text{权重字节}+\text{KV字节}}{\text{HBM 带宽}}$$
7B FP16 + 4k 上下文：14.0 GB（权重）+ 2.15 GB（KV，题 2）= 16.1 GB ÷ 2039 GB/s = **7.9 ms/token → 单流 ~126 tok/s 上限**；按 60–80% 有效带宽 → **80–100 tok/s 实际**。交叉验证计算强度：AI = 14 GFLOPs / 16.1 GB ≈ **0.87 FLOP/B**，而 ridge point = 312 TFLOPS ÷ 2039 GB/s ≈ **153 FLOP/B**——差一百多倍，decode 深陷 bandwidth-bound；
   **batching 的量化账本（自算）**：b=16 步耗时 = (14.0 + 16×2.15) GB ÷ 2039 = **23.7 ms/step** → **675 tok/s ≈ 5.4×**；但 TPOT 7.9 → 23.7 ms（3×）——吞吐换延迟，调度器的准入取舍（题 6）就用这张表；
6. **detokenize/流式回传** ~0.1–1 ms。

**延迟优化两分法（面试高频）**：
- **TTFT**：chunked prefill、prefix caching（公共 system prompt 不重算：RadixAttention，题 8）、kernel 效率（FA 题 4）；量化对 TTFT 帮助有限（W4A16 不省算力）；
- **TPOT**：权重量化（14 → 4–7 GB，题 14–19 直接改进本公式）、KV 量化/MLA（题 3）、投机解码（题 7，每步 ≥1 token）、batch 预算（题 6）。

**② 并行策略（原表 + 量化锚点）**

| 方式 | 原理 | 通信 | 量化锚点 |
|---|---|---|---|
| 张量并行 TP | 单层切多卡 | **每层两次 AllReduce**（attention 输出、MLP 输出；前向） | 通信量 ∝ b·s·d，**延迟敏感** → 锁单机 NVLink（≤8 卡） |
| 流水线并行 PP | 不同层在不同卡 | 仅 stage 边界一次 P2P | 跨机友好；bubble = (p−1)/(m+p−1)；decode TPOT 会加 stage 跳数 |
| EP（MoE） | 专家分卡 | All-to-All | 见 07-HotTopics DeepSeekMoE |

拓扑翻译原文件：`LB → 3×(vLLM TP=4)`——“节点内 TP / 跨机多副本”：TP=4 把 70B FP16 权重 140 GB 拆进 4×80G，多副本横向扩吞吐。数字补一刀：70B INT4 权重 ~35 GB，单 A100 roofline decode = 35/2039 ≈ **17.2 ms/tok**；TP=4 → 35/(4×2039) = **4.3 ms/tok ≈ 230 tok/s 上限**（带宽×4 = 并行收益来源）。

**③ 显存预算账（容量规划）**：权重（N×bytes，按量化位宽折）+ KV（题 2 公式×并发）+ 激活工作区。70B INT4：~36 GB + 1.25 GiB/序列@4k ≤ 80 GB → 并发 ~30（与题 2 同算）。

**3.【30秒答题框架】** “端到端六段：网关/tokenizer 几 ms；TTFT 是 prefill、compute-bound，FLOPs≈2Ns+4Ls²d，7B@1k 约 15 TFLOPs → 半效率 95 ms；TPOT 是 decode、bandwidth-bound，每步读权重+KV：7B@4k = 16 GB ÷ 2 TB/s ≈ 7.9 ms → 单流 ~126 tok/s，AI 0.87 离 ridge 153 很远，所以批越大吞吐越好——b=16 时 675 tok/s 但 TPOT 三倍，是准入控制的取舍。优化两分：TTFT 砍 prefill 计算与缓存，TPOT 砍权重字节（量化）+ KV（量化/MLA）+ 投机解码。TP 每层两次 AllReduce 只放单机；PP 边界一次 P2P 可跨机、bubble=(p−1)/(m+p−1)；MoE 加 EP；终态‘节点内 TP、跨机多副本’。一句话：prefill 的敌人是算力，decode 的敌人是带宽。”

**4.【追问与陷阱】**
- *TTFT 压到哪？* prefill compute-bound：治 kernel/MFU/前缀缓存；W4A16 量化帮不上算力（高频错误）。
- *TP 为什么不跨机？* 每层两次 AllReduce×近百层，RDMA P99 炸裂；跨机用 PP/EP。
- 陷阱：把 prefill 加速归功 PagedAttention（管显存不管理算力）；把“单流延迟随 batch 线性恶化”说死——权重读取是公共成本，只有 KV 部分线性涨（7.9→23.7 ms 的算例）；混淆 GiB/GB。
- *一句话收底*：量化治“每 token 讀的字节”，投机解码治“每步只出一个 token”的结构性低效——两者都作用在 TPOT 公式上。

**5.【知识串联】** 题 2（KV 预算 = 并发的账本）、题 5/6（内存与调度两刀）、题 7（TPOT 捷径）、题 14–19（量化直接进 TPOT 公式）、09-SystemDesign（本题是其微观物理底座）、07-HotTopics §1（MLA/MTP/EP 的系统影响）、10-RealQuestions（面经 3 第 6–9 题全部落点）。

---

# 本章高频追问 Top5

1. **“KV 显存公式 + 三算例”**（题 1/2/3）：现场写 MHA/GQA/MLA——2.0 GiB / 1.25 GiB / 0.27 GiB@4k 三个锚点；追问“MLA 为什么必须解耦 RoPE（576=512+64 从哪来）”“GQA 怎么从 MHA 转换”——考点是**因子级理解 + 反推容量与架构动机**，不是背公式。
2. **“FlashAttention 为什么快”**（题 4）：答案分两层——SRAM/HBM 数量级 + online softmax 的 max/sum 修正项白板归纳推导（再追：两遍法对比、FA-2 差异、decode 为什么不灵、什么时候不显著）。
3. **“投机解码为什么保证分布一致”**（题 7）：接受项 min(p,q) + 残差项 (p−q)₊ 恒等 p 的两步证明（含 Σ(p−q)₊=Σ(q−p)₊）;追问 γ 怎么选、α 低到哪是负加速（网格：0.4→1.18×、0.2→0.89×）、与 continuous batching 的配平。
4. **“vLLM 为什么把浪费压到 <4%”**（题 5/6）：三类浪费解剖 + 16-token 分页 + CoW 共享；追问抢占（swap vs recompute）、chunked prefill、静态批 47% 算例——serving 岗二面现场开题。
5. **“GPTQ vs AWQ / INT4 掉点怎么办”**（题 16/17/19）：Hessian 逐列补偿（事后）vs 激活感知缩放（事前）的路线差异 + 补偿瀑布（分组→校准→缩放→补偿→混合精度→QLoRA）；顺带接“蒸馏 vs 量化”的口径。

> 全章 20 题完整覆盖 01-KVCache-And-Acceleration.md（8 题）、02-Decoding-Strategies.md（5 题）、03-Quantization-And-Deployment.md（7 题），编号与原文件顺序一致；每题五段式（考察意图 / 深度解析 / 30 秒答题框架 / 追问与陷阱 / 知识串联）。所有代码块已在 numpy / 纯 Python 下实测通过（输出已内嵌）；论文口径数字（GPTQ 单 A100 4 小时 / AWQ 1% 显著通道 / vLLM 60–80% & <4% & 2–4× / 投机解码 2–2.5×、2–3×）均注明出处，非论文数字全部为已标注的可推导算例。原章节文件只读未改。
