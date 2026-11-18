# 01-Foundation 逐题详解

> 来源章节：`01-Transformer.md`（17 题）、`02-Tokenization.md`（7 题）、`03-Loss-And-Optimization.md`（8 题），共 **32 题**，覆盖率 100%。
> 编号沿用原文件顺序；原章节文件只读未改。数学推导对齐原始论文（Vaswani et al. 2017、RoFormer、AdamW、DeepSeek-V2 等），数字仅取论文/官方公开数据或本文可复现推导。

## 本章考点分布

| 主题 | 题号 | 面试权重 | 高频深挖点 |
|------|------|----------|-----------|
| Attention 机制与缩放 | 3, 4, 5, 17 | ⭐⭐⭐⭐⭐ | √d_k 的方差+梯度双解释；softmax 可否替换 |
| MHA/MQA/GQA/MLA | 6, 7 | ⭐⭐⭐⭐⭐ | KV Cache 显存公式、Decode 带宽瓶颈 |
| 位置编码（RoPE/ALiBi） | 8, 9 | ⭐⭐⭐⭐ | 复数推导、NTK/YaRN 外推 |
| Norm 与训练稳定性 | 10, 11, 12, 31 | ⭐⭐⭐⭐ | Pre/Post-Norm 梯度路径分析 |
| 架构选型与 FFN | 1, 2, 13, 14, 15, 16 | ⭐⭐⭐⭐ | FLOPs≈2×参数×token、8/3 法则 |
| 分词（BPE 系） | 18–24 | ⭐⭐⭐ | BPE 全流程样例、词表-压缩率权衡 |
| 损失与优化 | 25–32 | ⭐⭐⭐⭐⭐ | AdamW 解耦数学、loss scaling、warmup 必要性 |

面试分布规律：**纯八股题（架构描述）只出现在一面；二面起全部围绕"为什么"**——√d_k、多头收益、RoPE 推导、AdamW 解耦、混合精度这五个点是字节二面最常追问的深水区。

---

# Part A · Transformer 架构详解（01-Transformer.md）

### 题 1：请描述 Transformer 的整体架构

**【考察意图】**
热身题，一面第一问。验证三件事：(1) 是否真的背过/理解结构而不只是"用过 API"；(2) 能否在 1 分钟内画出模块级数据流；(3) 后续追问（为什么这样设计）的地基是否牢。

**【深度解析】**

原始论文《Attention Is All You Need》(Vaswani et al., 2017) 结构：

```
Encoder（N=6 层堆叠）:
  X ∈ R^{n×d_model}
  → Attention(自注意力, 双向) → Add & Norm
  → FFN → Add & Norm
  共 6 个子层操作

Decoder（N=6 层堆叠）:
  T（已生成序列，右移一位做 teacher forcing）
  → Masked Self-Attention（因果） → Add & Norm
  → Cross-Attention（Q=decoder 状态, K/V=encoder 输出） → Add & Norm
  → FFN → Add & Norm
  → Linear(d_model→|V|) → Softmax
```

Base 版关键数字：$d_{model}=512$，$h=8$，$d_{ff}=2048$，参数量约 65M；Big 版 $d_{model}=1024$、$d_{ff}=4096$、约 213M。

数据流关键点（面试必须主动说出）：
1. **Embedding 层与输出 softmax 前的 Linear 共享权重**（原文"share the same weight matrix between our embedding layers and the pre-softmax linear transformation"），缩放 $\sqrt{d_{model}}$。
2. **三种注意力**：encoder 自注意（双向）、decoder masked 自注意（因果）、cross-attention（Q 来自 decoder，K/V 来自 encoder）——cross-attention 中 K、V 的序列长度可以与 Q 不同（$m \neq n$），这正是 seq2seq 对齐能力的来源。
3. **每个子层包一层残差**：$\text{LayerNorm}(x + \text{Sublayer}(x))$，这是 Post-Norm（对比题 11）。
4. **残差 + LayerNorm 的存在意义**：让"裸 attention/FFN"可以是恒等初始化附近的扰动，保证深层可训。

现代 LLM 的改造（面试加分项，一句话说清差异）：
| 原始 Transformer | 现代 LLM（GPT/LLaMA 类） |
|---|---|
| Encoder-Decoder | Decoder-Only |
| Sinusoidal PE | RoPE（题 8） |
| Post-Norm + LayerNorm | Pre-Norm + RMSNorm（题 11、12） |
| ReLU FFN（4d） | SwiGLU FFN（~8/3 d，题 13、14） |
| Adam | AdamW（题 27） |

**【30秒答题框架】**
"整体是 Encoder-Decoder 双塔，各 N 层堆叠。Encoder 层 = 双向自注意 + FFN，Decoder 层多一个 cross-attention 且自注意带因果掩码；每个子层都有残差和 LayerNorm，万能公式是 LN(x + Sublayer(x))；入口是 Embedding 加位置编码，出口是 Linear+softmax 做词表分发。今天的 LLM 多数只用 Decoder 这条塔，把 Norm 换成 Pre-RMSNorm、FFN 换成 SwiGLU、位置编码换成 RoPE。"

**【追问与陷阱】**
- 追问：cross-attention 现在的 LLM 还有吗？——纯 Decoder-Only 架构没有显式 cross-attention，"检索内容"通过上下文窗口注入（见 RAG 章节）；但 T5/BART 类 enc-dec 仍保留。
- 追问：6 层里参数都花在哪？——FFN 约占 2/3（$8 d^2$ 每层 vs attention $4 d^2$ 每层，见题 14 推导）。
- 陷阱：以为 Add & Norm 是先 Norm 再加——那叫 Pre-Norm，原始论文是先加后 Norm。
- 陷阱：说不出 embedding 与输出层共享权重，或不知道 cross-attention 里 Q/K 来源不同。

**【知识串联】**
与 02-Inference/01-KVCache-And-Acceleration.md 的"prefill/decode 两阶段"直接相关——架构里的 causal mask 决定了 KV 可以缓存；与 03-FineTuning/01-PEFT.md 的 LoRA 相关——LoRA 冻主干、只训旁路，旁路就加在题 1 列出的这些线性层上（W^Q/W^K/W^V/W^O、FFN 的 W1/W2/W3）。

---

### 题 2：为什么 Transformer 比 RNN 好？

**【考察意图】**
验证能否把"并行化、长依赖"说成机制级理由（梯度路径、计算图结构），而不是背表格；顺便测是否知道 Transformer 付出的代价（O(n²)）。这题经常是"比较类问题"答题格式的校准题。

**【深度解析】**

四个维度逐一拆解：

1. **并行化——训练吞吐的量级差距**
   RNN 每步 $h_t = f(h_{t-1}, x_t)$，存在对 $t$ 的严格串行依赖，$T$ 步训练至少 $T$ 个串行 kernel 启动，GPU 大量算力空转。Transformer 中所有 token 的 QKV 投影、整个 $QK^T$ 都是一次 GEMM，可完全并行；训练时长从 O(T) 串行深度降到 O(log n) 深度（attention 里 softmax/reduction 的并行树深度）。
2. **长距离依赖——信号传播路径**
   RNN 两 token 交互要走 $\lvert i-j\rvert$ 步状态传递，梯度连乘 $\prod \frac{\partial h_t}{\partial h_{t-1}}$：|σ|<1 连乘即饱和（tanh/softmax 导数 ≤1 加剧），|σ|>1 爆炸。Transformer 任意两 token 在 attention 中一步直达，且残差主干（题 11、31）提供无衰减的梯度通道。
3. **计算复杂度——代价的正确表述**
   RNN: O(n·d²)（可写成 O(d²) 每步），Transformer: O(n²·d) + O(n·d²)。**n < d 时 Transformer 反而更便宜**；n 很大时二次项才成为瓶颈——所以"相比 RNN 多了 O(n²) 项"是精确说法，不是笼统的"更贵"。
4. **训练速度/收敛**：以上三条的结果——同数据量下 Transformer 用更少 wall-clock 收敛到更低 loss，任意位置信号更易形成，机器翻译 BLEU 显著提升（原始论文 28.4 WMT14 EN-DE，超所有已发表单模型）。

辩证补一句（二面加分）：Transformer 也**不是全胜**——
- Attention 是 o(n²) 显存/算力，推理需要 KV Cache（见 02-Inference 章）；
- RNN 推理每步 O(d²) 固定延迟，对超长流式任务端侧友好（如 RWKV、Mamba 这类"进化版 RNN"正在回潮，属于热点章节话题）；
- 完全并行让"任意位置都可访问"其实是**训练目标变难**（要学的依赖分布更广），所以才有 warmup、pre-norm 这些工程稳定化手段。

**【30秒答题框架】**
"核心是三点：第一，训练并行，RNN 困在时间维度的串行依赖里，Transformer 全程 GEMM；第二，长依赖，RNN 的信息交互要走 i−j 步且梯度连乘会消失/爆炸，Transformer 任意两点一步直达、残差保证梯度主干；第三，代价要说准——attention 是 O(n²d)，只在序列够长时比 RNN 的 O(d²) 贵。最后补一句 RNN 在推理端 O(1) 状态这点没输，所以现在 Mamba/RWKV 又在回潮。"

**【追问与陷阱】**
- 追问：既然 O(n²)，为何训练时没人抱怨？——训练看的是吞吐与并行度；且现代工程里 n² 已被 FlashAttention 的 IO 优化缓解（见 02-Inference 章）。
- 追问：为什么 RNN 训练慢不只是"算的少"？——串行深度导致 kernel 依赖链，GPU 利用率低，实际 wall-clock 差距远大于 FLOPs 差距。
- 陷阱：把"完全并行"绝对化——decode 阶段自回归依旧逐 token 串行；训练并行 ≠ 推理并行。
- 陷阱：只说"注意力好"不提梯度路径，二面会视为没懂。

**【知识串联】**
O(n²) 的细节题 5 推导；残差与梯度的双解释在题 11/31；RNN 回潮（Mamba/线性注意力）对应 07-HotTopics 的 2025-2026 热点；KV Cache 增长问题进入 02-Inference/01。

---

### 题 3：请详细推导 Self-Attention 的计算过程 ⭐⭐⭐⭐⭐

**【考察意图】**
本仓库标记为五星题，二面必考。验证：是否能把公式拆成"四步矩阵运算"并说清每步的张量形状；是否理解 Q/K/V 各自的角色（检索、索引、内容）；是否知道 softmax 的方向和因果掩码插入点。

**【深度解析】**

设输入序列 $X \in \mathbb{R}^{n \times d}$（n 个 token，每个 d 维）。

**第 1 步：线性投影**
$$Q = XW^Q,\quad K = XW^K,\quad V = XW^V$$
$W^Q, W^K \in \mathbb{R}^{d \times d_k}$，$W^V \in \mathbb{R}^{d \times d_v}$。多头时 $d_k = d_v = d_{model}/h$（题 6）。
**注意**：Q、K、V 来自同一个 X，但通过三套不同参数——语义上 Q 是"我在找什么"，K 是"我能被什么匹配到"（地址/索引），V 是"我实际携带的信息"（内容）。

**第 2 步：相似度打分**
$$S = QK^T \in \mathbb{R}^{n \times n}，\quad S_{ij} = q_i^T k_j$$
逐元素含义：第 i 个 query 与第 j 个 key 的内积，越大表示 token j 与 i 的"检索意图"越匹配。

**第 3 步：缩放**
$$S' = S / \sqrt{d_k}$$
原因见题 4（方差控制）。若使用因果掩码，在此步插入：
$$S'_{ij} = S'_{ij} + M_{ij},\quad M_{ij}=\begin{cases}0 & i\ge j\\ -\infty & i<j\end{cases}$$

**第 4 步：softmax（按行）**
$$A = \text{softmax}(S')，\quad A_{ij} = \frac{e^{S'_{ij}}}{\sum_{j'} e^{S'_{ij'}}}$$
softmax 沿**最后一维（key 维）**做——每个查询 i 得到一行和为 1 的权重分布 $\sum_j A_{ij}=1$。

**第 5 步：加权聚合**
$$\text{Output} = AV \in \mathbb{R}^{n \times d_v},\quad o_i = \sum_j A_{ij} v_j$$
输出位置 i 是全表 value 的凸组合，凸性由 softmax 保证。

**链接视角（二面加分）**：Self-Attention 本质是"软字典查询" query→(k,v) 的软检索：普通 dict 硬匹配一个 key，attention 以相似度 softmax 出所有 (k,v) 的加权平均，可微、可端到端学习。

**可运行参考实现（PyTorch）：**

```python
import torch
import torch.nn.functional as F

def self_attention(X, Wq, Wk, Wv, Wo, mask=None):
    """X: (n, d); Wq/Wk: (d, d_k); Wv: (d, d_v); Wo: (d_v, d)"""
    Q, K, V = X @ Wq, X @ Wk, X @ Wv          # (n, d_k) / (n, d_v)
    s = (Q @ K.T) / K.shape[-1] ** 0.5        # (n, n)，第2+3步
    if mask is not None:                       # 第3.5步：因果掩码，训练/推理一致性
        n_ = s.shape[0]
        s = s.masked_fill(torch.triu(torch.ones(n_, n_, dtype=torch.bool), 1),
                          float('-inf'))
    A = F.softmax(s, dim=-1)                   # 第4步：沿 key 维(每行)归一化
    return (A @ V) @ Wo                        # 第5步：加权聚合 + 输出投影
```

验证脚本（可直接跑）：
```python
import torch
torch.manual_seed(0)
X  = torch.randn(6, 512)
Wq, Wk, Wv = (torch.randn(512, 64) for _ in range(3))
Wo = torch.randn(64, 512)
out = self_attention(X, Wq, Wk, Wv, Wo)
assert out.shape == (6, 512)
print("OK", out.shape)   # OK torch.Size([6, 512])
```

复杂度：空间为 $S=A\in\mathbb{R}^{n\times n}$；naive 实现显存 $O(n^2)$，FlashAttention 改 IO 顺序后 $O(n)$（详见 02-Inference/01）。

**【30秒答题框架】**
"五步：X 先过三套权重得 Q/K/V；QK^T 算 pairwise 相似度并除以 √d_k；若有因果掩码在这加 −∞ 上三角；沿 key 维做 softmax 得到每行和为 1 的注意力矩阵 A；最后 AV 加权求和、过 W^O 出来，形状回到 (n, d)。语义上就是可微的软字典查询。"

**【追问与陷阱】**
- 追问：softmax 为什么在一维（dim=-1）？——因为权重分布是"对每个 query 的"按 key 归一；弄成 dim=-2 会得到列归一（每个 key 被消费的总量=1），那时输出与 query 强度无关，语义错。
- 追问：为什么 Q、K 不共用同一个矩阵？——q·k 需要不对称的打分空间；且没有理由让"查询集合"与"被查集合"同分布。
- 追问：能不能不用 X 投影、直接拿 embedding 当 QKV？——可以删掉 W^Q（如远连接实验），但表达力大幅受限；三投影是标准选择。
- 陷阱：把"除以 √d_k"算在 softmax 之后，或忘了掩码加在 softmax 之前。
- 陷阱：写出 attention 不带 W^O——少了输出混合层的 attention 是"每头独立"的，多头融合靠 W^O。

**【知识串联】**
√d_k 详见题 4；多头拆解详见题 6；因果掩码独立成题 17；本实现是 02-Inference/01（KVCache、FlashAttention、MQA/GQA）一切推理优化的基线；题 26 的交叉熵在分类 logits 上的 softmax 是同一函数。

---

### 题 4：为什么要除以 √d_k？⭐（方差-数值稳定性双解释）⭐⭐⭐⭐⭐

**【考察意图】**
金字塔尖的深挖题。一面答"简化点积大小"即可；二面面试官要听**两个独立解释互证**：方差推导（保持 softmax 输入的统计量）+ 梯度解释（one-hot 化导致梯度消失），并要能回应"为什么是 √d_k 而不是 d_k"这一陷阱题。

**【深度解析】**

**解释一：方差推导（保持输入分布）**

原论文 3.2.1 节理由原话："to illustrate why the dot products get large, assume the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product has mean 0 and variance d_k."

完整推导（不跳步）：
设 $q_i, k_i$ 独立、$E[q_i k_i]=0$、$\text{Var}(q_i k_i) = E[q_i^2]E[k_i^2] = 1$（使用独立性）。
$$\text{Var}(q\cdot k) = \text{Var}\!\Big(\textstyle\sum_{i=1}^{d_k} q_i k_i\Big) \overset{\text{独立}}{=} \sum_{i=1}^{d_k}\text{Var}(q_i k_i) = d_k$$
内积标准差量级 $\sqrt{d_k}$。
除以 $\sqrt{d_k}$ 后：$\text{Var}\!\big(\tfrac{q\cdot k}{\sqrt{d_k}}\big)=\tfrac{d_k}{d_k}=1$——**把 logit 的尺度与 d_k 解耦**。这是"方差保持（variance-preserving）"设计，与 Xavier 初始化思想同源（$\text{Var}(W_{ij})=2/(n_{in}+n_{out})$ 同样追求前向方差不变）。

**解释二：softmax 梯度解释（为什么要"方差=1"）**

softmax 的雅可比为 $J = \text{diag}(A) - AA^T$（逐行），其元素量级 ≤ $A_j(1-A_j)$。若输入 logit 尺度 $\sim\!\sqrt{d_k}$，间距也被放大，一行内的相对差距 $\Delta$ 量级 $\sqrt{d_k}$：
$$\frac{A_j}{A_i} = e^{\Delta},\quad \Delta \sim \sqrt{d_k}(\text{随机量级})$$
以 $d_k=64$：典型差距十级，$e^{-10}\!\approx\!4.5\times10^{-5}$，近乎 one-hot。此时：
- **前向**退化为 argmax——软字典查询变成硬查表，多 token 加权平均的"信息融合"能力丢失（一次只能消化一个 token）；
- **反向**雅可比 $J\to\mathbf{0}$（$\text{diag}(A)$ 与 $AA^T$ 两项相消），梯度 ≤1e−4 量级，训练几乎停滞；
- **数值**上 fp16 可表示的最大数 65504 = e^{11.09}，logit 超过 11 就 exp 溢出 inf→softmax 变 NaN。

除以 √d_k 把 Δ 控制在 O(1)，A 保持软分布，同时三面（表达力/梯度/数值）稳定。

**为什么不是除以 d_k？（陷阱必答）**
- 方差层面：$\text{Var}(q\cdot k)=d_k$，除以 $d_k$ 得方差 $1/d_k\to 0$——logit 全部挤到 0 附近，softmax 趋向**均匀分布**，同样梯度消失，且前向变成"平均主义"。
- 信息层面：q·k 中有真实信号，除 $\sqrt{d_k}$ 只去掉 d_k 引起的"维度膨胀"，保留信号对比度；除 d_k 会把信号一起压扁。

**工程补充**（数值稳定三件套之一）：缩放只是第一道防线；训练中还有 (a) log-softmax 稳定化（减 $\max$，见题 25/32）、(b) attention-logits 的 clip（如 Claude 论文的 attn 软截断）、(c) QK-Norm（对 q、k 各自做 RMSNorm 再点积，保证 logit 不会因数据长尾爆炸——现代超长上下文训练常用）。

**一句话双解释总纲**：
> 除 √d_k 让 logit 的方差与维度无关（解释一：前向分布正确），从而 softmax 既不过尖（保梯度）也不过平（保表达）（解释二：反向训练可行）。

**量化例子**（面试可口算）：$d_k=128$，不缩放时 $\text{std}(\text{logit})=\sqrt{128}\approx 11.3$；两个独立 logit 的差期望在 σ 量级（≈10+）。缩放后 std=1，差 ~1.5，竞争是"软"的。

**【30秒答题框架】**
"两句话双解释。第一，方差：q·k 每一项都是单位方差量的乘积，独立求和方差是 d_k，除以 √d_k 方差回到 1，让 logit 尺度和维度解耦——这是 Xavier 式的方差保持。第二，梯度：logit 尺度大时 softmax 相对间距被放大 e^{√d_k} 倍，输出趋近 one-hot，雅可比 diag(A)−AAᵀ 趋零、梯度消失，前向也退化成硬查表；缩放后保持软分布。追问为什么不是除 d_k：那就方差到 1/d_k，softmax 变均匀，同样死。"

**【追问与陷阱】**
- 追问（必问）：所以除 √d_k 是不是在把 attention "温度"调到 1？——对，等价于温度 τ=√d_k 的 softmax；后训练采样温度是同一门的另一面（见 02-Inference/02 解码）。
- 追问：q、k 已经过 W^Q/W^K，权重的初始化（Xavier/kaiming）能不能自动把方差压住？——线性投影有压缩作用，但训练中权重会漂移、且长 tail token 的 embedding 模长大（某些位置 norm 抖动），并不稳；QK-Norm 更直接。
- 追问：如果改用加性 attention 呢？——原论文也试过，加性在 d_k 小时相当，d_k 大时点积更快（BLAS 矩阵乘）；加性不必除，但速度差。
- 追问：KV 大 head 时要重新推导？——GQA/MQA 里 d_k=d_model/h 变小，如 7B 模型 4096/32=128 不变；但 MoE 里 head 数变化后 d_k 变小、√d_k 缩放系数跟着变。
- 陷阱：只背"方差 d_k"说不出两个独立解释；或说"防止 exp 溢出"却给不出 65504=fp16 max 这类锚点。

**【知识串联】**
方差保持思想与 Xavier 初始化（题 31）、Pre-Norm 的作用（题 11）一线；softmax 数值稳定与题 25/32 的 log-sum-exp 是同一招；"温度"概念接到 02-Inference/02 的采样；QK-Norm 属于 07-HotTopics 长上下文训练技巧。

---

### 题 5：Self-Attention 的计算复杂度是多少？（FLOPs≈2×参数量×token 数）⭐⭐⭐⭐⭐

**【考察意图】**
验证能否**从矩阵乘的 MAC 推出 attention 每个部件的 FLOPs**，并进一步跳出 attention，给出整个 LLM 的算力预算公式；这题是后续显存/算力估算、推理成本谈话的基础。二面常从"attention 是 O(n²) 吧？"开始，一路追问到"训练 7B 模型 1T token 要多少卡时"。

**【深度解析】**

**基元：矩阵乘的 FLOPs**
$\underbrace{(n \times d)}_{A}\ \cdot\ \underbrace{(d \times k)}_{B}\ \Rightarrow\ n\times k$ 输出，每个输出元素需要 d 次 MAC（乘加）。
**1 MAC = 1 乘 + 1 加 = 2 FLOPs**（工程口径）。故
$$\text{FLOPs}(A\cdot B) = 2 \cdot n \cdot k \cdot d$$

**Self-Attention 单层逐项 FLOPs（单头，n 序列长、d_model=d、d_k=d/h）**

| 操作 | 形状 | FLOPs（=2×n×…×…） |
|---|---|---|
| QKV 投影（三合并看） | $n\times d \cdot d\times 3d_k$ | $6\,n\,d\,d_k$ |
| $QK^T$ | $n\times d_k \cdot d_k \times n$ | $2\,n^2 d_k$ |
| softmax | $n^2$ 元素 | $O(n^2)$（相对可忽略） |
| $AV$ | $n\times n \cdot n\times d_v$ | $2\,n^2 d_v$ |
| 输出投影 $W^O$ | $n\times d_v \cdot d_v \times d$ | $2\,n\,d_v d$ |
| **小计（h 头，$d_k=d_v=d/h$）** | | $8\,n d^2 / h\cdot h \to$ **$O(4 n d^2 + 4 n^2 d)$** |

这里固定 $d_k$ 合并后：QKV+输出投影合计 $2nd\cdot(d/h)\cdot h \times 2 = 4nd^2$（h 头合并回 d 维），score/加权合计 $4n^2 d$。即 **单层 MHA：$4nd^2$（与 n 线性部分）+ $4n^2d$（与 n 二次部分）**。

**FFN 单层**：两个矩阵 $d\to d_{ff}\to d$：
$$\text{FLOPs}_{FFN} = 2\,n\,d\,d_{ff} + 2\,n\,d_{ff}\,d = 4\,n\,d\,d_{ff}$$
原始 $d_{ff}=4d$：$16\,nd^2$。SwiGLU 三矩阵（题 13）：$3\cdot2\,n\,d\,d_{ff}\to 6\,n\,d\,d_{ff}$，取 $d_{ff}=\tfrac{8}{3}d$ 抵平参数。

**每层合计（ReLU-FFN 型）：attention $4nd^2+4n^2d$ + FFN $16nd^2$ = $20\,nd^2+4n^2d$**，其中 attention 只占 1/5（FFN 占 4/5 的线性计算）——所以"attention 很贵"只在 n² 项主导时成立，即 n > 5d 时。

**全模型 & 量级锚点（n≪d 的常识区间）**
- 单层 ≈ $12\,d^2$ 参数（attention 4d² + FFN 8d²）+ 少量 norm/LN 参数；L 层总参数 ≈ $12\,L d^2$（不含 embedding/head）。LLaMA-7B：L=32、d=4096 → $12\cdot32\cdot4096^2\approx 6.44$ B，加 embedding+lm head（32000×4096×2≈0.26 B）≈ 6.7 B，与论文对上。
- **训练总 FLOPs ≈ 6·N·D**（N 参数量、D token 数）：前向每个"被激活"参数 1 MAC=2 FLOPs ⇒ 2ND；反向约为前向 2 倍 ⇒ +4ND，总计 6ND（经验常数 3~4 ND/前反向不确定，6 是 Kaplan et al. 2020 的标准口径）。
- 举例：7B×1T tokens ⇒ $6\times6.7\text{e}9\times10^{12}\approx 4\times10^{22}$ FLOPs。A100 (312 TFLOPS bf16 峰值，实际 MFU≈40% ⇒ 125 TFLOPS 有效)：$\tfrac{4\times10^{22}}{1.25\times10^{14}}\approx 3.2\times10^8\,\text{s}$ ÷ 86400 ≈ **3720 GPU·天**（数量级估算，环形并行加损耗会更高）。

**推理/显存侧（引出 02-Inference）**
- 推理单 token forward ≈ $2N$ FLOPs（7B ⇒ ~14 GFLOPs）。
- Decode 是**带宽瓶颈而非算力瓶颈**：每步要从 HBM 读全部权重+KV Cache，7B bf16 权重 14GB，A100 HBM 2TB/s ⇒ 理论 ≥7ms/步；FLOPs 只要 14e9/312e12≈0.05ms——**GPU 空转 99%**，这是所有 decode 优化的出发点。
- naive attention 激活显存：score $n\times n$（单头 fp16）：4096² ×2B ≈ 32 MiB/头；32 头×32 层 ≈ 33+ GB——FlashAttention 用 tiling 把它变 O(n)（省 ~1000×）。

**复杂度总结表（面试引用频率最高的表）**

| 组件 | 训练 FLOPs | 显存（激活，bf16） | 备注 |
|---|---|---|---|
| QKV 投影 ×3 | $6nd^2$ | $3nd$ | |
| score=QKᵀ | $2n^2d$ | $n^2 h$（naive）/ O(n)（flash） | 二次项来源 |
| softmax | $O(n^2)$ | $n^2 h$ | |
| AV | $2n^2d$ | $nd$ | |
| 输出投影 | $2nd^2$ | $nd$ | |
| FFN(4d) | $16nd^2$ | $8nd$ | |
| **单层合计** | $20nd^2+4n^2d$ | | |
| **全模型训练** | ≈ **6ND**（经验口径） | | N=参数，D=token |

**【30秒答题框架】**
"分三层说：一层内，QKV+输出投影是 4nd²，score 和 AV 各 2n²d，FFN 是 4·d·dff·n，ReLU 版 4d 时 16nd²，单层合计 20nd²+4n²d——attention 只占两成，FFN 是大头。全模型就是 12d²×L 参数，训练 FLOPs 用经验公式 6ND，2 是前向（每参数一次乘加）、乘 3 覆盖反向。推理 decode 是带宽 bound，读权重 14GB÷2TB/s≈7ms 一步，算力只用到 0.1%。"

**【追问与陷阱】**
- 追问：4n²d 是不是"总复杂度"？——粗口径 OK（O(n²d) 主导长序列），精确要加 4nd² 项；n<5d 时其实 FFN 更贵。
- 追问：为什么是 6ND 不是 2ND？——前向 2ND（每参数一次乘加）；反向"对输入的梯度"重算 2ND、"对权重的梯度"再 2ND，共 4ND；总 6ND。
- 追问：MFU 是什么、为什么只用 40%？——Model FLOPs Utilization=实际 FLOPs/(理论峰值)。损失来自访存瓶颈、kernel 调度、通信等。
- 陷阱：把 1 MAC 当 1 FLOP（差 2 倍）；或把 n² 项写成 O(n²) 而没有把 d 一起带出（比较跨层复杂度时 O(n²d) 才严谨）。
- 陷阱：混淆"参数量"与"FLOPs"——SwiGLU 用 3 个矩阵，计算 ×1.5 于同 d_ff；参数等价要 d_ff=2d/3·4d。

**【知识串联】**
6ND 公式与 09-SystemDesign 的训练成本/卡时估算同源；KV 显存公式在 02-Inference/01 展开（MQA/GQA/MLA 的动机、PagedAttention）；FlashAttention 属 02-Inference/01；GQA/MLA 显存推导即题 7。

---

### 题 6：Multi-Head Attention 的原理和作用 ⭐⭐⭐⭐⭐（子空间多样性 + 低秩代价）

**【考察意图】**
二面核心题。要听出三层：(1) 公式与参数量（"与单头相同"为什么成立）；(2) 收益的机制解释（多个低秩子空间并行 = 表达能力增量）；(3) 能否从"为什么 h 头合计参数等于单大 head"反推出本质是"矩阵分块并行"。

**【深度解析】**

**公式：**
$$\text{MultiHead}(Q,K,V)=\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$$
$$\text{head}_i=\text{Attention}(QW_i^Q,\,KW_i^K,\,VW_i^V)$$

**参数量恒等（必须能当场推）**：
单大头（$d_k=d_v=d$）：$W^Q,W^K,W^V$ 各 $d\times d$ + $W^O(d\times d)$ = $4d^2$。
多头（$d_k=d_v=d/h$，h 个头）：每个头 $3\times d\times(d/h)$，h 个头共 $3d^2/h\cdot h=3d^2$；$W^O$ 是 $d\times d$（concat 后投影回去）。合计 **4d²**——**h 头"h 头并行"等价于把一个大矩阵切 h 列做独立小 attention，再由 W^O 融合**。数学上等价于：
$$\text{MHA}(X)=\big[\!|_{i=1}^h \text{Attention}(XW_i^Q,\,XW_i^K,\,XW_i^V)\big] W^O$$

**收益来源：子空间多样性（subspace diversity）**
每个头在 $d/h$ 维低秩子空间中做 attention——原论文 3.2.2 原话："allows the model to jointly attend to information from different representation subspaces at different positions."
- 单个大 head 的 $QK^T$ 秩最高 $d$，但 softmax 后的**有效通信带宽**由注意力分布的熵决定，一个大 softmax 只能表达一个"平均"的检索模式；
- h 个独立小 softmax = h 个不同的检索假设**并行推理**，由 $W^O$ 线性组合——如同集成（ensemble），且各头输入子空间正交化（不同 $W_i$ 列空间）；
- 实证（Michel et al. 2019、Voita et al. 2019）：多数头可剪枝而 quality 变化很小，但少数"induction head"（Olsson et al. 2022）承担 in-context copying / 检索等关键功能，删了 ICL 崩——**头分工确实发生且有长尾**。
- 头的功能分类（面试可引用）：位置性头（只看前 1 个 token）、句法依存头、核心头（跨全句广播）、induction head（看"上一个 A 后面的 B"去预测当前 A 之后那个 token）。

**计算量与单头相同**（小 d_k 的 score 矩阵面积 n×d 一样）——这也是多头在"不影响 FLOPs"前提下白赚表达力的原因。

**代价**（二面必追问）：
- KV Cache 随 h 线性膨胀：$2\cdot n\cdot h\cdot d_k = 2\,n\,d$（每层）——多头带来的每层 K/V 存 $n\times d$×2，是 decode 带宽瓶颈的根源 ⇒ MQA/GQA/MLA（题 7）由此演进；
- h 越大 d_k 越小，单头"软检索"精细度下降（softmax 尖峰需要足够维度——题 4 的方差视角）；经验 d_k≥64 比较安全（LLaMA h=32, d_k=128；d_k<64 的模型头效变差）。

**一图流概括**：

```
一个大头： 一个 softmax（检索空间 = d 维全空间）
h 个小头： h 个 softmax（并行于 h 个 d/h 维子空间）→ W^O 线性融合
表达力：  集成效应 + 多视角 ≫ 单头，参数量与 FLOPs 不变
代价：    KV cache ×h 头（这是 MQA/GQA/MLA 要解决的事）
```

**【30秒答题框架】**
"公式是各头 Attention 后 concat 再过 W^O。参数量恒等式必须会推：单大头 4d²，h 小头每个 d²/h×3、合计还是 3d²，加 W^O 4d²——所以多头本质是把一个 d 维 attention 切成 h 个低秩子空间并行做，再线性融合。收益是集成：不同头学不同检索模式（句法/位置/induction head都有实证），FLOPs 没变。代价是 KV cache 随头数线性增长，这就是 MQA/GQA/MLA 演进的出发点。"

**【追问与陷阱】**
- 追问：为什么 W^O 是 d×d，而单头是 d×d/h 也能等价 reshape？——因为 concat 起来再乘 d×d，数学上等价于 h 个 d×(d/h) 的独立 O 上再 concat；W^O 吸收了"头间混合"。
- 追问：head 数和 d_k 怎么权衡？——参考文献结论：h 增加、d_k 减小，过小（<64）时 attention 会"模糊"；太少的头（h<4）又瓶颈。LLaMA h=32/d_k=128、GPT-3 96 头/128。
- 追问：能只保留 1 个头吗？——性能下降但并非不可用（小模型可以）；h 是一种"宽"的分配方式，和深度 trade。
- 陷阱：以为多头增参数/增 FLOPs——参数、FLOPs 都不变（恒等式会当场算错很掉分）。
- 陷阱：把多头好处说成"防止过拟合"——机制是表达能力强 + 子空间多样性，不是正则。

**【知识串联】**
KV cache 显存与带宽瓶颈→题 7 的 MQA/GQA/MLA 与 02-Inference/01；头分工与 induction heads→ 03-FineTuning 训练动态/07-HotTopics 的机理分析；W^O 融合是 rank-受约束的 LoRA 思想的远亲（03-FineTuning/01）。

---

### 题 7：MHA、MQA、GQA、MLA 的区别 ⭐⭐⭐⭐（KV Cache 显存公式 + 演进逻辑）

**【考察意图】**
二面工程深水区。考三件事：(1) 是否能推导每种的 KV cache 公式（而不是背"小、中、大"）；(2) 是否明白**演进的真实驱动力是 decode 阶段的 HBM 带宽**；(3) MLA 是否理解到"低秩压缩 + 旋转位置耦合"这层——区分背表格和真懂的分水岭。

**【深度解析】**

**0. 为什么这个问题重要（先立靶）**
Decode 阶段每个 token 都要读完整 KV cache（自回归，历史 KV 不能丢）。KV cache 大小：
$$M_{KV} = 2\ (\text{K},\text{V}) \cdot L \cdot n_kv \cdot d_h \cdot s \cdot \underline{\text{bytes}}$$
（L 层数、n_kv=KV 头数、d_h 每头维度、s 已生成 token 序列长）。
以 LLaMA-2-70B（L=80, n_kv=8 GQA, d_h=128, bf16）1 token：$2\times80\times8\times128\times2\,\text{B}=320\,\text{KiB}$；128k 上下文 ⇒ **320 KiB×131,072 ≈ 40 GiB**（≈42.9 GB，约为 bf16 权重 140GB 的三成），且每步都要从 HBM 全量读 → 这是解码瓶颈。

**1. MHA（标准多头）**
n_kv=h。LLaMA-2-70B 若不压缩：$2\cdot80\cdot64\cdot128\cdot2$B=2.56 MiB/token，128k 上下文 320 GiB——不可用。质量最好但 cache 最贵。
代表：GPT-2、BERT、LLaMA-1。

**2. MQA（Multi-Query Attention，Shazeer 2019）**
n_kv=1：所有 Q 头共享同一 K/V 头。KV cache 缩为 MHA 的 1/h。
- 好处：decode 每步读 KV 的字节数降 h 倍（64 头 → 1/64），带宽 bound 的 decode 大幅提速；
- 代价：所有头挤一个 (k,v) 空间——表达能力显著下降，质量损失实测 MQA 只能在小 / 中模型上被接受；
- 代表：PaLM、Falcon、Gemini 的部分配置。

**3. GQA（Grouped-Query Attention，Ainslie et al. 2023）**
n_kv=g（1<g<h）：h 个 Q 头分 g 组共享 KV。质量与 cache 的**插值**点：
- cache 比 MQA 大 g 倍但比 MHA 小 h/g 倍；
- LLaMA-2-70B 用 g=8：cache=8/64=1/8 MHA（对 128k 上下文：MHA≈320 GiB → GQA≈40 GiB），质量几乎无损（GQA 论文：质量退化在实验误差内、显著优于 MQA）；
- 还可以**从已训 MHA checkpoint 上游"uptraining"** 把 K/V 权重 mean-pool 到 g 组（省重训）。
代表：LLaMA-2/3 70B、Mistral、Qwen。

**4. MLA（Multi-head Latent Attention，DeepSeek-V2 2024）**
思路质变：**不是减头数，而是把 KV 压到低秩潜空间**。
$$c_t^{KV} = W^{DKV}\, h_t \in \mathbb{R}^{d_c}\quad(\text{每层每 token 只缓存 } c_t)$$
需要时再上投影：$k_t = W^{UK} c_t,\ v_t = W^{UV} c_t$。
- DeepSeek-V2 配置：$d_c=512$、RoPE 部分 $d_h^R=64$、60 层。**每层每 token 只缓存 512+64=576 个标量**（fp16=1152 B），vs GQA-8 的 $2\times 8\times 128=2048$（fp16 4096 B）——**每 token KV 降到 GQA 的 ~28%、MHA 的 ~3.5%**；论文口径"KV cache 降 93.3%、生成吞吐提升 5.76×"。
- 关键细节（区分懂与不懂）：RoPE 位置信息与低秩压缩**耦合冲突**（旋转矩阵作用在 k 的特定 2D 平面上，从 c 恢复后无法"先压缩再旋转"），所以 DeepSeek 把 k 拆成 content 部分（可压缩，$W^{UK}c$）+ rope 部分（不可压缩，单独存 $k_t^R$ 每头 64 维）；
- 等价技巧：训练时把上投影吸收进 attention 内积（$W^{UK} c$ 直接和 $W^{UQ}$…的计算融进 q·k），使前向 FLOPs 与 MHA 同量级；
- 训练/推理显存结构：激活也比 GQA 小（不必物化 2048 维 K/V），代价是上投影引入少量计算与一些 kernel 融合工程。
- 代表：DeepSeek-V2/V3、DeepSeek-R1 系列。

**对比总表（当场给数量级的表）**

| 方案 | KV头数 | KV cache/层/token(elems) | 相对 MHA | 质量 | 代表 |
|---|---|---|---|---|---|
| MHA | h | $2hd_h$（如 2×64×128=16384） | 1× | 基准 | GPT-2/BERT |
| MQA | 1 | $2d_h$ | 1/h | 最差 | PaLM/Falcon |
| GQA | g | $2gd_h$（8×128×2=2048） | g/h (1/8) | 接近 MHA | LLaMA-2/3 70B |
| MLA | h（但压缩） | $d_c + d_h^R$（512+64=576） | ~3.5% (V2配置) | 略优于 GQA¹ | DeepSeek-V2/V3 |

¹DeepSeek-V2 论文报告 MLA 质量略好于 GQA 同基线。

**演进的统一叙事（高分答案的"红线")**：
> 显存与带宽瓶颈在"每 token 的 K/V 元素数"。MQA 砍头 ⇒ 质量损失；GQA 分组 ⇒ 折中；MLA 跳出头数框架做低秩 ⇒ 把"K/V 元素数"从 $O(h\,d_h)$ 变成 $O(d_c)$，$d_c<d_h\cdot h$ 时质量靠共享潜变量保持。

**KV cache 计算小程序（可直接跑）**：
```python
def kv_cache_bytes_per_token(n_layers, n_kv_heads, head_dim, dtype_bytes=2):
    """K 与 V 各存 n_kv_heads×head_dim。MLA: n_kv_heads 置 1、head_dim=d_c+d_h^R"""
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes

print(kv_cache_bytes_per_token(80, 64, 128) / 1024)   # 2560.0 KiB/token (MHA@70B)
print(kv_cache_bytes_per_token(80, 8, 128)  / 1024)   #  320.0 KiB/token (GQA-8@70B)
# MLA@V2：缓存 c_kv(512) + k_rope(64) = 576 个元素、各存 1 份
# （K/V 联合压缩后同源，不再有"K 一份 V 一份"的 ×2，公式里那个 2 就此消失）
print(60 * 1 * 576 * 2 / 1024)                        #   67.5 KiB/token (MLA@V2)
# 128k 上下文：MHA≈320 GiB → GQA≈40 GiB → MLA≈8.4 GiB（V2 L=60 口径；V3 L=61 则 ≈8.6 GiB）
```
等等——MHA@70B = 2×80×64×128×2 = 2,621,440 B = 2560 KiB；GQA = 320 KiB；MLA(576) = 67.5 KiB。128k 上下文：MHA≈320 GiB、GQA≈40 GiB、MLA≈8.4 GiB；若按十进制 GB 口径则分别 ≈344 GB、≈42.9 GB、≈9.0 GB（此前把十进制数直接标成 GiB 会整体虚高约 7%，报数前先声明口径）。**这组数字务必自己复算一遍再出口。**

**【30秒答题框架】**
"出发点都是 KV cache：公式 2·L·n_kv·d_h·s·bytes。MHA 最贵；MQA 全共享 1 个 KV 头，cache 缩 h 倍但质量掉；GQA 分 g 组共享，LLaMA-2-70B g=8、cache 省 8 倍质量几乎不掉；MLA 是 DeepSeek-V2 的低秩压缩路线，每 token 只存 d_c=512+64 的潜码，cache 缩到 GQA 的约 28%、论文口径 93.3% 降幅、吞吐 5.76×；难点是 RoPE 与压缩耦合，要把位置项单独引出。"

**【追问与陷阱】**
- 追问：为什么不是干脆"复用一份 K/V 只变 Q 头数"等于 MQA？——MQA 就是 n_kv=1 的 GQA 特例；GQA 保留了多 KV 头 的表达多样性。
- 追问：MLA 的 cache 能再压（fp8/量化）吗？——可叠 KV 量化，但 MLA 已经把冗余以"低秩"方式压掉，量化收益空间变小（见 02-Inference/03）。
- 追问：MQA 为什么质量差那么多？——所有 Q 头看同一个 (K,V) 子空间，头间分工被强制同态；GQA 保留 g 份多样性。
- 追问：prefill 阶段受益一样吗？——prefill 也省 KV 写入与读取，但 prefill 是计算/算力 bound，收益占比与 decode 不同。
- 陷阱：背"MLA=压缩"但说不出 d_c、d_h^R 的量级或 RoPE 耦合问题。
- 陷阱：把 KV cache 公式里的"2"（K/V 两份）丢掉，或单位混用 bits/Bytes。

**【知识串联】**
decode 带宽 bound 的完整论证在 02-Inference/01-KVCache-And-Acceleration.md（含 PagedAttention、continuous batching）；KV 量化接 02-Inference/03-Quantization；MLA 与 LoRA 的"低秩共享"思想同构（03-FineTuning/01-PEFT）。

---

### 题 8：RoPE（旋转位置编码）的原理 ⭐⭐⭐⭐（复数推导 + 外推）

**【考察意图】**
RoPE 是现代 LLM 的默认位置编码，区分"知道名字"与"能推公式"的分水岭。面试官想听：绝对/相对位置的权衡、**复数/旋转矩阵推导的完整性**（"内积只依赖相对位置"必须能两行证出）、与线性注意力的兼容性、以及外推（PI/NTK/YaRN）能否接上。

**【深度解析】**

**设计目标**：希望存在函数对 $<f(q,m), f(k,n)>$ **只依赖相对位置 m−n**（不依赖绝对 m, n），同时 (a) 无参数、(b) 可外推、(c) 与线性注意力兼容（内积前对 q/k 各自做标量变换）。

**二维情形的复数推导（核心，必须能黑板上走）**

把 $q, k\in\mathbb{R}^2$ 看作复数 $q, k \in \mathbb{C}$。定义带位置的编码：
$$f(q,\,m) = q^{\,}\! e^{im\theta}\quad(\text{复数乘法 = 幅角旋转 } m\theta)$$
……写成实数矩阵形式就是：
$$f(q,m) = R_m q,\qquad R_m=\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}$$

**关键性质一行证明**：
$$\langle f(q,m),\, f(k,n) \rangle = \Re\!\big[(q e^{im\theta})\,\overline{k e^{in\theta}}\big] = \Re\!\big[q\bar{k}\,e^{i(m-n)\theta}\big] = \Re\big[q\bar{k}\,e^{-i(n-m)\theta}\big]$$
**只含 (m−n)**——这就是"内积只依赖于相对位置"。等价实数形式（原论文式）：
$$\langle R_m q,\ R_n k\rangle = q^T R_m^T R_n\,k = q^T R_{n-m}\,k$$
（用到 $R^T R = I$，$R_{-m}=R_m^T$——旋转矩阵的正交性，这里就是 n−m 相对相位的全部来源。）

**高维提升**：把 $q \in \mathbb{R}^d$ 切成 $d/2$ 个二维平面（如 $(x_0,x_1), (x_2,x_3),\dots$），每个平面配不同频率 $\theta_j$。主流配置（同 RoFormer）：
$$\theta_j = 10000^{-2j/d},\qquad j = 0,1,\dots,d/2-1$$
即**第 0 组频率最高（旋转最快、能分辨近邻 token）、第 d/2−1 组最慢（波长 2π·10000≈6 万 token，编码远程位置）**。例 d=128 时 $j=d/2-1=63$：$\theta=10000^{-126/128}\approx1.16\times10^{-4}$、波长 ≈ 54,000 token——*这是"RoPE 天然支持长距离"的空间结构*。

**实现代价**：$O(n\cdot d)$ 一次高性能逐元素 + 旋转（无矩阵乘）：预计算 $\cos(m\theta_j),\sin(m\theta_j)$ 表，q 的每 2 维做
$$\begin{pmatrix}q'_{2j}\\ q'_{2j+1}\end{pmatrix} = \begin{pmatrix}\cos m\theta_j & -\sin m\theta_j\\ \sin m\theta_j & \cos m\theta_j\end{pmatrix}\begin{pmatrix}q_{2j}\\ q_{2j+1}\end{pmatrix}$$

**为什么"外推性好但有上限"**：旋转是**连续、有界** 的相位函数，不像 learned PE（只见过 index 内）那样"没见过就乱码"。但**频率**$\theta_j$ 决定"多远的 token 才有可分辨相位"——训练上下文 $L$ 决定哪些 $\theta_j$ 被"充分转动过"，长于 $L$ 的远距离相位（慢频率）训练时几乎没变化 → 外推变差。**因此 RoPE 有原生外推能力，但不完美**——这是 PI/NTK/YaRN 的动机。

**外推三条路线（面试分水印）**：
1. **PI (Position Interpolation，2023)**：训练 4k、推理 32k 时把 $m$ 换成 $m/(32768/4096)=m/8$——把"大 m"压回训练见过的小区间，代价是**近邻 token 的相位差也被压缩 8 倍**，局部精细度受损；论文报告仅需约 1000 步长文本微调即可恢复（LLaMA 2k→32k 的实验口径），YaRN 进一步优化此成本曲线。
2. **NTK-aware（在 PI 与直接外推间插值）**：不动 m，改**频率基**（base 10000 → base'$=10000\cdot s^{d/(d-2)}$）。效果：低频分量（长程需要）缩放得多，最高频分量几乎不动（局部 token 区分度保住）——"频域选择性拉伸"。免训练即可 2×以上外推，7B 级模型可接受。
3. **YaRN（2023/2024）**：NTK 由频率插值 + attention logit 温度修正 组合，论文报告**千步级**微调即可扩展到 128k，且困惑度损失最小；是目前长上下文主流方案（多数开源 128k 模型都用了 YaRN 或其变体）。

**【30秒答题框架】**
"RoPE 把 q、k 每两维当复数，乘 e^{imθ} 做位置 m 的旋转；内积里两旋转共轭相消，只剩 e^{i(m−n)θ}——所以内积天然只依赖相对位置。高维是把 d 维切 d/2 个平面、频率 θ_j=10000^{−2j/d}，波长从 2π 到约 5 万 token、远近位置都有相位差覆盖。零参数、与线性注意力兼容。但外推不完美：长相位没训过，所以线上用 PI（缩位置）、NTK（缩 base，频域分级）、YaRN（NTK+温度修正，千步级微调扩展到 128k）。"

**【追问与陷阱】**
- 追问：为什么 θ_j 用 10000 基？——与 sinusoidal 对齐，2π·10000 ≈ 62k 远波；调 base 就是 NTK 方法的支点。
- 追问：RoPE 怎么与线性注意力兼容？——线性注意力 $E[\phi(q_i)^\top\phi(k_j)v_j]$ 同样由内积驱动，q/k 的各自标量旋转内嵌进特征即可，比加式 PE 干净。
- 追问：MLA 里为什么 RoPE 麻烦？——k 需要带位置旋转再压缩；若先压缩后旋转，旋转矩阵不可通过 c 线性表示（R_m 与 W 低秩合成不可交换），所以 DeepSeek 单独保存 rope 部分 64 维（题 7）。
- 追问：ALiBi 和 RoPE 谁好？——偏任务/模型；ALiBi 免参数免插值、远程衰减显式，RoPE 更通用且 YaRN 后外推不输；现代主流是 RoPE。
- 陷阱：说"RoPE 免微调可外推任意长"——错误，需要 PI/NTK/YaRN；纯外推（Extrapolation）是无损插入位置不调参，PI/YaRN 属于"插值+微调"。
- 陷阱：算不出 $R_m^T R_n = R_{n-m}$（正交性），或忘了切 2D 平面这件事。
- 陷阱：混淆"位置插值（缩 m）"与"base 插值（缩 θ）"。

**【知识串联】**
RoPE 与 MLA 的耦合在题 7（题 7 的 rope 通道细节可直接引用这里）；长上下文/位置编码微调对应 03-FineTuning 与 07-HotTopics；位置编码外推失败的工程表现（len extrapolation abrupt failure）见 02-Inference 长上下文一致。

---

### 题 9：各种位置编码对比

**【考察意图】**
考察"设计谱系"的认知——不是背表，而是把"绝对 vs 相对"背后的**参数量、外推、与 attention 兼容**三个设计轴总结出来，并说出主流选择（RoPE）与其对手。

**【深度解析】**

**一、Sinusoidal（原论文）**：
$$PE_{(pos,2i)}=\sin(pos/10000^{2i/d}),\qquad PE_{(pos,2i+1)}=\cos(pos/10000^{2i/d})$$
零参数；数学性质：$PE_{pos+k}$ 是 $PE_{pos}$ 的线性函数（因为 sin/cos 的差分公式 → "相对位置可达"——但这是**输入层注入**相对性，而 attention 内积并不自动保留它）。外推性：形式上可无限延展，实际（每维相位重复、长程反正弦歧义）较差。

**二、Learned Absolute PE（GPT-2/BERT）**：参数量 $L_{max}\times d$（GPT-2 1024×768≈0.8M）。外推：**结构性不可能**——超过 $L_{max}$ 无 embedding。优点：表达自由（可学到绝对先验如"句子开头"）。

**三、ALiBi（Press et al. 2022）**：
attention logit 加 $-m\cdot(i-j)$（因果，i≥j），m 是头相关斜率：$m_h=2^{-8h/L}$ 的几何序列（h 从 1..L，m ∈ (2⁻⁸, 2^{-8/L}])。
- 零参数、"线性偏置"直接嵌入 softmax——**远程衰减显式**；理论性质：可严格外推（新位置只是偏置变大、单调衰减）。BLOOM、CodeGen 用。
- 代价：远程 token 永远"打折"，对强依赖远程上下文任务偏保守；且 logit 尺度被位置耦合，温度调节困难。

**四、RoPE（详见题 8）**：相对、零参数、可插值外推，现代默认。

**五、延伸（一句：NoPE/无位置编码）**：Causal attention 也能学出位置（NoPE 论文），因为因果掩码本身不对称；但表达力受限，实用模型仍配位置编码。

**选型决策表**：

| 方法 | 类型 | 参数 | 外推 | 长程衰减 | 现代地位 |
|---|---|---|---|---|---|
| Sinusoidal | 绝对 | 0 | 差 | 无 | 教学价值/archaeology |
| Learned | 绝对 | $L_{max}d$ | 结构不可 | 无 | 退出主流 |
| ALiBi | 相对 | 0 | 大多可严格外推 | 显式线性 | 少数模型保留 |
| **RoPE** | **相对** | **0** | **PI/NTK/YaRN 支持到 128k+** | 隐式（低频相位远波） | **默认** |

**【30秒答题框架】**
"三个轴：绝对还是相对、有没有可学参数、能不能外推。Sinusoidal 零参数但相位歧义+绝对注入不强；learned PE 外推结构性不可能；ALiBi 是给注意力 logit 加 −m(i−j) 线性衰减，免参且外推好，但远程永远打折；RoPE 旋转式相对位置，配合 PI/NTK/YaRN 能稳定外推到 128k+，成为默认。我会顺带说 NoPE 说明 causal mask 本身有位置信息，但表达力不够。"

**【追问与陷阱】**
- 追问：为什么 GPT-2 就是学不来外推？——embedding 表按索引查，没有索引外的参数，插值在离散 index 无意义（连续插值对 one-hot 索引不可微）。
- 追问：绝对 PE 真无相对性？——不是，attention 本身在 query-key 交互时可学出相对先验，但外部注入的绝对 bias 限制外推。
- 陷阱：说"ALiBi 外推=任意长无损"——ALiBi 的远程衰减在超长（如 >8k）下远程信息过弱，效果也是任务相关。
- 陷阱：漏掉"位置编码加在 attention 输入"与"加在 logit"两类注入位置的区别（Rotary/sin 在输入；ALiBi 在 logit）。

**【知识串联】**
RoPE 细节在题 8；位置编码对长上下文/记忆的影响关联 02-Inference 长上下文与 03-FineTuning 长文本继续训练；外推问题在 04-RAG 的长文档处理同样出现（外推决定能塞进 prompt 的文档长度）。

---

### 题 10：LayerNorm vs BatchNorm 的区别

**【考察意图】**
基础题型，但要答到"哪一维统计"与"依赖 batch 与否"的机制层面，并指出 LN 公式（γ、β 仅方差收缩，无跨样本耦合）。测试是否能解释"NLP 为何不用 BN"而不是背 CV/NLP。

**【深度解析】**

**公式（LayerNorm）**：
$$\text{LN}(x) = \gamma \odot \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta,\qquad \mu=\frac{1}{d}\sum_{i=1}^d x_i,\ \sigma^2=\frac{1}{d}\sum(x_i-\mu)^2$$
对**最后一个维度（特征维）**做统计——每个 token、每个样本独立归一。

**BatchNorm**：$\mu_B, \sigma_B^2$ 沿 **batch 维（含空间维 CV 里到 H×W）**统计；训练用 batch 统计、推理用 EMA running stats。

**关键差异 4 点 + 为什么 NLP 不用 BN**：
1. **统计维度**：LN 沿特征（"这个 token 内部"友好）；BN 沿 batch（"同一通道跨样本"）。
2. **batch 依赖**：BN 训练中统计随 batch 抖动，小 batch 方差大（bs=1 完全失效），推理依赖 running-stats一致性；LN 与 batch 完全无关，训练/推理同一逻辑。
3. **变长序列**：NLP batch 内长度不一（padding），跨样本对齐统计引入噪音。
4. **语义上**：NLP 里"同一特征跨 token 共享统计"缺乏语义（不同 token 的 d 维语义分布不同源）；CV 里同一卷积通道跨空间位置共享统计并解释为"该特征图的激活幅度"，BN 有意义。
5. **训练/推理一致性**：LN 训推同构（无 running stats），更适合标准化 + 微调场景。

顺带（加分）：Transformer 曾试验 BatchNorm 失败；ViT 也用 LN 而非 BN——说明"是否时序"不是关键，关键是**跨样本共享通道统计的假设不成立**。

**【30秒答题框架】**
"核心一句：BN 沿 batch 维统计、LN 沿特征维统计。BN 依赖 batch 大小、训练推理两个 mode（running stats），对变长 NLP 序列以及小 batch 都不友好；LN 每 token 独立、训练推理一致、和 batch 解耦，是 attention/Transformer 的事实标准。公式 γ、β 都有，差别只在 μ、σ² 的统计方向。"

**【追问与陷阱】**
- 追问：LN 对每个 token 都做？——是的，per-token（per-position）；这也是 Pre-LN 稳定的原因之一。
- 追问：GroupNorm 在哪？——图像分割小 batch 场景（介于 LN/BN 之间），Transformer 不用。
- 追问：LN 别名"逐样本"还是"逐特征"？——逐样本、沿特征——有人答反。
- 陷阱：说"LN 推理需要 running stats"——错，那是 BN；LN 训推同一公式。
- 陷阱：把 $\sigma^2$ 写成依赖 batch。

**【知识串联】**
RMSNorm 是 LN 的瘦身版（题 12）；Pre/Post-Norm 中 LN 位置的讨论是题 11；训练稳定性理论依赖 LN 的梯度路径（题 31）。

---

### 题 11：Pre-Norm vs Post-Norm ⭐⭐⭐⭐（梯度路径分析）

**【考察意图】**
现代 LLM 常识题，但要答到**梯度路径可视化**层面：为什么 Pre-Norm 存在"无变换残差主干"、为什么 Post-Norm 表达力略强 / 训练需要 warmup。二面考"能否画出两种结构的反向流"。

**【深度解析】**

**两种结构**（每层一个子层 $F$）：
- **Post-Norm（原始 Transformer）**：$x_{l+1} = \text{LN}\big(x_l + F(x_l)\big)$
- **Pre-Norm（GPT-2/LLaMA）**：$x_{l+1} = x_l + F\big(\text{LN}(x_l)\big)$

**梯度路径对比**：
对第 $l$ 到 $L$ 层求 $\partial x_L/\partial x_l$：
- **Post-Norm**：主干被 LN 斩断——归一化在残差**之后**，梯度必须穿过每一层的 LN Jacobian：
$$\frac{\partial x_L}{\partial x_l} = \prod_{k=l}^{L-1} \left(I + J_F^{(k)}\right)\cdot J_{LN}^{(k+1)}$$
每个 $J_{LN}$ 的谱半径 ≤1（缩放 + 去均值），深层连乘让范数收缩、幅度方差大 ⇒ **信号衰减 + 训练初期不稳定**；经验修复 = warmup（把初始 lr 压小，等 v̂ 统计量建立）。
- **Pre-Norm**：残差主干上有 **identity，$I$ 无变换**——
$$\frac{\partial x_L}{\partial x_l} = I + \sum_k J_{F\circ LN}^{(k)}\ \text{（各子层路径的 Jacobian 之和）}$$
严格说：$x_L = x_l + \sum_{k=l}^{L-1} F(\text{LN}(x_k))$，所以 $\partial x_L/\partial x_l = I + \sum (\text{子层路径的 Jacobian})$——**恒等项始终保留**，梯度有一条"永远畅通"的高速公路（与 ResNet 相同的论证）；同时 LN 在 $F$ 前面只影响子层内部，不缩放主干信号。**因此训练稳定、可少 warmup 甚至免 warmup**。
- 表达能力：Post-LN 每层出口被强制标准化（各层在"同尺度"上堆叠），等深度下**有效深度利用更充分**（Xiong et al. 2020 与 GPT 系的实验共识：post-LN 表达力略优、但训练更难）；Pre-LN 各层出口尺度可漂移，理论上有"信号坍缩"风险，但实验与大模型实践均表明其稳定性收益 > 表达力微损。
- 数值实验（大部分面试官会信服"given code"），可跑如下检查两条梯度范数比：

```python
import torch, torch.nn as nn

def grad_ratio(depth=12, d=64, pre=True):
    x = torch.randn(8, 64, requires_grad=True)
    ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    ln = nn.LayerNorm(d)
    h = x
    for _ in range(depth):
        if pre: h = h + ffn(ln(h))          # Pre-Norm：主干无变换
        else:   h = ln(h + ffn(h))          # Post-Norm：主干被 LN 缩
    h.abs().sum().backward()
    return x.grad.norm().item()

for pre in (True, False):
    print("Pre" if pre else "Post", grad_ratio(pre=pre))
# 典型输出：Pre 梯度范数显著高于 Post（深度越深差距越大）
```
可复现实验在深（如 48 层）时 pre/post 比值放大到几十倍量级（数量级实验，具体值随 seed 波动——面试问答可说"实验跑一下落差 1~2 个数量级"，不要背固定数字）。

**工程现状**：
- 现代 LLM 几乎全 Pre-Norm：GPT-2/3、LLaMA 系列（Pre-RMSNorm）、Qwen、DeepSeek（最后那个层的 final norm 用 RMSNorm 输出残差流）；
- Post-Norm 未消亡：**Sandwich-Norm**（前+后都 norm）、DeepNet/DeepNorm（比例修正的 Post-Norm，训练千层深度）仍在学术竞争；但工业默认还是 Pre。

**【30秒答题框架】**
"差别在 LN 挡不挡梯度主干。Post-Norm 是 x+Sub 后再 LN，主干上每层都被 LN 的 Jacobian 收缩，连乘信号衰减+方差大，训练要 warmup；Pre-Norm 是 x 再加 F(LN(x))，残差主干上没有任何变换，∂x_L/∂x_l = I + Σ（子层贡献），梯度永远有恒等通路，训练稳、可免 warmup。代价是各层输出尺度可漂移、表达力略有损失，所以现代 LLM 全用 Pre（配 final norm 补救），DeepNet 尝试抢救 Post。"

**【追问与陷阱】**
- 追问：能既稳定又保持 Post 表达力吗？——Sandwich（Pre+Post），DeepNet 把 Post 每 FFN 的 init 用 α 因子重缩放；实际 deep 训练还是 Pre 为主。
- 追问：有没有 Pre-Norm 有_EXPR 残差幅值爆炸的实际问题？——有（head layer 附近 norm 会很大），一般靠初始化约束或 final norm 处理。
- 陷阱：把 Pre 写成 F(x+LN(x)) 之类位置错——**LN 只在子层输入侧**。
- 陷阱：说 Pre-Norm "完全解决"梯度问题——只是主干 identity 保证存在，子层内部的单层消失/爆炸仍可能。
- 陷阱：说 Pre-Norm 无参数优化——LN 自带 γ、β。

**【知识串联】**
LN 本体在题 10；创建梯度视角与题 31（残差、warmup 之外还有 clip）呼应；RMSNorm/最终 norm 见题 12；DeepSeek 用 WSD + Pre-Norm 组合的工程选择在 03-FineTuning 章节。

---

### 题 12：RMSNorm 是什么？

**【考察意图】**
小而精的题。考"去掉均值中心化"的数学含义、计算收益、以及为什么"mean 减除"其实是不必要的。

**【深度解析】**

**公式**：
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2+\epsilon}}\cdot\gamma$$

对比 LayerNorm：去掉了 (1) $\mu$ 的减法（re-centering）和 (2) 偏移 $\beta$，只保留 (3) **重缩放（re-scaling）**与逐维仿射的"缩放 γ"。

**为什么可以去掉均值中心化**（Zhang & Sennrich 2019 的论点）：
- 不变性分析：LN 的不变性分为 re-centering（平移不变）与 re-scaling（尺度不变）；他们通过实验证明**re-scaling 才是有效不变性**——对学习至关重要；re-centering 只贡献小。
- 换句话说，模型学到的东西对输入的整体尺度变化很敏感（尺度改变会影响 dot product of attention、routing 等），但**对输入整体上下平移** 不敏感（后续线性层自己能吸收偏差）。所以减 $\mu$ 是"多余"的计算。
- 具体收益（论文实验）：在多个任务上 RMSNorm 质量与 LN 相当或更好，推理/训练开销节省 **7%–64%（原论文测量）**；原因是少一次 mean reduce 寄存器访问+少一个 bias。与大模型工程上"norm 计算占前向耗时不少"的事实合拍。
- 没有 β：β 本可被后续线性层偏置吸收，多一个逐维偏置无必要（LLaMA 的 RMSNorm 也只有 γ）。

计算对比：LN 需两次 pass（mean + var），RMSNorm 只算 $\tfrac1d\sum x_i^2$ 一次。

**工程细节**：fp32 累加 prevents overflow（bf16 squares 易溢出）；分母在 fp32 计算。

**哪些模型用**：LLaMA 1/2/3 全家、Qwen、DeepSeek V2/V3、Mistral 等几乎所有现代开源 LLM。

**【30秒答题框架】**
"RMSNorm 是 LN 的去中心化版：只除以均方根 RMS，乘 γ，去掉减均值和偏移 β。理论依据是 LN 的有效不变性其实是尺度不变而非平移不变，减均值是多余的统计量；工程上少一次 pass，论文报告 7%~64% 运行时节省，质量与 LN 相当。LLaMA/Qwen/DeepSeek 全家都是 Pre-RMSNorm。"

**【追问与陷阱】**
- 追问：去掉 mean 不怕削掉"区分峰谷"的信息吗？——后续线性层自带 bias 的吸收；实验证明质量不降。
- 追问：eps 加在哪？——$\sqrt{\text{mean}(x_i^2)+\epsilon}$（分母内），顺序对数值有影响（加在外面会 double 校正）。
- 陷阱：写出 $x/\text{std}$——(标准差含 mean), diff RMS 无中心；这是常见笔误。
- 陷阱：说 RMSNorm "计算减半"——减一次 pass，不是减半 FLOPs（以访存为主）。

**【知识串联】**
与 LN（题 10）共享公式结构；Pre-Norm 位置（题 11）中 LN→RMSNorm 替换是现代标配；数值精度问题接 03-Loss 题的 fp16/bf16 讨论（题 32）。

---

### 题 13：标准 FFN vs SwiGLU

**【考察意图】**
考"现代 FFN 为什么这么设计"：门控机制的表达力来源、Swish 公式、"为什么换 SwiGLU 要把中间维度从 4d 改成 8/3 d"的参数守恒观念。区分背公式与真理解"门控"激活。

**【深度解析】**

**标准 FFN（原论文）**：
$$\text{FFN}(x)=\max(0,\,xW_1+b_1)W_2+b_2,\quad W_1\in\mathbb{R}^{d\times d_{ff}}$$
两层 MLP + ReLU，中间维 $d_{ff}=4d$，含 8d² 参数（含 bias 约 8d²+5d）。

**Swish**（Ramachandran et al. 2017，NAS 搜索得出）：
$$\text{Swish}(x)=x\cdot\sigma(\beta x)$$
β=1 时简化，形状：x<0 时轻度下凹（不像 ReLU 硬性零区间）、x>0 渐近线性。β 可学。相对 ReLU：负方向"温和"而非硬 0，梯度流好，下游深度网络收敛与 loss 更平滑（报告儿童的论文为深度网络的精度略优）。GLU 系列的关键是"门控"，不是 Swish 本身。

**GLU / 门控机制（Dauphin et al. 2017）**：
$$\text{GLU}(x) = (xW+b)\otimes\sigma(xV+c)$$
两个线性分支相乘：一个支路 Sigmoid 提供 [0,1] 的**重要性门**，另一个支路提供"内容"——乘法实现"选择性通过"，等价于对每个特征维度学一个数据相关的开关（相关性与注意力同源）。

**SwiGLU（Shazeer 2020 "GLU Variants Improve Transformer"）**：
$$\text{SwiGLU}(x)=\big(\text{Swish}(xW_3)\odot xW_1\big)W_2$$
（注：LLaMA 实现写成 $\text{down}(\text{SiLU}(\text{gate}(x))\otimes\text{up}(x))$；W_1/W_3 名字随实现不同）。三个矩阵：gate、up（合并两支路）、down。

**参数与计算**：
SwiGLU 3 矩阵 vs 2 矩阵——同中间维 4d 时参数 ×1.5。为保持参数量与 FFN(4d) 等效：
$$3\times d\times d_{ff}' = 2\times d\times 4d \Rightarrow d_{ff}' = \frac{8}{3}d$$
LLaMA-7B：$d_{ff}=11008=2.6875\cdot4096\approx\frac{8}{3}d$（凑整对齐 256 倍数）。亦有变体按激活内存 unit aligned（如 14336≈3.5d，13B 的新版对齐方便 kernel）。
（"为什么不是 2/3 而是 8/3"：与 FFN 4d 的比例保持一致：4d×2/3 = 8/3 d。）

**为什么门控有效（机制）**：
1. **乘性交互**：两个线性分支相乘产生**双线性**（bilinear）项，等价于引入了"特征×特征"二次信息——比加性非线性更丰富的函数类；
2. **选择性**：门在-token 基础上做稀疏激活（类比 MoE 的 routing 但完全软）——不相关维度被压 0（或负），有效容量分布更"尖"；
3. **梯度友好**：Swish 非饱和区+负区非零导数，缓解 dead ReLU。
4. 实验上 PaLM/LLaMA/T5(用GEGLU) 在等参数预算下 loss 一致略优于 ReLU-FFN（同量级、多任务稳定），故成为默认。

**实现代码（PyTorch）**：
```python
import torch, torch.nn as nn, torch.nn.functional as F

class SwiGLU_FFN(nn.Module):
    def __init__(self, d, d_ff):
        super().__init__()
        self.w_gate = nn.Linear(d, d_ff, bias=False)   # gate 支路（Swish 作用于此）
        self.w_up   = nn.Linear(d, d_ff, bias=False)   # up 支路（线性内容）
        self.w_down = nn.Linear(d_ff, d, bias=False)   # down（回归 d 维）
    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

# 参数量对照：SwiGLU(d_ff=8/3 d) == FFN-ReLU(d_ff=4d)
d = 4096
a = sum(p.numel() for p in SwiGLU_FFN(d, int(8/3*d)).parameters())  # 3*d*d_ff
b = 2 * d * 4 * d                                                    # ReLU-FFN 参数
print(a, b, abs(a-b)/b)   # 134.2M vs 134.2M —— 参数守恒成立（int 取整的 8/3 带来 <1% 偏差；
#                          LLaMA-7B 取 d_ff=11008 时 a=3*4096*11008≈135.3M，b=134.2M，差 ~0.8%）
```
（数值上 d_ff=11008 时：$3\times4096\times11008=135{,}266{\times}10^3$… 精确 $3\times4096\times11008=135{,}266,304$≈135M；ReLU 版 $2\times4096^2\times4=134{,}217,728$≈134M——差 0.8%，达标"参数量相当"的说法。）

**【30秒答题框架】**
"ReLU-FFN 是两层 MLP 中间 4d；SwiGLU 换成三矩阵：gate 支路 SiLU、up 支路线性，两者逐元素乘当门控，再 down 回 d。收益来自乘性交互（双线性）+ 数据相关的软稀疏开关，等参数下 loss 普遍略优于 ReLU。为保持参数守恒，中间维从 4d 降到 8/3 d（LLaMA 7B 是 11008≈2.69×4096）。"

**【追问与陷阱】**
- 追问：为什么是 SiLU 不是 GELU？——同为光骨 family，PaLM/LLaMA 用 SiLU；有些模型（Gemma）用 GEGLU（GELU 门控）——效果相近，选择历史性。
- 追问：门控会不会"杀死"信息（门全关）？——sigmoid 软门初始时约 0.5，训练后有稀疏但若训崩可通过初始化调整（gate 权重 0 初始化会把门压死）。
- 陷阱：公式写成 Swish(xW1)·xW2 不补 down——三矩阵结构写错。
- 陷阱：声称 SwiGLU 参数更少——不加中间维调整时参数更多（3 vs 2 矩阵）；面试官就是钓这个。

**【知识串联】**
8/3 法则在题 14 一并算参数预算；FFN 而 MoE 由 gate 路由交叉进化（07-HotTopics 的 MoE 部分）；GLU 与 attention 的"软开关"同思想约等于 06-Safety/07-HotTopics 的解释性素材。

---

### 题 14：FFN 中间维度为什么通常是 4d？

**【考察意图】**
考"经验值背后的算力账"：能否从参数量、FLOPs 与 attention 对称性给出 4d 的来历，以及 SwiGLU 时代 8/3 法则的正确换算（很多面试者乱说"8d"）。

**【深度解析】**

**平方神经元的算术账**：
每层参数（attention + FFN）：
- MHA：$W^Q,W^K,W^V,W^O$ 各 $d\times d$ → $4d^2$
- FFN(ReLU)：$d\times 4d + 4d\times d$ → $8d^2$
- **合计 $12d^2$，FFN 占 2/3**（见题 5 FLOPs 同比例：20nd² 中 16nd² 来自 FFN）。
"4×"的来历不是数学证明而是**工程平衡**：attention 与 FFN 计算各占一摊、避免某一项主导；4 显著大于1（表达充分）、又不太大（计算不爆炸）。与 attention 的 $4d^2$ 形成"对称"：FFN W1+W2 各 4d²，与 attention 的 4 个 d×d 矩阵同权重。

**SwiGLU 时代的守恒换算**：
SwiGLU 三矩阵是 $3 d\,d_{ff}$；要等价 $8d^2$：
$$d_{ff} = \frac{8d^2}{3d}=\frac{8}{3}d\approx2.67d$$
- LLaMA-7B $d=4096\Rightarrow d_{ff}=11008$（$=2.6875d$，11008/4096=2.6875；取 256 的倍数凑整数）。
- LLaMA-13B $d=5120\Rightarrow d_{ff}=13824=2.7d$。
- 混淆来源：有的模型（BLOOM, PaLM）用 4d 的 GLU（Parametr 矩阵变 4 个）→ 参数变 $16d^2$，FLOPs ×1.33；看似"更大"其实是他们没做 8/3 折算（PaLM 写明 hidden_dim=4d 的 parallel-FFN）。
- 本质：**用"中间维"换取"门控支路"——多出来的 1/3 预算花在"门"上**。

**更深一层（二面加分）**：FFN/中间维与"每 token 激活的专家数"在 MoE 里被重新定义（DeepSeek-V3 MoE：FFN 专家 top-m 激活、共享专家），但"Young 2024 论文 dense-transformer 中的 4d 是计算最优"的经验公式基本不变——它是**storage（参数）**与**communication（激活/显存）**与**compute（FLOPs）**三角平衡的产物。

**【30秒答题框架】**
"4d 是一个经验平衡数：让 FFN 参数 8d² 对 attention 4d² 形成 2:1，FLOPs 上 FFN 占每层 8 成，不会偏科太小或太大。SwiGLU 有 3 个矩阵，为参数守恒把 d_ff 从 4d 降到 8/3 d≈2.67d——LLaMA 7B 的 11008 就是 2.6875×4096（凑 256 倍数）。有时面试官故意说 GLU 用 4d"更大"——那其实是没做守恒折算的历史遗留。"

**【追问与陷阱】**
- 追问：d_ff 是不是越大越好？——过约束带宽（激活内存变 FFN 瓶颈）且收益边际递减；大 d_ff 会挤掉的 attention 头的预算（总参数约束下）。
- 追问：SwiGLU 上限是不是可以更高激活天空？——是门控带来的有效维度，不是直接堆 d_ff。
- 陷阱：把"4d"记成"4 倍 d_model"却不能算出 12d²/20nd² 的分解——二面会追问参数分配。
- 陷阱：声称 SwiGLU 用 8d 中间维——错（8/3 才守恒）。

**【知识串联】**
参数/FLOPs 换算即题 5；MoE 的"专家容量"扩展在 07-HotTopics 与 DeepSeek 系；FFN/attention 比例对推理成本的影响对应 09-SystemDesign 的训练/推理成本核算。

---

### 题 15：Encoder-Only / Decoder-Only / Encoder-Decoder 的区别

**【考察意图】**
基础分类题，但二面答案是"预训练目标+注意力模式+自然任务范式"三位一体，而不是背代表模型名。

**【深度解析】**

| 架构 | 注意力 | 损失/预训练目标 | 输入可用上下文 | 代表 | 天然任务 |
|---|---|---|---|---|---|
| Encoder-Only | **双向** | MLM/刷下文（BERT）/位置 span 预测 | 全上下文（两端都可见） | BERT、RoBERTa、DeBERTa | 分类/NER/句向量/embedding 检索 |
| Decoder-Only | 单向因果 | next-token prediction | 只看左侧 | GPT、LLaMA、DeepSeek、Qwen | 生成、对话、指令跟随、代码 |
| Encoder-Decoder | enc 双向 + dec 因果 + cross | span-corruption（T5）/denoising | enc 全文，dec 自回归 | T5、BART、mT5 | 翻译、摘要、seq2seq 受控输出 |

细节（面试要点）：
1. **注意力权重矩阵差异**：enc 双向为全可见 $(i,j)$ 任意填充；dec 因果裁掉 i<j；enc-dec 有三套（enc self / dec masked self / cross）。
2. **损失目标差异**：BERT MLM 约 15% mask、任务"读填空"；T5 span corruption（sentinel token）；GPT 每位置 next-token（每个 token 都是有监督信号，题 16 论点 6 的来源）。
3. **用途边界**：encoder-only 因"输出一个向量/分布而非序列"，**不能直接自回归生成**；decoder-only 可做理解（打分/分类在最后位置读 hidden state），只是"理解任务"上 encoder 在小模型同 token 预算下通常略优；enc-dec 在参数上每层为 encoder + decoder 之和，同算力下单边表达弱。
4. **现代格局**：生成/agent 全是 decoder-only（推理 cache 友好 + ICL 涌现）；encoder-only 活跃于 embedding/RAG 检索（bge/e5 等）；enc-dec 存活于翻译/蒸馏等专用任务。

**【30秒答题框架】**
"三个架构本质差别是'注意力可见范围+预训练目标'。Encoder双向+MLM，输出理解用表示，天然做分类/NER/embedding。Decoder因果+next-token，输出自回归序列，天然做生成对话，也是今天 LLM 主流。Encoder-Decoder 两头：enc 全文、dec 因果+cross attention，天然做翻译摘要。现在 decoder-only 因为数据效率高（每 token 都有 loss）、KV cache 友好成为主流，encoder 系退到 embedding/检索场景。"

**【追问与陷阱】**
- 追问：BERT 能做生成吗？——原生不能自回归；有变体（XLNet 抹平、T5 式、BERT2BERT 级联）但工程复杂、无 cache 增量。
- 追问：Decoder-only 能做 NER/分类？——fine-tune 头或用 last-token hidden；效果稍逊于同参 encoder，主要差在预训练目标与算力分配。
- 陷阱：说 T5 的 "enc 因果"或"dec 双向"——因果双方向别记反：enc 双向、dec 因果。
- 陷阱：忽略 BERT 类模型 MLM 只 watch 15%，数据效率低于 decoder-only。

**【知识串联】**
选型原因独立成题 16；"为什么只有 decoder 有 ICL"承接 03-FineTuning/02-Alignment 与 05-Agent 的 few-shot 基础能力；embedding/检索路线在 04-RAG。

---

### 题 16：为什么现在的 LLM 都是 Decoder-Only？⭐⭐⭐⭐

**【考察意图】**
5 个"事实+一句机制"的综合题。考的是能不能把 **scaling law 友好**、**KV cache 一致性**、**ICL 涌现**、**数据效率** 这四个面试官心中的高分点全部覆盖，而不是只说"因为 GPT 火了"。

**【深度解析】**

六条机制（原文表 + 补充 1 条）：

1. **统一的 next-token prediction 范式**：一个 loss 解决所有任务（翻译=目标在前缀后、问答、代码、对话）；任务是用**prompt 格式**切换而不是改架构——这是 function-level 的通用性。
2. **数据效率 100%**：每 token 每步都有交叉熵 loss（下一位置监督）；MLM 只有 ~15% 的 masked 位置出 loss——同算力下训练信号利用率 100% vs ~15%（这是 GPT-1（Radford et al. 2018）以来的公认结论，T5 论文也承认 decoder 系预训练效率高）。
3. **Scaling law 最友好**：Kaplan 2020、Hoffmann 2022（Chinchilla）都是在 decoder-only 语言模型上建立的；工业上调参、数据配比、learning-rate schedule 都沉淀为 decoder-only 的配方（比如 chinchilla-optimal 20 token/param、lr schedule 的 WSD 变体）。**生态配套**（框架、并行、MoE、量化）也建立在 decoder-only 上，形成正反馈。
4. **ICL 涌现**：few-shot/零样本 prompt 下的 in-context learning 在 decoder-only 大模型上出现（Olsson 的 induction heads 机制/“褒义模式匹配”），encoder 系（MLM）没有等价涌现——这与因果注意力的"前缀→预测"结构直接适配（不考证明，当作实证事实陈述）。
5. **KV cache 天然适配推理**：因果注意力保证"历史 KV 不变"，可以无冲突缓存复用；双向注意力每步都要重算全部 attention（因为新 token 改变了对所有历史位置的归一化）——**双向结构无法增量 decode**。这让 decoder-only 在 serving 成本上碾压（细节进 02-Inference/01）。
6. **指令/对话数据的天然格式**：(user, assistant) 的多轮对话本质是"前缀续写"，与 next-token 无缝；encoder 系列 需要把对话折叠成"输入-输出"对强行对齐。

加分反例（防"绝对化"陷阱）：
- **理解类任务**（大规模分类、句匹配）小模型 dense encoder 仍更高效（BERT 式）；
- **检索/ würden embedding 主战场**：双塔 encoder + 因果不相干；
- **翻译/摘要的 dense 专用模型**：T5 系仍在使用；
- **prefix-LM/前缀化架构**（UL2 等）试图统一，但工业未选择。
所以严谨的说法：**生成/agent 的 LLM 全部是 decoder-only，而非"所有模型都该是"**。

**【30秒答题框架】**
"四条硬理由+一条生态理由。硬理由：一、每 token 都有 loss，数据效率碾压 MLM 的 15%；二、KV cache 增量 decode 只有因果注意力可以做，双向做不到；三、ICL 和对话格式天然是'前缀续写'；四、scaling law 和所有训练 recipe、并行框架都在 decoder-only 上沉淀。生态理由：GPT 系验证 + 工具链正反馈。补一句绝对化防御：embedding/检索还是 encoder 的地盘，说'LLM 主流都是 decoder-only'而不是'全部模型都该是'。"

**【追问与陷阱】**
- 追问：BERT 系在如今哪里还在赚钱？——embedding 检索/排序（RAG 双塔）、小参数量分类服务、代码补全的 encoder 部件等。
- 追问：双向注意力不是更"好"吗？——训练理解上确实信息更多，但 (a) 无法增量 decode (b) 每步全量重算；"训练好"≠"部署好"。
- 追问：prefix-LM 为什么没起来？——工程上与 decoder 的 cache 优化、生态不兼容；UL2 混合模式只在实验。
- 陷阱：只答"GPT 火"不给"数据效率/KV cache"——二面算不及格。
- 陷阱：把"decoder-only 推理快"错说成"训练快"（训练也快是因为数据效率，不是 cache）。

**【知识串联】**
KV cache 细节 02-Inference/01；数据效率与 loss 结构在题 25 补充；ICL/协议续写接 03-FineTuning/03-Instruction-Tuning、05-Agent；embedding 常见工程实现 04-RAG/01。

---

### 题 17：Causal Mask（因果掩码）是什么？

**【考察意图】**
看似简单，但被打分点是**训练-推理一致性**与**-inf 的数值细节**（加在 softmax 之前还是之后、−inf 在 fp16 的处理、行全 −inf 的 NaN 边界）。

**【深度解析】**

**目的**：自回归训练用 teacher forcing（一次喂全序列并行算 loss），但生成时第 t 个 token 只能看见 <t 的历史——**训练分布必须匹配推理分布**，否则训练学到了"偷看未来"的捷径，推理时崩。

**实现**：在 softmax **之前** 把上三角 logit 置 −∞：
$$\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}+M\right),\qquad M_{ij}=\begin{cases}0&i\ge j\\ -\infty&i<j\end{cases}$$
softmax 后上三角权重精确为 $e^{-\infty}/\sum = 0$（在数学上）；同时也不影响下三角位置的相对 softmax（finite logit）。

**数值细节（易错点集中区）**：
1. **为什么是 −∞ 不是 0**：mask 在 softmax 之前（logit 域），若置 0 反而变成"权重 1"（logit 0 也是可竞争项）。
2. **fp16/bf16 没有 −∞ 吗**：都有（float('inf')），但 $-\infty + $ finite = $-\infty$，稳定；$\exp(-\infty)=0$。问题出现在**整行都 −inf**（如 padding 后 query 无合法 key）：softmax(全 −∞) = 0/0 = **NaN**。工程上：整行至少保留 self-attention 对角项（j=i 允许），或者 masked_fill 用 −1e9 而非 −inf，或者最后 finalize 那些行（flash attention 的 handle）。
3. **掩码 + RoPE 共存**：RoPE 在 P/Q 投影后立刻旋转（query/key 各自加位置）；mask 与之无冲突。
4. **加性掩码 vs 布尔掩码**：现代实现都用 additive bias（掩码在 softmax 前加）。取值约定：fp32 下常用 −1e9；fp16 的 min ≈ −65504 会被反向乘大或与 0 相乘出 NaN，工程上用 −1e4 量级或把 attention 分数在 fp32 里计算（bf16 的 finfo.min≈−3.4e38 可直接用）。
5. **训练=推理等价性**：训练时并行计算（大 mask）与推理时逐 token 计算在数学上**完全一致**（对角以下自适应），这是 transformer 自回归训练的"魔法"——一次前向同时展开 T 个训练步。
6. padding mask vs causal mask：两者叠加使用（key 的 padding 位置额外 −inf）。

**【30秒答题框架】**
"Causal mask 是把 logit 的上三角置 −∞、softmax 前加上去，让第 i 个 token 只能看到 ≤i 的历史；目的就是让并行训练（teacher forcing）和逐步推理的分布完全一致。坑在数值：0 不是掩码（softmax 后权重 1），−inf 全行的 NaN 边界要兜底，fp16 用 −inf 或 −1e4 量级的加性 bias。"

**【追问与陷阱】**
- 追问：为什么不在 softmax 之后强制置 0？——可数学上等价（置 0 再重归一化），但 (a) 浪费 exp 计算 (b) 归一化分母包含未来 token 会导致实现混乱；且斑块化 kernel（Flash）必须有 logit 域 mask 才能切块 streaming。
- 追问：cross-attention 有 mask 吗？——dec→enc 通常无因果（encoder 输出全部可见），但 enc 的 padding mask 需要加。
- 追问：能"部分因果"吗？——prefix-LM（前缀全可见 + 后段因果）是 UL2/prefix-LM 变体，工程少见。
- 陷阱：把 mask 加在 softmax 之后（−1 变 -0 权重）；或把 −∞ 放在 fp16 里直接当数值做 attention 相乘产生 NaN。

**【知识串联】**
训练-推理一致性与 KV cache 的"历史不变"性质同源（02-Inference/01）；损失计算（题 25）的 shift 依赖 mask；decode 输出（02-Inference/02）的分布假设由 mask 定义。

---

# Part B · 分词算法（02-Tokenization.md）

### 题 18：BPE 分词算法的原理 ⭐⭐⭐（训练/推理全流程样例）

**【考察意图】**
BPE 是 LLM 字节级分词的默认算法，面试不是背四步流程，而是**完整走一遍训练样例**+ 说明编码时如何用学好的 merge 表，以及 byte-level 变体（GPT-2）为何出现。本题按"逐步样例"深挖。

**【深度解析】**

**算法思想**：一种自底向上的**数据压缩式**子词学习（Sennrich et al. 2016 从机器翻译引入）：从字符级词表开始，反复把语料中**出现频率最高的相邻符号对**合并成新符号、加入词表与 merge 规则表，直到词表达到预算大小。

**训练过程（4 步循环）**：
1. 预分词（pre-tokenize，一般按空格/标点切开，词内不再跨界合并）；
2. 把每个词转成字符序列（加结束符 </w> 或空格前缀 Ġ）——**词表 = 全部字符**；
3. 统计**相邻符号对**频次 → 合并最高频对 $(a,b)\to ab$，写入 merge 表；
4. 在全语料上应用该合并（替换所有相邻对），回到 3，直到词表达额定大小。

**逐步样例**（对齐原文件的语料 "low lower lowest"：先按**词型计数**（每个词出现 1 次），复现原文件的 Step1~3；随后给出真实语料**按词频加权**的差异说明）：

**按词型计数（示例口径）**：
```
low      → l o w </w>
lower    → l o w e r </w>
lowest   → l o w e s t </w>       （每个词计 1 次）
```
相邻符号对计数：
- (l,o)=3、(o,w)=3、(w,e)=2
- 全为 1：(w,</w>)、(e,r)、(r,</w>)、(e,s)、(s,t)、(t,</w>)

**Step 1**：最高频对 (l,o) 与 (o,w) 计数 3 并列——tie-break（实现里常按首次出现序/字典序，Counter 的 most_common 稳定排序取先插入者）选 (l,o)：**合并 → "lo"**，词表 +{lo}，merge 表记录 ('l','o')→'lo'。语料变为：
`lo w </w>` | `lo w e r </w>` | `lo w e s t </w>`

**Step 2**：重算计数：(lo,w)=3 最大 → **合并 → "low"**。语料：
`low </w>` | `low e r </w>` | `low e s t </w>`

**Step 3**：计数：(low,e)=2，其余全 1 → **合并 → "lowe"**（与原文件示例一致）。续推：(lowe,r)→"lower"、(lowe,s)→"lowes"→(lowes,t)→"lowest"→(lowest,</w>)、(low,</w>)→"low</w>"……

**按词频加权的差异（真实训练口径，面试加分点）**：若语料是 `low×5、lower×2、lowest×2`，计 voll变化为 (l,o)=(o,w)=9、(w,</w>)=5、(w,e)=4、其余 2。Step3 时 (low,</w>)=5 > (low,e)=4，合并顺序会**先变**成 (low,</w>)、下一轮才 (low,e)。**示例与真实实现不同不是"错"，而是计数口径不同（词型 vs 词频）+ tie-break 约定不同**——两句都能在面试里讲清，是 BPE 理解深度的信号。

**本质**：merge 表就是**按训练优先级排序的实现清单**，编码的时候按顺序回放。

**编码（推理）过程**：把新文本先预分词成初始字符序列，然后**按 merge 表的learned order 依次应用合并**（每次扫描当前序列，若相邻对匹配表中下一个未消耗的 merge 就合并；循环直到没有可合并对）。**贪心、确定性**；OOV 不存在（字符总在词表内），byte-level 的 BPE 更是 256 字节全覆盖 ⇒ **零 <UNK>**。
伪代码：
```
encode(w):
    syms = chars(w)
    for (a,b) in merges_in_learned_order:      # 按训练优先级顺序回放
        if [a,b] contiguous anywhere in syms:  syms = merge_all(syms, a→b)
    return ids(syms)
```
（HF tokenizers 的实现细节：优先级队列 + 按出现的 rank 找最优合并点，复杂度约 O(m log m)，m 为词长；整句 BPE 编码约线性于 token 产出数。）

**Byte-level BPE（GPT-2 起）**：
- 以**byte（0–255）**为初始符号而非 unicode 字符；训练前把文本用可逆 unicode trick（GPT-2 用 utf-8→可打印表映射，避免控制字符），推理无损还原。
- 好处：**任意语言/emoji/乱码序列都不会 OOV**（256 字节必覆盖）；中文/日文也被自然收敛为多字节 token。
- 代价：低资源语言一个字 ≈ 1–3 bytes，中文被 utf-8 编码成 2–3 byte token 起步——**中文压缩率吃亏**（题 24）。
- LLaMA 用 SentencePiece-BPE（byte fallback 模式），Qwen 用 byte-level BPE（tiktoken 风格）。

**复杂度**：训练时 step 3 统计 O(|corpus|)、合并传播 O(|corpus|)；总 O(|V|·|corpus|)。缓存优化的堆/计数器可降（HF 实现在数百万词的语料数十分钟到数小时量级）。

**BPE 训练+编码可运行最小实现（Python，可直接跑）**：

```python
from collections import Counter

def bpe_train(corpus: dict, num_merges: int):
    """corpus: {tuple_of_symbols: freq}，符号是被 tuple 化的字符串"""
    vocab = corpus
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for sym, f in vocab.items():
            for i in range(len(sym) - 1):
                pairs[(sym[i], sym[i+1])] += f
        if not pairs: break
        best, freq = pairs.most_common(1)[0]
        merges.append(best)
        vocab = {tuple_merge(seq, best): f for seq, f in vocab.items()}
    return merges

def tuple_merge(seq, pair):
    """按从左到右顺序合并一次给定 pair"""
    out, i = [], 0
    while i < len(seq):
        if i < len(seq)-1 and (seq[i], seq[i+1]) == pair:
            out.append(seq[i] + seq[i+1]); i += 2
        else:
            out.append(seq[i]); i += 1
    return tuple(out)

def bpe_encode(word: str, merges):
    """按 learned order 回放 merges，确定性编码"""
    syms = list(word)
    for a, b in merges:
        i = 0
        while i < len(syms) - 1:
            if syms[i] == a and syms[i+1] == b:
                syms[i:i+2] = [a + b]
            else:
                i += 1
    return syms

# === 对齐原文件示例：按词型计数（每词 1 次）===
corpus = {("l","o","w","</w>"):1,
          ("l","o","w","e","r","</w>"):1,
          ("l","o","w","e","s","t","</w>"):1}
merges = bpe_train(corpus, num_merges=6)
for m in merges:
    print(m)
# 输出（与正文 Step1~3 一致；Counter 严格按计数排序、并列时取先插入者）：
#  ('l', 'o')        -> lo      [Step1]
#  ('lo', 'w')       -> low     [Step2]
#  ('low', 'e')      -> lowe    [Step3]
#  ('low','</w>'), ('lowe','r'), ('lower','</w>') ...（此后全为计数 1 的并列）
print(bpe_encode("lowest", merges))   # ['lowe', 's', 't']（6 次 merge 未合完时的确定性结果）
# === 换成词频加权（low×5, lower×2, lowest×2）时，第 3 次合并变为 ('low','</w>') ===
```
（编码示例：`bpe_encode("lowest", merges)` 先被 ('l','o')→('lo','w')→('low','e') 三条规则逐步合并成 'lowe'，其余符号在 6 次 merge 预算内没有更高优先级的可合并对。真实工程编码用 HF tokenizers 实现（rank 优先贪心 + 优先队列），语义等价。）

**【30秒答题框架】**
"BPE 从字符词表开始，每一轮统计相邻符号对频次、合并最高频对进词表并记录 merge 规则，直到词表达到预算。训练好后编码就是把新文本拆成字符、**按训练时的 merge 顺序回放**贪心合并，确定性、无 OOV。byte-level BPE（GPT-2 以来）初始词表是 256 字节，中文生僻字也永远不被拆出词表外；代价就是中文 utf-8 起步 3 字节较碎。复杂度 O(|V|·corpus)。"

**【追问与陷阱】**
- 追问：为什么按"频次最高"合并而不是"信息量大"？——BPE 的目标是**用少数规则覆盖高频初段**（压缩优先），WordPiece/Unigram 才是概率/似然导向（题 20/21）。
- 追问：为什么必须预分词（空格切开）？——防止跨词 merge 把 "e r"（跨词）合成一个 token；SentencePiece 通过空格的唯一前缀符号（▁）替代词内的"独立表示"要求。
- 追问：词频一样怎么办？——tie-break 一般按 dict 序/首次出现序；不同工具实现不同，这在严格的可用性评估里要说明。
- 追问：merge 表在 decode（id→text）时精确可逆吗？——确定可逆，因为 str 词表与 ids 双射；但**编码不唯一**的原则按"贪心合并"确认，不能"随机拼"。
- 陷阱：说"BPE 需要神经网络"——无参数，纯统计；别和 unigram 混淆。
- 陷阱：把 "</w>"（词尾）与 "Ġ"（词首空格，GPT-2）混为一谈——两种词界约定都存在。

**【知识串联】**
byte-level 的中文吃紧问题在题 22/24；分词对推理长度的影响（压缩率）直接关联 02-Inference 的 KVCache/attention n² 与 09-SystemDesign 的成本预算；词表设计（embedding 大小）与题 23 的权衡呼应。

---

### 题 19：BPE vs WordPiece vs SentencePiece

**【考察意图】**
考"这三个词是不是同一个东西的三个名字"——很多人混淆。要能清楚：两种维度（合并准则/训练框架；框架 vs 算法），SentencePiece 是**工程框架**不是算法。

**【深度解析】**

先拆维度——这是二面答辩要点：
- **算法层**：学哪种子词表：BPE（高频合并）/ WordPiece（似然增益合并）/ Unigram（大词表裁剪）。
- **工程层**：怎么处理原始文本（预分词、空格、归一化、OOV policy、可逆性）→ 这是 SentencePiece 的位置。

| 维度 | BPE | WordPiece | SentencePiece |
|---|---|---|---|
| 合并/选择准则 | 最高频对 | 最大化似然增益 score | 内嵌 BPE 或 Unigram（可配置） |
| 预分词 | **需要**（空白/标点） | 需要 | **不需要**（把"空格"当普通字符 ▁） |
| 空格表示 | 保留（或 Ġ 前缀 hack） | `##`词中缀 | `▁`（U+2581）统一替换空格 |
| 解码可逆性 | 弱（依赖预分词 rules 复原） | 需要重建（`##`拼接） | **强**（字节 fallback + 无歧义） |
| Trained on | 空格切完的 word 序列 | 空格切完的 word 序列 | 原始字符串（含 CJK 无空格） |
| 使用者 | GPT-2/3、RoBERTa（byte-level）、tiktoken 系 | BERT、DistilBERT、Electra | T5（Unigram）、LLaMA（BPE 模式）、XLM-R |

**SentencePiece 的两个独特卖点**（面试加分）：
1. **语言无关**：预处理把空格视为普通字符（替换为 U+2581 "▁"），因此**中文/日文/泰文**这类无空格语言与英文同一条流水线；不再依赖 whitespace tokenizer 的语言假设。
2. **可逆性**：encode(decode(x)) == x 在 byte-fallback 下严格成立——**没有 "unknown" 概念**（退化为 byte fallback），修复了 BERT "##" 拼接不能完美还原（如 unicode 归一化前后差异、控制字符丢失）的老 bug。

**词间"##" 与 "Ġ" 的差别**（常见追问）：
- WordPiece：`playing` → `play ##ing`（## 标记"我是词中间续接"）；decode 时 `##` 前无空格。
- GPT-2/BPE：`Ġplaying`（Ġ=空格前缀）——空格跟 token 走，decode 直接拼接。两种约定等价，但 GPT-2 的方式**流式 decode 无需 rejoin 逻辑**，更适合生成。

**【30秒答题框架】**
"先分层：合并准则是算法层（BPE 频次、WordPiece 似然增益），SentencePiece 是工程框架（含 BPE 或 Unigram 两个内核）。SentencePiece 两大亮点：把空格换成 ▁ 当普通字符，语言无关、中日韩无碍；byte fallback 保证 encode/decode 严格可逆、无 UNK。词中标记上 WordPiece 用 ##、GPT-2 用空格前缀 Ġ，后者生成流式 decode 更自然。"

**【追问与陷阱】**
- 追问：BERT 的 tokenizer 严格叫什么？——WordPiece（HF 里 ` BertTokenizer`），但**其实是 greedy longest-match**（训练用 score 公式，编码用最长匹配而不走动态规划）。
- 追问：LLaMA 的 tokenizer 叫什么？——SentencePiece **框架** 运行 **BPE**（byte fallback mode），不是"Unigram"。
- 陷阱：说"SentencePiece 是一种分词算法"——它是 pre-tokenization-free 的处理框架，算法仍可选。
- 陷阱：说不清"## 拼接 vs Ġ 前缀"的 decode 差异。

**【知识串联】**
选型后果（中文压缩率、embedding 尺寸）在题 23/24；多语言 tokenizer 问题在 04-RAG 多语言文档处理涉及；tokenizer 对 prompt 长度的乘数效应在 05-Agent 工具调用协议上也有影响。

---

### 题 20：WordPiece 与 BPE 的核心区别

**【考察意图】**
原文件给出 score 公式，考的是能不能从**互信息/似然**角度把"选对 vs 选频"讲透，并由此解释英文高频字母对（如 th/er）在 BPE 下为何总被先合并、WordPiece 如何避免这类碎片。

**【深度解析】**

WordPiece (Schuster & Nakajima 2012；BERT 实现) 的合并评分：
$$\text{score}(x,y)=\frac{\text{freq}(xy)}{\text{freq}(x)\cdot\text{freq}(y)}$$
分母是"独立假设下的期望共现"，分子是实际共现——该分数本质上是**逐点互信息（PMI）去掉 log**：
$$\log \text{score} = \log P(xy) − \log P(x) − \log P(y) = I(x;y)\ (\text{点互信息})$$
**等价解释**：选"两者共现显著高于独立假设"的对——**frequency AND association** 双条件。

**与 BPE（纯频率）的分歧后果**：
- BPE 合并 (e, r) 这类全语料超高频对（er/re/th…）——**碎片很多**，进而"英文的大众 token"全被单字/双字占据；
- WordPiece 合并更像"成词"（如 `##ing`、`中华人民共和国` 一词），词中词尾的**粘着**合并得更"语义化"；
- 代价：WordPiece 的 merge 会让低频 token 被切很碎（数据不足的高-cost 区），实际 BERT 的 OOV rate 与"碎片长"在某些 domain（医学/代码）比 byte-BPE 差。
- **训练目标不同**：BPE 目标是最小化"词典大小对语料的描述代价"（压缩视角）；WordPiece 目标是"最大化对语料中的语言模型似然"（LM 视角——score 即在两极 LM 假设下的 log-likelihood 增益的近似）。

**BERT 的实现陷阱**：训练时按 score 挑 merge；**推理/编码却用 greedy longest-match**（从词表里找最长前缀），并不真正运行动态规划最优分割——严格说 BERT 的 encode ≠ 全局最优 WordPiece 分割。这是面试"细节加分点"。

**【30秒答题框架】**
"一句话：BPE 按频率、WordPiece 按互信息。score = freq(xy)/(freq(x)freq(y)) 就是 PMI——不只要共现多，还要'共现显著超过独立假设'，所以 WordPiece 容易合成'词状'子词而 BPE 堆'er/th'这种高频碎片。额外细节：BERT 训练用公式打分，但实际 encode 用 greedy longest match，不是全局最优。"

**【追问与陷阱】**
- 追问：score 不取 log 有影响吗？——与取 log 排序不同（因为 freq 比值的大小差异在 log 下会被压缩），官方就用比值；提一句即可。
- 追问：所以 WordPiece 一定比 BPE好用吗？——不是，领域迁移高频碎片（byte-BPE）在代码/多语上反而稳，BERT 用 WordPiece 是 2018 语境；GPT-2 后 byte-BPE 成为生成模型主流。
- 陷阱：说 WordPiece 的 score 是"最大似然"就完了——要说得出是**频率比/PMI 近似**、以及它和 Unigram 的"真似然"的区别（题 21）。
- 陷阱：误以为 BERT 的 encode 跑了 Viterbi——没有；Unigram 才是 Viterbi。

**【知识串联】**
与题 19 表格相互印证；PMI/似然思想在 07-HotTopics 的 tokenizer evaluation（比如"多语公平性"评测）也用；分词碎片率对 embedding 检索的影响在 04-RAG。

---

### 题 21：Unigram 分词与 BPE 的区别

**【考察意图】**
Unigram 名气小二面却问——因其代表"从大词表做剪枝"的反向思路 + 概率模型 + Viterbi 分割三件套的真正"生成式"分词观。考的是能不能从"方向性"把它与 BPE 的差异说成方法论差异。

**【深度解析】**

**Unigram LM（Kudo 2018；SentencePiece 内置）**：

算法骨架（自顶向下）：
1. **初始化超大奖池**：种子方式多样（如用一段 BPE/启发式把高频 sub-string 全列进来，几十万候选）。
2. **设概率模型**：一个分词 $w = x_1\cdots x_m$ 的概率为**独立链**乘积：
$$P(w) = \prod_{i=1}^{m} P(x_i),\qquad \sum_{x\in V} P(x)=1$$
（无顺序、无位置信息——"unigram"）。
3. **EM 估计 P(x)**：语料总似然 $\sum_w \log P(w|w\text{的最优分割})$，用 **Viterbi**（动态规划）找每个词的最优分割；由分词的频率重估 $P(x)$；EM 循环到收敛。
4. **剪枝**：对每个候选 x 计算**移除后的语料似然损失** $\Delta\mathcal{L}_x$（删掉后 Viterbi 只能选次优分词），删除 $\Delta\mathcal{L}$ 最小的 20% 剩余 chunk（保留单字符种子永不删）；回到 3。循环至词表大小达到预算。

**VS BPE 逐项对比**：

| 维度 | BPE | Unigram |
|---|---|---|
| 方向 | 自底向上（合并）| 自顶向下（裁剪）|
| 词表假设 | 无概率、merge 序列 | **显式概率模型** |
| 训练目标 | 压缩式（合并价值） | **最大似然**（EM） |
| 词表过程 | 逐步增 | 从大到小删 |
| 分词/编码 | 确定性贪心（merge 回放/最长匹配）| **Viterbi 全局最优**；可**采样**多个分割 |
| ambiguity 处理 | 无 | 有—— 概率分布可采 |
| 级联价值 | 训练快、工程简单 | l'不确定性处理更理论化（T5 选它）|
| 使用者 | GPT-2/3（byte）、LLaMA（SP-BPE） | T5、ALBERT、mBART（SP-Unigram） |

**为什么"可采样"是重要属性**：同一句子可有多个合法分词（如 "个体 经 济" vs "个体经济"），训练时**随机采样**不同分词＝隐式数据增强；也会引出"分词不唯一"的问题（比 grounding 更鲁棒）。BPE 完全不动。
**为什么 T5 选 Unigram**：多语言/大词表下自顶向下剪枝更稳健（BPE 在 filter 低资源语料时容易堆碎片）；且无 merge 顺序依赖（可并行、可增量）。

**复杂度**：EM 每轮一次全语料 Viterbi（O(语料·平均片段数)），实际 SentencePiece 在百万句级语料上的训练是小时级。剪枝强度：论文表述为每轮删除使整体似然损失最小的约 20% 词汇（SentencePiece 实现的 shrinking_factor=0.75 即每轮保留 75%、删除 25%，同一件事的两种记法）。**数值三连：初始 seed 几十万个、每轮删约 1/4、Viterbi DP**——面试能说出机制与量级即可，别死记常数。

**【30秒答题框架】**
"BPE 是自底向上往外合，Unigram 是自顶向下往里删：先起超大候选表，给每个子词一个概率、假设独立链，EM + Viterbi 找最优分词并重估概率；每轮删除'删掉后总似然损失最小'的一批，直到词表达标。结果差异三点：真最大似然 vs 压缩近似；Viterbi 全局最优 vs 贪心；分割可采样（软）vs 确定（硬）。T5 用它，LLaMA 用 BPE。"

**【追问与陷阱】**
- 追问：Viterbi 是对每个"词"做一次吗？——预分词后的每个 chunk（句子片段）独立 DP；不是全句一个 DP（因为空格先切）。
- 追问：Unigram 为什么更好处理低资源语言？——候选池大，低频字符串仍入池，删除按似然损失决策，不像 BPE 高频单字对压制一切。
- 陷阱：说"Unigram 一定比 BPE 好"——生成模型生态（byte-BPE + tiktoken 系）与 FlashAttention 那些并行 bias 使 BPE 仍是主流；各有生态位。
- 陷阱：说 Unigram 的 EM 与 Transformer 的参数训练"类似"——一个是词表概率重估，无梯度。

**【知识串联】**
分词不确定性与 LLM 输入的鲁棒性在 06-Safety/Evaluation 的 prompt 扰动鲁棒性测试中常被忽视；EM/Viterbi 思想与 02-Inference 的 beam search/解码（Viterbi 之于 HMM=search 之于 CRF）概念同族。

---

### 题 22：中文 LLM 的分词有什么特殊之处？

**【考察意图】**
中文场景加分项：考 tokenizer 的多语公平性、词表预算中"中文 token 数"的设计取舍、以及 vocab 扩展（Chinese-LLaMA）这类工程史。面试官想听"字/词/byte 三级的权衡"。

**【深度解析】**

**挑战三点**：
1. **无空格**：词边界歧义（「中华人民共和国」是 1 词/3 词/7 字都可自洽），语言学界本身没有唯一答案——**先分词再 BPE** 的方案依赖外部分词器的错误传播。
2. **字符集/长尾**：常用汉字 ≈3500（一级）/6763（二级 GB2312）/ Unicode CJK ≈ 2万+；英语 26 letter vs 汉字千级"原子"——若"每字一 token"中文基础 vocab 就要 8k+ 起步。
3. **字节编码**：UTF-8 中汉字占 3 字节 → **byte-level BPE 若中文语料不足，一个汉字会被切成 2–3 个"UTF-8 半字节" token**，序列长 ×2–3、推理成本同增（二面可给：LLaMA-1 纯英文词表对中文大约 token:汉字 ≈ 1.5–2.5:1，中文扩表后接近 1 汉字 1 token——量级，注明是经验值、非固定数）。

**三层解决（原文给的）+ 现代 ground truth**：
1. **字级**：BERT-Chinese（char-level，无 wordpiece merge）；优点粒度稳定、易学；序列长。
2. **SentencePiece**：Unigram/BPE 直接在原始字符流（空格→▁）上训练，**无外部依赖**——中文也不错；GPT-2 的 byte-BPE 也是"语言无关但中文压缩率差"。
3. **现代中文 LLM 路线**：**原生多语词表 + 中文语配额**——
   - ChatGLM 65,024、Qwen（cl100k 变体 152k）、DeepSeek 128K、LLaMA-3 128,256 ——普遍把 vocab 扩 2–4×，其中**中文字符专用 token 大几千到一万+**；
   - Chinese-LLaMA-Alpaca 一系：**在 LLaMA 32K 词表上「增量加中文 token」+扩 embedding 行 + 短期 continued pretraining**，把中文压缩率提约 1 倍余——vocab 扩展本质是"给 embedding 矩阵加行 + 初始化 + 继续训"，与模型其余部分兼容。
   - Qwen 采用 tiktoken-BPE（byte-level）+ 巨大 vocab（~152K）→「英文 1 token≈4 chars、中文≈1.5 汉字/token」级别的压缩率。

**延伸（显式可量化的"中文分词影响"）**：
- **压缩率**影响 KV cache 与 prefill 成本线性、attention 成本二次——中文转 token 越碎，同输入内容推理成本越高；
- **信息密度**：token≠字义单位，中文"成语/术语"复合在一个 token 内的模型内部表征好，碎成 byte 后语义被稀释（需要更多层来"汇聚"）；
- **训练配额**：中文 token 占比与词表中文字数的匹配不当会导致"中文 embedding 行训练不足"（学不好）或"字符级退化"。

**【30秒答题框架】**
"中文三大挑战：没空格、字符集大、UTF-8 3字节起步。历史上三层方案：字级（BERT-Chinese）、SentencePiece 框架、现代路线是'大词表+byte-level BPE+中文语料加额'——ChatGLM 65K、Qwen 152K、DeepSeek 128K。还有工程史上的 Chinese-LLaMA 做法：在 LLaMA 词表增量加中文 token、扩 embedding 行再短期续训，压缩率明显提升。核心影响：中文 token 化的碎不碎直接决定推理成本（KV/线性之和）。"

**【追问与陷阱】**
- 追问：词表里中文 token 多了会不会挤掉英文质量？——大数据下互相影响小，但 embedding&head 输出矩阵参数会被中文"分走"，lm head 与 vocab 同大的模型 lm head 略增参数占比（Quantization 的分组处理，画架构时注意）；多语平衡靠语料配额，不靠 vocab。
- 追问：jieba 分词+BPE 的路线现在还有人用吗？——被 SentencePiece 取代——外部依赖+错误传播，没必要。
- 陷阱：把 UTF-8"3字节/汉字"记成 UTF-16 或 2 字节——中文基本平面汉字 UTF-8 是 3 字节。
- 陷阱：说中文 LLM 都是从 char-level 开始；现代全部是 BPE/Unigram 的子词，字级只是 BERT 时代的做法。

**【知识串联】**
词表大小权衡独立成题 23；压缩率↔推理成本换算在题 24；多语 token 公平性的训练投入在 03-FineTuning 的中文继续预训练章节；LLM 中文的 tokenizer 经济在 09-SystemDesign 的成本估算非常关键。

---

### 题 23：词表大小如何选择？

**【考察意图】**
考"trade-off 量化"：embedding 参数、softmax 开销（lm head 的 $2 d V$ FLOPs）、压缩率三方联动的正确公式化，外加词表对多语、代码、数字等 token 类型的影响。二面追问"词表越大越好吗"要有定量反驳。

**【深度解析】**

**三个直接成本**（每 token 均摊）：
1. **参数**：embedding 表 $V\times d$ + lm head $d\times V$（tie 则一份）≈ $2Vd$（不 tie）/ $Vd$（tie）。按 7B（$d=4096$）折算：V=32K → $2Vd$≈0.26 B 参数（占总参数 ~4%）；V=152K → ≈1.25 B 参数（~18%）——**占比差异巨大**。
2. **lm head 计算**：每 token 输出 logit $2dV$ FLOPs。V=152k、d=4096 → 1.25 GFLOPs/token vs 模型每 token 前向 2N≈13 GFLOPs——**约 10%**。
3. **序列长度（反向收益）**：大词表 → 压缩率高 → **同文本 token 数更少** → attention 的 n² 项和 KV cache线性下降。1M token 文本：compress rate 从 3 char/tok → 4 char/tok，token 数 −25%，prefill 算力 −25%。

**词表过大（V→∞，软上限）**：
- 单 token freq 过低：**embedding/head 训练不足**（长尾 token 的梯度样本少，表示退化）——BPE 的"高频初段"原理决定这是自然惩罚；
- softmax 的**数值/显存**（logit 矩阵 V×bs seq）与采样算力变大；某些分布式实现 lm head row-parallel 重排开销；
- **嵌入矩阵碎片化**（多语挤兑）：token 间融合不好，下游 zero-shot 恶化。
- 有趣的实证：V=64K 之后 loss 边际收益递减；V 超 100K 主要在多语/代码场景回本。

**现代实践数字**（原文件表+补充）：
- GPT-2 50,257；BERT 30,522；LLaMA/2 32,000；LLaMA-3 128,256（tiktoken-style BPE，重点扩代码/多语）；ChatGLM 65,024；Qwen ≈152K；DeepSeek-V3 128,000。**趋势：从 32K → 128–256K**，生成长上下文+多语+代码需求把压缩率的权重抬高。

**与"每 token 意义"的联动**：
- 词表大小决定"每个 token 的平均语义/亚信息量"：
   - 小词表（8K）：token 碎、语义细但序列长——弱模型/端侧（节省 head 参数）；
   - 中 32K–64K：通用平衡；
   - 100K+：长/多语/代码高吞吐。

**【30秒答题框架】**
"三个账：一、参数账——embedding+head 是 2Vd，词表翻倍它们的内存与 lm head FLOPs 同翻倍，7B 模型 V 从 32K 到 152K 的 lm head FLOPs 每token会从 ~2% 涨到 ~10%。二、序列账——词表大、压缩率高，token 数下降，attention 二次项和 KV cache 都省。三、长尾账——V 大后单 token 频次变低，embedding 训练不足。所以 32K→64K 是通用甜区，128K+ 是多语/代码/长上下文的自觉选择，再大就长尾_gamma。"

**【追问与陷阱】**
- 追问：embedding 和 lm head 权共享好不好？——小模型省参数、质量略损；GPT-2 tie、LLaMA **不 tie**（scale 大模型不差这点参数）。
- 追问：词表大小影响什么"非参数"成本？——softmax 显存、Vocab 上分布 parallel all-gather、多语 token 之间的迁移等。
- 追问：什么情况下反而要小词表？——端侧（head 是参数大头的小模型）、task 特化（只有英文的场景）。
- 陷阱：说"大词表=更强模型"——是**吞吐**优化，不是能力上限的决定因素；能力主要由数据/参数决定。
- 陷阱：忽略 lm head 的 FLOPs 是 per-token（decode 每步都要算），把它当一次性成本。

**【知识串联】**
token 压缩率↔推理成本精确账在题 24 & 02-Inference/01；多 vocab tokenizer 在 04-RAG（检索端与生成端 tokenizer 不一致的问题）也有同类讨论；ledger 09-SystemDesign 的 token→成本表直接引用此题数字。

---

### 题 24：token 数量与序列长度的关系？

**【考察意图】**
工程感知题：考对不同语言不同内容的 token/char 换算手感 + "压缩率是影响推理成本的一阶变量"这一认知；也考数据从 tokenizer 视角做成本预算的能力。

**【深度解析】**

**经验换算（带出处）**：
- 英文（GPT-2/tiktoken 系）：**1 token ≈ 0.75 word ≈ 4 chars**（OpenAI 官方说明：100 tokens≈75 words）。
- 中文：**与词表强相关**：纯英文词表（LLaMA-1/2 32K）≈1–2.5 token/汉字；中文扩词表后（Qwen/DeepSeek-128K）≈**0.75–1.5 token/汉字**。粗略口诀"**1 汉字 ≈ 1.5–2 token**"是无中文优化的 GPT-2 系，"1 token ≈ 1–1.5 汉字"是 Qwen 级中文扩表。
- 代码：**比自然语言更密**（标识符/缩进/重复符号）——GPT-4 官方口径"1 token ≈ 3 chars of code in English"，但 Python 片段常 1 token≈2–3 chars；JSON/key-heavy 内容甚至更碎。
- 特殊符号/数字：数字常按 1–3 字符一 token（GPT-2 每 1–3 digits 一 token；GPT-4 更好）；数学表达式碎。

**对成本/延迟链条的影响（量化）**：
- **KV Cache 线性**：$2\,L\,n_{kv}\,d_h\cdot s\,$bytes；同语义内容 s↑ 1.5×，所有线性项 ×1.5。
- **Attention 计算二次**：prefill 的 score/AV 部分按 s² 增长——序列翻倍，预算 4×。
- **decode 延迟**：输出 token 数直接就是时延（每 token ~几十 ms 量级），**输出端长 = 慢**。
- **API 定价**：per-token 均价 × 每 query token 数。
**例**：同一 5000 字中文文档，LLaMA-1 32K 词表 ≈7500–12500 token；Qwen 152K ≈ 4000–7500 token——**同样的 RAG top-k 容量多塞 1 倍内容**（配 RAG 场景直接换算上下文容量）。

**"压缩率 vs 细粒度"权衡**（原文点睛）：
- 压缩率大 → 短序列 → 快；但**过碎合并**（尤其代码/低资源文体）会让"子词的语义单位"碎片化——模型要有更多层/更多头做"重新拼装"，在数据分布 shift 时退化（如全新域名缩写）。
- 反向极端（char-level）：表达自由但序列暴涨 —— 历史上 TPU 时代 char-level 尝试，皆因长序列成本失败。

**从业人员小抄**：成本估算公式（面试能背出）：
$$\text{cost} \approx \text{tokens} \times (\text{per-token price})，\quad \text{tokens} \approx \frac{\text{chars}}{\text{compress rate}}$$
其中 compress rate 英文≈4 chars/token、中文（新词表模型）≈1.5–2 chars/token、代码≈2–3 chars/token。

**【30秒答题框架】**
"英文 1 token≈4 字符/0.75 词，中文 1 汉字 1–2 token（取决于词表中文优化），代码更碎。这个换算的真正意义是它是推理成本的第一因：token 数进 KV cache 线性、attention 二次、decode 串行延迟直接相乘。原文的'压缩率影响'要辩证：高压缩率省成本，但碎粒度语义被稀释，新 domain 会退化——所以 2–4 char/token 是甜区。估算成本时 chars÷compress rate 即 token 数。"

**【追问与陷阱】**
- 追问：同样 1000 万 token 训练语料，中文和英文谁的"信息"更多？——token 相同下英文 char 数 ×4 而中文 ×1/2，**中文 1000 万 token 塞的汉字更少**——语料配比要按"字/token"校正，不是纯 token。
- 追问：tokenizer换了对已有 fine-tune 的模型影响？——**embedding/head 全变**，不能只换 tokenizer；换 tokenizer 意味着重训或大 Vocabulary Resizing（题 22 的 Chinese-LLaMA 做法是折衷）。
- 陷阱：把"≈4 chars/token"用到中文（×5 估算错误）。
- 陷阱：忽视了**代码/JSON** 的特殊碎化（RAG 塞代码文档时上下文预算按 chars／2–3 估）。

**【知识串联】**
本换算直接服务 02-Inference/01（KVCache/attention 长度效应）、09-SystemDesign 的成本模型、04-RAG 的窗口预算；分词与 embedding/头固定的"模型兼容性"接 03-FineTuning。

---

# Part C · 损失函数与优化器（03-Loss-And-Optimization.md）

### 题 25：LLM 预训练使用什么损失函数？

**【考察意图】**
验证"损失函数→训练信号密度→评估指标"这条主链是否打通：能不能写出公式、说清为什么选它、以及 PPL 如何与 bits/token 换算。这是所有 LLM 岗的开场基础题。

**【深度解析】**

预训练目标是**自回归 next-token prediction 的负对数似然**（teacher forcing，训练时一次并行算出全部 T 个位置的损失）：

$$\mathcal{L}(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log P_\theta(x_t \mid x_{<t})$$

关键机制拆解：

1. **每个位置都是一个样本**：一条 T 长度的语料产生 T 份监督信号（对比 BERT 的 MLM 只在约 15% 的 [MASK] 位置有信号）。这是 decoder-only 数据效率高的核心原因（呼应题 16）。
2. **与 softmax 分类的联系**：设 logits $z_t\in\mathbb{R}^V$，则
   $$\mathcal{L}_t = -\log\frac{e^{z_{t,x_t}}}{\sum_v e^{z_{t,v}}}$$
   对 logits 的梯度（经典推导，答 KL 题时复用）：
   $$\frac{\partial \mathcal{L}_t}{\partial z_{v}} = \text{softmax}(z_t)_v - \mathbb{1}[v=x_t]$$
   推导两行不跳步：$\mathcal{L}_t = -z_{x_t} + \log\sum_k e^{z_k}$，对 $z_v$ 求导 $=-\mathbb{1}[v=x_t] + e^{z_v}/\sum e^{z_k}$。
3. **数值实现**：实际用 `log_softmax`（等价 $z-\log\sum e^z$）+ label gather，配合减 max 防溢出；大模型还常加 **z-loss**（如 PaLM 用系数 $10^{-4}$ 项 $c\big(\log\sum_k e^{z_k}\big)^2$）把 logits 总体幅度约束住、抑制 fp16 下的 softmax 溢出/loss spike（也是 logit drift 控制手段）。
4. **Perplexity 与信息量换算**：PPL $=\exp(\mathcal{L})$（loss 以 nat 计），含义是"模型在每个位置等价于从这么多个等概率候选中做选择"。与 bits/token 互换：
   $$\mathcal{L}\,[\text{bits/token}] = \mathcal{L}\,[\text{nats}] / \ln 2 \approx 1.443 \times \mathcal{L}_{nats}，\qquad \text{PPL} = 2^{\mathcal{L}_{bits}}$$
   例：loss=2.0 nats → PPL≈7.39 → 约 2.89 bits/token（相当于每个 token 承载 2.89 bit 信息压缩）。
5. **label smoothing 不是预训练标配**：会把硬目标摊软，破坏概率校准与 PPL 可比性；常见于机器翻译/部分 SFT 场景，预训练不用。

一次能跑的最小实现（PyTorch，含 shift 细节——最常被问的坑）：

```python
import torch, torch.nn.functional as F

def lm_loss(logits, targets, ignore_index=-100):
    # logits: (T, V)  已含全部位置的下一词分布；targets: (T,)
    # 惯例：logits[t] 预测 targets[t]；若模型输出的是"每个位置的编码"，需要右移一位
    return F.cross_entropy(logits.float(), targets, ignore_index=ignore_index)

# 等价手写（帮助面试白板）：
def ce_manual(logits, targets):
    z = logits - logits.max(-1, keepdim=True).values   # 减 max 防溢出
    logZ = torch.logsumexp(z, -1)                      # = log Σ exp
    return (logZ - z.gather(-1, targets[..., None]).squeeze(-1)).mean()

@torch.no_grad()
def perplexity(logits, targets):
    return lm_loss(logits, targets).exp()
```

**【30秒答题框架】**
"预训练就是 next-token prediction 的交叉熵：负对数似然对整个序列求平均。每个 token 都是监督信号，所以数据效率高。数值上走 log_softmax 加标签 gather，减 max 防溢出，大模型常配 z-loss 稳 logits。评估指标是 PPL=exp(loss)，跟 bits/token 是差一个 ln2 的换算，PPL 越低代表压缩得越好。"

**【追问与陷阱】**
- 追问：为什么不用 KL/均方误差？（答案见题 26：one-hot 下 CE=KL，且 CE 对 logits 梯度形式最简）
- 追问：loss 和 PPL 哪个用来 monitor 训练？loss（加法可微、稳定），PPL 显示给人看（乘性，对早期 loss 变化更敏感）。
- 追问：instruction tuning 后 loss 还能当 PPL 报吗？不能直接比——数据分布不同；要同域 test set 才可比（评估详见 06-Safety-Evaluation）。
- 陷阱：忘了一个长度为 T 的序列有 T 个 loss 位置（最后一个 token 也预测终止符），即 shift 细节；把 shift 写反是最常见 bug。
- 陷阱：混淆"每 token loss"（×T）与"每序列 loss"（×1），影响 batch 平均的口径。

**【知识串联】**
CE/KL/熵关系是题 26；z-loss 与混合精度稳定性是题 32；"每个 token 都是信号"支撑题 16 的 decoder-only 论证；PPL 是 06-Safety-Evaluation 里评测章节的基础指标；loss spike 处理（skip/rollback）在题 31 深挖。

---

### 题 26：交叉熵、KL 散度、信息熵的关系

**【考察意图】**
信息论基本功 + "为什么工程上用 CE 而不是 KL"的辨析能力；顺带检测候选人能否当场推导 CE 对 logits 的梯度（二面常客）。

**【深度解析】**

三个定义：
$$H(p)=-\sum_i p_i\log p_i,\qquad H(p,q)=-\sum_i p_i\log q_i,\qquad D_{KL}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}$$

1. **恒等式**（展开即证，无跳步）：
   $$D_{KL}(p\|q)=\sum_i p_i\log p_i-\sum_i p_i\log q_i = -H(p) + H(p,q)
   \;\Rightarrow\; H(p,q)=H(p)+D_{KL}(p\|q)$$
   且由 Gibbs 不等式（$\log x \le x-1$ 代 $x=q_i/p_i$ 逐项求和）得 $D_{KL}(p\|q)\ge 0$，等号当且仅当 $p=q$。这就是"交叉熵 = 熵 + 距离"，最小化 CE 等价于最小化 KL（当 p 固定）。
2. **one-hot 特例**：标签分布 $p$ 是 one-hot 时 $H(p)=0$，故 $H(p,q)=D_{KL}(p\|q)$。预训练的分类目标恰好是 one-hot，这就是"预训练用 CE 而不说 KL"的原因——**两者此刻重合**。
3. **梯度等价性**（有意思且常被追问）：设 $q=\text{softmax}(z)$，对 logits 求导：
   - CE（目标 one-hot p）：$\partial H/\partial z_j = q_j - p_j$
   - KL（一般 p）：$D_{KL}=\sum p_k\log p_k-\sum p_k\log q_k$，$\frac{\partial (-\sum p_k\log q_k)}{\partial z_j} = -p_j + q_j\sum_k p_k = q_j - p_j$
   - **两者梯度完全相同**——因为 $H(p)$ 那一半与 $z$ 无关。所以"用 CE 还是 KL"在梯度上无关紧要，差异只在**常数项的数值监控**。
4. **什么时候真用 KL 而不是 CE**：
   - **知识蒸馏**：教师分布 p 非one-hot，此时 CE=H(p)+KL，KL 项才是真正被优化的"师生逼近"；
   - **RLHF/对齐**：对参考模型的 KL 正则 $D_{KL}(\pi_\theta\|\pi_{ref})$ 防 reward hacking（详见 03-FineTuning/02-Alignment）；
   - **VAE**：$\mathcal{L}=\text{recon}+\beta\,D_{KL}(q(z|x)\|p(z))$。
5. **与 PPL 的闭环**：$H(p,q)$ 的单位是 nats，换 bits 除以 $\ln 2$；模型完美拟合时 CE→H(p)（语料的固有熵），**这就是"loss 有下界"的非工程解释**——再强的模型也压不掉数据本身的不确定性。

**【30秒答题框架】**
"关系一句话：交叉熵 = 真实分布的熵 + KL 散度，KL≥0 所以交叉熵是 KL 的平移上界，最小化二者等价。one-hot 标签时熵为 0，CE 就等于 KL，所以预训练都写 CE。梯度上两者对 logits 同形：softmax 减目标分布。真会用 KL 的是蒸馏、RLHF 正则、VAE 这类 p 不是 one-hot 的场景。"

**【追问与陷阱】**
- 追问：交叉熵和负对数似然什么关系？（NLL 是"数据的 −$\log q$"，按数据分布期望即 CE：$H(p,q)=-E_p[\log q]$；单样本 NLL 是 CE 的蒙特卡洛估计）
- 追问：为什么 KL 不对称？（$D_{KL}(p\|q)$ vs $D_KL(q\|p)$ 是 mode-covering/mode-seeking 之别，QA 里一句话即可）
- 追问：基是什么？（log₂ → bits；e → nats；GPT 报告多是 nats）
- 陷阱：把 KL 说"越小越好且可以为负"——KL 恒非负。
- 陷阱：宣称"CE 优于 KL 因为梯度更稳"——梯度其实同形；真实差异是常数项和语义。

**【知识串联】**
CE 梯度公式在题 25 使用；KL 用于 RLHF 在 03-FineTuning/02-Alignment；H(p) 下界解释"语料熵不可消除"关联 04-RAG 里的检索噪声论述。

---

### 题 27：Adam vs AdamW 的区别 ⭐⭐⭐⭐

**【考察意图】**
是否能**从公式层面**指出 L2-in-Adam 与解耦权重衰减的差别（二面最高频优化器题）。只背"We decouple the weight decay"不够——面试官会追问"为什么 L2 在 Adam 里就失效了"。

**【深度解析】**

**Adam 的完整机制**（先摆正！很多人答 AdamW 时连 Adam 步骤都缺）：
$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$
$$\hat m_t=\frac{m_t}{1-\beta_1^{t}},\qquad \hat v_t=\frac{v_t}{1-\beta_2^{t}}$$
$$\theta_t=\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

**偏差修正的推导**（面试官爱问"为什么要除以 $1-\beta^t$"，必须能写）：
以 $m$ 为例，$m_0=0$，展开递推：
$$m_t=(1-\beta_1)\sum_{i=1}^{t}\beta_1^{\,t-i}\,g_i$$
若梯度稳态（$E[g_i]=g$ 常数），则
$$E[m_t]=(1-\beta_1)\,g\sum_{i=1}^{t}\beta_1^{t-i}=(1-\beta_1)\,g\,\frac{1-\beta_1^{t}}{1-\beta_1}=g\,(1-\beta_1^{t})$$
所以 $E[\hat m_t]=E[g]$：除以 $(1-\beta_1^t)$ 恰好抵消零初始化造成的**前期低估**（$t$ 小时 $(1-\beta_1^t)$ 小）。$v_t$ 对 $g^2$ 同理。**不做偏差修正的后果**：训练初期 $\hat v$ 被低估 → 步长 $\hat m/\sqrt{\hat v}$ 爆大 → 前几十步 loss 尖峰。验证锚点：$t=1$ 时 $\hat m_1=m_1/(1-\beta_1)=g$，精确。

**L2 进 Adam 的病灶（核心推导）**：
标准 L2 正则是把罚项梯度加进 $g$：$g_t \leftarrow \nabla\mathcal{L}(\theta) + \lambda\theta_{t-1}$。在 Adam 里这个 $\lambda\theta$ 会**流进 $v_t$**（因为 $v$ 是对 $g^2$ 做 EMA）：
$$v_t \approx (1-\beta_2)v_{t-1}+\beta_2^\times(\nabla\mathcal{L}+\lambda\theta)^2$$
于是自适应分母被"正则项的平方"撑大，等效正则强度变成
$$\eta\lambda\,\frac{\text{(m 侧对 }\lambda\theta\text{ 的反应)}}{\sqrt{\hat v_t}+\epsilon}\;\propto\;\eta\lambda\cdot\frac{1}{\sqrt{v_t}}$$
即：**本来梯度就大（$v$ 大）的参数，L2 衰减被自动削弱**；梯度小的参数反而被相对更重地衰减。这在 Transformer 里恰是灾难组合——embedding/前几层与某些头的 $g^2$ 跨多个数量级，同一 λ 名义上对所有参数正则，实际强度各不相同。

**AdamW 的修复（Decoupled Weight Decay, Loshchilov & Hutter 2017/2019）**：
把衰减从梯度里拿出来，直接乘在权重上：
$$\theta_t=\theta_{t-1}-\eta\lambda\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$
- 衰减**不经过 $m/v$**：强度对所有参数统一为 $\eta\lambda$，与各自梯度大小无关；
- 学习率调度（warmup/cosine）同时缩放两臂，lr=0 时衰减也归零（物理一致：不更新就不该衰减）；
- 收敛性：AdamW 在适当 β₂ 下恢复"Adam≈SGD-with-momentum 的正则行为"，MLM/LM 实测泛化更稳（论文在 GLUE 与 image 分类上验证）。

**工程红线（LLM 实践共识）**：
- **norm 参数（LN/RMSNorm 的 γ、bias）不做 weight decay**：它们是标度/平移参数，衰减会把 RMSNorm 的增益拉向 0；
- embedding 是否 decay 各家不一（LLaMA 系列对 embedding 也 decay，但有些配方排除）；
- 典型配置：β₁=0.9、β₂=0.95、ε=1e−8、wd=0.1、clip 1.0（LLaMA 系）、lr 3e−4（7B 级）。

**可运行参考实现**（与 PyTorch 官方 AdamW 数学一致的极简版）：

```python
import torch

@torch.no_grad()
def adamw_step(p, grad, state, t, lr=1e-3, b1=0.9, b2=0.95, eps=1e-8, wd=0.1):
    if p not in state:
        state[p] = (torch.zeros_like(p), torch.zeros_like(p))  # (m, v)
    m, v = state[p]
    m.mul_(b1).add_(grad, alpha=1 - b1)                 # 一阶矩
    v.mul_(b2).addcmul_(grad, grad, value=1 - b2)       # 二阶矩
    m_hat = m / (1 - b1 ** t)                           # 偏差修正
    v_hat = v / (1 - b2 ** t)
    p.mul_(1 - lr * wd)                                 # ★解耦衰减：不进 m/v，与梯度无关
    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr) # 自适应更新

# L2-in-Adam 的错误写法（对照记忆）：
# grad = grad + wd * p        → wd 项流入 v，被 √v̂ 稀释，失效
```

**对比表（面试可直接画）**：

| | Adam + L2 | AdamW |
|---|---|---|
| λθ 走哪 | 进梯度→进 v | 直接乘权重 |
| 有效衰减 | $\eta\lambda/\sqrt{\hat v_t}+\epsilon$，随参数梯度大小漂移 | 统一 $\eta\lambda$ |
| 大梯度参数 | 正则被弱化 | 与梯度无关 |
| 与 lr 解耦 | 否（衰减被 η 与 v 双重调制） | 是 |
| LLM 训练 | 不用 | **默认** |

**【30秒答题框架】**
"Adam 是 m/v 两个 EMA 加偏差修正再除以 √v̂ 更新。AdamW 的差异一句话：权重衰减不再加进梯度、不流进 v，而是直接乘权重——所以衰减强度对所有参数统一是 ηλ，不再被各参数自己的梯度大小稀释。为什么不 decay 的参数要排除？LN 的 γ 是标度参数，衰减会把 norm 压向 0。LLM 训练标配 AdamW：β₂ 0.95、wd 0.1。"

**【追问与陷阱】**
- 追问：偏差修正不做会怎样？（初始 v 小 → 步长爆炸 → 前期 loss spike；与 warmup 的动机重叠——一个治标一个治根，同主题题 30）
- 追问：β₂ 为什么从 0.999 降到 0.95？（EMA 窗长 $1/(1-\beta_2)$：0.999→1000 步，训练早期/loss spike 后恢复太慢；0.95→20 步。GPT-3 与 LLaMA 都用 0.95）
- 追问：AdamW 与学习率调度有没有耦合？（衰减臂乘 η，所以 scheduler 同时影响两者；存在"decoupled & no-lr-scaled"的变体但非主流）
- 陷阱：把 AdamW 说成"Adam + 正确的 L2 位置"就够了——要能写出 $\hat v_t$ 公式并解释 λθ 进 v 的机制。
- 陷阱：说"AdamW=带动量的 SGD"——不对，自适应分母仍在。

**【知识串联】**
偏差修正与 warmup 的因果在题 30 展开；β₂ 窗口与训练稳定性在题 28/31；wd 与泛化关联 03-FineTuning（SFT 常用较小 wd）；解耦思想与分InParameter Group（norm/embedding 排除）是 08-Coding 屡考的写题细节。

---

### 题 28：常用超参数设置

**【考察意图】**
不是考记忆，而是考"每个数字背后有没有机制解释"。面试官会随机抽一个数（比如 β₂）问"为什么是这个值"。

**【深度解析】**

有一致文献出处的主流水位（GPT-3 论文 Table 2.1 与 LLaMA/LLaMA-2 系论文）：

| 参数 | 常用值 | 机制解释（能说出这个才算二面水平） |
|---|---|---|
| β₁ | 0.9 | 动量 EMA 时间常数 1/(1−β₁)=10 步，平滑梯度方向 |
| β₂ | **0.95**（大模型）/ 0.999（小任务） | 缩短 v 的 EMA 窗（1000 步→20 步）：LLM 训练有 spike，需要 v 快速适应；GPT-3 与 LLaMA 均 0.95 |
| ε | 1e−8 | 分母防 0；纯 fp16 更新时 ε 太小会下溢，但优化器状态保持 fp32，1e−8 安全 |
| weight_decay | **0.1**（LLaMA/GPT-3）；SFT 常 0.0–0.1 | 只作用在矩阵权重，排除 norm/bias（题 27） |
| lr | 预训练 3e−4（7B 级）/ 1.5e−4（70B 级）；SFT 1e−5–2e−5 | 随模型变大降 lr（70B 更敏感）；与 batch size 按 token 记账（大 batch 可稍大） |
| grad clip | ‖g‖₂ ≤ 1.0 | 防单步异常梯度支配（题 31） |
| warmup | **2000 steps**（LLaMA-2）/ 1%–5% 总步数；GPT-3 用 375M tokens 线性 | 题 30 五点论证 |
| 调度 | cosine → 10% peak（LLaMA）或 WSD（DeepSeek 系） | 题 29 |
| dropout | 预训练 0 / SFT 0.05–0.1 | 大模型预训练基本关 dropout（数据规模够大） |

**量化补充**（能报出下面两组数，可信度立刻上一档）：
- GPT-3 175B：lr 0.6e−4、cosine 衰到 10%、warmup 375M tokens、β₁ 0.9、β₂ 0.95、ε 1e−8、clip 1.0、batch 3.2M tokens。
- LLaMA-2 系列：AdamW（β₁ 0.9、β₂ 0.95）、ε 1e−8、cos 到 10% peak、**warmup 2000 steps**、weight decay 0.1、**clip 1.0**；lr：65B/70B 用 1.5e−4，7B/13B/34B 用 3e−4、2e−4、1.5e−4。

**为什么 LLM 的 β₂ 从 0.999 降到 0.95 是"标志性"配置**：
$v_t$ 的有效平均窗口 $\tau\approx 1/(1-\beta_2)$：β₂=0.999 → 1000 步内梯度二阶矩几乎不更新；一个 batch 的异常尖峰会让 $\hat v$ 用 1000 步才能稀释掉——中间所有 update 步长被压扁。0.95 只需 ~20 步恢复，配合更低的 lr 更稳。这同时解释了题 30 warmup 的一部分（早期 v 不可靠）。

**【30秒答题框架】**
"报数+给理由：AdamW β₁ 0.9、β₂ 0.95（大模型缩短二阶矩 EMA 窗口抗 spike）、ε 1e−8、wd 0.1 且排除 norm/bias、clip 1.0、warmup 2000 步左右、cosine 衰到 10% peak，lr 3e−4 到 1.5e−4 随规模下调。SFT 更小 lr 1e−5–2e−5、可能关 decay。每个数字都能说一句机制，不是背的。"

**【追问与陷阱】**
- 追问：ε 1e−8 在 bf16 下安全吗？（优化器状态必须 fp32；若纯 fp16 更新，ε 和 √v̂ 精度都不够——见题 32 的 master weights 论证）
- 追问：wd=0.1 会不会太大？（wd 乘 lr 生效：3e−4×0.1=3e−5 每步衰减 0.03%；cosine 后期 lr 低 → 衰减也随之减弱，与 warmup 对偶）
- 追问：为什么 pretraining 不用 dropout？（数据量 >> 模型容量时正则需求小；dropout 会破坏 KV/cache 一致性不适用于推理路径…主要是前者）
- 陷阱：口误 β₂=0.999 是"LLM 配方"——那是 Adam 论文默认，LLM 已改 0.95。
- 陷阱：报 lr 时忘说"per-token batch 大小与其联动"（lr 与 batch 的线性缩放规则）。

**【知识串联】**
warmup 的五理由是题 30；调度细节是题 29；β₂ 窗口公式在题 31（梯度噪声与 spike）复用；loss scaling 的 ε/精度联动在题 32。

---

### 题 29：常用的学习率调度策略

**【考察意图】**
考 Shape 与用途的匹配：为什么预热+余弦是预训练默认、WSD 为什么在多阶段/继续训练场景胜出、终止时的 lr 终值有什么讲究。

**【深度解析】**

1. **Warmup + Cosine Decay（最常用，LLaMA 系原文公式）**：
$$\eta_t=\begin{cases}\eta_{\max}\cdot\dfrac{t}{T_{w}} & t\le T_{w}\\[2mm]\eta_{\min}+\dfrac{1}{2}(\eta_{\max}-\eta_{\min})\Big(1+\cos\dfrac{t-T_{w}}{T_{total}-T_{w}}\pi\Big) & t>T_w\end{cases}$$
   - $\eta_{\min}$ 通常 = 0.1×η_max（LLaMA-2：cos 衰到 10%）——不是 0：末端小 lr 让最终 loss 平滑收敛，同时保留"还有一点 lr"防止过早冻结。
   - 余弦的形状：前半段缓（大 lr 稳定学大结构）、后半段陡然下降（精修）。
2. **WSD（Warmup–Stable–Decay）**：线性 warmup → **恒定 lr 的 stable 段（占大头）** → 快速 decay（10%–20% 预算内线性/余弦骤降）。优点：
   - **可中途停**：任何 stable checkpoint 拉出来做几十步 decay 就能拿到"低 loss 终点"，特别适合**预算不定**与**继续训练/mid-training**（DeepSeek 系列、MiniCPM 证实 10% 左右的 decay 预算即可达到与 cosine 相近的最终 loss）；
   - stable 段更便于关停/恢复/多阶段数据配比调整；cos 的 lr 连续变化使不同 checkpoint 不可比。
   - 缺点：stable 段中后段 loss 下降速度略逊于 cosine 的"精修"形状（实验差异很小）。
3. **其他**：线性 decay、多阶段平方根、cycle；**constant** 用于 SFT 的低 lr 也成立。
4. **Warmup 进入本题的理由**（也是以下题 30 的预演）：warmup 与"稳定衰减"配合成"慢-稳-快"三段式，是所有现代配方的母体。

**可运行片段**：

```python
def cosine_lr(step, total, warmup, peak, min_ratio=0.1):
    if step < warmup:                      # 线性 warmup
        return peak * step / warmup
    t = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + __import__('math').cos(t * __import__('math').pi)))

def wsd_lr(step, total, warmup, decay_ratio=0.1, peak=3e-4):
    d_start = int(total * (1 - decay_ratio))
    if step < warmup:  return peak * step / warmup
    if step < d_start: return peak
    return peak * (1 - (step - d_start) / (total - d_start))  # 线性速降到 0
```

**【30秒答题框架】**
"预训练标配 warmup+cosine：线性升到 3e−4，余弦降到 10% peak；LLaMA warmup 2000 步。DeepSeek 用 WSD：warmup、恒定、最后 10% 骤降——最大好处是 stable 段随时可停可续，适合预算不定与继续训练，最终 loss 与 cosine 相当。选型一句话：一次性训完用 cosine，可能停/续训用 WSD。"

**【追问与陷阱】**
- 追问：为什么余弦末端要留 10% 不归零？（完全 0 之后不可恢复；且 loss 平滑需要一个非零小 lr 扫尾；WSD 则直接降到接近 0 是因为它是"终态冻结"思路）
- 追问：warmup 到 peak 的步数按什么定？（绝对步数或总步数比例都有人用；GPT-3 按 tokens 更稳健——与 batch size 解耦）
- 追问：scheduler 与 batch size 增大的关系？（大 batch 可承受更大 peak lr；训练稳定后逐步增大 batch 的"前 k 倍 batch"策略也常见）
- 陷阱：把 warmup 写成指数—主流就是线性。
- 陷阱：说 WSD 的 stable 段 lr"越大越好"——越大越接近发散边界，spike 风险同步升。

**【知识串联】**
warmup 的存在性证明是题 30；与 β₂/wd 的数值耦合在题 27/28；WSD 与继续训练/多阶段配比在 03-FineTuning 和 07-HotTopics 的 long-tail 训练话题呼应。

---

### 题 30：为什么需要 Warmup？

**【考察意图】**
考"把一个经验配方还原成机制"的能力。五条理由要能按重要性排序，且能关联 Adam 偏差修正与 Pre/Post-Norm 的梯度结构。

**【深度解析】**

1. **Adam 二阶矩的冷启动**：$v_0=0$，即使有偏差修正（除以 $1-\beta_2^t$，题 27），$\hat v_t$ 在最初几十步对"真实梯度平方的平均水平"估计仍有较大方差——单步异常梯度会被放大成大步长。小 lr 先把参数推进"平滑区"，等 v 的 EMA 窗（β₂=0.95 → 20 步左右）填满再放开步长。
2. **初始参数远离流形/早期损失大**：随机初始化下模型输出接近均匀分布，早期 loss 高且 batch 间梯度方向反差大；一步大 lr 可能直接把 embedding 归一化破坏掉（float 边界、RMSNorm 数值问题），少量小步做"退火进场"。
3. **Post-LN 深层结构训练初期的弱梯度**：Post-Norm（原始 Transformer）在第 l 层的梯度包含连乘的 LN Jacobian，深层梯度范数收缩（题 11 的分析），用大 lr 一步打过去相当于"高噪声低信号"，warmup 让它先建立稳定规范，llen 与 Adam 统计量和 LN 的 scaling 一同就位。
4. **二阶矩与学习率的时间常数失配**：EMA 的遗忘窗 ≈ 1/(1−β₂) 步；在窗内 $\hat v$ 偏低、步长偏大。warmup 长度与这个窗匹配（几千步 ≫ 20 步窗，足够收敛）。
5. **与 lr schedule 的复合平滑**：warmup 是整条 lr 曲线的"左端平滑"，避免 step0 从 0 直接跳 peak 的冲击。

**经验量级**：原论文 4000 步（原始 Transformer）→ GPT-2/3 按 tokens（GPT-3：375M tokens linear warmup）→ LLaMA-2 固定 **2000 steps**~1%总步数。**SFT/fine-tuning 常常可以不 warm 或短 warm**（参数已在合理区域，且 lr 本来小 1e−5 级）——这是区分"预训练为什么要 warmup"与"微调是否需要"的二面考点。

**【30秒答题框架】**
"主理由三条：一是 Adam 的 v 冷启动，偏差修正只能纠均值、纠不了方差，前面几十步 ûv 不可信，小步进场；二是随机初始化下早期 loss 大、batch 间梯度方向不一致，大 lr 第一步就可能把参数打出平滑区；三是对 Post-LN 深层弱梯度（Pre-Norm 缓解这一点）。经验 2000 步到 1%，SFT 可以不 warm，因为参数本来就在合理区。"

**【追问与陷阱】**
- 追问：Pre-Norm 模型还需要 warmup 吗？（仍建议但可放宽：LLaMA 用 warmup 2000 步；Pre-Norm 主要是把"不要 warmup 就崩"改成"要 warmup 更稳"，不是免死金牌）
- 追问：warmup 太长有什么坏处？（有效训练预算减少；早期低 lr 段模型收敛到次优 basin 的风险；一般 1% 总步）。
- 追问：能不能把 warmup 理解成"防止一步把权重打到 NaN"？（部分正确：数值溢出与 fp16 有关，见题 32 的 loss scaling；但 warmup 是指"统计量没就位"的办法）
- 陷阱：把 warmup 仅仅归因于"Adam 偏差修正"——修正项存在，但方差问题才是主因；二面常压这一点。
- 陷阱：说"solver 需要预热"这类含糊话；要能点名 v 的 EMA 窗与 LN 结构两个具体机制。

**【知识串联】**
偏差修正的数学是题 27；Post-LN 梯度路径分析是题 11；混合精度下的"早训 NaN"与 loss scaling 是题 32；调度公式本身是题 29。

---

### 题 31：梯度消失和梯度爆炸如何解决？

**【考察意图】**
考"架构级 vs 数值级 vs 优化级"分层的解决思维；以及 Transformer 里爆炸为什么比消失少、真正的敌人是 loss spike。

**【深度解析】**

**Transformer 里的两种现象（先纠正预期）**：
- **消失**（更常见）：mainly 深层连乘 + 激活函数死区；LN 位置也会影响（Post-LN 每层缩放，题 11）。
- **爆炸**（不太常见但致命）：异常 batch/长序列聚合 logit；表现为 loss spike 或 fp16 inf。现代 LLM 训练的实际痛点是 **loss spike** 而非 textbook 爆炸。

**梯度消失的四类解法（逐条给机制）**：
1. **残差连接**：$\partial x_L/\partial x_0$ 里有恒等项 $I$（题 11 推导），梯度直通。
2. **Pre-Norm**：残差主干放进无 LN 通路（题 11 深挖；现代 LLM 默认）。
3. **权重初始化**：Xavier/He——让每层前向方差保持在 O(1)；GPT-2 还有残差路径缩放。
4. **激活函数**：ReLU 系（Swish/SwiGLU 门控）无饱和负区（对比 sigmoid/tanh 的 0.25 导数连乘）。
5. （结构级）**QK 缩放** $\sqrt{d_k}$（题 4）：其实 attention 里最关键的"梯度保持"就来自它。

**梯度爆炸/异常梯度的解法（工程层）**：
1. **梯度裁剪**：$\displaystyle g \leftarrow \min\!\Big(1,\ \dfrac{c}{\|g\|_2}\Big)g$，$c=1.0$（按全局范数，不是逐元素；逐元素会扭曲方向）。监控"clip 触发率"作为训练健康指标。
2. **weight decay**：把权重范数控制在有界区间（题 27；非主要手段）。
3. **小 lr + warmup**（题 29/30）：穿入动力系统稳定区。
4. **数值层**：logits 的 z-loss（题 25，PaLM 1e−4）；attention 的 −inf mask（题 17）；bf16 取代 fp16（题 32）。
5. **训练系统级**（DeepSeek-V3 做法）：检测到 spike → **回滚到最近健康 checkpoint + 跳过异常数据/step** 再继续，而不是硬扛——工业化 LLM 训练的标配。
6. **架构级防爆炸**：MLA/GQA 均不改变此题状态；**RoPE 的高频分量**与 **SSM**（Mamba）在极长序列上更稳，属于 07-HotTopics 话题。

**【30秒答题框架】**
"消失靠残差直通+Pre-Norm 把恒等通路让出来+良好初始化+不饱和激活；attention 里的 √d_k 缩放也算这类。爆炸靠全局范数 clip 到 1.0、weight decay、warmup 小 lr，数值上 bf16 和 z-loss 兜底；工程上遇到的是 loss spike——现代做法是 checkpoint 回滚加脏数据跳过。"

**【追问与陷阱】**
- 追问：clip 用 L2 还是 L∞？（norm-based 全局：保持方向、只缩模长）
- 追问：clip 触发率多少算异常？（>0 就是信号，持续>0 说明 lr/数据/模型有病；偶发 1 次不一定）
- 追问：残差让梯度"不消失"的严格说法？（∂x_L/∂x_l = I + Σ 子层链，恒等项保证下界；不是证明每条路径都大）
- 陷阱：说"爆炸用 dropout 解决"——无关。dropout 是正则不是稳定器。
- 陷阱：把"梯度见 0"同"权重不见更新"混淆（Adam 会用动量兜底一段，仍有风险）。

**【知识串联】**
残差/Pre-Norm 推导即题 11；√d_k 是题 4；clip 与 loss spike 的联动在题 28（超参）与题 30（warmup）；训练系统级的 skip/rollback 在 09-SystemDesign 与 07-HotTopics 的 infra 训练章有专题。

---

### 题 32：混合精度训练

**【考察意图】**
考"显存×吞吐×数值安全"三账一起算的能力，以及 loss scaling 的存在理由（这一段是 fp16 专属、bf16 免费的细微处，二面必抽）。

**【深度解析】**

**1. 精度格式（先摆数）**：

| 格式 | 位宽 | 尾数（含隐位） | 最大值 | 最小 normal | 典型用途 |
|---|---|---|---|---|---|
| FP32 | 32 | 24 | ±3.4e38 | ~1e−38 | master weights、优化器 m/v、数值敏感累加 |
| FP16 | 16 | **11** | ±65504 | 6.1e−5（2⁻¹⁴），subnorm 到 ~6e−8 | 前向/反向（需 loss scaling） |
| BF16 | 16 | **8** | ±3.4e38（同 FP32 指数域！） | ~1e−38 | 前向/反向（无需 scaling）；现代 LLM 默认 |

一句话差异：**FP16 是"高精度窄动态"，BF16 是"低精度宽动态"**。fp16 尾数 10 位+1 隐 ~3–4 位十进制有效，bf16 尾数 7+1 位 ~2–3 位十进制；但 bf16 的指数域与 fp32 相同（±3.4e38），所以 loss/梯度的大数不会溢出——**这就是 LLaMA/现代 LLM 默认 bf16 的根本原因**。

**2. 混合精度的标准组件（fp16 场景）**：
- **主权重（master weights）FP32**：更新累加（小 ±lr×update）必须高精度，fp16 累加会丢小数个 ulp 就停滞；
- **前向/反向用 FP16/BF16**：矩阵乘吃 tensor core（A100 bf16 312 TFLOPS vs fp32 19.5 TFLOPS——**16×** 峰值；H100 bf16 ~989 TFLOPS、fp8 1979 TFLOPS），激活显存减半；
- **FP32 优化器状态 m/v**；
- **loss scaling（fp16 专属）**：fp16 的最小 normal ≈ 6e−5，而实际损失梯度经常 1e−7 量级以下——**下溢成 0**。流程：
  1) 前向后把 loss 乘 $S$（初始 ~65536）
  2) 反向各梯度被放大 $S$ 倍，避开 subnormal 区
  3) optimizer.step 前先 unscale（把梯度除回 $S$）
  4) 动态调 $S$：出现 inf/nan → skip 本步并把 $S$ 减半；连续 N 步无溢出 → $S$×2（PyTorch GradScaler 内置）
  **BF16 全程免这一套**（宽指数域不存在下溢问题，这是 bf16 更"省心"的原因）。
- **梯度累加**发生在"unscale 之后"（或每一步分别 scale）——顺序错会让大累计 batch 的等效 scale 漂移。

**3. 显存账（能写出公式+报数）**：
每参数字节数（标准 bf16 混合精度）：
$$\underbrace{2}_{\text{bf16 权重}}+\underbrace{2}_{\text{bf16 梯度}}+\underbrace{4}_{\text{fp32 master}}+\underbrace{4+4}_{\text{m,v fp32}}=\textbf{16 B/param}$$
7B 模型 → 112 GB（还没算激活）。ZeRO/分片优化：ZeRO-1 把 m/v 分片到 N 卡（每卡 8+8B/param），推理/训练尽显存压力——详见 02-Inference/03-Quantization 的量化叠法。
激活（不含 optimizer）：每层 ≈ n_seq×d×(34 Bytes) 的多头+FFN 中间值量级（FlashAttention 后大头是 FFN 的 $n\times 4d$），序列 4k、7B 单卡也能塞下 bf16 训练（结合梯度 checkpointing）。

**4. 吞吐账（MFU）**：
$$\text{MFU}=\frac{6\,N\,D}{n_{GPU}\times\text{peak FLOPs}\times T}$$
（6ND 训练 FLOPs 见题 5；PaLM 报 ~46%、常见 35%–55% 之间。）fp16/bf16 让 6ND 不变的张大分子，直接抬 MFU 上限。

**5. 可运行参考（PyTorch 原生 amp）**：

```python
import torch
from torch.cuda.amp import autocast, GradScaler

model, opt = build_model(), torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9,0.95), weight_decay=0.1)
scaler = GradScaler(enabled=torch.cuda.is_bf16_supported() is False)  # bf16 关 scaler，fp16 开

for step, (x, y) in enumerate(loader):
    opt.zero_grad(set_to_none=True)
    with autocast(dtype=torch.float16):                # fp16 前向（bf16 时用 dtype=torch.bfloat16 且不用 scaler）
        loss = model(x, y)
    scaler.scale(loss).backward()                      # 反向梯度已 ×S
    scaler.unscale_(opt)                               # clip 前必须先还原（否则 clip 阈值失真）
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)                                   # 有 inf/nan 则自动跳过更新
    scaler.update()                                    # 连续无溢出 → S×2；溢出 → S/=2 且跳步
```

（bf16 时：`autocast(dtype=torch.bfloat16)` + `scaler = GradScaler(enabled=False)`；其余流程不变——这是面试里"bf16 相比 fp16 省哪一步"的准确答案：**免 scale/unscale/skip 流程**。）

**【30秒答题框架】**
"三句话：前向/反向走低精度吃 tensor core（A100 bf16 峰值 312 TFLOPS，fp32 只有 19.5），权重/优化器状态保 fp32 防累加损失。fp16 窄动态有下溢问题所以需要 loss scaling：乘 S 放大反向梯度、step 前再除回、溢出跳步 S 减半。bf16 指数域同 fp32 就免这套，代价是尾数只有 7+1 位。显存账 16 bytes/param，7B 就是 112GB 优化器+主权重，要靠 ZeRO 分片。"

**【追问与陷阱】**
- 追问：bf16 有丢精度问题吗？怎么补？（尾数少 3 位 → 累加误差大；缓解：残差/layernorm/QKV 内积**内部用 fp32 累加**（chunked fp32 accumulate），logits 的 softmax/CE 在 fp32——题 25 的 `logits.float()` 就是这个位置的接线）
- 追问：为什么不全 fp32？（慢：tensor core 翻 8–16 倍差距 + 显存/带宽 ×2）
- 追问：fp8？（Hopper 架构支持 fp8 训练，两种格式 E4M3/E5M2：尾数更少，需要 per-tensor scaling 与更精细的 recipe，属前沿话题 07-HotTopics）
- 陷阱：说"bf16 都比 fp16 稳所以没有代价"——精度位数少是有代价的（训练 loss 曲线更毛糙，靠 fp32 累加兜底）。
- 陷阱：忘了"optimizer 的 m/v 必须 fp32"，答"全 bf16 训练"（那叫 pure-bf16，需要特殊 recipe 如 stochastic rounding，非常规）。

**【知识串联】**
6ND 与 2ND 的 FLOPs 账在题 5；z-loss 与 fp16 softmax 溢出在题 25/31；fp32 master 与 AdamW 的 ε 关系在题 27/28；量化（推理侧）延伸在 02-Inference/03-Quantization；显存分片（ZeRO）属于 09-SystemDesign。

---

# 全章收束

## 本章高频追问 Top 5

1. **"为什么要除以 √d_k？除 d_k 行不行？"**（题 4 全文）
   - 方差解释：$\text{Var}(q\cdot k)=d_k$ → 除 √d_k 回到单位方差；
   - 梯度/数值解释：logit过大 → softmax 饱和成 one-hot，雅可比 $\text{diag}(A)-AA^T$ 趋零、梯度消失，fp16 还会 exp 溢出；
   - 除 d_k 会把方差压到 1/d_k，注意力退化为均匀平均，两边都不行——√ 是唯一"方差=1"的对称解。
2. **"softmax 能不能去掉/换掉？attention 必须是 softmax 吗？"**（题 3/4/8 联动）
   - 必须有"归一化的竞争机制"才能形成凸组合输出；线性注意力 $\phi(q)^\top\phi(k)$ 可替换（复杂度 O(n·d²)、与 KV 递推兼容），但尖峰结构/表达力下降，且要求位置编码与核化兼容——这正是 RoPE（正交旋转可看作核内函数）胜过绝对 PE 的地方（题 8 兼容性论证）。
3. **"MHA→MQA→GQA→MLA 为什么这么演进？KV cache 到底多大？"**（题 7）
   - KV cache 公式 $2\cdot L\cdot n_{kv}\cdot d_h\cdot s\cdot\text{bytes}$ 是入口；LLaMA-2 70B GQA-8 是 320 KiB/token，128k 上下文 40 GiB；
   - MQA 砍质量，GQA 以 h/g 平衡，MLA 用低秩压缩（DeepSeek-V2：cache 降 93.3%、吞吐 5.76×；RoPE 与压缩耦合是技术难点）。
4. **"Pre-Norm 和 Post-Norm 到底差在哪？warmup 为什么必须配 Post-LN？"**（题 11、30）
   - Pre-Norm：残差主干是恒等通路，$\partial x_L/\partial x_0$ 含 I；Post-Norm：主干每层被 LN Jacobian 缩放，深层弱梯度 → 训练初期必须小 lr（warmup）。表达力依旧是 Post 略强，工程全是 Pre + final norm。
5. **"AdamW 解耦的数学含义？哪些参数不该 decay？混合精度为什么 fp16 需要 loss scaling 而 bf16 不用？"**（题 27、32）
   - λθ 进 v 会被 $\sqrt{\hat v_t}$ 稀释 → 有效衰减随参数梯度漂移；解耦后统一 ηλ；norm/embedding 参数排除 decay。
   - fp16 最小 normal 6.1e−5 会让 1e−7 级梯度下溢 → loss×S/unscale/动态 S；bf16 指数域同 fp32 无下溢问题，但尾数少 3 位要靠内部 fp32 累加兜底。

> 全章 32 题已完整覆盖 01-Transformer.md（17 题）、02-Tokenization.md（7 题）、03-Loss-And-Optimization.md（8 题），编号与原文件顺序一致，每题五段式：考察意图 / 深度解析 / 30 秒答题框架 / 追问与陷阱 / 知识串联。
