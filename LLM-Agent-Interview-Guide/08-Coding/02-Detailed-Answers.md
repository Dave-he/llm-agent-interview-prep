# 08-Coding 手撕代码 · 逐题详解

> 源文件（只读，未改动）：`08-Coding/01-Coding-Problems.md`。本文为其配套详解。
> 研究背景文档：任务未指定（undefined）→ 所有推导以原始论文/官方实现为准：
> *Attention Is All You Need*、LoRA（Hu et al., 2021）、RoPE（Su et al., 2021）、LLaMA 1/2/3 技术报告、GQA（Ainslie et al., 2023）、Kaplan et al. 2020。

## 本章考点分布

| 题号 | 主题 | 一级考点 | 关键细节 | 频率 |
|------|------|----------|----------|------|
| 1 | Self-Attention | 手撕核心机制 | mask 在 softmax 之前、除以 √d_k | 5/5 |
| 2 | Multi-Head Attention | 头拆分/合并 | view+transpose 次序、contiguous | 5/5 |
| 3 | Transformer FLOPs 估算 | 数值推导 | GEMM=2mnp、训练 6ND / 推理 2ND | 4/5 |
| 4 | BPE 分词器 | 分词工程 | 训练/推理一致性、预切分 | 3/5 |
| 5 | Beam Search | 解码算法 | 长度惩罚、EOS 管理 | 3/5 |
| 6 | KV Cache Attention | 推理显存 | KV 显存公式、prefill/decode | 5/5 |
| 7 | LoRA | 微调组件 | B=0 初始化、α/r 缩放、权重合并 | 5/5 |
| 8 | 余弦相似度检索 | 向量检索 | 归一化、TopK 复杂度 | 3/5 |
| 9 | RoPE | 位置编码 | 相对位置证明、复数 reshape 顺序 | 5/5 |
| 10 | LeetCode 8 连发（10.1–10.8） | 基础算法 | LRU / TopK(含堆与 AVL 对比) / 链表 / 树 / DP / 图 / 排序 / 单调队列 | 4/5 |
| 11 | GQA（补充题：原章节缺实现，此处补全） | 注意力变体 | 头对齐 repeat_interleave、KV 显存折扣 | 5/5 |

分布结论：手撕结构 5 题（1/2/6/9/11）、算力/显存推导 1 题（3）、工程组件 2 题（4/8）、解码与微调 2 题（5/7）、基础算法 8 小题（10.x）。字节二面的典型组合是"一道 Attention 手撕 + 一道显存/FLOPs 口算 + 一道 LeetCode Medium"，本章按此深度覆盖。

---

### 题 1：Self-Attention

**【考察意图】**
验证三件事：(1) 是否真正理解 Q/K/V 三条路径各自"生产什么、消费什么"，而不是背公式；(2) mask 加在 softmax **之前还是之后**、除以 √d_k 的数学原因——这两个点几乎是手撕环节 100% 出现的追问；(3) 能否现场报出 attention 的计算量与显存复杂度（O(L²·d)），判断你是否了解它为什么是大模型推理的瓶颈之一。

**【深度解析】**

*思路推导：*

Scaled Dot-Product Attention 的完整定义（论文 3.2 节）：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

推导三步：

1. **打分**：Q·Kᵀ 的第 (i,j) 元素 = 第 i 个 query 与第 j 个 key 的点积，衡量"位置 i 想从位置 j 取多少信息"。矩阵形状 (B, L, L)。
2. **缩放**：为什么要除以 √d_k？设 q, k 的各分量 i.i.d.、均值 0、方差 1，则点积
   `q·k = Σᵢ₌₁^d_k qᵢkᵢ`，其方差 = Σᵢ Var(qᵢkᵢ) = Σᵢ E[qᵢ²]·E[kᵢ²] = **d_k**（独立随机变量和的方差等于方差之和）。
   即点积的标准差是 √d_k。d_k=128 时分数天然在 ±12 左右波动，softmax 会进入饱和区（输出趋近 one-hot），softmax 的 Jacobian `diag(p) − p pᵀ` 近似全零 → 梯度消失。除以 √d_k 把分数方差拉回 1，保证 softmax 工作在线性敏感区。
3. **加权求和**：softmax 按行（dim=-1）归一化，每行是位置 i 对所有位置 j 的注意力分布（和为 1），再对 V 加权，输出 (B, L, d_v)。注意 attention 是**非对称的**：`softmax(QKᵀ)` 不等于其转置。

*完整可运行参考实现（含维度注释）：*

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """单头自注意力：原文件骨架 + causal/padding mask + dropout 补全。"""
    def __init__(self, d_model, d_k=None, d_v=None, dropout=0.0):
        super().__init__()
        self.d_k = d_k or d_model             # Q/K 投影维度（缺省 d_model）
        self.d_v = d_v or d_model             # V 投影维度 = 输出维度
        self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v, bias=False)
        self.attn_drop = nn.Dropout(dropout)  # 原 Transformer 在 softmax 后有 dropout

    def forward(self, x, mask=None, is_causal=False):
        # x: (B, L, d_model)
        Q = self.W_Q(x)                       # (B, L, d_k)
        K = self.W_K(x)                       # (B, L, d_k)
        V = self.W_V(x)                       # (B, L, d_v)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)   # (B, L, L)

        if is_causal:                         # 因果 mask：位置 i 只能看 ≤ i
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~causal, float("-inf"))
        if mask is not None:                  # padding mask：(B,1,1,L) 或可广播到 (B,L,L) 的 bool，True=保留
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)      # 行内归一化，(B, L, L)
        attn = self.attn_drop(attn)
        out = attn @ V                        # (B, L, d_v)
        return out, attn

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 5, 16)
    out, attn = SelfAttention(16)(x, is_causal=True)
    print(out.shape)        # (2, 5, 16)
    print(attn.sum(-1))     # 每行和 = 1（验证 softmax 正确性）
```

*逐行讲解：*

- `W_Q/W_K/W_V`：三条独立投影（bias=False 是 Transformer 惯例，加 bias 对结果无本质影响）。
- `Q @ K.transpose(-2,-1)`：在最后两个维度上做批量矩阵乘，batch 维自动广播；等价 `torch.matmul(Q, K.transpose(-2, -1))`。
- `masked_fill`：在 **softmax 之前** 把非法位置置 -inf（softmax 后这些位置权重为 0）。注意 mask 的 bool 语义要与代码约定一致：本文约定 `True=保留`，所以用 `~mask`。
- `F.softmax(scores, dim=-1)`：dim=-1 是对"key 维度"归一化；写错成 dim=-2 意味着"每个 key 收到的 query 注意力和为 1"，语义完全错误。
- `attn @ V`：对每行做凸组合，输出 shape 由 V 决定 (B, L, d_v)。

*复杂度（量化）：*

- 时间：scores 一次 O(L²·d_k) + attn@V 一次 O(L²·d_v) → 总 O(L²·d)。
- 显存：scores 矩阵 (B, 1, L, L)。fp16 时 = 2B·L² bytes。B=1、h=32 头、L=4096 → 32×4096²×2B ≈ **1.07 GB**。这就是 FlashAttention 要把 softmax 中间矩阵从 O(L²) 显存压到 O(L) 的原因。
- 参数量：3·d_model·d_k（+ d_model·d_v）。

**【30秒答题框架】**
"Attention 分三步：先把 x 分别投影成 Q/K/V；score 是 Q 和 K 的点积，除以 √d_k——因为点积方差随 d_k 线性膨胀，不缩放 softmax 会饱和、梯度消失；然后 mask 在 softmax 之前把非法位置置 -inf，softmax 行归一化后再乘 V。复杂度 O(L²d)，显存瓶颈在 L×L 的 attention 矩阵，所以有 FlashAttention 这类 IO 感知实现。"

**【追问与陷阱】**

*常见手写 Bug：*
1. **mask 加在 softmax 之后**：最常见的错误。softmax 对 -inf 得到 0，之后再乘 mask 只是乘零，非法位置根本没被屏蔽。mask 必须在 softmax 前做 `masked_fill(-inf)`。
2. **漏除 √d_k**：功能不报错但训练效果差，且面试官一眼看出。
3. **mask 语义反了**：有的库约定 True=屏蔽（如 PyTorch `attn_mask` 的 bool 版本），有的约定 True=保留。拿不准就写注释并自测一行全 False。
4. **整行全 -inf → NaN**：softmax 全 -inf 行输出 NaN（torch 表现为 nan）。兜底方案：用 `float("-inf")` 换成 -1e9 这类大负有限值，或保证 padding query 行至少保留自身位置。
5. `dim=-1` 写成默认或写反方向。
6. causal mask 用 `torch.triu`（上三角）而不是 `tril`（下三角），方向反了模型学不到任何依赖。

*面试官追问：*
- " doping mask 的 (B,1,1,L) 是怎么广播到 (B,L,L) 的？" → 尾维对齐：B 对 B、两个 1 广播 L、L 对 L。
- "softmax 为什么按行不按列？" → 分布定义在"一个 query 对所有 key"上，和为 1 的是 key 方向。
- "FlashAttention 为什么快？" → 在线 softmax 分块重缩放，不必物化 L×L 矩阵，显存 O(L)，HBM 读写次数大幅下降。
- "能不能不用 softmax？" → 有线性注意力（kernel/RFA）路线，复杂度 O(L·d²)，但近似质量有 tradeoff。

**【知识串联】**
- 机制细节（多头、QKV 权重共享、为何点积不用 tanh）→ `01-Foundation/01-Transformer.md`。
- 除 √d_k 引出的"初始化方差控制"思想 → `01-Foundation/03-Loss-And-Optimization.md`。
- decode 阶段 K/V 独立于新 Q → KV Cache（题 6）。batch mask 与推理显存 → `02-Inference/03-Quantization-And-Deployment.md`。

---

### 题 2：Multi-Head Attention

**【考察意图】**
考察两处最容易当场卡住的 reshape 细节：(1) `view + transpose` 拆头、`transpose + contiguous + view` 合并头的**次序和原因**；(2) mask 如何从 (B,L) 或 (1,L,L) 广播到 (B,h,L,L)。同时验证你理解"多头 = h 个 d_k=d_model/h 的子空间并行做 attention，再拼回去过一个 W_O"，而不是"把模型复制 h 份"。

**【深度解析】**

*思路推导：*

- 单头 attention 用一个 d_k 维子空间。多头（论文 3.2.2）：h 个头，每个头 d_k = d_h = d_model/h，各自独立投影、独立打分，最后 concat 后过 W_O。数学上：
  `MultiHead(x) = Concat(head₁,…,head_h)·W_O，headᵢ = Attention(x·W_Qᵢ, x·W_Kᵢ, x·W_Vᵢ)`。
- **参数量与单头 qkv(d_model→d_model) 完全相同的实现**：用一个 `Linear(d_model, d_model)` 一次算出所有头的向量，再 reshape 拆头——因为 concat(h 个 d_k) = d_model，总参数 4·d_model²（QKV+O 各 d×d）。LLaMA-7B 每层 attention 参数 = 4×4096² ≈ 67.1M。
- 拆头的正确变形链：
  `(B, L, d_model) → view(B, L, h, d_k) → transpose(1,2) → (B, h, L, d_k)`
  `view` 是把最后一维 d_model 按"头 × 头内维度"切开；`transpose` 把头维换到 batch 后面让每个头独立成矩阵。合并是逆过程，必须 `contiguous()`（transpose 打断了内存连续性，`view` 需要连续内存；等价写法 `reshape`）。
- **√d_k 用的是 d_k 而不是 d_model**：每个头内部打分用的是该头的小维度。

*完整可运行参考实现（含维度注释）：*

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads       # 每头维度
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None, is_causal=False):
        B, L, _ = x.shape
        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B,h,L,d_k)
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B,h,L,d_k)
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)  # (B,h,L,d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,h,L,L)

        if is_causal:
            causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~causal, float("-inf"))
        if mask is not None:                # 接受 (B,1,L,L)/(B,1,1,L)/(1,1,L,L)，自动广播到 (B,h,L,L)
            scores = scores.masked_fill(~mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)                                   # (B,h,L,L)
        context = torch.matmul(attn, V)                                    # (B,h,L,d_k)

        context = context.transpose(1, 2).contiguous().view(B, L, -1)      # (B,L,d_model)
        return self.W_O(context)
```

*逐行讲解：*

- `view(B, L, h, d_k)` 前提是 Q 最后一维的内存布局确实是 [head0 的 d_k | head1 的 d_k | …]，Linear(d_model→d_model) 输出天然满足（前 d_k 列是头 0）。
- `transpose(1, 2)` 把 (B,L,h,d_k) 换成 (B,h,L,d_k)，头维和 seq 维交换——之后 `matmul(Q, Kᵀ)` 的最后两维就是头内的 (L,d_k)×(d_k,L)。
- 合并时 `transpose(1,2)` 先换回 (B,L,h,d_k)，此时张量不连续，必须 `.contiguous()` 才能 `view`成 (B,L,d_model)——写成 `view` 不加 contiguous 会直接 RuntimeError；用 `reshape` 也行（内部等价做拷贝）。
- mask 广播：causal 用 `(1,1,L,L)` 或本实现先 tril 的 `(L,L)` 触发广播；padding 用 `(B,1,1,L)`。

*复杂度（量化）：*

- 时间/层：四条投影 4·(2·B·L·d²) + 两次注意力矩阵乘 2·(2·B·h·L²·d_k) = 8·B·L·d² + 4·B·L²·d（乘加都算 FLOP 时系数再乘一，见题 3）。
- 参数：4·d_model²，且与头数 h 无关（h 只决定切法）。
- 显存：scores (B,h,L,L)，B=1、h=32、L=4096、fp16 ≈ **1.07 GB**（同题 1）。

**【30秒答题框架】**
"多头就是把 d_model 切成 h 份，每份 d_k=d_model/h 独立做 attention，concat 后过 W_O。实现上一次 Linear(d_model→d_model) 再 view(B,L,h,d_k)、transpose 成 (B,h,L,d_k)；合并反向操作且必须 contiguous。参数量和投影计算都是 4d² 每层，attention 矩阵乘是 O(L²d)。除以 √d_k 而不是 √d_model。"

**【追问与陷阱】**

*常见手写 Bug：*
1. **view 与 transpose 次序搞反**：先 transpose(1,2) 再 view 会把不同位置的值切错组——拆头是"先切后换"。
2. **合并时忘 contiguous**：transpose 后 view 直接报错（或用 reshape 静默拷贝）。
3. **mask 形状错误**：把 (B,L) 的 padding 掩码直接 `masked_fill` 到 (B,h,L,L)，尾部维对不上会广播出错误语义；正确是插入两个头维变 (B,1,1,L)。
4. **除以 √d_model**：多头内部打分维度是 d_k。
5. **n_heads 不整除 d_model**：要在 `__init__` 里 assert。
6. 输出投影 W_O 常被手写者漏掉，导致 concat 的多头输出直接作为层输出——结构与残差回路的容量都会错。

*面试官追问：*
- "为什么多头有效？" → 每头学习不同的表示子空间/关系模式（论文 3.2.2 的可视化：一个头抓语法依存、一个头抓远距离共指）；点积注意力本身是线性算子多次组合，多头增加了函数类多样性。
- "头数越多越好吗？" → h↑ 则每头 d_k↓，dim 太小时投影瓶颈（d_k<64 无明显收益，常见 64/128）；有 head pruning 研究表明部分头可剪。
- "MQA/GQA 是什么？" → 减少 KV 头数、省 KV cache——引出题 11。
- "为什么 concat 后还要 W_O？" → 各头独立产出，需要一个跨头混合投影；等价于对 block-diagonal 的多头输出做一次全连接。

**【知识串联】**
头拆分/合并 → `01-Foundation/01-Transformer.md`；KV 头数缩减(题 11) 与解码显存 → `02-Inference/01-KVCache-And-Acceleration.md`；FLOPs 口算（题 3）依赖本层的 4d² 参数结构。

---

### 题 3：Transformer FLOPs 估算

**【考察意图】**
看你是否具备"训练/推理算力口算"能力——这是字节二面区分候选人的必考点。核心是：(1) 矩阵乘 FLOPs = 2·m·n·p（乘加各算一次）的口径要固定；(2) 会把一层拆成 QKV/attention/输出投影/FFN 四块；(3) 指出"训练 ≈ 6·N·D、推理 ≈ 2·N·D"这条经验公式并给出推导；(4) 对具体模型（LLaMA-7B）给出 TFLOPs 量级、并与算力利用率 (MFU) 联动。

**【深度解析】**

*思路推导：*

1. **口径**：矩阵乘 A(m×n)·B(n×p)：共 m·p 个输出元素，每个要 n 次乘 + n 次加 = 2n FLOPs → **2·m·n·p FLOPs**。线性层 y = W·x（W: out×in）作用在 B·L 个 token 上：FLOPs = 2·(B·L)·in·out。
2. **单层分解**（d = d_model，d_k = d/h，标准 4×d 的 FFN）：
   - QKV 投影：3 条 (d→d) → 3 × 2BLd² = **6BLd²**
   - scores（Q·Kᵀ）：(B,h,L,d_k)×(B,h,d_k,L) → 2·B·h·L²·d_k = **2BL²d**
   - attn·V：同样形状的矩阵乘 → 又是 **2BL²d**
   - 输出投影 W_O：**2BLd²**
   - FFN（d→4d→d，两次矩阵乘）：2 × 2BLd·4d = **16BLd²**；若用 SwiGLU（3 个矩阵，d_ff≈8d/3）：6·BL·d·d_ff = 16BLd²，恰好近似持平。
   合计每层 ≈ **24·B·L·d² + 4·B·L²·d**（注：原文件只算了 scores 一次矩阵乘（2BL²d），漏了 attn·V；面试时主动补上这一点是加分项。）
3. **Embedding / LM Head**：embedding 是查表（gather），FLOPs 记 0；LM Head 是 (B·L, d)×(d, V) 的矩阵乘 = **2·BL·V·d**（tied/untied 不影响这个口径；原文件写 2·2·B·L·V·d，是把 lookup 也当乘法的保守口径，建议统一到 2BLVd 并现场说明差异）。
4. **宏观经验公式**：每个非 embedding 参数在一次前向中约贡献 2 FLOPs（一次乘一次加）→ 前向 ≈ 2·N·D（N 参数量，D token 数）；反传 ≈ 前向 2 倍（dW 一份、dx 一份）→ **训练 C ≈ 6·N·D**（Kaplan scaling law 的口径）。
   - 校验：LLaMA-7B（d=4096，32 层）：每 token 前向 FF/投影 = 24·d²·32 = 4.03×10⁸，加 attention 项 4·L·d·32 与 LM Head 2·32000·4096≈2.6×10⁸，单 token 前向 ≈ 1.4×10¹⁰ FLOPs，与 2·N = 1.34×10¹⁰（N≈6.7B）一致。
5. **示例（原文件的调用）**：LLaMA-7B, d=4096, layers=32, L=2048, B=1：
   - 每层 = 24·2048·4096² + 4·2048²·4096 ≈ 8.25×10¹¹ + 6.87×10¹⁰ ≈ 8.93×10¹¹
   - ×32 层 ≈ 2.86×10¹³，加 LM Head 2·2048·32000·4096 ≈ 5.37×10¹¹ → 未做因果修正的保守口径 ≈2.92×10¹³；含 causal×0.5 修正后 ≈2.87×10¹³，**约 29 TFLOPs / 一次前向**
   - 对照硬件：A100 bf16 dense 峰值 312 TFLOPS，若实测 4 ms/step → 有效算力 7.3 TFLOPS → MFU ≈ 2.3%（这就是"长序列推理算力利用率极低"的定量说法）。

*完整可运行参考实现（含维度注释）：*

```python
def estimate_transformer_flops(
    vocab_size, d_model, n_layers, n_heads, seq_len, batch_size,
    d_ff=None, n_attn_matmuls=2, causal_half=True, swiglu=False,
):
    """前向 FLOPs 估算。口径：乘加各算 1 FLOP → GEMM (m,n)·(n,p) = 2mnp。
    参数：
      n_attn_matmuls=2        # scores(QKᵀ) 与 attn·V 两次矩阵乘（原文件只计了 scores）
      causal_half=True        # causal 场景 QKᵀ 的下半无效，×0.5
      swiglu                  # True 时 FFN 用 3 个矩阵（gate/up/down），d_ff≈8d/3
    返回：一次前向总 FLOPs
    """
    d_k = d_model // n_heads
    d_ff = d_ff if d_ff is not None else 4 * d_model
    B, L, d = batch_size, seq_len, d_model

    qkv_flops   = 3 * 2 * B * L * d * d                       # (B,L,d)·(d,d) ×3 → 6BLd²
    scores_gemm = 2 * B * n_heads * L * L * d_k               # 2BL²d
    attnv_gemm  = 2 * B * n_heads * L * L * d_k               # 2BL²d
    if causal_half:
        scores_gemm *= 0.5                                     # causal 只算上三角
    attn_flops  = scores_gemm + attnv_gemm
    out_flops   = 2 * B * L * d * d                           # 2BLd²

    n_proj = 3 if swiglu else 2                                # SwiGLU: gate/up/down 三条
    ffn_flops  = n_proj * 2 * B * L * d * d_ff                 # 标准 4d: 16BLd²

    per_layer = qkv_flops + attn_flops + out_flops + ffn_flops
    total     = per_layer * n_layers

    embed_flops = 2 * B * L * vocab_size * d                   # 仅 LM Head；embedding 查表 = 0
    return total + embed_flops, {
        "per_layer": per_layer, "attn": attn_flops, "ffn": ffn_flops}


if __name__ == "__main__":
    # LLaMA-7B：vocab=32000, d=4096, 32层, 32头, d_ff=11008(SwiGLU)
    total, parts = estimate_transformer_flops(
        vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
        seq_len=2048, batch_size=1, d_ff=11008, swiglu=True)
    print(f"FLOPs: {total/1e12:.1f} TFLOPs")       # ≈28.7 TFLOPs（causal×0.5 修正后；不修正 ≈29.3，保守上界）
    print("per token:", int(total // 2048), "≈ 2N =", int(2 * 6.7e9))
```

*逐行讲解：*

- 系数 `n_proj` 把标准 FFN（2 矩阵）与 SwiGLU（3 矩阵）统一起来；SwiGLU 时若 d_ff=8d/3，3·2·d·(8d/3)=16d²，与标准 4d 等宽——所以 LLaMA 才选 11008≈8·4096/3。
- `causal_half`：训练与 LLM 推理的 QKᵀ 实际只用上三角，严格口径要 ×0.5；不除也不算错（保守上界），面试时说明即可。
- 返回 dict 便于现场逐块报数（面试官常常要求 breakdown 而不是一个数）。

*复杂度：*
- attention 部分 O(B·L²·d)，投影/FFN 部分 O(B·L·d²)。L < d 时 FFN 主导（LLaMA-7B 单步 ≈ 92% FLOPs 在投影+FFN），L ≫ d 时 attention 主导（长上下文瓶颈）。KV cache 显存则是 O(L·d) 线性（见题 6）。

**【30秒答题框架】**
"先固定口径：GEMM 是 2mnp。每层四块：QKV 6BLd²，attention 两个矩阵乘共 4BL²d，输出投影 2BLd²，FFN 16BLd²——合计 24BLd²+4BL²d，乘层数，再加 LM Head 2BLVd（embedding 是查表不算）。宏观校验：前向 ≈2ND、训练 ≈6ND，LLaMA-7B 2048 上下文一次前向约 29 TFLOPs，拿它除硬件峰值就能算 MFU。"

**【追问与陷阱】**

*常见手写 Bug：*
1. 乘加只算一次（忘了 +1 FLOP），整体半价。
2. attention 只算 QKᵀ 漏了 attn·V（原文件口径的坑）。
3. 把 FFN 写成 `4·d²` 而不是 `16·B·L·d²`（丢了 8× 系数：两个矩阵 × 4d 宽度）。
4. embedding 当成矩阵乘计 FLOPs（实际是 gather = 0）；LM Head 反而漏算（untied 时它是最大单矩阵之一，7B 上 32000×4096）。
5. 用 d_model 除 h 得 d_k 时算了两次幂（d_k 参与 L² 项时约掉了，结果一致但要能说清）。

*面试官追问：*
- "训练总 FLOPs？" → 反传 ≈ 2× 前向 → 6·N·D；7B 模型训 1T tokens ≈ 4×10²² FLOPs。
- "长上下文先爆什么？" → 显存先爆 attention 激活/KV cache，算力后爆 attention 矩阵乘；工程上靠 FlashAttention(省激活) + KV 量化/GQA(省 cache) + MQA(极限省)。
- "MFU 怎么算？" → 实测 FLOPs 吞吐 / 硬件峰值；训练好系统 40%–55%，推理 decode 阶段常 <5%（memory-bound）。
- "为什么推理 decode 是 memory-bound？" → 单 token 前向 ≈ 2N FLOPs，但要读 N×2 bytes 权重 → 算术强度 ≈1 FLOP/byte，远低于硬件 ridge point（A100 ≈ 200）。

**【知识串联】**
KV 显存与 decode 吞吐 → `02-Inference/01-KVCache-And-Acceleration.md`、`02-Inference/03-Quantization-And-Deployment.md`；预训练 scaling → `01-Foundation/03-Loss-And-Optimization.md`；算力板凳与系统设计 → `09-SystemDesign/01-System-Design.md`。

---

### 题 4：BPE 分词器

**【考察意图】**
(1) 是否理解 BPE 是"**从语料里学合并规则**"的贪心统计过程，而不是查表；(2) 训练与推理的**一致性**（merges 按学习顺序重放）；(3) 复杂度意识：朴素法 O(M·N) 为什么工业界不可用、怎么优化；(4) BPE vs WordPiece vs Unigram 的差异。

**【深度解析】**

*思路推导：*

BPE 训练循环：
1. 初始词表 = 所有单字符 + 结束符 `</w>`（区分词尾、还原词边界）。
2. 每轮统计语料中所有**相邻符号对**频次，把最高频对 (a,b) 合并成新符号 ab，存入 merges 表（**顺序即优先级**）。
3. 重复直至词表达到目标大小（或无对可合）。
推理（tokenize）：把文本切成字符序列，按 merges 的**训练顺序**贪心重放合并。顺序重要：训练时第 k 步的符号只能由前 k−1 步的符号构成，逆序或乱序会产生训练中不存在的分词。

与 WordPiece（BERT）区别：WordPiece 按"提高语言似然最大"选对（分数 = pair频次 / (left频次×right频次)），不是纯频次贪心；Unigram（SentencePiece 常用）反向：先大词表再剪枝。

*完整可运行参考实现（含维度注释）：*

```python
from collections import Counter

class SimpleBPE:
    """简化 BPE（词内按字符 + </w>）。注释里给出工业级优化点。"""
    def __init__(self, vocab_size=300):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []   # [(a, b)，按训练顺序 = 优先级]
        self.vocab: set[str] = set()

    def train(self, corpus: str):
        # 语料预切分：按空白成词；每词结构 [c1, c2, ..., '</w>']
        words: list[list[str]] = []
        for word in corpus.split():
            words.append(list(word) + ['</w>'])
            self.vocab.update(word)
        self.vocab.add('</w>')

        while len(self.vocab) < self.vocab_size:
            pairs = Counter()                     # 相邻符号对频次
            for word in words:                    # O(N)：N = 语料符号总数
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i+1])] += 1
            if not pairs:
                break
            best_pair, _ = pairs.most_common(1)[0]
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab.add(new_token)

            new_words = []                        # 全量重写：把 best_pair 在每个词里合并掉
            for word in words:
                merged, i = [], 0
                while i < len(word):
                    if i < len(word)-1 and (word[i], word[i+1]) == best_pair:
                        merged.append(new_token); i += 2
                    else:
                        merged.append(word[i]); i += 1
                new_words.append(merged)
            words = new_words

    def tokenize(self, text: str) -> list[str]:
        tokens = list(text) + ['</w>']
        for a, b in self.merges:                  # 必须按训练顺序重放merges
            out, i = [], 0
            while i < len(tokens):
                if i < len(tokens)-1 and tokens[i] == a and tokens[i+1] == b:
                    out.append(a + b); i += 2
                else:
                    out.append(tokens[i]); i += 1
            tokens = out
        return tokens

if __name__ == "__main__":
    bpe = SimpleBPE(vocab_size=50)
    bpe.train("low lower nonzero newest widest low lower")
    print(bpe.tokenize("lower"))   # 按学到的 merges 顺序贪心切分
```

*逐行讲解：*
- `train` 的双重 while：外层每轮学一条新 merge，内层对每个词做一次"扫描合并"。注意合并匹配是**从左到右贪心、跳二**（i += 2），相邻两个可合并对不重叠。
- `tokenize` 逐条重放 merges：每条 merge 全序列扫描一次，与训练保持同序，保证词表内 token 均为训练产物。
- 全量 `words` 重写是复杂度大头。

*复杂度：*
- 训练（朴素，如上）：每次合并 O(N)（N=语料符号数）→ 总 **O(M·N)**，M≈词表增量。工业优化：只统计"仍含最佳对的词"（pair→词倒排索引 + 频表增量更新），实际 GPT-2 训练 50K merges 在 GB 级语料小时级完成；理论可达 O(M·N̄)（N̄ 为涉及词的总长， 远小于 N）。
- 推理：O(T·M)（T=文本符号数）；可用 trie/规则索引优化到近 O(T·M/并行度)，生产库（tiktoken/rust tokenizers）按词预缓存 radix 结构。

**【30秒答题框架】**
"BPE 三件事：初始字符表、每轮统计相邻对频次、把最高频对合并进词表并存 merges。推理按 merges 训练顺序贪心重放。朴素实现训练 O(M·N)，工业版靠 pair 倒排索引做增量统计。真实系统是 byte-level BPE（GPT-2/LLaMA），词前先 regex 预切分，避免空格/中英文差异。"

**【追问与陷阱】**

*常见手写 Bug：*
1. **训练和推理预处理不一致**：训练按空格切词，推理直接整串切 → `</w>` 边界错乱；真实实现（GPT-2）先 regex 预切分再 BPE。
2. merges **乱序重放**：打乱顺序会产生训练中不存在的 token。
3. `most_common` 平局不稳定：真正的实现有 tie-break（按发现时间），否则复现性差。
4. 中文/无空格语言没有词边界 → 你需要在预切分层面补规则，否则 `</w>` 语义失效。
5. 忘记为 token 添加 `</w>` 版本或忽略大小写规范化 → 词表对词内/词尾形态不一致。

*面试官追踪：*
- "BPE vs WordPiece vs Unigram？" → 频次贪心 / 似然增益贪心 / 反向剪枝；LLaMA 用 BBPE（字节级，永不 OOV）；T5 用 SentencePiece Unigram。
- "vocab_size 怎么权衡？" → 大词表：embedding 参数 V·d暴涨（LLaMA 32000×4096≈131M；GPT-4 传闻 ~100K），但压缩率高、序列短、prefill 省算力；小词表：OOV 少但序列长、decode 步数多。
- "数字为什么单独切？"（LLaMA-2 的 digit splitting / digits→each）→ 数字组合爆炸，单独切提升数学任务表现。

**【知识串联】**
分词原则与子词选型 → `01-Foundation/02-Tokenization.md`；tokenizer 词表与 lm_head 联动（LoRA 也可能把 lm_head 加入目标模块）→ 题 7；OOV 与中文场景 → `04-RAG/01-RAG-Complete-Guide.md` 的文本管道部分。

---

### 题 5：Beam Search

**【考察意图】**
验证：(1) 是否能正确维护"进行中的 beam"和"已完成的 beam"两池；(2) 理解**累积 log 概率**与**长度惩罚**——为什么没有长度归一化时 beam search 天然偏爱短句；(3) 效率问题：手撕时是否意识到朴素实现在原文件中每次 `model(seq)` 重算整条序列；(4) 能说清 beam search 与采样的适用边际。

**【深度解析】**

*思路推导：*

- 自回归解码是一个树搜索问题：每步分支 V。贪心 = 每步取 argmax（宽度 1）；beam search = **宽度 B 的剪枝搜索**：每步把 B 个存活 beam 各扩 B 个候选（2B 个，其中含 EOS 的 beam 移入 completed 池），按累积分数取前 B。因为深度优先的整树搜索是指数级，beam search 用 B² 候选剪枝到 B，保证分数单调性与局部最优性。
- 分数：`score(y) = Σ log P(tᵢ|y<i)`。log 概率累加为负 → 越长越"扣分"，不改就会偏爱短句 → 长度惩罚：`score_norm = score / |y|^α`（α∈[0,1]，α=0 不惩罚，α=1 平均 logP）；GNMT 用 `|y|` 换成 `((5+|y|)/6)^α`，对小句子更温和。
- EOS 处理：分数应在遇到 EOS 时**带上 EOS 的 logP** 完整结算（先把 EOS candidates 生成出来进 completed，比分母长度 +1）。
- 每步成本：B 次 decode 前向（使用 KV cache，每次 O(层)·(2d² + L·d) 算力）+ B² 候选排序 O(B² log B²)；总 O(T·B·F_decode)。

*完整可运行参考实现（含维度注释）：*

```python
import torch

def beam_search(model, input_ids, beam_width=3, max_len=50, eos_id=2, alpha=0.7):
    """input_ids: List[int] → List[int]
    契约：model(seq: List[int]) 返回最后位置的 logits (vocab,)
    生产实现应使用 KV cache 只前向新增 token（题 6），本接口便于手撕讲解。
    alpha：长度惩罚指数，alpha=0 完全不惩罚（偏短句），alpha=1 = 平均 logP
    """
    beams = [(list(input_ids), 0.0)]          # (序列, 累积 log P) 初始 beam
    completed = []                            # 已到 EOS 的 beam

    for _ in range(max_len):
        if all(seq[-1] == eos_id for seq, _ in beams):
            break                             # 全部完成则提前结束
        cands = []
        for seq, score in beams:
            if seq[-1] == eos_id:             # 已完成：结算，不再扩展
                completed.append((seq, score))
                continue
            logits = model(seq)               # (vocab,)，建议含 KV cache
            logp = torch.log_softmax(logits, dim=-1)
            top_p, top_i = logp.topk(beam_width)      # 每 beam 取 B 个候选
            for p, tok in zip(top_p.tolist(), top_i.tolist()):
                cands.append((seq + [tok], score + p))  # 累积 log 概率
        if not cands:
            break
        cands.sort(key=lambda x: x[1], reverse=True)    # 未归一化排序（同长度，同公平）
        beams = cands[:beam_width]

    pool = completed + [(s, sc) for s, sc in beams if s[-1] != eos_id]
    def norm(pair):                            # 长度归一化选最终答案
        s, sc = pair
        return sc / (len(s) ** alpha)          # GNMT 变体：sc / ((5+len(s))/6)**alpha
    pool.sort(key=norm, reverse=True)
    return pool[0][0]

class RandomLM:                                # 最小可运行 demo stub
    def __init__(self, vocab=100, seed=0):
        torch.manual_seed(seed); self.w = torch.randn(vocab, vocab)
    def __call__(self, seq): return self.w[seq[-1]]

if __name__ == "__main__":
    ids = beam_search(RandomLM(), [1], beam_width=3, max_len=10, eos_id=2)
    print(ids)
```

*逐行讲解：*
- 双池结构：`beams`（活跃）与 `completed`（EOS 结算）分开，避免已完成序列继续被打分/扩展。
- `topk(beam_width)`：从 V 个 token 里取 B 个候选 → 每步候选池 B²（活跃率和 EOS 也可能出现在这 B² 里）。
- 排序用**未归一化**分数：候选间长度至多差 1，且排序目的只是截断；最终再长度归一化比较"不同长度候选"。
- 最终 pool 要合并 completed 与未完成（max_len 截断的兜底），归一化后取 argmax。

*复杂度：*
- 时间 O(T · B · F_decode + T · B² · log B)；空间 O(T · B)。
- 若无 KV cache：每步对长度 L 的序列整条 forward → O(T·B·L·F/token)，差一个 L 因子。

**【30秒答题框架】**
"beam search 维护两池：活跃 beam 和已完成 beam。每步每个活跃 beam 取 top-B 扩展，B² 候选按累积 logP 截断回 B；EOS 进 completed 池不再扩展但保留分数。最后用长度归一化（score/len^α，GNMT 用 (5+len)/6 幂）在两池里挑最优。成本 O(T·B) 次带 KV cache 的前向。注意它偏短句、易重复，开放生成一般不用，翻译/摘要等有明确终点的任务用。"

**【追问与陷阱】**

*常见手写 Bug：*
1. 已完成 beam 没移出扩展循环 → 会在 EOS 后继续生成垃圾 token。
2. 长度惩罚用错位置：**必须只在跨长度比较时归一化**，步内排序不宜用（会把不同 beam 的部分序列不可比）。
3. EOS 的 logP 忘了累加再结算 → 分数系统性偏低。
4. `all_candidates.sort` 每次 sort 全池 O(B²logB²) 没问题，但若 beam B=1 要退化为贪心（等价正确）。
5. 朴素的 `model(seq)` 无 KV cache：复杂度膨胀 L 倍（生产用增量解码，题 6）。
6. 没处理 max_len 内全部 beam 未 EOS 的兜底分支。

*面试官追问：*
- "为什么 LLM 时代很少用 beam search？" → 采样 + 温度更契合开放生成（多样性、可控性）；beam 产生的是高似然的**模板句**，重复率高；另外 RLHF 后的模型分布已经校准，argmax-ish 搜索性价比低。nucleus/top-p 采样 → `02-Inference/02-Decoding-Strategies.md`。
- "beam width 对质量的影响？" → 中等 B（4~10）在 MT 上 BLEU 提升；过大收益递减且重复增多。
- "batch 化 beam 怎么写？" → (2B, L) 张量打头，选完 token 后 scatter 回 B 条；上下文用 beam 索引 gather KV。
- "diverse beam search？" → 在候选里加 beam 间惩罚项，鼓励多样性。

**【知识串联】**
解码策略家族 → `02-Inference/02-Decoding-Strategies.md`；KV cache 前向 → 题 6；对齐/奖励后的采样行为 → `03-FineTuning/02-Alignment.md`；Agent 系统里生成规划 ≤ beam1+temperature 的常见实践 → `05-Agent/01-Agent-Complete-Guide.md`。

---

### 题 6：KV Cache Attention

**【考察意图】**
以梅论，这是**必须必须必须**拿满的一题：(1) 能说出"为什么肯 KV cache"——自回归因果性下历史 token 的 K/V 与新 query 无关；(2) 能现场写 KV cache 显存公式并带数字（LLaMA-7B fp16：**512 KB/token**，2048 ctx ≈ 1.05 GB）；(3)知道 prefill/decode 两阶段、decode 为什么是 memory-bound；(4) 能自己诊断原文件实现的工程缺陷（cat 拷贝、batch 支持、多轮对话忘清 cache）。

**【深度解析】**

*思路推导：*

- 设第 t 步生成 token x_t。attention 需要：Q_t（当前 token），K/V 历史 (√所有位置。因果性 → 历史位置的 K、V 不依赖未来，且与"谁在询问"（新 Q）无关 → **可缓存**。
- 每步缓存增量：本步新算 K,V 各 (B, h, 1, d_k) 拼到已有 cache 尾部；scores = Q_t·K_cacheᵀ /√d_k (B,h,1,L_total)；不需要 causal mask（新 query 天然看全部历史）。
- **显存公式**：KV cache = 2·L·h_kv·d_k·n_layers·bytes·B
  − LLaMA-7B（MHA，各 hf 32 层、d=4096）：2·4096·32·2B(fp16) = **512 KB/token**；2048 ctx → **1.05 GB/条请求**。
  − LLaMA-2-70B（GQA-8：h_kv·d_k=8×128=1024；80 层）：2·1024·80·2 = 320 KB/token，4k ctx 每条 1.31 GB。
  − GQA-8 使 cache 缩小 h/h_kv = 4 倍（题 11）；int8 KV 再省 2×。
- **两阶段**：prefill（全 prompt 一次前向，FLOPs O(L²d) 上面类似，算力的主峰阶段）；decode（每步 O(L·d) attention + O(d²) 投影，显存带宽限制阶段）。
  - decode 算术强度：每 token 前向 ≈ 2N FLOPs，需读 N·2 bytes 权重 → ≈1 FLOP/byte；A100 带宽 1.5–2TB/s → 单流吞吐上限 ≈ 150+ token/s（7B fp16）。要提升吞吐靠 batch 并行共享权重读取。

*完整可运行参考实现（含维度注释）：*

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCacheAttention(nn.Module):
    """带 KV cache 的 decode 注意力。支持 batch；cache 按 (B, h, L_total, d_k) 追加。"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.k_cache = None    # (B, h, L_past, d_k)
        self.v_cache = None

    def forward(self, x, use_cache=True):
        # x: (B, L_new, d_model)。decode 时 L_new=1；prefill 时 L_new=prompt_len
        B, L, _ = x.shape
        Q = self.W_Q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)   # (B,h,L,d_k)
        K = self.W_K(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)   # (B,h,L,d_k)
        V = self.W_V(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        past_len = 0
        if self.k_cache is not None:
            past_len = self.k_cache.shape[2]
            if use_cache:
                K = torch.cat([self.k_cache, K], dim=2)                      # (B,h,past+L,d_k)
                V = torch.cat([self.v_cache, V], dim=2)

        if use_cache:
            self.k_cache = K.detach()     # detach：cache 不进反传图，防止显存随 autograd 累积
            self.v_cache = V.detach()     # 生产实现：预分配 buffer 按 slice 写入，避免 cat 拷贝（vLLM PagedAttention）

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,h,L,L_total)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)                                           # (B,h,L,d_k)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)                 # (B,L,d_model)
        return self.W_O(out), past_len

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

if __name__ == "__main__":
    torch.manual_seed(0)
    layer = KVCacheAttention(64, 4)
    prompt = torch.randn(2, 5, 64)               # prefill
    o1, _ = layer(prompt, use_cache=True)        # (2,5,64)；cache 长度 5
    step  = torch.randn(2, 1, 64)                # decode 一步
    o2, past = layer(step, use_cache=True)       # (2,1,64)；past=5 表示新 token 覆盖了 5 个历史
    print(o1.shape, o2.shape, past)
```

*逐行讲解：*
- 关键分支：`if self.k_cache is not None` → 只有**非首步**才拼 cache；首步（prefill 或 cache 清空后）全长序列直接算并写 cache。
- `past_len`：手撕时能自觉返回 past_len，说明你知道外层要靠它做 RoPE 位置偏移（题 9）与位置 ID 递增。
- `K = cat([k_cache, K])` 把新 token 追加到**序列维 dim=2**——手写这里极易把 dim 写成 1（头维）或 0（batch），一写错 shape 报错立刻暴露。
- `detach()`：训练态跑 decode（如 teacher forcing 检查）时防止 cache 挂着生成图。
- 生产差异（面试用语言说出来）：真实实现预分配 max_len buffer + `index_copy_` 写 slice，或 vLLM PagedAttention 按 block 复用，cat 的全量拷贝 O(L·d) 每步可避免。

*复杂度（量化）：*
- 无 cache：每步 O(L²·d)；有 cache：每步 O(L·d)（attention 部分）+ O(d²)（投影/FFN）。
- KV cache 显存：见上方公式；B 条并发乘 B。归一如表：单条 LLaMA-7B 8K ctx fp16 KV = 512KB×8192 ≈ 4.2 GB（对比权重 13.4 GB）。

**【30秒答题框架】**
"因果 self-attention 里历史位置的 K/V 与新 query 无关，可以缓存。每步只算新 token 的 QKV，K/V 追加进 cache，Q 与全体 K 算一次注意力。KV 显存 = 2·L·d_model·layers·bytes（MHA 口径 LLaMA-7B fp16 是 512KB/token，2k 上下文 1GB 左右），GQA/量化能再压 4×/2×。prefill 算力重、decode 带宽重，所以 PagedAttention/continuous batching 成了推理框架标配。"

**【追问与陷阱】**

*常见手写 Bug：*
1. **cache 忘 detach / 忘 no_grad**：自回归循环图越挂越大，显存爆（即使只推理）。
2. **cat 的 dim 写错**：序列维是 dim=2（(B,h,L,d_k)）。
3. **多轮对话忘 clear_cache**：上一轮的 K/V 混进本轮，生成污染。
4. decode 时新 query 只有 1 个位置：**不要**再叠 causal mask（新 token 天然可看全部历史）；在 prefill 才需要 causal。
5. RoPE 配 cache 忘了位置偏移：新 token 的位置 = past_len（题 9 联动），否则 K 已旋转过到过去的角度、Q 却转错角度。
6. batch 各序列 cache 长度不同要 padding：真实实现用 varlen/flash-attn 的 cu_seqlens 处理，手撕至少要提。

*面试官追问：*
- "KV cache 存哪、多大？" → 公式 + 上述数字；70B GQA 数字展示你知道大模型（80 层是 320KB/token）。
- "PagedAttention 是什么？" → 把 KV cache 分块成 page，逻辑连续物理离散，消除碎片，吞吐 2–4×（vLLM 论文数据）。
- "KV cache 怎么量化？" → int8 KV 几乎无损（少数模型需校准），int4 有损但翻倍容量；也有 per-channel scale 方案。
- "MQA/GQA 与 cache 的关系？" → KV 头数减为 h_kv，cache 按 h/h_kv 缩小（题 11）。
- "decode 为什么慢？怎么提吞吐？" → memory-bound；continuous batching 让不同请求分享权重读取；投机采样（speculative decoding）用小模型猜、大模型验证一次接受多 token。

**【知识串联】**
这是 `02-Inference/01-KVCache-And-Acceleration.md` 的代码落地；GQA/Rema → 题 11；RoPE 偏移 → 题 9；投机采样/bea → `02-Inference/02-Decoding-Strategies.md`；多路部署 → `09-SystemDesign/01-System-Design.md`。

---

### 题 7：LoRA 实现

**【考察意图】**
(1) 是否真正理解 **ΔW = B·A 的低秩参数化**与参数/算量公式；(2) 最深一问：为什么 **B 初始化为 0**（保持训练起点与预训练函数逐点相等），以及 A、B 同取 0 会发生什么（梯度死锁）；(3) scaling = α/r 的作用与合并推理（`W' = W + α/r·BA`）的等价性；(4) 融入 QLoRA 场景（基座 4bit）。

**【深度解析】**

*思路推导：*

- 假设（LoRA 论文）：微调的权重更新是**低秩**的 → ΔW = B·A，A∈R^{r×in}（"降维侧"），B∈R^{out×r}（"升维侧"）。
  参数量：out·in → r·(out+in)。行内常规：r=8、in=out=4096 → 65,536 参数/矩阵；若对 7B 模型每层的 q/k/v/o/gate/up/down 共 7 矩阵启用：65,536×7×32层 ≈ **14.7M（≈0.22%）**。
- 前向：`h = W·x + (α/r)·B(A·x)`。训练起点要求 h ≡ W·x：B 初始化为 0 即满足（A 随机值也无所谓，B·A=0）。**若 A=0 且 B=0**：∂L/∂A ∝ B=0、∂L/∂B ∝ A=0，乘积梯度恒零 → 网络不学习（死锁）。所以半边随机、半边零。
- scaling α/r：把影响幅度与 r 解耦 —— 调 r 时等效学习率不变。α 常取 2·r（如 r=16, α=32，scaling=2）。
- 推理合并：`W' = W + (α/r)·B·A`，一次 O(out·in·r) 矩阵乘完成，推理零额外延迟。不合并则每层多两条小 GEMM（x→A→B），r≪d 时开销也可忽略但毕竟多一跳。
- 训练显存（7B, rank 8, 目标模块如上）：冻结基座 bf16 13.4GB（若无梯度） + 可训练参数 최적화状态：14.7M × (fp32 copy 4B + m 4B + v 4B + grad 2~4B) ≈ **~0.2 GB**；主要开销在激活。QLoRA 则把基座压到 4bit（~3.4GB）+ NF4/双重量化，24GB 卡可调 33B 级别。

*完整可运行参考实现（含维度注释）：*

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALinear(nn.Module):
    """W(冻结) + (alpha/r)·B·A(x)。对标 LoRA 论文 eq.(4) 与 HuggingFace peft 实现风格。"""
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.05):
        super().__init__()
        self.rank, self.alpha = rank, alpha
        self.scaling = alpha / rank                       # 缩放因子，与 r 解耦
        self.weight = nn.Parameter(                       # 冻结的原始权重
            torch.empty(out_features, in_features), requires_grad=False)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))   # 降维侧 (r, in)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))  # 升维侧 (out, r)，零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))        # peft 同款初始化
        self.dropout = nn.Dropout(dropout)               # 只作用在 A 分支输入

    def forward(self, x):
        # x: (B, ..., in)；base_output: (B, ..., out)
        base_output = F.linear(x, self.weight)
        lora_output = F.linear(
            F.linear(self.dropout(x), self.lora_A),      # (B, ..., r)
            self.lora_B)                                 # (B, ..., out)
        return base_output + self.scaling * lora_output

    @torch.no_grad()
    def merge(self):
        """合并为等价单矩阵：W' = W + (α/r)·B·A。推理零延迟。"""
        delta = self.lora_B @ self.lora_A                 # (out,in) = (out,r)·(r,in)
        self.weight.data += self.scaling * delta
        self.lora_A.data.zero_(); self.lora_B.data.zero_()  # 合并后分支归零保持等价

if __name__ == "__main__":
    torch.manual_seed(0)
    layer = LoRALinear(in_features=4096, out_features=4096, rank=8, alpha=16)
    x = torch.randn(2, 4096)
    y0 = layer(x)                                         # 训练起点：B=0 → 输出与纯线性层一致
    layer.merge()
    y1 = layer(x)                                         # 合并后函数不变
    print(torch.allclose(y0, y1, atol=1e-6))              # True（等价性验证）
    n_params = (8*4096*2)                                  # 每矩阵可训练参数
    print(n_params, f"{n_params*7*32/6.74e9:.2%}")         # 65536, ≈0.22%（7B 场景估算）
```

*逐行讲解：*
- `requires_grad=False`：冻结基座，只训 A/B；optimizer 只收 LoRA 参数。
- 初始化分工：A 用（近似正态的）kaiming 初始化，B 恒 0 → 起点逐点等于原模型，loss 曲线从预训练位置出发；同时避免 A 变 0 的死锁（A 有随机扰动，梯度能传）。
- `forward` 顺序是先 A 后 B（x→r→out），注意维度：A 的 in 维先降；等价于 ΔW·x 分解，与 (B·A) 先合并数学一致。
- `dropout` 按 LoRA 论文加在主输入路径里（peft 的 lora_dropout 在 A 前面），不作用于 base 分支。
- `merge`：合并后两种写法数值等价；注意合并后不能继续训练（要 unmerge 回来）。

*复杂度：*
- 前向额外时间 O(x·r·(in+out))，r≪d 时占比 ~2r/d ≈ 0.4%（r=8, d=4096）。
- 训练可训练参数 14.7M（7B/rank 8 七矩阵）≈ 0.22%；AdamW 状态 ~0.2GB。梯度不需要为冻结权重分配。
- 合并一次 O(out·in·r) == 一次 d×d GEMM 的一半量级，离线一次性完成。

**【30秒答题框架】**
"LoRA 的核心假设是微调的权重更新低秩：ΔW=B·A，参数从 out·in 降到 r(out+in)。前向 h=Wx + α/r·B(Ax)。初始化要点：B 初始化 0 —— 这样训练起点与预训练输出完全一致；A 随机初始化；如果两个都 0，梯度全是 0 会死锁。推理把 BA 乘 α/r 加回 W 合并，零额外延迟。7B 模型 rank 8 对七条投影才 14.7M 参数约 0.22%，QLoRA 把基座压 4bit 后能单卡调大模型。"

**【追问与陷阱】**

*常见手写 Bug：*
1. **B 忘了零初始化**（如两个用 randn）：起点就偏移预训练分布，最优 loss 曲线变差、复现实验失败。
2. **A、B 同时 0**：梯度恒零，静默不学习；断言/检查首次 step 的 norm 可抓到。
3. 冻结权重的 `requires_grad=False` 漏掉：显存翻倍且optimizer 状态错事。
4. **只对 A 加 dropout 或加错位置**（应该加在 A 分支的输入，即 x 进 A 之前）。
5. target modules 只挂 attention → 有论文（QLoRA）显示 **把 FFN（gate/up/down）也挂上**对质量显著有利；只挂 q_proj 的对比实验会吃亏。
6. **大 vocab 场景把 lm_head 挂进 LoRA**：32000×4096 的矩阵 rank 8 才 0.4M 可训练但产生ٕ的 A/B 计算大小正常，但 embedding/lm_head 参数本身大且输出 logit 语义精细，若干论文建议慎重。
7. 合并时忘乘 scaling → 上线后效果"变味"。

*面试官追问：*
- "LoRA/Adapter/Prefix-tuning 对比？" → LoRA 零推理延迟、无额外上下文；Adapter 加串行层有延迟；Prefix/P-tuning 占 KV cache 空间。
- "rank 怎么选？" → 论文 r=1~64 内变化不大（低秩假设成立）；任务复杂/多语可 16-64；r 大过 128 一般容易过拟合。
- "LoRA+？" → A/B 用不同学习率（B 更大）收敛更快。
- "QLoRA 与 LoRA 不同是什么？" → 冻结基座 4bit NF4 + per-channel 量化 + 双重量化 + paged optimizer；LoRA 部分仍为 bf16 高精度。
- "为什么说 LoRA 没有推理成本？" → 合并为 W'，一次前向 GEMM 完成；若不合并也只是加 2 条 r 宽的小 GEMM。

**【知识串联】**
LoRA 理论/训练技巧全景 → `03-FineTuning/01-PEFT.md`；与 RLHF/PPO 参数对比 → `03-FineTuning/02-Alignment.md`；量化的基座与 LoRA 精度联动 → `02-Inference/03-Quantization-And-Deployment.md`；指令微调选型 → `03-FineTuning/03-Instruction-Tuning.md`。

---

### 题 8：余弦相似度检索

**【考察意图】**
（RAG 落地题）(1) 归一化的意义：**不看向量的模长**，cosθ 只比方向；而且**归一化后 L2 距离与余弦等价**（‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b = 2 − 2cosθ），所以 ANN 索引拿 L2 用完全一致；(2) TopK 复杂度意识：argsort O(n log n) 是浪费，`argpartition`/堆 O(n)/O(n log k) 才是正确方案；(3) 数值健壮性（eps、dtype）与工程升级路径（FAISS/HNSW）。

**【深度解析】**

*思路推导：*

- 定义：`cos(u,v) = u·v / (‖u‖‖v‖)` ∈ [−1,1]。预处理：离线把文档向量 L2 归一化一次（O(n·d)），在线把 query 归一化后一次矩阵-向量乘即可：
  `sims = doc_normed @ q_normed`（形状 (n,d)·(d,1) → (n,)，BLAS 高效）。
- TopK 的选择：`np.argsort(sims)[::-1][:k]` = 全排序 O(n log n)；只需要 k 个优化：
  - `np.argpartition(-sims, k)[:k]`：introselect，平均 **O(n)**；再对这 k 个位排序 O(k log k)。
  - 堆（小顶堆维护 k 大）：流式 O(n log k)、O(k) 内存；支持逐块读盘召回。
  - 大规模检索：Exact MIPS（n 亿级）不可行 → ANN：HNSW（平均 O(log n) 查询），IVF-PQ（先聚类粗筛+量化压缩）；与归一化数据配 L2/IP 等价索引均可。
- eps 的作用：零向量除零 → NaN 污染; 通常 1e-8 常数即可。
- dtype：float32 足够（fp16 相似度排序有 tie 风险，需要小心）；得分我自己 batched query 用矩阵乘一条 `(Bq,d)@(n,d).T`。

*完整可运行参考实现（含维度注释）：*

```python
import numpy as np
import heapq

def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

def cosine_similarity_search_argpartition(query_vec, doc_vecs, top_k=5):
    """query_vec: (d,)   doc_vecs: (n, d)。TopK: O(n) 分区 + O(k log k) 排序"""
    q = normalize(query_vec)                     # (d,)
    docs = normalize(doc_vecs.astype(np.float32))# (n, d)   建议离线预归一化只做一次
    sims = docs @ q                              # (n,)     矩阵-向量乘 O(nd)
    k = min(top_k, sims.shape[0])
    part = np.argpartition(-sims, k - 1)[:k]     # O(n)     选出前 k 的下标（无序）
    order = part[np.argsort(-sims[part])]        # O(k log k) 排序
    return [(int(i), float(sims[i])) for i in order]

def cosine_similarity_topk_heap(query_vec, doc_vecs, top_k=5):
    """流式/分块版本：小顶堆维护 top-k，O(n log k)，O(k) 额外内存。
    面试可同时给出 AVL/平衡 BST 版本：TreeMap 维护 k 大小窗口，插入/删除均 O(log k)，
    适合"流式入库还要动态删改"的场景；纯静态 topK 用堆即可。"""
    q = normalize(np.asarray(query_vec, np.float32))
    heap = []                                    # (sim, idx) 小顶堆
    for i in range(0, doc_vecs.shape[0]):        # 支持分块：换成 for block in chunks
        s = float(normalize(doc_vecs[i].astype(np.float32)) @ q)   # O(d)
        if len(heap) < top_k:
            heapq.heappush(heap, (s, i))         # O(log k)
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, i))      # O(log k)
    return sorted(heap, key=lambda t: -t[0])     # O(k log k)

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    docs = rng.standard_normal((1000, 64)).astype(np.float32)
    q = rng.standard_normal((64,)).astype(np.float32)
    a = cosine_similarity_search_argpartition(q, docs, 5)
    b = cosine_similarity_topk_heap(q, docs, 5)
    assert [i for i, _ in a] == [i for i, _ in b]    # 两种实现结果一致
    print(a[:2])
```

*逐行讲解：*
- `normalize`：最后一维归一化；`keepdims=True` 保证广播形状对；eps 防 0 向量。
- `docs @ q`：一次 BLAS 调用比循环快 1–2 个数量级。
- `argpartition(-sims, k-1)`：负号把"最大 k"转成"最小 k"，因为 partition 取前 k 小；先无序再对 k 个排序，避免全排。
- heap 版本：小顶堆的堆顶是当前第 k 大，新元素比堆顶大才值得入堆——`heapreplace` 一步完成 pop+push。

*复杂度（量化）：*
- 精确检索：一次查询 O(n·d)（相似度）+ TopK O(n)/O(n log k)。n=10⁷、d=768：一次查询 ≈ 7.7 GFLOP（fp32）≈ 数百 ms CPU 量级 → 所以要用 ANN。
- HNSW：查询平均 O(log n)，内存 n·(d·4 + M·8~16) bytes；IVF-PQ：查询 O(√n·d' )，内存压到 n·(每向量几 d 字节)。

**【30秒答题框架】**
"先 L2 归一化，余弦相似度就变成一次矩阵向量乘，O(nd)。TopK 别全排序，用 argpartition O(n) 或堆 O(n log k)；流式入库要支持删改时用平衡树（AVL/TreeMap）维护 k 大小集合。归一化后 L2 距离与余弦单调等价，FAISS 索引随便挑。大规模上 HNSW/IVF-PQ 近似检索，换常数级查询延迟。"

**【追问与陷阱】**

*常见手写 Bug：*
1. **忘了 keepdims** 导致形状 (n,) 除 (n,) 触发 broadcast 错位（(n,) vs (n,d) 会错报）。
2. **argsort 全排序**当 TopK 用：O(n log n)，量化上向量的百万级检索多这一个对数没死，但面试要能主动指出 argpartition/堆更优。
3. **eps 漏加**：零向量出现 NaN，排序直接乱。
4. dtype 用 float64 或 list：内存×2~4、速度降数倍；统一 float32。
5. 一边归一化一边比较原始模长（比如"相关性还打分权重"）会破坏单调假设。
6. 多 query 忘却 batch 运算：循环每 query 调函数没有 BLAS 批乘。

*面试官追问：*
- "为什么 RAG 用 cosine 不是欧氏？" → 归一化下等价；未归一化时 L2 会惩罚模长大（信息不密集的 chunk 语义弱）—— 一句话答出等价条件是高分答案的标志。
- "ANN 为什么可接受？" → 召回的是进入候选池的近似 top-k，后续接 cross-encoder 精排，准确率损失被精排兜住（两阶段检索：召回+精排）。
- "HNSW/IVF/PQ 差异？" → 图索引广优先搜索（查询快、内存大、建图慢）；倒排+量化（省内存，召回率可控参数）；混合方案（IVF-HNSW、OPQ）。
- "embedding 训练目标？" → 对比学习 InfoNCE：拉近正对、推远负对；batch 内负采样与温度 τ；对应损失推导结合 `01-Foundation/03-Loss-And-Optimization.md`。
- "检索前为什么要 chunk？" → 上下文窗口与召回精度：长文切 256~512 token，重叠 10-20%。

**【知识串联】**
RAG 全链路（chunk → embedding → 检索 → rerank）→ `04-RAG/01-RAG-Complete-Guide.md`；embedding 训练损失 → `01-Foundation/03-Loss-And-Optimization.md`；向量库选型与系统设计 → `09-SystemDesign/01-System-Design.md`。

---

### 题 9：RoPE 旋转位置编码

**【考察意图】**
This is the位置编码高频考题：(1) 能否现场推导 **RoPE 的相对位置性质**：`q_m·k_n = f(q, k, m−n)` —— 二维情形旋转向量闭包性质最简明；(2) 复数实现中 `view_as_complex` 的 **(…, −1, 2) reshape 顺序**（实部/偶位在前）——写错位置编码就乱；(3) **与 KV cache 的配合**：cache 里的 K 已旋转过，decode 只需给新 token 的 freqs 加 offset (past_len)；(4) 外推问题与修正方案（PI/NTK/YaRN）。

**【深度解析】**

*思路推导：*

- 目标：把位置 m 的信息注入 q_m，n 的信息注入 k_n，使**内积只依赖相对位移 m−n**（不是绝对位置）。
- 构造（二维情形）：把 (q_{2i}, q_{2i+1}) 当二维向量，旋转角 m·θ_i：
  `q̃_m = R(mθ)·q`，`k̃_n = R(nθ)·k`。二维旋转矩阵满足 **R(a)·R(b) = R(a+b)** 且正交（R(−θ)=R(θ)ᵀ）：
  `q̃_m · k̃_n = qᵀ·R(mθ)ᵀ·R(nθ)·k = qᵀ·R((n−m)θ)·k`
  右边只含 n−m → 内积是相对位置的函数，且等价于复数乘法 `Re[ q̃·conj(k̃) ]`（幅角差 m−n）。三维以上把 d_head 维**两两一组**（第 2i、2i+1 维），每组分配不同频率 θ_i = base^{−2i/d}（base=10000）→ 不同维度以不同"角速度"旋转，低维高频、高维低频，多尺度编码位置。
- 复数实现：把 (q_{2i}, q_{2i+1}) 视作复数 `x_{2i} + i·x_{2i+1}` → `view_as_complex(reshape(..., -1, 2))`。错序（例如把 (odd, even) 互换或用 `reshape(-1, 2).T` 后再配对）会悄悄打乱频率映射。注意 HF/GPT-NeoX 风格实数版是**先分上下两半**（前 d/2 与后 d/2 配对）再 rotate_half，与 interleaved 版在数值上不同排列——**二者不能混用 ckpt**，这是真实的坑（转换 LLaMA ckpt 时总有这个对齐问题）。
- precompute：`freqs_cis[m] = e^{i·m·θ_i}`，复数 (L, d/2)；apply 时广播到 (B, L, h, d/2)。
- 与 KV cache：K 在进 cache 前已旋转到其绝对位置 → decode 新 token 直接用 `freqs_cis[past_len : past_len+1]`，历史无需重算（位置偏移见题 6 的 past_len）。
- 外推：θ=10000 的低频维度在 L>train_len 时角度组合未见过。方案：**PI**（位置插值：m→m·L_train/L_new，等比缩放频率）、**NTK-aware**（增大 base）、**YaRN**（按维度分段插值+温度修正注意力熵）。这解释了为什么 LLaMA 最初的 2k 硬限、以及社区 8k/32k 扩展全靠这几种。

*完整可运行参考实现（含维度注释）：*

```python
import torch

def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    """dim 必须等于 head_dim 且为偶数。返回 (max_seq_len, dim//2) 复数。"""
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))  # (d/2,) 逐对频率
    t = torch.arange(max_seq_len)                                     # (L,)
    freqs = torch.outer(t, freqs)                                     # (L, d/2) 角度 m·θ_i
    return torch.polar(torch.ones_like(freqs), freqs)                 # e^{i·mθ}（模长1）

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq/xk: (batch, seq_len, n_heads, d_head)
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # (B,L,h,d/2) 相邻两维→复数
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    fc = freqs_cis[None, : xq.shape[1], None, :]      # (1,L,1,d/2) 广播 batch 和 head 维
    xq_out = torch.view_as_real(xq_c * fc).flatten(-2)   # 复数乘=逐对二维旋转
    xk_out = torch.view_as_real(xk_c * fc).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def rotate_half(x):            # 实数版（GPT-NeoX 风格，与上面数学等价但配对方式不同）
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope_real(xq, xk, cos, sin):
    # cos/sin: (L, d)  由 m·θ_i 预计算并复制成对
    return xq*cos + rotate_half(xq)*sin, xk*cos + rotate_half(xk)*sin

if __name__ == "__main__":
    torch.manual_seed(0)
    fc = precompute_freqs_cis(32, 64)                          # head_dim=32, max_len=64
    q = torch.randn(2, 7, 4, 32)                               # (B,L,h,d_head)
    k = torch.randn(2, 7, 4, 32)
    q1, k1 = apply_rotary_emb(q, k, fc)
    print(q1.shape)                                            # (2,7,4,32)
    # 相对位置性检验必须用"同一个向量"放到不同位置：
    # 同一 q0/k0 旋转到 (m,n) 与 (m',n')，只要 m−n == m'−n' 内积就相同
    def rotate_vec(x, pos):                                    # x: (h, d)  同一向量旋转到位置 pos
        cpl = torch.view_as_complex(                           # 偶位实部、奇位虚部
            torch.stack([x[..., 0::2], x[..., 1::2]], dim=-1)) # (h, d/2, 2)
        return torch.view_as_real(cpl * fc[pos]).flatten(-2)   # (h, d)
    q0, k0 = q[0, 0], k[0, 0]                                  # 取同一个 q/k 向量
    s1 = (rotate_vec(q0, 1) * rotate_vec(k0, 4)).sum()         # m−n = 1−4 = −3
    s2 = (rotate_vec(q0, 4) * rotate_vec(k0, 7)).sum()         # m−n = 4−7 = −3
    print(s1, s2, torch.isclose(s1, s2, atol=1e-5))            # True
```

*逐行讲解：*
- `torch.arange(0, dim, 2)/dim`：频率下标偶数位 0,2,4,…，指数为负 → 高频到低频。
- `torch.outer(t, freqs)`：外积 (L, d/2)，第 (m,i) 元素 = m·θ_i —— 注意是 `outer(t, freqs)` 不是 `outer(freqs, t)`（转置不算错位但对齐 head 维时不顺）。
- `torch.polar(ones, angles)`：构造单位复数 e^{i·angle}。
- `reshape(*shape[:-1], -1, 2)`：**最后一维切两半做复数**，约定偶位→实部、奇位→虚部；`view_as_complex` 要求最后一维为 2 且 stride 合法。
- demo 里 `s1 ≈ s2` 验证了相对位置性质：**同一对向量**放到相对位移相同（−3）的两个位置对上，内积相同——这是 RoPE 的"面试杀手锏"现场证据。

*复杂度：*
- 预计算 O(max_L·d/2)；apply 逐 token 复数乘 O(B·L·d)（≈一次逐元素运算量级），远小于 attention 本身。
- KV cache 旋转后即固定 → cache 大小与 RoPE 无关（存的是位置已编码的 K）。

**【30秒答题框架】**
"RoPE 把每对相邻维视作二维向量，用位置相关角度 m·θ_i 旋转 Q 和 K，θ_i 是按维指数降的频率（base 10⁴）。因为二维旋转矩阵满足 R(a)R(b)=R(a+b) 且正交，q_m·k_n = q·R((n−m)θ)·k —— 内积只依赖相对位移。复数实现就是 view_as_complex 前把最后一维 reshape 成 (…,2)。与 KV cache 配合：cache 里的 K 已旋转，新 token 用 freqs_cis[past_len:] 即可。长上下文外推要用 PI/NTK/YaRN。"

**【追问与陷阱】**

*常见手写 Bug：*
1. **reshape 顺序错**：`reshape(-1, 2)` 前没有保证"相邻维成对"，或手动 `stack([x[..., ::2], x[..., 1::2]], -1)` 与复数约定顺序不一致 → 位置编码错乱（模型看似能训，长文崩）。
2. **dim 传错**：precompute 的 dim 必须是 `head_dim` 而不是 `d_model`（d_model 会让每组频率重复 h 次）。
3. **decode 偏移**：cache 场景直接 `freqs_cis[:L_new]` 重新从 0 开始 → 破坏相对位置一致；要用 past_len 作为偏移。
4. **旋转加在与 embedding 上**而不是 Q/K 投影后（RoPE 只作用于 attention 的 Q/K，不作用于 V，也不加残差流）。
5. **interleaved vs half-split 混用**：LLaMA 官方（interleaved% 复数法）与 GPT-NeoX/HF 转换版（先劈两半）数值不同；用错 ckpt 的小文本差异微妙、长文崩坏，属真实工程事故。
6. GQA 下 freqs_cis 广播时 head 维改动没对齐（1, L, h_kv → repeat 到 h）。

*面试官追问：*
- "现场推一下为什么只依赖 m−n？" → 用二维 R(a)R(b)=R(a+b)+正交性 2 行内推完（上面推导）。
- "为什么长文本外推会崩？" → 低频 θ_i 对应角度超出训练分布；缓解：PI（直接线性缩频率）、NTK（调大 base）、YaRN（分段+注意力温度），LLaMA 官方社区 32k 方案是 NTK/YaRN 系。
- "RoPE 与 ALiBi？" → ALiBi 在 attention 打分加线性 bias（−m·|i−j|），不修改向量本身；RoPE 是旋转（保范数、分布式插值容易）；两者都支持外推但机制不同。
- "为什么 V 不旋转？" → V 只被加权求和，相对位置语义在 Q/K 打分处就完成了，旋转 V 无意义且破坏值语义。
- "θ=10⁴ 能更大吗？" → base 越大低频越高分辨率长程信息，但短程相对精度略降；NTK 变体就是调 base 的应用。

**【知识串联】**
RoPE 与 attention 实现（题 1/2/6/11）的位置偏移联动 → `02-Inference/01-KVCache-And-Acceleration.md`；长上下文（SSM/RWKV/长序列外推）与条第 → `07-HotTopics/01-Hot-Topics-2025-2026.md`；预训练数据长度策略 → `01-Foundation/01-Transformer.md`。

---

### 题 10：LeetCode 高频算法题

> 原文件第 10 节是 8 道表格题（LRU Cache、TopK、合并 K 个有序链表、二叉树层序遍历、最长公共子序列、岛屿数量、快速排序/归并排序、滑动窗口最大值）。以下按 10.1–10.8 逐题详解，编号沿用表格顺序。重点覆盖原表全部 8 道，并把 TopK 一题扩展到"堆 vs AVL"（手撕高频）。

#### 题 10.1：LRU Cache

**【考察意图】**
"哈希表 + 双向链表 = O(1) 组合"的工程条件反射；同时考边界（capacity=0、put 已存在 key）与"能不能写出不使用 `OrderedDict` 的手写版"。

**【深度解析】**

*思路推导*：要 get/put 都是 O(1)：哈希表负责 O(1) 定位；双向链表负责 O(1) 把节点移到头/尾。环状哨兵（dummy head/tail）让边界不再特判。

*完整可运行参考实现（含维度注释）：*

```python
class Node:
    __slots__ = ("k", "v", "prev", "next")      # 双向链表节点
    def __init__(self, k=None, v=None):
        self.k, self.v = k, v
        self.prev = self.next = None

class LRUCache:
    def __init__(self, capacity: int):
        assert capacity > 0
        self.cap = capacity
        self.map = {}                            # key -> Node
        self.head, self.tail = Node(), Node()    # 哨兵（环状夹持）
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, n):                        # O(1) 摘链
        n.prev.next, n.next.prev = n.next, n.prev

    def _push_front(self, n):                    # O(1) 头插（MRU 侧）
        n.next, n.prev = self.head.next, self.head
        self.head.next.prev, self.head.next = n, n

    def get(self, key: int) -> int:
        if key not in self.map:
            return -1
        n = self.map[key]
        self._remove(n); self._push_front(n)     # 访问即提升为最新
        return n.v

    def put(self, key: int, value: int) -> None:
        if key in self.map:
            n = self.map[key]; n.v = value
            self._remove(n); self._push_front(n)
            return
        if len(self.map) == self.cap:            # 逐出 LRU（尾哨兵前一格）
            evict = self.tail.prev
            self._remove(evict)
            del self.map[evict.k]
        n = Node(key, value)
        self.map[key] = n
        self._push_front(n)
```

*复杂度：* get/put 均 **O(1)** 时间；空间 O(cap)。
*备选极简版*：`OrderedDict.move_to_end/popitem(last=False)` 两行搞定——面试先写手写 DLL 主版本，末尾补一句"生产可用 OrderedDict"证明 API 熟悉。

**【30秒答题框架】**
"哈希表存 key→链表节点定位 O(1)，双向链表维护访问序，get/put 后都把节点移到头；容量满时逐出尾部（LRU）。用哨兵头尾省边界判断。最忌用的是每次删除时线性查找 prev。"

**【追问与陷阱】**
Bug：摘链只改一遍指针；put 已有 key 且 value 相同时也要 move-to-front；cap=0 未 assert；用单链表（想要 O(1) 删除必须存 prev）。
追问："LFU 怎么改?"（频次桶 + 同频 LRU => O(1) 生产版本比较复杂，说明思路即可）；"Redis 的近似 LRU？"（随机采样近似逐出，不用精确结构，工程 tradeoff）；"与 KV 换页的关联？"（PagedAttention 思想类似，题 6）。

**【知识串联】**
缓存逐出的工程类比 → `09-SystemDesign/01-System-Design.md`；PagedAttention 换页思想 → 题 6 与 `02-Inference/01-KVCache-And-Acceleration.md`。

#### 题 10.2：TopK 问题（含堆与 AVL 对比）

**【考察意图】**
四个层次递进：全排序 O(n log n) → 堆 O(n log k) → quickselect O(n) 平均 → 海量数据分布式 ≥ Top-K。同时把 "AVL 堆" 一类模糊术语澄清：流式 TopK**

**【深度解析】**

*思路推导*：
1. 静态数组 TopK：**快排分区（quickselect）**平均 O(n)（每次期望折半：n + n/2 + n/4 + … = 2n）；最坏 O(n²)（随机 pivot 规避）。
2. 流式 TopK：**小顶堆**维护"当前 k 个最大"，堆顶为最小；新元素 > 堆顶即替换。堆 = **完全二叉树**，push/pop O(log k)。
3. 需要**动态插入+删除+任意时刻 TopK 查询**的流式集合：小顶堆只能 O(log k) 插、删除任意元素 O(k) —— 此时选**平衡 BST（AVL/红黑/TreeMap）**维护大小为 k 的集合：insert/delete/query 都是 O(k)（输出时），保证最坏 log k。这就是"AVL 堆"这类叫法想考的知识对比。
4. 亿级分布式：hash 分桶（Count-Min Sketch 计数）→ 每台机器本地 topk → 归并 topk。

*完整可运行参考实现（含维度注释）：*

```python
import heapq
import random

def top_k_heap(nums, k):
    """静态数组 TopK：小顶堆 O(n log k) 时间 / O(k) 空间"""
    heap = nums[:k]
    heapq.heapify(heap)                        # O(k)
    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)         # O(log k)
    return sorted(heap, reverse=True)          # O(k log k)

def top_k_quickselect(nums, k):
    """quickselect：平均 O(n)，原地修改 nums"""
    lo, hi = 0, len(nums) - 1
    target = len(nums) - k                     # 升序第 target 位即第 k 大的起点
    while True:
        i, p = lo, nums[random.randint(lo, hi)]
        lt, gt = lo, hi
        while i <= gt:                         # 三路划分
            if nums[i] < p:
                nums[lt], nums[i] = nums[i], nums[lt]; lt += 1; i += 1
            elif nums[i] > p:
                nums[i], nums[gt] = nums[gt], nums[i]; gt -= 1
            else:
                i += 1
        if target < lt:   hi = lt - 1
        elif target > gt: lo = gt + 1
        else:             return sorted(nums[target:])   # target..末尾恰为 top-k（target落在等于区 [lt,gt] 内）

if __name__ == "__main__":
    nums = [7, 1, 3, 9, 4, 8, 2, 6]
    print(top_k_heap(list(nums), 3))           # [9, 8, 7]
    print(top_k_quickselect(list(nums), 3))    # 有序后 3 个：[7, 8, 9]
```

*逐行讲解：*
- 堆版：`heapify` 先建 O(k) 堆，`heapreplace` 合并 pop+push；只在意 top-k 不必全排。
- quickselect：三路划分（感兴趣的面试加分项）处理大量重复元素，避免退化 O(n²)。

*复杂度（量化）：* 堆 O(n log k)/O(k)；quickselect 平均 O(n)、最坏 O(n²)；全排 O(n log n)。流式增删用平衡树 O(log k)/op；Count-Min Sketch 计数 O(1)/op，ε-误差换空间。

**【30秒答题框架】**
"静态用 quickselect 平均 O(n)；流式用 k 小顶堆 O(n log k)；数据要动态增删就用平衡 BST（AVL 口径：TreeMap）维护 k 大小集合，插删查都 log k。海量数据先分桶 sketch 计数再分治归并。注意 Python heapq 是小顶堆，求 top 最大用小顶很容易写反。"

**【追问与陷阱】**
Bug：求最大 k 却用大顶堆（应小顶堆）；quickselect 递归写错区间；没说明不稳。
追问："为什么找第 k 大用小顶堆？"（维护准入门槛）；"HashMap 计数 + 堆呢？"（TopK 词频 = 计数后同上）；"LLM 里哪里出现？"（推理 logits top-k 采样、RAG 召回 top-k、检索 top-k —— 算法思想与 Decoding 里的 top-k 完全同一：见 `02-Inference/02-Decoding-Strategies.md`）。

**【知识串联】**
与解码 top-k/top-p 的同源思想 → `02-Inference/02-Decoding-Strategies.md`；检索 topk 与 argpartition 工程版（题 8）；系统级海量 TopK → `09-SystemDesign/01-System-Design.md`。

#### 题 10.3：合并 K 个有序链表

**【考察意图】**
Hard 题考的是"堆的泛化比较键"写法（ListNode 不可比时用 `(val, i, node)` 元组）与分治法两种写法的复杂度说服力。

**【深度解析】**

*思路推导*：K 条链表共 N 节点。
- 方案 A（小顶堆）：堆里始终只剩 K 个候选（每链当前头），每次弹出全局最小 O(log K)，共 N 次 → **O(N log K)**，空间 O(K)。
- 方案 B（分治归纳）：两两合并 log K 轮，每轮总代价 O(N) → O(N log K)，额外空间 O(1)（迭代版）。
- 方案 C（暴力顺序合并）：第 i 次合并代价 O(累计长度) → 最坏 O(N·K)，面试拿这个对比凸显 A/B。

*完整可运行参考实现（含维度注释）：*

```python
from heapq import heapify, heappop, heappush

class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def merge_k_lists(lists):
    """lists: K 条有序链表的头。时间 O(N log K)，空间 O(K)，N=总节点数"""
    dummy = cur = ListNode()
    heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
    heapify(heap)                              # O(K)
    while heap:
        val, i, node = heappop(heap)           # O(log K) 全局最小
        cur.next = node; cur = node
        if node.next:
            heappush(heap, (node.next.val, i, node.next))
    cur.next = None
    return dummy.next
```

*逐行讲解：*
- 堆元素必须放 `(val, i, node)`：node 是对象不可比较（Python3 无 `__lt__` 会 TypeError），加 i 破 tie 并顺带拼写稳定。
- 弹出后把**该链的下一个**压回堆——维持"每链至多一个候选"不变式。

*复杂度：* 时间 O(N log K)；空间 O(K)（堆）。分治版同复杂度但空间 O(1)/无递归栈。

**【30秒答题框架】**
"维护一个 K 大小的最小堆，每个链表当前头都在里面；每次弹全局最小接到结果链，把它的 next 推回。N 个节点全部走一遍，O(N log K)。面试备选分治两两合并也是 O(N log K)、常数小。关键细节是堆元素加链表索引破除节点不可比。"

**【追问与陷阱】**
Bug：push `(val, node)` 两元素时 tie 引发 TypeError；尾指针忘收 `cur.next=None`（可能成环）。
追问："和外部归并排序的关系？"（k 路归并范式一致、调度磁盘 IO）、"元素热重排用什么堆？"（索引堆/fibonacci 堆提法即可）。

**【知识串联】**
"k 路归并"在 beam search（题 5：B² 候选归并）与外部 sort 里同构；系统设计的数据管道 → `09-SystemDesign/01-System-Design.md`。

#### 题 10.4：二叉树层序遍历

**【考察意图】**
BFS 的队列 size 分层技巧 + 高频变体（之字形/右视图）。

**【深度解析】**

*思路推导*：队列逐层处理：记录当层元素数 `n=len(q)`，弹 n 个收集一层 → 下一层孩子依次入队。`queue.popleft()` O(1) 必须用 `collections.deque`（list.pop(0) 是 O(n)，高频扣分点）。

*完整可运行参考实现（含维度注释）：*

```python
from collections import deque
from dataclasses import dataclass

@dataclass
class TreeNode:
    val: int = 0
    left: "TreeNode" = None
    right: "TreeNode" = None

def level_order(root):
    """时间 O(n)，空间 O(w)（w=最大层宽）"""
    if not root: return []
    q, res = deque([root]), []
    while q:
        level = []
        for _ in range(len(q)):            # 只消费当层：len(q) 必须先固化
            node = q.popleft()             # O(1)
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        res.append(level)
    return res
```

*复杂度：* 时间 O(n)（每节点进出各一次）；空间 O(w)（队列）。
变体：之字形（奇偶层反序 append）；右视图（每层最后一个，DFS 深度优先记录首个到达深度的节点也拿下）。

**【30秒答题框架】**
"双端队列做 BFS：每轮先记 len(q) 就是当层节点数，弹满一轮输出层。注意用 deque 而不是 list，list.pop(0) 是 O(n)。变体之字就按层号反转，右视图取每层末位。"

**【追问与陷阱】**
Bug：在 `for _ in range(len(q))` 中动态 push 使 len 变化（用先固化）；DFS 递归层序反而绕弯。
追问："DFS 能不能做层序？"（可以，按深度分桶，一层 list）；"最大宽度？"

**【知识串联】**
图 BFS → 10.6 岛屿；层次遍历与模型层级 vs 数据结构，字符串串场话题即可。

#### 题 10.5：最长公共子序列（LCS）

**【考察意图】**
经典二维 DP：是否知道子序列 ≠ 子串、状态定义、空间滚动优化。评测指标的同族思想（编辑距离）也常被借来考。

**【深度解析】**

*思路推导*：
状态：`dp[i][j]` = a 的前 i 个字符与 b 的前 j 个字符的 LCS 长度。
转移：`a[i-1]==b[j-1] ? dp[i-1][j-1]+1 : max(dp[i-1][j], dp[i][j-1])`（一次只跳过一个/两个指针都跳；两个都跳会被 dp[i-1][j-1]+1 覆盖——经典证明）。
初始：dp[0][j]=dp[i][0]=0；答案 dp[n][m]。
空间滚动：只需上一行与 dp[i][j-1] → 保留 `prev` 注记。

*完整可运行参考实现（含维度注释）：*

```python
def lcs(a: str, b: str) -> int:
    """时间 O(mn)；空间 O(min(m,n))（滚动数组）"""
    n, m = len(a), len(b)
    dp = [0] * (m + 1)               # dp[j] 表示 dp[i][j]
    for i in range(1, n + 1):
        prev = 0                     # 记 dp[i-1][j-1]，进入 j=1 时为 0
        for j in range(1, m + 1):
            tmp = dp[j]              # dp[i-1][j]
            dp[j] = prev + 1 if a[i-1] == b[j-1] else max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]

if __name__ == "__main__":
    print(lcs("abcde", "ace"))   # 3 ("ace")
```

*复杂度：* 时间 O(m·n)；空间未优化 O(mn)，滚动后 O(min(m,n))（先 swap 使 b 为较短者）。

**【30秒答题框架】**
"状态 dp[i][j] 表示两前缀的 LCS；相同字符 → 左上+1，否则取上/左较大者；答案 dp[n][m]。要还原序列需保留全表回溯；空间用滚动数组压到 O(min(m,n))。子序列不要求连续，子串会不同——别与"最长公共子串"混淆。"

**【追问与陷阱】**
Bug：滚动更新时未用临时变量保存 dp[i-1][j-1]（直接 dp[j-1] 已被改写）；把"子序列"当"连续子串"转移。
追问："编辑距离一齐对比？"（同族 DPs，加/删/改三向）；"sequence 评测指标？BLEU/ROUGE 的编辑距离同源"（`06-Safety-Evaluation/01-Safety-And-Evaluation.md` 评测章节）。

**【知识串联】**
评测指标相关（ROUGE/BLEU 的序列对比）→ `06-Safety-Evaluation/01-Safety-And-Evaluation.md`；seq2seq 生成与 LCS 类指标 → `01-Foundation/01-Transformer.md`。

#### 题 10.6：岛屿数量

**【考察意图】**
图连通分量的 flood-fill：递归 vs BFS 的栈深考虑、原地标记、并查集加分。

**【深度解析】**

*思路推导*：扫格点，遇 '1' 计数 +1 并 flood-fill（DFS/BFS）把整块 '1' 清 '0'（或 visited 数组标记）。递归 DFS 简洁但最坏栈深 O(mn)（大盘面上万行递归可能爆栈）→ BFS 用队迭代更稳。

*完整可运行参考实现（含维度注释）：*

```python
def num_islands(grid) -> int:
    """grid: List[List[str]]（'0'/'1'）。原地标记。时间/空间 O(mn)"""
    if not grid: return 0
    m, n = len(grid), len(grid[0])

    def sink(i: int, j: int) -> int:
        if 0 <= i < m and 0 <= j < n and grid[i][j] == "1":
            grid[i][j] = "0"                    # 标记访问，一步完成
            stack = [(i, j)]                    # 迭代 DFS：防爆栈
            while stack:
                x, y = stack.pop()
                for dx, dy in ((1,0), (-1,0), (0,1), (0,-1)):
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == "1":
                        grid[nx][ny] = "0"      # 入栈前标记，防重复入栈
                        stack.append((nx, ny))
            return 1
        return 0

    return sum(sink(i, j) for i in range(m) for j in range(n))
```

*复杂度：* 每格至多进出一次 → 时间 O(m·n)；空间递归最坏 O(mn)、BFS 队列最坏 O(min(m,n))。
变体："岛屿的最大面积"（同 flood-fill 计面积）；并查集版本支持增量插岛。

**【30秒答题框架】**
"扫二维格子，遇陆地计数并把整连通块 flood-fill（DFS/BFS 标 '0'）。递归最坏栈深 O(mn) 可能爆，用显式栈/队列；入栈前先标记防止重复。也可以用并查集支持动态插入场景。"

**【追问与陷阱】**
Bug：`'1'` 与整数 1 混淆（LeetCode 是 string 矩阵）；入栈后标记 → 同格多次入栈；递归爆栈没讨论。
追问："最大岛屿？"（flood-fill 过程中计数取 max）；"动态加格子？"（并查集 union 增量维护）。

**【知识串联】**
连通分量/graph 分桶 → `09-SystemDesign/01-System-Design.md` 相关；agent 图规划（任务依赖图）概念 → `05-Agent/01-Agent-Complete-Guide.md`。

#### 题 10.7：快速排序 / 归并排序

**【考察意图】**
一手写不越界。同场追问 stability、三路划分、为什么工业库用 Timsort/hybrid 内省排序。

**【深度解析】**

*思路推导：*
- 快排 partition：选 pivot，一次遍历达成 `< p | ==p | > p` 三段（三路（Dutch flag），处理大量重复元素避免退化）；递归两段。
- 复杂度：quick sort 平均 O(n log n)、最坏 O(n²)（随机 pivot 概率意义规避）、**不稳定**、原地 O(log n) 栈；merge sort 恒 O(n log n)、稳定 O(n) 空间（链表版 O(1) 且是链表排序最优）。

*完整可运行参考实现（含维度注释）：*

```python
import random

def quick_sort(a, lo=0, hi=None):
    """原地三路快排；平均时间 O(n log n)，空间 O(log n) 递归栈"""
    if hi is None: hi = len(a) - 1
    if lo >= hi: return a
    p = a[random.randint(lo, hi)]           # 随机 pivot 防构造最坏用例
    lt, i, gt = lo, lo, hi
    while i <= gt:                          # 不变式：[lo,lt) < p | [lt,i) == p | (gt,hi] > p
        if a[i] < p:
            a[lt], a[i] = a[i], a[lt]; lt += 1; i += 1
        elif a[i] > p:
            a[i], a[gt] = a[gt], a[i]; gt -= 1   # i 不动：换来的数未检查
        else:
            i += 1
    quick_sort(a, lo, lt - 1)
    quick_sort(a, gt + 1, hi)
    return a

def merge_sort(a):
    """稳定；时间 O(n log n)，空间 O(n)"""
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    l, r = merge_sort(a[:mid]), merge_sort(a[mid:])
    out, i, j = [], 0, 0
    while i < len(l) and j < len(r):
        if l[i] <= r[j]:                    # <= 保证稳定：相等取左半
            out.append(l[i]); i += 1
        else:
            out.append(r[j]); j += 1
    out += l[i:]; out += r[j:]
    return out

if __name__ == "__main__":
    print(quick_sort([3,5,1,3,3,2])
     == sorted([3,5,1,3,3,2]))            # True
    print(merge_sort([3,5,1,3,3,2]))          # [1,2,3,3,3,5]
```

*复杂度：* quicksort avg O(n log n)/worst O(n²)，space O(log n)；mergesort O(n log n)/O(n) 稳定。

**【30秒答题框架】**
"快排随机 pivot + 三路划分防重复退化，平均 O(n log n) 不稳定原地；归并恒 O(n log n) 稳定，链表版 O(1) 空间。生产语言库用混合（Timsort/内省）兼顾稳定、有序数据利用和退化保护。"

**【追问与陷阱】**
Bug：三路 partition 中 `a[i] > p` 交换后 `i` 应停留（新 a[i] 未检查）；quicksort 不稳定却拿去做"按主序+次序"的双键排序；recursion depth 默认1000（Python 默认 Python 默认——超长数组需迭代 hedged（Python 栈限制）或递归深度法。
追问："为什么 Python/C++ 标准库不直接快排？"（Timsort：稳定 + 对已序部分接近 O(n)；C++ introsort：quicksort+heapsort+insertion 混合）；"GPU/LM 领域排序？"（Radix/TopK 融合 kernels 出现在 beam 与采样的 vocab 排序中）。

**【知识串联】**
Merge/heap 在 beam search（题 5）、TopK（10.2）里复用；系统设计的海量外部排序 → `09-SystemDesign/01-System-Design.md`。

#### 题 10.8：滑动窗口最大值

**【考察意图】**
单调队列模板题：为什么堆 O(n log n) 而单调双端队列 O(n)；窗口过期管理。

**【深度解析】**

*思路推导：*
单调递减双端队列（存**下标**）：
- 入队前从队尾弹出所有 ≤ 当前值的下标 —— 它们永远不可能再成为某窗口最大值（当前值更新、更靠右）。
- 队头是当前窗口最大：仅当队头下标 ≤ i−k（出窗）时 popleft。
每元素至多入队/出队各一次 → O(n)。

*完整可运行参考实现（含维度注释）：*

```python
from collections import deque

def max_sliding_window(nums, k: int):
    """单调减双端队列（下标）。时间 O(n)，空间 O(k)"""
    dq, res = deque(), []                    # 存下标，保证 nums[dq[0]] >= nums[dq[i]]
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:      # 尾部弹：数值不占优的淘汰
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                   # 头部过期淘汰（下标出窗）
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]])          # 队头即当前窗口最大
    return res

if __name__ == "__main__":
    print(max_sliding_window([1,3,-1,-3,5,3,6,7], 3))   # [3,3,5,5,6,7]
```

*复杂度：* 每个下标最多 push/pops 一次 → **O(n)** 时间、O(k) 空间。
对比：multiset/平衡树 O(n log k)；堆(懒删除) O(n log n)。

**【30秒答题框架】**
"单调递减双端队列存下标：入队前把尾部所有 ≤ 新值的弹掉，队头下标出窗时 popleft；队头就是窗口最大。每元素进出队各一次 O(n)，对比显然的堆是 O(n log n)。"

**【追问与陷阱】**
Bug：**队里存值而非下标**（出窗时无法判断过期——唯一可靠做法是存下标）；过期判断写成 `dq[0] < i-k+1`（等价但易错，用 `<= i-k` 更直观）；用 `<=` 还是 `<` 弹尾都可能正确，但要能解释 tie 的处理。
追问："CNN max-pooling？"（同构，kernal=k stride=1）；"二维滑动最大？"（一维滑动+列分块级联）；"为什么不是堆？"（懒删除堆其实也可用，O(n log n) 常数大）。

**【知识串联】**
cnn pooling 与卷积的滑窗思想 → `01-Foundation/01-Transformer.md`（卷积对比）；流式统计在监控类系统 → `09-SystemDesign/01-System-Design.md`。

---

### 题 11（补充题）：GQA 分组注意力（Grouped-Query Attention）

> 原文件没有 GQA 的实现（它紧邻 MHA 与 KV-Cache 两节，属于"原章节欠实现"，此处按任务要求补全，编号顺延）。

**【考察意图】**
2023 起的必考题：(1) MHA/MQA/GQA 三档 KV 头数的关系与**KV cache 显存折扣因子 h/h_kv**；(2) **在代码里正确实现"头对齐"**：n_rep = h/h_kv 组内头共享同一 KV —— `repeat_interleave(dim=1)` 与 `torch.repeat(dim=1)` 的差异（后者是平铺 tiling，分组错位，经典手撕 bug）；(3) 能报出主流模型的具体选型（LLaMA-2-70B GQA-8、LLaMA-3 全系 GQA-8、Mistral-7B GQA-8、PaLM-540B MQA）。

**【深度解析】**

*思路推导：*

- 显存动机（题 6 的延伸）：KV cache 显存 ∝ KV 头数。MHA 每层每 token 存 2·d_model；若 KV 头压缩到 h_kv，
  `KV显存/token/层 = 2·h_kv·d_k·bytes`，折扣因子 **(h_kv/h)**。
  LLaMA-7B（d=4096、h=32、d_k=128、32 层）：
  − MHA：2·4096·32·2B = **512 KB/token**（2k 上下文 ≈ 1.05 GB/请求）
  − GQA-8：KV 维 = 8×128=1024 → 2·1024·32·2B = **128 KB/token**（**4× 折扣**）
  − MQA（h_kv=1）：2·128·32·2B = **16 KB/token**（32× 折扣，质量反而最损失）
- GQA 的质量折衷：论文结论"刀口"：h_kv≈8 时质量几乎不掉、吞吐显著提升；训练时把相同位置 xq 加相同组内 K/V 表示（参数少，KV 只学 h_kv 组），推理 Q 用组内共享 KV → 头对齐。
- 关键 head mapping：query 头 i 属于组 `i // (h/h_kv)`。实现上：KV 经过 `expand/reshape` 复制 n_rep 份。`repeat_interleave(x, n_rep, dim=1)` 产生 `[k0,k0,k0,k1,k1,k1,...]`（正确：同组头相邻）；而 `torch.repeat(x, n_rep, dim=1)` 产生 `[k0,...,k_{h_kv-1},k0,...]` 的平铺 tiling（错误分组——**这是手撕 GQA 最常见 bug**，因为 Q 头的排序是 0..h 连续，交叉平铺会把错误的 K 配对）。

*完整可运行参考实现（含维度注释）：*

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    """Grouped-Query Attention（LLaMA-2-70B/LLaMA-3/Mistral 风格）。
    d_model 必须能被 n_heads 整除；n_heads 必须能被 n_kv_heads 整除。
    """
    def __init__(self, d_model, n_heads, n_kv_heads, causal=True):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "GQA 头对齐前提：n_heads 整除 n_kv_heads"
        self.h, self.h_kv = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads          # 每组共享 KV 的 query 头数
        self.d_k = d_model // n_heads
        self.causal = causal

        self.wq = nn.Linear(d_model, d_model, bias=False)             # h 组：全量 Q
        self.wk = nn.Linear(d_model, self.h_kv * self.d_k, bias=False)  # 关键：只产 h_kv 组 KV
        self.wv = nn.Linear(d_model, self.h_kv * self.d_k, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.k_cache = None                                           # (B,h_kv,L,d_k)
        self.v_cache = None

    @staticmethod
    def expand_kv(x, n_rep):
        """x: (B,h_kv,L,d_k) → (B,h,L,d_k)
        用 expand+reshape（等价 repeat_interleave）：第 i 组 KV 复制到 query 头 [i*nr,(i+1)*nr)"""
        B, h_kv, L, d_k = x.shape
        x = x[:, :, None, :, :].expand(B, h_kv, n_rep, L, d_k)   # 仅视图，无拷贝
        return x.reshape(B, h_kv * n_rep, L, d_k)                # reshape 触发拷贝
        # 错误写法（平铺错组）：x.repeat(1, n_rep, 1, 1)
        #   repeat 在 head 维平铺 [k0..k7,k0..k7,...]；与 Q 头 0..h-1 顺序错位
        # 也可写 torch.repeat_interleave(x, n_rep, dim=1)，结果同 expand+reshape

    def forward(self, x, use_cache=True):
        B, L, _ = x.shape
        Q = self.wq(x).view(B, L, self.h,    self.d_k).transpose(1, 2)   # (B,h,L,d_k)
        K = self.wk(x).view(B, L, self.h_kv, self.d_k).transpose(1, 2)   # (B,h_kv,L,d_k)
        V = self.wv(x).view(B, L, self.h_kv, self.d_k).transpose(1, 2)

        past_len = 0
        if self.k_cache is not None and use_cache:
            past_len = self.k_cache.shape[2]
            K = torch.cat([self.k_cache, K], dim=2)                      # (B,h_kv,past+L,d_k)
            V = torch.cat([self.v_cache, V], dim=2)
        if use_cache:
            self.k_cache, self.v_cache = K.detach(), V.detach()          # cache 只存 h_kv 组！

        Kf, Vf = self.expand_kv(K, self.n_rep), self.expand_kv(V, self.n_rep)  # (B,h,L+past,d_k)

        scores = torch.matmul(Q, Kf.transpose(-2, -1)) / math.sqrt(self.d_k)   # (B,h,L,L_total)
        if self.causal:                                   # prefill 需要；decode(L=1)天然只看历史
            L_total = scores.shape[-1]
            causal = torch.tril(torch.ones(L_total, L_total, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~causal[past_len:, :L_total], float("-inf"))
        attn = F.softmax(scores, dim=-1)                                     # (B,h,L,L_total)
        out = torch.matmul(attn, Vf)                                         # (B,h,L,d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)                # (B,L,d_model)
        return self.wo(out)

    def clear_cache(self):
        self.k_cache = self.v_cache = None

if __name__ == "__main__":
    torch.manual_seed(0)
    m = GQA(d_model=256, n_heads=8, n_kv_heads=2)   # h=8, h_kv=2 → 折扣 4×
    y = m(torch.randn(2, 6, 256), use_cache=True)   # prefill
    y2 = m(torch.randn(2, 1, 256), use_cache=True)  # decode 一步
    print(y.shape, y2.shape)                        # (2,6,256) (2,1,256)
    kb_per_tok = 2 * 2 * 128 * 2                    # 2(KV)×h_kv×d_k×fp16 字节
    print(f"KV cache: {kb_per_tok/1024:.3f} KB/token/layer")   # 0.5 KB/层 ≈ 正确折扣
```

*逐行讲解：*
- `wk/wv` 输出维度是 `h_kv·d_k`：**把 KV 的参数/显存一次性减到 h_kv 组**，这是 GQA 的本质；`expand_kv` 不改变已存 cache，只在算 attention 时临时展开。
- `expand+reshape` 分组展开是否正确：expand 视图表示"组内 n_rep 个相同 K"，reshape 把 (h_kv, n_rep) 折成 h → 头序为 [组0头0,组0头1,…,组1头0,…]，与 Q 头 0..h-1 天然对齐。
- causal mask 的切片 `~causal[past_len:, :L_total]`：decode 的新 query 位于整段的第 past_len.. 行，只允许看前 L_total 列——手写时容易忘了按 past_len 切行（L=1 时等价退化即可全列，prefill 时必须切）。
- KV cache 直接以 (B,h_kv,L,d_k) 形态保存，显存即按 h_kv 折扣計。

*复杂度（量化）：*
- attention 矩阵乘不变（Q 头数照常 h）：O(B·L²·d) prefill / O(B·L·d) decode。
- **KV cache 显存降 h/h_kv 倍**；参数侧：wq/wo 不变（各 d²），wk/wv 从 d² 降到 d·(h_kv·d_k)=d²·h_kv/h → 单层 attention 参数从 4d² 变为 d²·(2+2h_kv/h)。LLaMA-70B 量级（h_kv/h=8/64=1/8）：wk+wv 从 2d² 压到 d²/4，省 3/4。
- LLaMA-2-70B（h=64,h_kv=8）：KV 显存 320KB/token（MHA 版会是 1.28MB/token，4×）。

**【30秒答题框架】**
"GQA 让 h 个 query 头共享 h_kv 组 KV，h_kv=1 就是 MQA，=h 就是 MHA。收益全在 KV cache：按 h/h_kv 缩小，LLaMA-7B FP16 MHA 是 512KB/token，GQA-8 变 128KB。实现重点是头对齐：KV expand 时必须是 repeat_interleave（组内相邻复制），写成 torch.repeat 平铺就配错组；动手前先 assert n_heads % n_kv_heads == 0。Mistral/LLaMA-3 都是 GQA-8，PaLM 用 MQA。"

**【追问与陷阱】**

*常见手写 Bug：*
1. **`torch.repeat` 平铺 instead `repeat_interleave`/expand**：头对齐错位，模型能跑但质量崩——手撕面试的杀手细节。
2. cache 按 (B,h_kv,…) 保存却按 h 展开——不展开直接做 matmul 广播也可以（`Q(B,h,L,d)·K(B,h_kv,L,d)ᵀ` 不广播，必须显式展开或用 einsum 分组算）。
3. `n_heads % n_kv_heads != 0` 没 assert。
4. GQA 下 WK/WV 输出维写成 d_model（变相 MHA，cache 没省）。
5. decode 时 causal mask 按整段 tril 或无 mask 混杂（新 token 单 query 对 prefill 与 decode 的差异——题 6 同款坑）。

*面试官追问：*
- " 为什么 GQA-8 是甜点位？" → 论文网格搜索：h_kv≈8 时质量损失可近似忽略，吞吐接近 MQA；更少 KV 头质量掉头，更多收益饱和。
- "训练原有 MHA ckpt 怎么变 GQA？"（GQA 论文的 mean-pooling KV 头 + 短训恢复）；"KV 头数选择？"（保持 head_dim 64/128，动 h_kv）。
- " cache 现在还能再省？"（int8/int4 KV、 sliding window（Mistral）/speculative、MLA（DeepSeek 多头隐注意力，压缩 KV 到一个共享潜向量，2024 热点））—— MLA 一句带过连到 `07-HotTopics/01-Hot-Topics-2025-2026.md`。
- "drawback？"（K/V 表达力受限，相近任务上 GQA-vs-MHA 差距自热点任务上可见）。
- "MLA 与 GQA 的关系？"

**【知识串联】**
GQA/MQA/MLA 的背景与显存收益 → `02-Inference/01-KVCache-And-Acceleration.md`；长上下文与 MLA/SSM → `07-HotTopics/01-Hot-Topics-2025-2026.md`；实现细节承接题 2（头拆分）、题 6（cache）、题 9（RoPE 偏移）；字节真题里的 GQA 手撕出现频次高 → `10-RealQuestions/01-ByteDance-Questions.md`。

---

## 本章高频追问 Top5

1. **"mask 为什么必须在 softmax 之前？漏 √d_k 会怎样？整行 -inf NaN 怎么救？"**
   标准答案：softmax 是行内竞争，先竞争后屏蔽等于没屏蔽；√d_k 防分数方差线性膨胀（Var[q·k]=d_k 推导要能现场写）；全 -inf 行用 -1e9 或确保对角位；顺带说出 FlashAttention 用重缩放解决 O(L²) 显存。

2. **"现场算 KV cache 显存：LLaMA-7B / 2048 上下文 / fp16 是多少？"**
   公式 2·h_kv·d_k·layers·bytes/token：MHA 512KB/token → 2k ctx ≈ 1.05GB；GQA-8 减 4 倍 = 128KB/token；连续乘上 batch 与并发请求数就是生产容量规划。下一问大概率是"PagedAttention/GQA/MLA/量化各能省多少"。

3. **"训练和推理的 FLOPs 经验公式会推吗？"**
   GEMM=2mnp → 前 2ND、训练 6ND。要能从"每参数一次前向 2 FLOPs、backward 2× forward"推出来，并顺手算过 LLaMA-7B 2048 token 前向 ≈ 29 TFLOPs，除硬件峰值 → MFU；decode 算术强度 ≈1 FLOP/byte → memory-bound。

4. **"LoRA 的 B 为什么要初始化为 0？合并推理为什么无损？"**
   B=0 → 起点函数与预训练逐点相等（A 随机、B 零的组合同时避免 A=B=0 的梯度死锁）；α/r 把等效步长与秩解耦；ΔW=(α/r)BA 可合并回权重矩阵 W′=W+ΔW，前向一次 GEMM 零额外延迟。追问链：rank 选择、QLoRA、target 模块（attention+FFN 都挂）。

5. **"手撕多头的 reshape 链讲一遍，为什么合并要 contiguous？GQA 的 KV 又怎么展开？"**
   拆：view(B,L,h,d_k)→transpose(1,2)；合并：transpose(1,2)→contiguous→view（transpose 打断内存连续性）；GQA 头对齐用 expand+reshape/repeat_interleave（组内相邻），错用 torch.repeat 平铺就配错组——这是题 2 与题 11 的组合追问，近两年出现率飙升。
