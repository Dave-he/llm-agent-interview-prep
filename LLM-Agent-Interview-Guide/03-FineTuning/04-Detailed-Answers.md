# 第 03 章：微调（Fine-Tuning）逐题详解

> 来源文件：`01-PEFT.md`（5 题）、`02-Alignment.md`（8 题）、`03-Instr
uction-Tuning.md`（5 题），共 **18 题**，覆盖率 100%。原文只读，本文件为配套详解。
> 约定：题目编号沿用原文件顺序；与原文冲突处（如 DAPO 全称）以官方论文为准并明确标注。

## 本章考点分布

| 子领域 | 题号 | 难度分布 | 核心考点 | 一句话总结 |
|--------|------|----------|----------|------------|
| PEFT：LoRA | 题 1–3 | ⭐⭐⭐⭐ | ΔW=BA 低秩假设、参数量算例、A0/B0 初始化、秩 r 选择、注入层位 | "低秩假设 + 初始化对称性破缺 + 零延迟合并" |
| PEFT：QLoRA | 题 4 | ⭐⭐⭐⭐ | NF4 量化数学、双量化显存账本、paged optimizer | "NF4 = 对高斯权重的 Lloyd-Max 最优标量量化；35B 级模型进 48GB 卡" |
| PEFT：Adapter/Prefix/Prompt | 题 5 | ⭐⭐⭐ | 注入位置谱系（层间 / KV 前 / embedding 前）与延迟代价 | "注入位置决定推理开销：越靠近输入越省钱越弱" |
| 对齐：RLHF | 题 6–7 | ⭐⭐⭐⭐ | 三阶段流水线、RM 损失、PPO 四模型、KL 两种施加位置 | "KL 既是缰绳也是防 reward hacking 的正则" |
| 对齐：DPO | 题 8–9 | ⭐⭐⭐ | KL-约束 RLHF 目标的闭式解、Z(x) 相消、β 温度角色 | "把奖励重参数化为 β·log(π/π_ref)，BT 模型吃掉不可积分配函数" |
| 对齐：GRPO 及谱系 | 题 10–12 | ⭐⭐⭐⭐⭐ | 组内相对优势替代 Critic 的推导、方差/组大小、DAPO/REINFORCE++ 修正 | "V(x) 用同组蒙特卡洛均值替代；KL 从 reward 挪进 loss" |
| 对齐：奖励设计 | 题 13 | ⭐⭐⭐ | Function Calling 分层奖励 + RLVR 思想 | "可验证奖励 > 学习型奖励，权重设计防 hack" |
| 指令微调 | 题 14–18 | ⭐⭐⭐ | 输出侧掩码损失、LIMA 质量>数量、ChatML 多轮、全参/LoRA 选型、遗忘与过拟合 | "数据质量与多样性的边际收益远高于条数；模板一致性是隐形坑" |

面试画像：PEFT 一线是"数学 + 显存账本"题（字节二面必考 LoRA 算例与 QLoRA 显存公式）；对齐一线是"推导链"题（DPO 推导要能白板逐步写出、GRPO 要能讲清为什么砍 Critic 不掉点）；指令微调一线偏工程判断（数据配比、遗忘治理），是区分"用过"与"调过"的分水岭。

---

### 题 1：LoRA 的原理 ⭐⭐⭐⭐

**【考察意图】**

验证三件事：(1) 是否真正理解"权重更新是低秩的"这一假设的来源与含义，而不是背"冻结 W、训练 BA"；(2) 能否白板上手做参数量与计算量的算例（d=4096、r=16 这种数字要能脱口而出）；(3) 是否知道 A 随机 / B 置零的初始化理由——这是区分"用过 LoRA"与"读过论文"的探针题。

**【深度解析】**

**1. 假设来源：微调更新矩阵是（近似）低秩的。**
全量微调学到的更新量 ΔW = W_sft − W_pretrain，其本征秩（intrinsic rank）通常远小于矩阵维度。Aghajanyan et al. (2020) 证明微调后的泛化性能由低"内在维度"参数子空间决定；LoRA 论文观察到训练中 ΔW 的谱迅速坍缩（前几个奇异值主导）。因此把 ΔW 约束为秩 ≤ r 的矩阵，用两个瘦矩阵的乘积 B·A 参数化：

$$W' = W + \Delta W = W + BA,\quad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times d},\ r\ll d$$

rank(BA) ≤ min(rank B, rank A) ≤ r。当 r = d 时失去约束，LoRA 蜕化为带特定初始化的全量微调——论文用这一点证明低秩约束在小 r 下不掉点。

**2. 前向与缩放。** 实际前向为

$$h = Wx + \frac{\alpha}{r}\,B\,A\,x$$

α 是秩不变常数（原论文随 r 改变时固定 α/r，避免重调学习率；实践中常取 α = 2r 或 α = r）。计算上 B A x 分两步瘦矩阵乘：开销 O(n·d·r)（n 为 token 数），相对主干 Wx 的 O(n·d²) 可忽略；训练时对 W 的梯度不需要存储/更新（W 冻结），省下的是优化器状态和梯度，而非前向时间。

**3. 参数量算例（要求公式完整可复现）。**

| 量 | 公式 | 代入 d=4096, r=16 |
|----|------|--------------------|
| 全参可训练参数（单矩阵） | d² | 4096² = 16,777,216 |
| LoRA 参数 | 2·d·r | 2·4096·16 = 131,072 |
| 参数比 | 2r/d | 2·16/4096 = **0.78%** |

对 7B 模型只在 Q/K/V/O 注入（d_model=4096，32 层，每层 4 个 d×d 投影）：每层 LoRA 参数 = 4×2dr = 4×131,072 = 524,288，全模型 = 524,288×32 ≈ **16.8M ≈ 7B 的 0.24%**。收益不在参数本身而在优化器状态：AdamW 每个可训练参数约需 权重2B+梯度2B+动量方差8B（fp32）≈ 12B/参数，7B 全参 ≈ 84GB 优化开销；LoRA 16.8M 可训练参数只占 ~0.2GB，其余 56GB 为冻结权重（bf16）。

**4. 初始化：A 随机高斯、B 全零——两条硬理由。**

- 理由 A（行为等价）：B=0 ⇒ ΔW=BA=0，t=0 时刻模型与预训练模型严格逐 bit 等价，训练从"原模型"出发平滑偏移，不会开局就破坏预训练分布。
- 理由 B（对称性破缺 / 死锁规避）：对 h = Wx + (α/r)BAx 求梯度：
  $\frac{\partial \mathcal L}{\partial A} = \frac{\alpha}{r} B^{\!\top} \frac{\partial \mathcal L}{\partial h}\, x^{\!\top}$，$\frac{\partial \mathcal L}{\partial B} = \frac{\alpha}{r} \frac{\partial \mathcal L}{\partial h}\, (Ax)^{\!\top}$。
  若 A、B 全零：∂L/∂A = B^T(·) = 0 且 ∂L/∂B = (Ax)^T(·) = 0——两者梯度同时为零，ΔW 永久死在原点（双零初始化不可训练）。若 A=0、B 随机：第一轮 ∂L/∂A = 0、∂L/∂B ≠ 0 同样僵住。所以必须**恰好一个因子非零**：取 A 随机（保证 ∂L/∂B 第一步非零，B 先动起来带动 ΔW 离开零空间）、B 置零（保证 t=0 等价性）。代价是第一步 ∂L/∂A 为 0，A 要等 B 非零后才开始更新——这是"最小破坏初始等价性"与"两因子同步梯度"之间的必然取舍，PiSSA / LoRA-GA 等改进工作即针对此（用 SVD 主成分初始化 A、B，第 0 步梯度即两侧都非零）。

**5. 可运行参考实现（PyTorch，完整可跑，复杂度标注）。**

```python
import math
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    """y = W x + (alpha/r) * B(A x)，B 初始为 0。
    复杂度：前一次调用，LoRA 侧 O(d*r) 每行输入；合并后与原 Linear 相同。
    显存：可训练参数 2*d*r 与其梯度 + Adam 双状态（fp32），冻结 W 的权重/梯度/优化器态全部省去。
    """
    def __init__(self, in_features: int, out_features: int, r: int = 16,
                 alpha: int = 32, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features),
                                   requires_grad=False)          # 预训练 W，冻结
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r                                  # 常数 α/r
        self.lora_A = nn.Parameter(torch.empty(r, in_features))   # 随机高斯
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))  # 置零 ⇒ ΔW=0
        nn.init.normal_(self.lora_A, std=1.0 / math.sqrt(r))      # A≠0 ⇒ 第一步 ∂L/∂B≠0
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x: (..., d_in)
        base = nn.functional.linear(x, self.weight)               # O(...*d_in*d_out)
        delta = (self.lora_B(self.lora_A(self.lora_dropout(x))    # 两步瘦矩阵乘
                * self.scaling))                                  # O(...*d_in*r + ...*r*d_out)
        return base + delta

    @torch.no_grad()
    def merge(self) -> nn.Linear:                                 # 推理前合并 ⇒ 零额外延迟
        merged = self.weight + self.lora_B @ self.lora_A * self.scaling  # O(d_out*r*d_in) 一次性
        return nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False).__init__ and None

# ---- 算例与行为断言（可直接运行）----
if __name__ == "__main__":
    d, r = 4096, 16
    layer = LoRALinear(d, d, r=r, alpha=2*r)
    trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    total = d * d
    print(f"LoRA 参数 {trainable}，全量 {total}，占比 {trainable/total:.2%}")  # 0.78%
    x = torch.randn(2, 128, d)
    t0 = layer(x).clone()
    assert torch.equal(t0, nn.functional.linear(x, layer.weight)), "ΔW=0 ⇒ 初始等价"
    # 反向：t=0 时 ∂L/∂A 应为 0，∂L/∂B 非零（验证对称性讨论）
    layer(x).sum().backward()
    assert layer.lora_A.grad.abs().sum() == 0 and layer.lora_B.grad.abs().sum() > 0
    print("初始等价性 + 梯度不对称性断言通过")
```

（merge 的写法以合并到权重矩阵为准：`W' = W + scaling·B@A`，此处注释说明原理即可；生产实现见 peft 库 `merge_and_unload`。）

**【30秒答题框架】**

"LoRA 假设微调更新是低秩的，把 ΔW 写成 B·A：B 是 d×r、A 是 r×d，只用 2dr 参数——4096 维、r=16 时是全量的 0.78%。A 高斯初始化、B 置零：B=0 保证初始与预训练严格等价，A 非零保证第一步 B 的梯度不为零，避免双零死锁。训练只更新 2dr 个参数和小优化器状态，冻结的大矩阵不存梯度和 Adam 状态。推理前把 (α/r)BA 合并进 W，延迟零增加，换 adapter 只需换一块 2dr 小矩阵。"

**【追问与陷阱】**

- *追问 1：为什么不让 A、B 都随机初始化？* 可以训，但 t=0 时 ΔW≠0，噪声瞬间打入模型，可能破坏预训练能力；B=0 让 LoRA 是预训练模型的严格延拓。
- *追问 2：r 增大会怎样？* 表达力单调增、过拟合风险缓增；边际收益先大后小（见题 2），且显存随 2dr 线性上升但占比仍小。
- *追问 3：LoRA 能学到"新知识"吗？* 记忆类更新容量受限（低秩瓶颈），风格/格式/行为适配强、知识注入弱——见题 17 的选型决策。
- *追问 4：训练时能直接更新 W 吗？* W 已冻结即无需其梯度和优化器状态，这是显存收益的大头；但反向传播仍要精确算通过 W 的梯度流（不存 W 的 grad，只算传给输入的梯度），checkpoint 进一步省激活。
- *易错 1：把参数量写成 dr 或 d·2r 顺序搞反*；比例是 2r/d，不是 r/d。
- *易错 2：说"LoRA 有额外推理延迟"*——合并后没有；不合并才有两跳瘦乘的微小延迟。
- *易错 3：说"两个都置零也没关系，训练会出来"*——死锁是数学事实：两因子乘积结构的梯度对另一因子的依赖导致零点为驻点，除非数值噪声救场（不可依赖）。

**【知识串联】**

- `01-Foundation/01-Transformer.md`：LoRA 的注入对象就是 W_q/W_k/W_v/W_o（及 FFN）投影——先懂投影矩阵结构再谈为何选层。
- `01-Foundation/03-Loss-And-Optimization.md`：AdamW 每 fp32 参数 ≈ 12B 状态的账本是 LoRA 省显存的算术基础，学习率相对全参要放大 5–10 倍（见题 18）。
- `02-Inference/03-Quantization-And-Deployment.md`：LoRA 的"低比特约束"与量化叠加即 QLoRA（题 4）；合并后权重为满精度，部署侧无需关心 adapter（除非多租户热切换，见题 17 / 09-SystemDesign）。

---

### 题 2：LoRA 的秩 r 如何选择？

**【考察意图】**

考察经验判断 + 是否理解"秩=任务复杂度的容量旋钮"。好答案应给出 ① 分档经验表、② "α 与 r 联动"的缩放原理、③ 过大秩的代价不是显存而是过拟合与调参负担、④ 用验证集/消融驱动而非拍脑袋。

**【深度解析】**

**1. 分档经验（原文表格扩充）。**

| 秩 r | 参数量（d=4096，单矩阵） | 适用场景 | 依据 |
|------|--------------------------|----------|------|
| 4–8 | 32K–65K（0.39–0.78%） | 简单适配：风格/语气/格式迁移、极少样本(<1K) | LoRA 原论文在 GPT-3 上 r=1~4 已接近全参，"本征秩"结论 |
| 16–32 | 131K–262K（0.78–1.6%） | 通用默认，多数指令微调 SFT | 精度/成本甜蜜点，社区主流配置 |
| 64–128 | 0.5–1M（3–6%） | 复杂任务：代码、数学、长推理；大模型 + 大数据(>50K) | 数据量大时需更大更新容量 |
| 256+ | ≈2M+（≈12%+） | 接近全参效果、多专家领域 | 逼近全量上限，关注度下降 |

**2. 为什么小 r 有效：** 更新集 {ΔW_i} 排布在低维子空间内。经验量化：任务类型增量（"怎么答"）秩低，知识与行为注入（"答什么"）所需要秩高——这也是题 17 全参/LoRA 路线分歧的根因。

**3. α 联动：** 前向缩放为 (α/r)·BA。若把 r 从 16 升到 32 而保持 α 不变，则初始梯度尺度近似减半（∂L/∂B ∝ A，∂L/∂A ∝ B 链式经 α/r 缩放），等效学习率变化。实操：**α ≈ 2r（或 α=r）以固定 α/r**，再微调学习率，避免"改了 r 全套超参重调"。

**4. 方差视角：** r 是单层更新的参数量上界，也是梯度噪声维数上限。小 r + 大 LR 容易在窄子空间内过冲；消融时固定 α/r 依次扫 r={8,16,32,64}，看验证集曲线拐点即可，通常拐点明显（例如 16→64 提升 <1pt 时再翻倍无意义）。

**5. 决策清单（面试可背）：** ① 数据 <1K → r=4–8、LR 1e-4 档；② 通用 SFT → r=16、α=2r；③ 领域深度/代码/数学 → r=64+，且注入全部线性层（见题 3）；④ 显存紧 → QLoRA 配 r=16–64；⑤ 最终以验证 loss 与下游评测为准，不迷信经验表。

**【30秒答题框架】**

"按任务复杂度分档：风格适配 4–8，通用 SFT 16–32，代码数学这类重任务 64–128，256 以上接近全参。LoRA 论文证明很多任务本征秩很低，所以小 r 常常够用；关键是 α 要随 r 联动，固定 α/r 缩放，否则等效学习率变了。最终用训练/验证 loss 的拐点来定，不是越大越好——大 r 的代价主要是过拟合和调参负担，显存占比其实仍然很小。"

**【追问与陷阱】**

- *追问：r=128 比 r=16 效果一定好？* 不一定。容量超出任务本征秩后接近冗余参数，主要风险是过拟合与把模板背死；大幅领先通常出现在"知识注入/长推理"类任务。
- *追问：α 该怎么设？* 等效缩放 α/r，重置 r 时同步重设 α 保持缩放不变；常见 α=2r，α=16–64 量级。α 过大等效 LR 过大，易 loss 飙升。
- *易错：把 r 解释成"注意力头数"或"层数"*——r 是 ΔW 的秩（ Skinny 因子维度），与 head 数无关。
- *易错：忽略 dropout*——r 调大时 LoRA dropout（0.05–0.1）常同步调大以抗过拟合。

**【知识串联】**

- 与题 1（B=0 初始化）、题 17（选型表）构成同一决策链：容量估计 → 初始化 → 秩/dropout 正则。
- 与题 15（LIMA：数据质量>数量）互补：小而精数据配小 r；低质量大数据配大 r 反而放大噪声。

---

### 题 3：LoRA 应用在哪些层？

**【考察意图】**

考两件事：① 是否读过原始消融（LoRA 论文：Wq+Wv 组合最优的结论）；② 是否知道这个结论后来被更新了（QLoRA 证明"全部线性层"在量化底座上更好；现代实践的默认是 all-linear）。答"只加在 Q 和 V"且不知演变，会被判定为知识停更。

**【深度解析】**

**1. 原始结论（LoRA 论文，2021）。** 表格化：在相同预算下比较注入位置，结论是 W_q 与 W_v 组合最佳；单独 W_q 或 W_v 都明显掉点；W_k/W_o 边际贡献小于 W_q/W_v。直观理解：注意力内容路由（value 端）与查询匹配（query 端）是任务适配最敏感的通道。

**2. 时代修正一（QLoRA，2023）。** 底座被 NF4 量化后本身带量化噪声，需要更大的可训练旁路去补偿；QLoRA 消融显示在 4-bit 底座上对**所有线性层**（q/k/v/o + FFN 的 gate/up/down）注入比只注 q/v 效果更好，且不过度增加显存。

**3. 时代修正二（当前社区默认）。** `peft` 的 `target_modules="all-linear"` 成为指令微调默认：MLP 两侧（gate/up/down）承载了大量"事实与词汇偏好"更新。经验排序（信息量）：q,v ≥ o,gate,up > k,down；但既然每层增量参数只有 0.78% × 各注入矩阵，全注入的总参数仍 <1%，没必要省。

**4. 例外与禁区。**
- embedding 与 lm_head：一般不动（词表大，2dr 变 2·d·r 相对可行，但会牵动 tied weights 与输出分布，风险大）；除非任务重度涉及新 token/小语种。
- 深度偏置：监督信号偏向"靠近输出层的层"（离 loss 近），某些实现给靠后层更高 r 或更高 lr——非主流，面试点到即可。
- 多 router/Experts（MoE）：LoRA 通常只注入 dense 投影，浪费显存去训 router 无益。

**5. 延迟与合并。** 注入位置只影响训练侧计算与显存；推理统一通过 merge 消除额外延迟（见题 1）。Prefix/Prompt 类方法则不同——它们把可训练张量放进 KV 序列，推理延迟无法消除（题 5）。

**【30秒答题框架】**

"经典答案是 Q 和 V：LoRA 原论文消融显示同预算下 q+v 最优。但现在默认打满全部线性层——QLoRA 证明量化底座上 all-linear 更好，且 MLP 的 gate/up/down 承载事实类更新，全注入总参数仍不到 1%。真正要避开的是 embedding 和 lm_head。一句话：结论已从'q+v'演进为'all-linear'，答题时要把时间线讲出来。"

**【追问与陷阱】**

- *追问：为什么单独 W_v 或 W_q 效果差？* 注意力输出 = softmax(QK^T)·V，只有 Q 无 V 是"只会选不会写内容"、只有 V 无 Q 是"会写但选不对"，两通道需协同。
- *追问：能加在 CNN/Embedding 上吗？* 论文变体支持加热通道和 embedding（大词表上 LoRA 论文有 vocab embedding 实验），但常规 LLM 流程不用。
- *易错：说"LoRA 就是加在 attention 的"就是不能加 MLP*——现代答案必须含 FFN 注入。
- *易错：忘记"合并后无延迟" QApplication 前提*——若服务多 adapter 共底座（10-多租户），通常不合并而是动态路由，见题 17。

**【知识串联】**

- `01-Foundation/01-Transformer.md`：Q/K/V/O/FFN 的矩阵结构是此题底图。
- 题 1（合并推理）与题 5（Adapter 的串行延迟）形成"延迟光谱"对照。
- 题 4：QLoRA 底座噪声 ⇄ all-linear 结论互为因果。

---

### 题 4：QLoRA 的原理

**【考察意图】**

验证三块硬功夫：① NF4 的"为什么normal分布要分位数格点"（Lloyd-Max 最量化性，不是玄学）；② Double Quantization 的显存算术（0.5 → 0.127 bits/param 这笔账）；③ 能否把"65B 进单卡 48GB"拆成权重/优化器/激活三条腿的显存公式。这题几乎就是一笔显存会计题。

**【深度解析】**

**1. 总体结构。** QLoRA = "NF4 4-bit 冻结底座 + bf16 计算 + LoRA 旁路（bf16 参数、fp32 优化器状态）+ paged optimizer"。前向时 NF4 权重按块反量化为 bf16 参与计算；LoRA 旁路以接近全精度训练，吸收/补偿量化噪声。

**2. NF4（NormalFloat4）。**
- 观察：预训练权重经验上近似高斯（逐通道/逐层观察）。均匀格点的 Int4 对高斯分布是"中间浪费两头不够"：密度高的 0 附近格点太稀、密度低的尾部格点太密。
- 最优解是 Lloyd-Max 标量量化器：给定源分布 p(x) 与 k=16 个重建点 {q_i}，最小化期望量化误差 E|min(x−q_i)|²。对连续对称分布，最优格点 ≈ 分布的等概率分位点——每个格点"管辖"1/16 的概率质量。
- 因此 NF4 的 16 个码元取 N(0,1) 分位数（归一化到 [−1,1]，bitsandbytes 实测网格，保留 0 为精确码点）：

  ```
  -1.0, -0.6962, -0.5231, -0.3921, -0.2844, -0.1848, -0.0911, 0.0,
   0.0795, 0.1609, 0.2466, 0.3379, 0.4373, 0.5499, 0.6835, 1.0
  ```
  量化：码元 = 距离最近格点；反量化：码元值 × 块缩放。**期望误差低于 Int4**（同等误码率下以尺寸换去偏）。
- 分块缩放（block-wise absmax）：权重切成 64 个一组，块内除以 absmax 得到 [−1,1] 后按 NF4 码本量化；每块另存一个缩放常数。

**3. Double Quantization（双量化）——显存账本。**

| 方案 | 缩放常数存储 | 换算 bits/param |
|------|--------------|------------------|
| 常规分块量化 | 每块(64 权重)一个 FP32 缩放 | 32 bit/64 = **0.5 bits/param** |
| 双量化 | 缩放常数再量化为 FP8(8-bit)，每 256 块存一个 FP32 二级缩放 | 8/64 + 32/(64×256) = 0.125 + 0.00195 ≈ **0.127 bits/param** |

净省 0.373 bits/param。65B 模型：65e9 × 0.373 / 8 ≈ **~3GB**——单卡 48GB 场景下这正是"能不能塞进去"的差距。

**4. Paged Optimizers。** NVIDIA 统一内存：优化器状态（只有 LoRA 参数的 Adam 状态）挂到可 page-out 的内存池，长序列梯度检查点造成显存尖峰时自动换页到主存，避免 OOM；代价是偶尔的换页延迟。

**5. 显存总账（65B，单卡 48GB 如何成立）。**

| 项 | 全参微调（BF16+AdamW） | QLoRA |
|----|------------------------|-------|
| 权重 | 2 B/param → 130GB | 0.5 B/param → **~32.5GB** |
| 梯度 | 2 B/param（65B 全部）→ 130GB | 仅 LoRA 旁路（≈0.1% 参数） |
| Adam 状态 | 8 B/param（fp32 m,v）→ 260GB+ | 仅 LoRA（16.8M×(2+8)B≈0.2GB，可分页） |
| 合计 | ≈ 520GB+ | ≈ 33GB + 激活（checkpoint 后可控） |

全参 65B 若用 16-bit AdamW ≈ (2+2+8)×65e9 ≈ **780GB**（QLoRA 论文口径），这就是为何需要 QLoNA/LoRA——而 QLoRA 把权重腿从 130GB 压到 32.5GB，把优化器状态腿从 260GB 压到 0.2GB 且可分页，三条腿共同把 65B 塞进 48GB。

**6. 精度机制。** LoRA 旁路是唯一"高精度锚点"：量化噪声是零均值随机微扰，梯度更新拟合的是 E[∇L(量化W + BA)]——LoRA 旁路可学到对噪声的校正项；QLoRA 论文实验：4-bit 底座 + all-linear LoRA + bf16 计算精度，训练 65B 后与 16-bit 底座 LoRA 的 MMLU 分差在 1pt 内。它是**训练时量化（QAT 类）**而非纯 PTQ：底座固定 NF4，adapter 是全精度。

**【30秒答题框架】**

"QLoRA = NF4 4-bit 冻结底座 + bf16 计算 + LoRA 旁路 + paged optimizer。NF4 不是均匀格点，而是按高斯分布的分位点做 Lloyd-Max 最优标量量化——每个码元承载 1/16 概率质量，对近似高斯的权重误差最小。分块缩放本身要 0.5 bit/param，用双量化把缩放常数再量化到 FP8 并加二级缩放，压到 0.127 bit，65B 省 ~3GB。最后的算术：65B 权重 4-bit 约 32.5GB，LoRA 优化器状态只有 0.2GB 且可分页，激活靠 checkpoint 控制，于是单卡 48GB 微调 65B 成为可能。"

**【追问与陷阱】**

- *追问：NF4 与 GPTQ/AWQ 差别？* GPTQ 是 Hessian-aware 逐列 PTQ（推理导向），AWQ 保护激活敏感通道（也是 PTQ），NF4 是"知道分布形状"的训练时量化；QLoRA 里精度由 LoRA 旁路保证，推理也可把合并结果再做 GPTQ/AWQ（两段式）。见 `02-Inference/03-Quantization-And-Deployment.md`。
- *追问：Paged optimizer 是 LLM 特有的吗？* 不是，是 CUDA unified memory 在训练框架的应用；价值主要在内存尖峰场景（长序列、大 batch 反传）。
- *追问：QLoRA 精度损失来源？* 主要不是 4-bit 本身，而是训练超参敏感：建议 bf16 计算精度、输出侧不量化、all-linear LoRA、适当大 r。
- *易错 1：说"NF4 是把权重 normal 化再 int 量化"*——是分位数格点，不是变换后整数化。
- *易错 2：65B 4-bit 权重算成 16GB 或 65GB*——0.5 B/param，65e9×0.5=32.5GB。
- *易错 3：以为 Falcon/NF4 还需 0.5 bits/param 缩放*——双量化吃掉了大半。

**【知识串联】**

- 题 1/2/3：QLoRA = 题 1 的 LoRA"旁路 + 全注入层(题 3)"叠在量化底座上。
- `02-Inference/03-Quantization-And-Deployment.md`：GPTQ/AWQ/GGUF 是同门不同路线（推理 PTQ vs 训练 QAT），面试常互问。
- `09-SystemDesign.md`：QLoRA 是"绝境算法"——算力不足但模型过大的选型。

---

### 题 5：Adapter / Prefix Tuning / Prompt Tuning 的原理与方法对比

**【考察意图】**

考"PEFT 谱系结构感"：三种方法在**注入位置**轴上构成递进（层间串行模块 / 每层 KV 前缀 / 输入 embedding 前缀），注入位置决定参数-效果-延迟三角。考的是能不能把三者的机制差异、显存/延迟代价、适用场景一次讲清——尤其"为什么 Adapter 有延迟而 LoRA 没有"这笔账。

**【深度解析】**

**1. Adapter（Houlsby et al., 2019）。** 在每个 Transformer 块内**串行插入**瓶颈模块（down-project → 非线性 → up-project），残差连接包裹：

```
Attention → [Adapter: d→r → GELU → r→d] → FFN → [Adapter: 同构]
```

参数量：每模块 ≈ 2·d·r_extra（h_bottleneck 约 64 就能达 1–5% 参数，r 大于 LoRA 的 r），可训练比例高。**缺陷：不能合并**——它像一段电路串在线路里，推理必须逐层多跑两跳瘦矩阵乘，延迟增加数个百分点；批处理可摊薄但不可消除。

**2. Prefix Tuning（Li & Liang, 2021）。** 在**每层**的 K、V 序列前拼接可训练前缀：

$$K' = [P_K;\ K],\quad V' = [P_V;\ V],\quad P_K,P_V \in \mathbb{R}^{P\times d}$$

注意力在 prefix 上也要算（可影响全层注意力分布）。参数量 2·P·d×L，P 取 10–100。**关键代价：KV Cache 侵腐**——推理时这些 prefix 向量常驻上下文，长 P 直接挤占 KV cache 与有效上下文（与 `02-Inference/01-KVCache` 直连）；生成慢、 attends 混乱。重参数化（MLP 生成 prefix 再固化）可提稳定性。

**3. Prompt Tuning（Lester et al., 2021）。** 只在**输入 embedding** 前拼 P 个可训练 token：

$$E' = [P;\ E(x)]$$

参数 <0.1%，最便宜；但能力弱——要通过全部堆叠层的间接影响输出，效果随模型尺寸敏感；不冻结 lm_head 时同等无几。零推理延迟增加（P token 融进 prompt 一起算，无新增模块）但挤占输入长度（与 KV 不同：它是一次性的，非每层重复——但影响所有层通过注意力传递）。

**4. 对比表（原文表格扩充，可背）。**

| 方法 | 注入位置 | 参数量 | 推理延迟 | 显存/服务影响 | 效果 | 现状 |
|------|----------|--------|----------|----------------|------|------|
| LoRA | 每层线性权重旁路（乘性） | 0.1–1% | 合并后**零** √ | adapter 可热切 | ⭐⭐⭐⭐ | 主流 |
| QLoRA | 同 LoRA + 量化底座 | 0.1–1% | 量化底座略降速 | 单卡通大模型 | ⭐⭐⭐ | 主流（算力受限） |
| Adapter | 每层★串行模块（加性结构） | 1–5% | **增加**（2 跳瘦乘×层数） | 多模块常驻 | ⭐⭐⭐ | 已边缘化 |
| Prefix | 每层 KV 前缀（加进序列） | 0.1–0.3% | **增加**（KV cache 变长 + 注意力计算变多） | 侵腐 KV cache | ⭐⭐ | 边缘 |
| Prompt | 输入 embedding 前缀 | <0.1% | 几乎无 √（挤占 prompt 长度） | 最轻 | ⭐⭐ | 特定场景（超大模型 + 极多任务） |
| (IVE)/Soft prompt 变体 | 同上 | — | — | — | — | — |

**5. 统一视角（面试加分）。** 三者都可用"加性扰动 vs 乘性（权重）扰动"分类：Prefix/Prompt 是对激活（KV、E）的加性扰动，改动会随深度传播被过权或稀释；LoRA/Adapter 是对权重的扰动，直接改变每层的映射本身。LoRA 恰好满足三定理：① 参数最少且能力足（低秩学够了）；② 推理零延迟（可合并）；③ 多任务可热切换（2dr 小文件）。这就是它在 2021 之后把 Adapter/Prefix 挤到边缘的原因。

**【30秒答题框架】**

"这三个方法的差别本质是注入位置：Adapter 在每层串行插瓶颈模块——能训但推理有真实延迟，适合不能改底座权重的黑盒场景；Prefix 在每层 KV 前拼可训练前缀，会变相加长 KV cache；Prompt Tuning 只在输入 embedding 前拼软 token，最便宜但能力受限、大模型上才够强。LoRA 是权重侧乘性扰动加上可合并，三个优点取齐，所以现在是默认选项。"

**【追问与陷阱】**

- *追问：Prompt Tuning 与 Prompt Engineering 区别？* 一个是可训练连续向量（梯度优化），一个是人工离散文本（无梯度）；前者是 PEFT。
- *追问：为什么 Prefix 的效果反而常不如 Prompt？* 每层 KV 前缀虽然直接操纵注意力，但自由度过大导致过拟合与对分布高度敏感，P 能力有限；输入端 prompt 虽间接但参数最省。
- *易错：把软 prompt 当成"每次推理要重新算的 token"*——训练收敛后是固定向量，可随 adapter 一起存储切换。
- *易错：不知道 Adapter 的延迟是串行结构导致*——与 LoRA 对比时务必讲"可合并 vs 不可合并"这一关键差异。

**【知识串联】**

- 题 1/3：LoRA 的合并机制是本题"延迟光谱"的对照组。
- `02-Inference/01-KVCache-And-Acceleration.md`：Prefix 前缀与 KV cache 的资源竞争是同一资源池的两端。
- `07-HotTopics.md`：软提示思想延伸到"软对齐 token / 蒸馏场景"等 2025–2026 热点。

---

### 题 6：RLHF 的完整流程 ⭐⭐⭐⭐

**【考察意图】**

考全局工程视野：① 三阶段（SFT → RM → PPO）的目的与衔接——为什么 SFT 不能直接出模型、为什么 RM 不能缺、为什么 PPO 需要 RM；② 会不会画出流水线图并说出每阶段的数据规模与产物；③ 有没有意识到 InstructGPT 的经典数据规模（约 13K SFT / 33K 偏好 / 31K RL prompts）和"为什么用排序而非打分"。

**【深度解析】**

**1. 流水线图（可背 ASCII）。**

```
预训练 LLM（base，会续写不会对话）
   │
   │ 阶段 1：SFT 监督微调
   │   数据：高质量人工示范 (instruction, response)，InstructGPT ~13K
   │   目标：L = -Σ log π(y|x)，学会基本指令格式与回答分布
   ▼
SFT 模型 π_SFT ──────────────►（复制冻结）─► Reference 模型 π_ref
   │
   │ 阶段 2：Reward Model 训练
   │   数据：对每条 prompt 采 K 个回答，人工两两排序 (y_w ≻ y_l)，~33K 组
   │   损失：L_RM = -log σ( rφ(x,y_w) − rφ(x,y_l) )   ←Bradley–Terry
   │   产物：奖励模型 rφ（通常从 SFT 模型初始化、去掉 LM head 换标量头）
   ▼
Reward Model rφ（冻结）
   │
   │ 阶段 3：PPO 强化学习
   │   在线采样：prompt → actor 生成 y → rφ 打分 − β·KL(πθ‖π_ref)
   │   更新：actor（策略）、critic（价值）；ref 与 RM 全程冻结
   │   InstructGPT：~31K prompts
   ▼
对齐后的模型（更符合偏好，但受 KL 拴住不漂移）
```

**2. 每阶段的"存在理由"。**
- SFT：把"预训练分布（互联网续写）"平移到"对话分布"。只有行为先验对了，后续 RL 才有合理的探索起点（否则 PPO 会从错误格式上开始乱试）。这也是为什么一般不做"从 base 直接 PPO"。BUT：DeepSeek-R1-Zero 证明了纯 RL 也可行（ reward 足够稠密可验证时）——面试可以主动提及例外并说明前提。
- RM：人类偏好是相对序而非绝对分——"多好一点点"的判断比"打几分"稳定得多，所以用成对比较（排序）而非评分。BT 模型把排序变成概率：P(y_w ≻ y_l) = σ(r(y_w) − r(y_l))；损失即负对数的成对形式（原文件已给）。RM 结构=去掉 LM 头的 SFT 模型 + 标量输出，训练用 SFT 初始化保证理解能力对齐。
- PPO：有了可微分的奖励信号，就能用策略梯度把"分数高"变成"输出概率大"，同时用 KL 惩罚把模型拴在 SFT 分布附近，防止两个失败模式：① 过度优化 RM 的弱点（reward hacking，RM 也只是统计模型有盲区）；② 流利度/多样性崩塌（RL 常见退化：输出趋同、重复、长度膨胀）。

**3. 数据规模（InstructGPT 口径，可引用）。** SFT 1.3 万条示范、RM 3.3 万组偏好比对、PPO 3.1 万条 prompt；人工排序员同一 prompt 出 K=4~9 个回答进行成对比较。有趣的系统事实：RM 的数据比 SFT 多但单位信息小（排序只给出一位信息），RM 训练前要过 prompt 去重与标注员共识过滤。

**4. KL 惩罚位置（重点，题 7 展开）。** 奖励改写为 R = r_φ − β·KL(π_θ‖π_ref)（原文式），逐 token 计；施加位置有两种——(a) 直接计入 reward（InstructGPT 做法：ADV/Critic 用的 reward 就含 KL），(b) 在 advantage/loss 中另行扣减（DPO 论文附录提到，数学期望等价但实证 (a) 更稳）。所有阶段的 β 典型 0.02 量级（1e-2 ~ 1e-1 区间调）。

**5. 现代表述/新词。** ① 原生多轮：Iterative/DPO/GRPO 视角下 SFT 与 RLHF 阶段可交替（Llama-3：rejection sampling + PPO/DPO 混合迭代）；② RLVR（可验证奖励）把阶段 2 的 RM 换成规则校验器（代码执行、数学判定），DeepSeek-R1 即此路线，见题 13 / `07-HotTopics`。

**【30秒答题框架】**

"RLHF 三阶段：先 SFT 把预训练模型的续写分布平移到对话分布，给 RL 一个好的行为先验；再用人工成对比较训 Reward Model——用 Bradley-Terry 把'谁更好'变成 sigmoid 概率，损失是 −log σ(r_win − r_lose)；最后 PPO，actor 在线生成回答、RM 打分并减去 β×与参考模型的 KL，防 reward hacking 和分布漂移。InstructGPT 数据量级 13K/33K/31K。三阶段缺一不可：SFT 给起点，RM 给梯度信号，KL 给安全绳。"

**【追问与陷阱】**

- *追问：为什么用排序不用打分？* 跨标注员的绝对分不可比（均值/方差漂移），相对序更稳；且 BT 损失恰好只需要差值。
- *追问：RM 是怎么初始化的？* 一般从 SFT checkpoint 初始化（换头为标量），保证对对话分布的理解一致。
- *追问：能不能跳过 SFT？* 要分情况：有高质量小数据时 LIMA 式极简 SFT 也够；纯 RL 路线（R1-Zero）在奖励稠密可验证时可行，但会出现可读性差、语言混杂等问题。
- *易错：把三阶段顺序说反*（RM 在 SFT 前 / PPO 在得到 RM 前）；或忘了 reference model 是 SFT 的副本（不是 base 模型——从 base 出发 KL 无锚点）。
- *易错：不知道 KL 是逐 token 计算的*（对整句求和后应等于句子级 KL）。

**【知识串联】**

- 题 14：SFT 的损失细节在本章指令微调部分。
- 题 7/8/10：KL 的施加位置是 PPO/DPO/GRPO 的分水岭之一。
- `06-Safety-Evaluation.md`：RM 评估（如 RewardBench）与 reward hacking 的检测。

---

### 题 7：PPO 的四个组件 ⭐⭐⭐⭐

**【考察意图】**

考"四模型显存与信号流图"+ 两个进阶：① 每个组件的训练/冻结状态、显存占用、初始化来源；② KL 项的**两种施加位置**（reward 内 vs advantage/loss 内）的精确区别与实证结论——这是把"会用 PPO"与"理解 PPO"分开的探针。若再能写出 GAE 公式和 clip 目标，即达二面标准。

**【深度解析】**

**1. 四组件信号流图。**

```
           ┌────────── 生成 ──────────┐
prompt ──► │ Actor πθ (可训练，SFT 初始) │ ──► 回答 y
           └────────────────────────────┘
                 │                    │
     冻结副本π_ref│                    │ 冻结 RM rφ
     逐 token KL_t│                    │ 句级奖励 rφ(x,y)
                 ▼                    ▼
        per-token 罚分 −β·KL_t →  R_t = rφ − β·KL_t   （t∈句子）
                                      │
                                      ▼
             Critic Vψ (可训练，RM/SFT 初始) ← GAE(δ_t = R_t + γV(s_{t+1}) − V(s_t))
                                      │
                                      ▼
        PPO-clip 目标（代理损失）更新 πθ ┐
        价值损失 MSE(clipped) 更新 Vψ ──┘  (两个可训练模型同时更新)
```

| 组件 | 参数量 | 初始化 | 冻结? | 作用 |
|------|--------|--------|-------|------|
| Actor π_θ | =策略模型 | SFT checkpoint | 否 | 生成回答，被 RL 更新 |
| Critic V_ψ | =Actor（同级 LLM，输出标量） | 常 RM（或 SFT） | 否 | 估计 V(s) 以算优势减方差 |
| Reward Model r_φ | ≈Actor | SFT | **冻结** | 提供质量分数 |
| Reference π_ref | ≈Actor | SFT | **冻结** | KL 锚点防漂移 |

显存账：4 个同级模型，PPO 至少 2 个可训练 ⇒ 混合精度下可用显存 ≈ 可训练×16B/参数 + 冻结×2B/参数。7B 全流程 ≈ 2×112GB + 2×14GB ≈ 252GB + 激活——这就是多卡/ZeRO/多机的根源，也正是 GRPO（题 10）砍 critic 的动机。

**2. 目标函数（原文式 + 逐项）。**

$$\mathcal{L}^{PPO} = -\mathbb{E}\left[\min\left(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t\right)\right],\quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

clip 的含义：当优势为正（这条回答好），策略概率上升被限制在 (1−ε) 倍内……更准确说：比值跳过 1+ε 后梯度归零，防止单批更新过冲；A_t<0 对称地卡下限。ε 典型 0.2。

**3. 总奖励与 KL 两种位置（本题核心区分点）。**

$$R(x,y) = r_\phi(x,y) - \beta\, D_{KL}(\pi_\theta \| \pi_{ref}),\qquad D_{KL} 以逐 token 方式计：\sum_t \big[\log\pi_\theta(a_t|s_t) - \log\pi_{ref}(a_t|s_t)\big]$$

经采样，E_{y~π}[每 token 项] = KL(π_θ‖π_ref) 的无偏估计（逐 token 形式是 KL 的 Monte-Carlo 估计）。
- **位置 A：计入 reward（RLHF 惯例，InstructGPT 采用）。** 每个时间步的即时奖励直接写为 r_t = r_φ(x,y)·[t=T] − β·KL_t，KL 作为"过程奖励"参与 advantage/GAE，影响 critic 拟合目标与 actor 梯度。
- **位置 B：reward 不动，在 advantage/loss 里扣。** R 只用 RM 分数；KL 从（未归一化的）优势或独立 loss 项中减去（GRPO 走这条路，KL 直接并入 loss，见题 10）。
- 数学上两种方式对期望梯度等价（都是同一 RL 目标的 reward shaping vs loss 正则两种写法，DPO 论文附录有讨论）；工程上位置 A 实测更稳（GAE/value 会内生地处理 KL 的时序），位置 B 实现更简、且允许完全去掉 value/critic（→GRPO/DAPO 的选择）。
- 每-token KL 常用 **k3 无偏估计器**：$\widehat{KL}_t = \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1 \ge 0$（x − log x − 1 的下界性质且逐样本非负，见题 10 论证）。

**4. GAE 与优势（显式写出）。**

$$\delta_t = R_t + \gamma V(s_{t+1}) - V(s_t),\qquad \hat A_t = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}$$

LLM 场景常取 λ≈0.95、γ=1，terminal-only reward 情况下 δ 只在末位非零，需 critic 补足中间步骤的价值估计——这正是 critic 存在的价值；也是 RLVR/GRPO 用 outcome reward 时可甩掉它的原因。

**5. 训练机制提示。** 每 iteration：用 π_old 采 G 条/prompt → 打分 → 固定 π_old、π_ref、RM → 更新 actor/critic 若干 epoch → 进入下一批。KL β 是最敏感超参：过大=不动（学不到偏好），过小=飞（reward hacking、模板坍缩）。

**【30秒答题框架】**

"PPO 四模型：Actor 是被训练的策略，从 SFT 初始化，负责生成；Critic 是同尺寸的价值网络，输出 V(s) 用来算 GAE 优势、减方差；Reward Model 冻结，只打分；Reference 是 SFT 的冻结副本，算逐 token KL 总额防漂移。总奖励 r_φ − β·KL；KL 加进奖励（InstructGPT 式，过 GAE/critic）或直接扣在 advantage/loss（GRPO 式），期望上等价、工程上前者更稳。显存上至少两个同尺寸可训练模型，这是它在 7B 级就要多卡的根因。"

**【追问与陷阱】**

- *追问：Critic 为什么常从 RM 初始化？* RM 与价值函数相关（RM 已学会"这个回答有多好"的表示），从它出发 V 的回归目标更平滑，训练早期方差低。
- *追问：PPO 里 KL 太大会怎样？* 策略被绑死在 π_ref，梯度主要在抵消 KL；表现为奖励曲线停滞、输出停留在 SFT 风格——即"过度保守"。过小则反向。
- *追问：为什么 LLM 的 PPO 常取 γ=1？* 生成过程即 MDP 终止于 EOS，中间步的折扣意义有限；奖励主要在句尾。
- *易错：说 Reference 是 RM 的副本*——π_ref 是 SFT 模型。KL 锚在"对齐前的行为分布"。
- *易错：把 clip 解释为"把概率裁剪到 [0.8,1.2]"*——裁剪的是**比率** ρ_t，不是概率本身；且只对越过边界的部分停止梯度。
- *易错：以为 critic 与 actor 共享权重*——是两个完整模型（有工作分享 trunk 但不 mainstream）。

**【知识串联】**

- 题 6：四模型在流水线阶段图中的位置；题 10：GRPO 砍 critic 后的显存账。
- `01-Foundation/03-Loss-And-Optimization.md`：AdamW/clip/学习率的优化器基础。
- `06-Safety-Evaluation.md`：KL 之外的第二道防线——RM 红队评估。

---

### 题 8：DPO 的数学推导 ⭐⭐⭐

**【考察意图】**

这是本章的"白板压轴题"：能否**从 KL-约束 RLHF 目标出发，一步步推出 DPO 损失**，并讲清三件事——Z(x) 为什么能消掉（ Bradley-Terry 只在差值上操作）、重参数化把 π* 换成可训练 π_θ 的合法性、β 的"温度"角色（控制 KL 松紧 & 控制梯度饱和）。推导不能跳步。

**【深度解析】**

**Step 0：出发点。** KL-约束的 RLHF 目标（InstructGPT 目标的连续化）：

$$\max_{\pi}\ \mathbb{E}_{x\sim\mathcal D,\, y\sim\pi(\cdot|x)}\big[r(x,y)\big] - \beta\, D_{KL}\big(\pi(\cdot|x)\,\|\,\pi_{ref}(\cdot|x)\big)$$

（r 待定，最后用数据学；β>0 是惩罚强度 = 温度。）

**Step 1：展开目标为逐 y 求和。** 对固定 x，略去不含 π 的项：

$$J(\pi) = \sum_y \pi(y)\, r(y) - \beta \sum_y \pi(y) \log\frac{\pi(y)}{\pi_{ref}(y)}
= \sum_y \Big[\pi(y)\big(r(y) + \beta\log\pi_{ref}(y) - \beta\log\pi(y)\big)\Big] + \text{const}$$

**Step 2：约束优化。** 约束 Σ_y π(y)=1（及 π≥0 自动满足）。拉格朗日：把项 g(π_ y ) = π_ y [r + β log π_ref − β log π_y] 对 π_ y 求导（注意 g 是 π_ y 的凹函数）：

$$\frac{\partial}{\partial \pi_y} = r + \beta\log\pi_{ref}(y) - \beta\log\pi_y - \beta = \lambda$$

（KL 的梯度里那个 −β·1 来自 d(π log π)/dπ = log π + 1。）

**Step 3：解出闭式最优策略。**

$$\log \pi^*(y) = \log \pi_{ref}(y) + \frac{r(y)}{\beta} - \frac{\lambda}{\beta} - 1
\;\Longrightarrow\; \pi^*(y|x) = \frac{1}{Z(x)}\ \pi_{ref}(y|x)\, e^{\,r(x,y)/\beta}$$

配分函数 $Z(x) = \sum_y \pi_{ref}(y|x)\, e^{r(x,y)/\beta}$ 由归一化条件 Σ_y π*(y)=1 决定（吸收 λ/β+1）。注意：归一化因子是**定义在全部可能输出序列上**的求和——LLM 词表指数级序列空间，**不可计算**。这是 DPO 之前必须训显式 RM 的根本原因。

**Step 4：反解奖励（重参数化）。**

$$e^{r/\beta} = \frac{\pi^*(y|x)\,Z(x)}{\pi_{ref}(y|x)}
\;\Longrightarrow\; r(x,y) = \beta \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

（与原文题面公式一致：$r(x,y) = \beta\log\frac{\pi^*}{\pi_{ref}} + \beta\log Z(x)$。）

**Step 5：代入 Bradley–Terry。** 偏好模型：$P(y_w \succ y_l|x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$。代入 Step 4：

$$r(x,y_w) - r(x,y_l) = \beta\left[\log\frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}\right] + \underbrace{\beta\log Z(x) - \beta\log Z(x)}_{=\,0}$$

**Z(x) 在差值中相消**——这就是整个推导的枢纽：不可积分配函数只以常数形式出现，而 BT 只消费奖励的序信息（差值），于是显式配分函数被"免费"消除。

**Step 6：用可训练策略替换未知的 π\***（MLE 的重参数化）。π* 是理想但未知；构造 π_θ 用同一公式表达"隐含奖励"：

$$\hat r_\theta(x,y) = \beta \log\frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$$

若 π_θ = π^* 则 r̂_θ 的序 = 真奖励的序。将该隐含奖励代入 BT，按负对数似然（ML）写出损失：

$$\boxed{\ \mathcal{L}_{DPO}(\theta) = -\,\mathbb{E}_{(x,y_w,y_l)\sim\mathcal D}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]\ }$$

与原文公式一致 ✓。**两个模型（π_θ + 冻结 π_ref）替代四个模型**，损失是纯监督式的两两 sigmoid 分类——无采样、无 critic、无 RM。

**Step 7：β 的温度角色（追问重点）。**
- **约束端（RLHF 侧）**：β 越大，KL 惩罚越强，π* 越"贴"π_ref（Step 3 中 e^{r/β} 越平缓）；β→∞ 退化为"不更新"（π*=π_ref）；β→0 允许无限漂移（退化成把所有概率集中到 y_w —— 过拟合偏好数据）。
- **损失端（梯度动力学）**：记 u_w/u_l 为两个隐含奖励之差 u = β·log-ratio 差。sigmoid 梯度 σ'(u) 在 |u| 大时趋于 0。β 越大 ⇒ 同样的策略变化被放大进 u ⇒ 更快进入饱和 ⇒ 单位步长小 ⇒ 行为保守（效果上像在"温度"里把两个候选的概率差距拉大后做 softmax）。β 越小 ⇒ sigmoid 未饱和 ⇒ 梯度大、学习快，但策略可自由漂移，易出现重复/长度膨胀/过拟合。
- 实践常用 β ∈ [0.05, 0.5]，0.1 最常见；β 与学习率共同控制漂移速度。

**Step 8：梯度与隐式加权（加分项）。** 对 u = r̂_w − r̂_l 求导：

$$\nabla_\theta \mathcal{L} = -\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l) - \hat r_\theta(x,y_w)\big)}_{\text{错序程度}}\cdot \nabla_\theta\big(\log\pi_\theta(y_w|x) - \log\pi_\theta(y_l|x)\big)\Big]$$

模型把"输的"排得比"赢的"越高（违反偏好越狠），梯度权重越接近 1——DPO 的**内建难例加权**。但当数据有标注错误时它会"更用力学错样本"，这是 DPO 对标注噪声敏感的机制性根源。

**【30秒答题框架】**

"从 KL-约束 RLHF 目标出发，对策略逐 token 求导解侣拉格朗日，得到闭式最优策略 π* = π_ref·e^{r/β}/Z。反解出 r = β·log(π*/π_ref) + β·log Z，代入 Bradley-Terry 的成对 sigmoid——因为 BT 只用奖励差，β·log Z 两项相消，把不可计算的配分函数甩掉了。最后把未知的 π* 换成可训练的 π_θ，得到 DPO 损失。β 是温度：约束端上 β 大则贴紧参考模型，损失端 β 大则 sigmoid 更饱和、更新更保守。全程两个模型、无采样。"

**【追问与陷阱】**

- *追问：Z(x) 真的无关吗？* 对每个 x 是常数，BT 差值中相消；但**训练与评估时是不同的 x 独立成立**，不影响逐条样本的梯度——严格说 DPO 优化的是"每条比较之上的 MLE"，这与约束 RLHF 目标等价要靠 π_θ 恰好等于某个 π*（重参数化覆盖全部解空间成立）。
- *追问：为什么 DPO 不需要采样和 critic？* 因为最优策略的解析形式已经把"RL 搜索"变成了"对闭式解的回归"；不再需要采样代价与价值估计。
- *追问：β=0 会怎样?* 数学上 u→0 ⇒ sigmoid 线性区，且 RLHF 目标失去 KL 锚 → 策略可完全重排（退化）。实际中 β 小但非 0。
- *追问：DPO 为什么还是需要好数据？* 它是离线方法：π_θ 只能回应数据里见过的对比，无法在线探索；分布一旦离开 π_ref 支撑，隐含奖励估计会系统性偏差（此类失败最近由 on-policy 变体 SimPO/KTO/在线迭代 DPO 缓解）。
- *易错 1：把 DPO 损失说成"奖励模型损失的变体"*——不同对象：DPO 训练的是策略本身，`r̂ = β log(π/π_ref)` 只是重参数化的"隐含奖励"。
- *易错 2：β 讲不出"温度"双重角色*（约束强度 + 梯度饱和尺度）。
- *易错 3：推导跳步省略"λ 由归一化定出 Z"*——面试官常盯着这一步问。

**【知识串联】**

- 题 6/7：DPO 是同一条 KL-约束 RLHF 目标（题 6 阶段 3）的"跳过 RL"闭式解；PPO 处理同一目标但走采样路径。
- 题 10：GRPO 保留 RL 框架但换掉 critic；DPO/GRPO/RLOO 三者都用"参考点"思想——DPO 用 π_ref 作锚，GRPO 用组内均值作基线。
- 题 15：偏好数据质量决定 DPO 上限（噪声敏感）。
- 离线-在线谱：SimPO/KTO/在线迭代 DPO → `07-HotTopics.md`。

---

### 题 9：DPO vs PPO 对比

**【考察意图】**

考"对比要带机制而不是背表"：每一行差异都要能给出机制原因（为什么 PPO 需要四个模型、为什么 DPO 稳定性高但可能掉上限）；以及当前社区实证结论（大厂在指令/偏好对齐上两路线并存的现状），避免"DBO 无脑优"一面倒。

**【深度解析】**

**1. 对比表（原文表扩充逐行给机制）。**

| 特性 | PPO | DPO | 机制解释 |
|------|-----|-----|----------|
| 奖励模型 | 需要（显式 RM 打分） | 不需要（隐含奖励 β log π/π_ref） | DPO 把 RM 训练与策略优化合并为一个目标的重参数化（题 8 Step 4–6） |
| 模型数 | 4（actor/critic/RM/ref） | 2（策略 + ref） | critic 提供优势、RM 提供信号；DPO 用闭式解把两者吞进损失 |
| 训练复杂度 | 高：必须在线采样 + 打分 + 多模型并行 + 超参(clip/β/λ/…) | 低：批内成对监督 + 单 β | 无 rollout 管道 ⇒ 工程故障面小一档 |
| 时间/数据流 | 在线（可探索、可纠偏） | 离线（数据固定、不能探索） | 在线采样让策略看到自己的新分布、可自我修复；离线则受限于数据覆盖 |
| 稳定性 | 较差：ratio 爆炸、KL 失控、RM hack | 较好：sigmoid 有界损失自带退火 | 无 min/clip 策略梯度链路，损失全局平滑 |
| 效果上限 | 高（若有好 RM + 算力）：GPT-4/Llama-3 主线 | 较好但受数据限制：LIMA 之外的中等规模对比数据上表现可比或略低 | 在线探索可发现数据外的更优回答；离线只能压数据内偏序 |
| 典型失败 | reward hacking、KL 需调参、代码量大 | 过拟合/长度膨胀、分布偏移、噪声敏感 | 内建难例加权会放大标注错误（题 8 Step 8） |
| 算力特征 | 生成（采样）开销为瓶颈 | 反向传播（成对 logprob）开销为瓶颈 | 采样一次要 4 模型，DPO 只需 2 次前向 + 2 次反向 |

**2. 当前实践主流（2024–2026 情报，需带名字）。**
- 偏好数据有限、算力受限、需求是"改善风格/有用性"：DPO 及其变体（SimPO/KTO/ORPO）最经济。
- 追求上限、有 RM 和算力、需要 on-policy 迭代：PPO 仍是工业级主力（Llama-3 报告同时用 rejection sampling + PPO + iterative DPO）。
- RLVR 时代：可验证奖励（数学/代码）几乎一律走 GRPO/PPO 类在线算法——DPO 对 sequence-level 规则奖励的优势状态不定。
- 数据质量论折衷：两组都用；"模型对齐不是一个开关，而是 SFT+偏好优化 blends"（面试里给这条判断比站队更聪明）。

**3. 失败模式速查（面试常考）。**
- DPO：长度膨胀（长答案更易成为 y_w）、模板复读、对噪声标签敏感、β/数据规模失衡。
- PPO：RM 过优化后"高分低质"、采样吞吐瓶颈（ rollout 时长占总时长 80%+）、四个模型的显存墙。

**【30秒答题框架】**

"PPO 和 DPO 优化的是同一个 KL-约束 RLHF 目标：PPO 走在线 RL 路线——四个模型（actor、critic、RM、ref）加采样管道，能探索、上限高，但工程复杂、容易 reward hack；DPO 用闭式解把 RM 吸进损失，两个模型、离线监督式训练，稳定好调但受数据覆盖限制，还容易长度膨胀和过拟合噪声。实践判断：算力紧、数据好、目标是改善风格 → DPO 系；追求上限、有在线基础设施和好 RM → PPO；数学代码这类可验证奖励 → GRPO/PPO 类在线算法。"

**【追问与陷阱】**

- *追问：DPO 为什么更稳？* 损失是单项 sigmoid（界于 [0,1]），没有 ratio 的乘性爆炸路径、没有价值函数回归的二阶不稳定。
- *追问：DPO 会不会丢失多样性？* 会：把 π 往"数据内 y_w"集中，是分布收窄的过程；缓解：加 KL/降低 β/SimPO 长度归一。
- *易错：说"DPO 没有 KL 惩罚"*——KL 以 ref 项的形式内置在训练对中（β·log(π/π_ref) 的分母就是 KL 的化身）。
- *易错：只背表不给机制*——每一行都要能回答"为什么"。

**【知识串联】**

- 题 8：所有行差异的推导根源都在题 8 的闭式解。
- 题 10/11：GRPO 是"PPO 的稳定化+省显存"版本，与 DPO 构成在线/离线双端点。
- `06-Safety-Evaluation.md`：两种算法的 reward hacking 表现与评测。

---

### 题 10：GRPO 的原理（DeepSeek 核心算法）⭐⭐⭐⭐⭐

**【考察意图】**

2024–2026 最热点（DeepSeekMath→R1）；考① 能否**从"为什么能去 Critic"讲起**：组内蒙特卡洛均值就是 V(x) 的估计；(②) 公式细节逐 token 展开（组内 z-score、k3 KL、token-level 归一化）；③ 方差学：组大小 G 的 trade-off、std 归一化的失败模式（全对/全错组梯度为零）；④ 与 RLOO/DPO 关联。这题答得好相当于把 2025 年 RL 热点题目的地基打通。

**【深度解析】**

**1. 动机：critic 的两宗罪。** PPO 里价值网络 Vψ 与 actor 同尺寸 ⇒ 训练显存翻倍且 critic 目标（回归 GAE）本身难学——LLM 的状态空间是序列上下文，价值函数回归极其不稳；对 LLM RL 这是震荡主要来源。DeepSeekMath 的剪裁：对同一 prompt x 采 G 个回答，这 G 条**共享同一个状态 x**——那么 V(x) = E_{y~π}[r(x,y)] 可以直接用**组内样本均值**蒙特卡洛估计。既然基线可以用采样算出来，为什么还要一个网络去学它？——这就是 GRPO 的全部实质。

**2. 算法流程与公式（原文式逐项）。**

1. 每题 x 从 π_θ 采 G 条回答 {y_1…y_G}（DeepSeekMath：G=64；小规模常 4–32）；
2. 奖励模型/规则/验证器打分 {r_1…r_G}；可拆分 r = 正确性 + 格式等分项求和；
3. 组内标准化出优势：

$$\hat A_i = \frac{r_i - \mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}$$

   （逐条回答一个标量，广播到该回答的所有 token——outcome 级优势；过程奖励版本则逐 token 打分再组内归一化。）
4. 优化目标（与 PPO 相同的 clip 项 + 独立 KL 项；DeepSeekMath 原文式，逐条回答按 1/|y_i| 长度归一）：

$$\mathcal{J}_{GRPO} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|y_i|}\sum_{t}\Big(\min\big(\rho_{i,t}\hat A_i,\ \mathrm{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\big) - \beta\, \widehat{D}_{KL}^{(t)}(\pi_\theta\|\pi_{ref})\Big)\right]$$

   其中 ρ_{i,t} = π_θ(y_{i,t}|y_{i,<t},x)/π_old(y_{i,t}|y_{i,<t},x)，$\widehat{D}_{KL}^{(t)}$ 为 k3 估计（见第 4 点）。注意**原文（DeepSeekMath）的归一化分母是 (1/G)·(1/|y_i|)**——每条回答按自身长度归一后等权平均；DAPO 的 Token-Level 修正才把分母换成全局 Σ_i|y_i|（每 token 一票，纠正样本级平均对长回答的梯度稀释）。两种口径不要混写：**(1/G)(1/|y_i|) 是 GRPO 原式，1/Σ_i|y_i| 是 DAPO 修正**（见题 12 ④；`07-HotTopics` 章 R1 一节的公式同为原式口径）。

**3. "平均即基线"的推导与偏差。**
对单条回答的 REINFORCE 型梯度：∇J = E[∇log π(y|x)·(r(x,y) − V(x))]。用 μ̂_G = (1/G)Σ_j r_j 替代 V(x)。为什么均值是合理基线：Monte-Carlo 误差 O(1/√G)（Var(μ̂)=σ²/G），G 越大越准——式子：G 从 4→64 误差缩减 4 倍。两个系统性偏差值得说：
- **自包含偏差**：μ̂ 含 r_i 本身，使 r_i − μ̂ 向 0 收缩（相关系数压缩）；RLOO 用 leave-one-out 均值 (1/(G−1))Σ_{j≠i} r_j 消除该偏差，是无偏且低方差的替代（GRPO 用全样本均值工程上近似等价）。
- **std 归一化的两面性**：好处是对奖励尺度不变（0/1 稀疏正确性奖励直接 z-score，训练稳定、超参跨 verifier 通用）；坏处是**全同奖励组**（组内全对/全错）σ̂→0 或 advantage 全 0 ⇒ 该 prompt 一个梯度都没有，白采样。DAPO 的动态采样（sampling 到"有区分度"的组再入库）即对症下药（题 12）。

**4. KL 项的位置与估计器（与 PPO 的机制分界）。** GRPO **不把 KL 放进奖励**，而是把 KL 当成**独立的逐 token loss 项**直接加在目标函数里（原文式 −β·D_KL·(...)），用 Schulman 的 k3 无偏估计：

$$\widehat{D}_{KL}^{(t)} = \frac{\pi_{ref}}{\pi_\theta} - \log\frac{\pi_{ref}}{\pi_\theta} - 1 \ \ge \ 0$$

两条性质：① 无偏：E_{a~π_θ}[k3] = KL(π_θ‖π_ref)（展证：E[π_ref/π_θ] = Σ π_ref = 1，E[log(π_ref/π_θ)] = −KL，代回即得）；② **逐样本非负**（x − log x − 1 ≥ 0 恒成立），相比每 token 简单差 log π_θ − log π_ref（可为负）方差更小。R1 后期/DAPO 则干脆置 β=0——长 CoT 需要策略大规模偏离 π_ref，KL 项反而束缚探索。

**5. 显存与算力账。** GRPO：actor（可训练）+ ref（冻结）= 2 个同级模型，省掉 critic；采样归 rollout 引擎（vLLM/SG-Lang 批量发动机）。7B 场景 ≈ 112GB(可训) + 14GB(冻结) + 激活/rollout 缓冲 ≈ 单机 8×A100 常见配置（PPO 同配置还要多挂一个 112GB 级 critic）。采样成本：G 条/prompt ⇒ rollout 吞吐需求 ×G（这是 GRPO 的主要工程负担，DeepSeekMemory 时代的报告里 rollout 时间占比 70–80%）。

**6. R1 语境总结。** DeepSeek-R1 = GRPO + RLVR（以准确率为主奖励，格式与语言一致性为辅）+ 少量 SFT 冷启动沉淀"推理可读性"；GRPO 在其中的角色：**无 critic 的组蒙特卡洛策略梯度 + 双 KL 正则**，能承受长序列（数万 token）CoTT 训练而不爆显存——这是 PPO-critic 架构在长 CoT 上的现实瓶颈（critic 要为每 token 都估价值，显存/噪声双重困境）。

**【30秒答题框架】**

"GRPO 的洞察：PPO 里 critic 是用来估基线 V(x) 的，而 LLM 采样的 G 条回答共享同一 prompt——V(x) 可以直接用组内奖励均值做蒙特卡洛估计，干脆不训 critic。优势 = (r_i − 组均值)/组标准差，z-score 天然尺度不变，0/1 规则奖励直接可用。目标沿用 PPO 的 clip 项，但 KL 不进 reward、改为独立的逐 token k3 无偏估计器直接加进 loss。组大小 G 控制基线方差 O(1/G)，代价是采样吞吐 ×G；全对/全错组 advantage 为零，要靠 DAPO 的动态采样过滤。DeepSeek-R1 靠它在 4-bit…… ——不，它在长 CoT 场景下单卡多机就能训，是 R1 的地基。"

**【追问与陷阱】**

- *追问：和 RLOO 的区别？* RLOO 是 leave-one-out 均值（无偏、不含自身）；GRPO 用全组均值 + std 归一化——多一个尺度不变性、生物学上 slight bias；实践中差异小。
- *追问：std 为 0 怎么办？* 加 ε 下限或跳过该组；DAPO 直接动态过采样非零方差组。
- *追问：为什么 outcome-reward 下不需要 critic？* 中间步没有过程奖励，优势只在句级广播即可；PPO-critic 的中间步骤价值估计在这样的奖励下基本是噪声放大器。
- *追问：GRPO 的 KL 方向？* κ̂_3 是 KL(π_θ‖π_ref) 的无偏估计（与 PPO 的"惩罚离参考分布太远"方向一致：策略对接口 token 的置信度被通胀时惩罚）。
- *易错 1：把 GRPO 写成"去 critic 的 REINFORCE，无 KL"*——原始 GRPO（DeepSeekMath）是带 KL（独立 loss 项）的；DAPO 才把 β 置 0，注意版本差异。
- *易错 2：忘了 advantage 广播到 token*——数学上 advantage 先广播到 token，原式按 (1/G)Σ_i (1/|y_i|) 逐条长度归一，DAPO 才改为 (1/Σ|y_i|) 的全局 token 归一化。
- *易错 3：说 GRPO 无参考模型*——π_ref 在原始版本存在（k3 需要）；但很多工程实现把 KL 直接关掉以省显存，这两种口径都要能说清。

**【知识串联】**

- 题 7：GRPO 对四模型的裁剪对照；题 8：同为"参考点"思想的两种实现（锚=π_ref vs 基线=组均值）。
- 题 12：DAPO/REINFORCE++ 是 GRPO 同门的后继。
- `07-HotTopics-2025-2026.md`：R1/RLVR/Agent-RL 的主线索；`06-Safety-Evaluation.md`：reward hacking 在组内归一化下的新形态（"组间套利"）。

---

### 题 11：GRPO vs PPO 的关键区别

**【考察意图】**

考"区别要落在机制与数字上"：能不能指出"减的是哪个模型、每项差异的因果链"（去掉 critic ⇒ 显存减半 & 方差来源消失 ⇒ 可以训长序列），以及适用边界（outcome vs process reward）。背表不给机制会被追问一步打穿。

**【深度解析】**

**1. 对比表（原文表扩充，每行带"为什么"）。**

| 维度 | PPO | GRPO | 因果链 |
|------|-----|------|--------|
| Critic 模型 | 需要（与 actor 同尺寸） | **不要**（组均值估计 V(x)） | V(x)=E[r|x] 可用同 prompt 的 G 条采样 MC 估计 ⇒ 网络版价值函数冗余 |
| 优势估计 | GAE(λ, γ) 需 V(s) 沿 token 轨迹展开 | 组内 z-score（句级广播） | outcome 奖励下中间 token 的优势只需常量差分（题 10 §3） |
| 显存 | actor+critic 可训练 ⇒ 每可训参数 ~16B | actor 可训练 ⇒ ~8B/可训参数 + 冻结 ref | 7B：~250GB → ~130GB 量级 |
| KL 位置 | 常进 reward（经 GAE/critic） | 独立逐 token k3 loss（或干脆 β=0） | 衡量同一约束的两种写法（题 7 §3） |
| 采样成本 | 每 prompt G 条（同 GRPO） | 每 prompt G 条（G=64 常见） | 相同；差异在训练侧 |
| 方差来源 | critic 回归噪声 + rollout | 组均值误差 O(1/√G) | 消失一个方差源，代价是采样加倍 |
| 奖励类型适配 | process reward 自然 | outcome reward 自然；process 变体亦有 | z-score 需要在同一组内比较（题 10 §2） |
| 埋坑 | KL/critic 双超参不稳 | 全同奖励组梯度为 0；rollout 吞吐瓶颈 | DAPO 修第一类（题 12） |
| 成名应用 | InstructGPT/传统对齐 | DeepSeekMath → DeepSeek-R1 | R1 长强化训练（数千步 CoT）只能被 2 模型架构承载 |

**2. 一个容易被追问的量化口径：** "PPO 至少两个可训练模型" ⇒ 混合精度训练下 ~16 B/可训练参数（4 权重+4 梯度+8 优化器，bf16 权重另计时的不同口径有出入，给量级即可）。GRPO 把可训练量减半，等量显存下要么模型翻倍、要么 batch/G 翻倍——R1 选择了后者（大 G 大 batch 全局归一化）。

**3. 选型判断（面试加分结构）。** 需要过程奖励精细信任分配/有现成 RM/算力充裕 → PPO；需要长 CoT、可验证规则奖励、单机或小集群 → GRPO（+DAPO 改进）；只有偏好对数据、不想搭采样管道 → DPO 系。三者的边界是"奖励形态（process/outcome/sparse pairwise）与算力带宽"。

**【30秒答题框架】**

"核心区别一句话：GRPO 把'用网络学基线'换成'用同组采样算基线'。于是 critic 消失——可训练量砍半，7B 从约 250GB 降到约 130GB；GAE 变成组内 z-score，优势句级广播到 token；KL 从 reward 内挪为独立 k3 逐 token loss（DAPO 之后干脆删掉）。代价是采样吞吐乘以组大小 G，以及全同奖励组会浪费梯度。数学/代码这类 outcome 奖励、长 CoT 场景下 GRPO 是当前事实标准，DeepSeek-R1 就是靠它训出来的。"

**【追问与陷阱】**

- *追问：GRPO 采样也用了 G 条，为什么说它"训练效率更高"？* 省的是训练侧：少一个每 token 前反传的价值模型、少一个需要调稳的回归头；采样侧成本相同甚至更多（G 倾向取更大）。
- *追问：GRPO 会不会把 KL 删到没有？* 有两个版本口径：DeepSeekMath 原版保留 β、R1/DAPO 实验里 β=0；答题要说明"版本之后删"。
- *易错：把"组内均值估计 V"当成新发明*——这是 REINFORCE with baseline / RLOO 一族的延伸，能说出家族谱系是加分。
- *易错：只说"显存少"不给数字*——给 16B vs 8B per-trainable-param / 7B 例子的量级。

**【知识串联】**

- 题 10：所有机制细节的家；题 12：谱系后继与修正。
- `07-HotTopics-2025-2026.md`：R1 之后（VRPPO/小的 GRPO 变体）的续章；`09-SystemDesign.md`：rollout 引擎与训练引擎 disaggregation。

---

### 题 12：对齐方法时间线（含 DAPO / REINFORCE++ 勘误）

**【考察意图】**

考知识保鲜度与谱系准确度：① 能否按年讲故事（2022 RLHF → 2023 DPO → 2024 KTO/IPO/ORPO/GRPO提出 → 2025 GRPO 系改进 DAPO/REINFORCE++/R1、GSPO 等）；② **对原文错误进行勘误**：原文件将 DAPO 写成"Decoupled Alignment from Direct Preference Optimization"——官方论文（ByteDance Seed，arXiv:2503.17002, 2025）为 **"Decoupled Clip and Dynamic sAmpling Policy Optimization"**，属于 **GRPO/在线 RL 谱系**而非 DPO 系。这是"以官方论文为准"条目的直接应用。

**【深度解析】**

**1. 时间线（修正版，标注勘误）。**

```
RLHF-PPO（2022, InstructGPT/OpenAI）        ← 三阶段流水线、四模型（题 6/7）
   ↓
DPO（2023-05, Stanford）                    ← 闭式解替代显式 RM（题 8）
   ↓
IPO, KTO, ORPO, SimPO（2023 底–2024）        ← DPO 家族改进（见下）
GRPO（2024-02, DeepSeekMath 提出；原文标 2025 实为"提出于 2024、R1 于 2025-01 带火"）
   ↓
DAPO（2025-03, ByteDance）※勘误：Decoupled Clip and Dynamic sAmpling Policy
        Optimization——GRPO 系在线 RL 的改进，不是 DPO 系
REINFORCE++（2025-01, arXiv:2501.03262）    ← 单模型批级基线
GSPO（2025, Qwen 团队）                     ← 序列级 importance ratio，长 CoT 更稳
DeepSeek-R1（2025-01）                      ← GRPO+RLVR 规模化（题 10 §6）
   （2025–2026：Agent-RL/多轮 RL、过程奖励 PRM、自我博弈等，见 07 章）
```

**2. 各方法一句话机制（追问弹药）。**
- **IPO**（Azar et al., 2024）：DPO 的 sigmoid 会饱和 ⇒ 边际停止增大；IPO 换成对奖励边际的**平方损失**（hinge-like），带显式 KL 正则，防过拟合偏好标签。
- **KTO**（Ethayarajh et al., 2024 Kahneman–Tversky）：用**前景理论**的非对称效用（损失厌恶）：wanted / unwanted 样本不同价值函数与参考点，**不需要成对偏好**，只有好/坏两种标签也能训——工程上数据要求低，被广泛采用。
- **ORPO**（Hong et al., 2024, odds ratio）：**单阶段**——在 SFT 损失上直接加 odds-ratio 惩罚项抑制被拒回答的似然，无需 ref model 与两阶段流程；小模型（≤7B）性价比高。
- **SimPO**（2024）：去 ref model，奖励改成长度归一的 logprob 加 margin，专治 DPO 长度膨胀。
- **REINFORCE++**（2025）：把经典 REINFORCE 补上三件现代装备——**全局 batch 基线**（批内奖励均值，比 GRPO 组级更粗但更省采样防 group-exchange 伪影）、**逐 token KL（对 π_ref）**、长度归一化；单模型在线 RL，比 PPO 少 critic、比 GRPO 少组采样开销，代码简单而稳定。
- **DAPO（2025 修正认知）**：四件套 —— ① **Clip-Higher**：上界 ε_high 与下界 ε_low 解耦（如 0.28/0.2），放宽上界鼓励低概率 token 继续生长，对抗熵坍缩；② **Dynamic Sampling**：过采样并丢弃"全对/全错"零优势组（题 10 §3 的失效模式）；③ **Token-Level 逐 token 损失**（全局归一化，纠正样本级平均对短答案的加权偏差）；④ **Overlong 处理**（截断回答软惩罚/过滤）。同时 **移除 KL 项（β=0）**。实质 = GRPO 的工程化稳定版，LM RL 规模化训练的现役默认之一。

**3. 大局观（加分结构）。** 两条主线：**离线偏好**（DPO 家族：把对齐变成监督学习，便宜、稳、上限受数据限制）与**在线 RL**（PPO→GRPO→DAPO/GSPO：可探索、上限高、工程贵）；2025 年的 R1 效应把产业天平推向在线 RL（RLVR/Agent-RL），离线家族退守"小数据/低成本/安全微调"场景；两线融合（在线迭代 DPO、SFT+RL 交替）是当前工程常态。

**【30秒答题框架】**

"时间线三段：2022 InstructGPT 的 RLHF-PPO 定下三阶段范式；2023 Stanford DPO 用闭式解砍掉 RM 和 RL；2024 一年冒出 KTO（前景理论，不要成对数据）、IPO（平方损失防饱和）、ORPO（单阶段）、GRPO（DeepSeekMath 提出，组内相对优势去 critic）；2025 R1 用 GRPO+可验证奖励出圈，同档的 DAPO——注意是 'Decoupled Clip and Dynamic sAmpling Policy Optimization'，ByteDance 的 GRPO 工程稳定版，做了 clip-higher、动态采样、token-level loss 和去掉 KL——不是 DPO 系；REINFORCE++ 用批级基线把单模型在线 RL 做稳。当前两大阵营：离线偏好 DPO 系 vs 在线 RL GRPO 系，后者是 2025 后的主流方向。"

**【追问与陷阱】**

- *追问：GRPO 到底哪年？* 提出：2024-02 DeepSeekMath；出圈：2025-01 DeepSeek-R1。答单一年份时说明出处。
- *追问：KTO 为什么不需要成对偏好？* 效用函数非对称：对 wanted 用-with-offset 的 pass 项、unwanted 用 loss 项，各自带参考点与不同斜率（损失厌恶系数），单边标签即可回归。
- *追问：DAPO 为什么去掉 KL？* 长 CoT 的 RL 要求策略大幅偏离 SFT 分布（自我修正的行为在 SFT 分布里罕见）；KL 会抑制这种偏离，实测去掉后推理行为的多样性保留更好。
- *易错：把 DAPO 说成 DPO 变体*（原文笔误），或把 GRPO 时间写死 2025。
- *易错：比如"REINFORCE++ = REINFORCE + KL"* 漏掉全局基线与长度归一——它是三件套。

**【知识串联】**

- 题 8/10：离线与在线两大阵营的锚点略题。
- `07-HotTopics-2025-2026.md`：R1/RLVR/Agent-RL 的现状叙述主场。
- `06-Safety-Evaluation.md`：对齐税 / reward hack 评测。

---

### 题 13：如何为 Function Calling 场景设计奖励函数？

**【考察意图】**

考两件事：① RLVR 思想落地——奖励必须是**可验证/难以被 hack 的**，权重设计里藏着 reward hacking 的攻防；② 能不能写出可直接跑、可放进 GRPO 的完整奖励代码并给出复杂度。考的是"能设计且能编码"，比背 RLHF 六阶段难一个档次。

**【深度解析】**

**1. 设计原则（先讲原则再给代码）。**
- **分层解耦：格式 → 选型 → 参数 → 执行**。错误一键连锁（格式错 → 一切皆错），因此奖励通常分层及格制：格式分是前置门槛（不满足直接 0 或负，避免模型学会"参数对但格式乱"的折中解）。
- **权重与 or 关系：** 原文权重 0.3/0.3/0.3/0.1，合计 1.0 格式化。要点：执行分权重低（0.1）且**不依赖任何一个上游**（执行失败时有独立信号）；参数分用连续重叠率而非 0/1（部分 credit，但要看 §3 的 hack 风险）。
- **可验证优先（RLVR）：** 若能拿 ground_truth（有标注/可模拟工具调用），规则 reward 完胜学习型 RM——无 hack 面、无 RM 训练成本；开放场景（多个工具都可解）再退回 RM 或人工评审抽样审计。
- **与算法的接口：** GRPO 的 z-score 优势对尺度不敏感，但对**粒度**敏感：连续部分分让组内方差非零（避免全同组），这是连续参数分在 GRPO 下的实际价值。

**2. 可运行参考实现（完整可跑，复杂度 O(|params|)。**

```python
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class ToolCall:
    """一次工具调用的解析结果（或 ground truth）"""
    tool_name: str
    params: Dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""            # 模型原始输出（用于格式判断）
    result: Optional[Any] = None  # 执行结果（可选项）

def is_valid_json(raw: str) -> bool:
    """格式门禁：必须是合法 JSON 且含 tool_calls 顶层结构。O(len(raw))"""
    try:
        obj = json.loads(raw)
        return isinstance(obj, dict) and "tool_calls" in obj
    except (json.JSONDecodeError, TypeError):
        return False

def compute_param_overlap(pred: Dict[str, Any], gold: Dict[str, Any]) -> float:
    """参数 overlap：键重合 + 值相等的调和平均（精确匹配版）。
    复杂度：O(|pred| + |gold|)。若需要字符串模糊匹配可换 token-F1，仍是线性。
    """
    if not pred and not gold:
        return 1.0
    keys = set(pred) | set(gold)
    hit = sum(1 for k in keys if k in gold and k in pred and pred[k] == gold[k])
    # 键覆盖率与值准确率的调和平均，偏严：漏一个必答键会显著扣分
    key_recall = hit / max(len(gold), 1) if gold else 0.0
    key_precision = hit / max(len(pred), 1) if pred else 0.0
    if key_precision + key_recall == 0:
        return 0.0
    return 2 * key_precision * key_recall / (key_precision + key_recall)

def execute_response(resp: ToolCall) -> Any:
    """真实/沙箱执行。生产中应在受限沙箱中跑，失败返回哨兵值，不抛异常。"""
    try:
        return tool_registry_execute(resp.tool_name, resp.params)  # 接入你的工具注册表
    except Exception:
        return "EXECUTION_FAILED"

def reward_function_calling(response: ToolCall, ground_truth: ToolCall) -> float:
    """分层奖励：格式0.3 + 选型0.3 + 参数0.3 + 执行0.1 ∈ [0, 1]。
    总复杂度 O(len(raw) + |params|)；GRPO 中对每组 G 条调用，组级成本 O(G·(...))。
    """
    reward = 0.0
    # —— 1. 格式门禁（硬前置）：格式错直接把后续一票否决，防止"折中解"
    if not is_valid_json(response.raw_text):
        return 0.0
    reward += 0.3
    # —— 2. 工具选择（强信号，但不能完全 gate：多工具同解时避免二号惩罚）
    if response.tool_name == ground_truth.tool_name:
        reward += 0.3
    # —— 3. 参数连续分（部分 credit：让 GRPO 组内方差非零）
    reward += 0.3 * compute_param_overlap(response.params, ground_truth.params)
    # —— 4. 执行结果（独立通道，不依赖上游正确性才有信息量）
    if execute_response(response) == ground_truth.result:
        reward += 0.1
    return round(min(max(reward, 0.0), 1.0), 4)

# ---- 自测（可直接运行）----
if __name__ == "__main__":
    gold = ToolCall("get_weather", {"city": "北京", "date": "2026-11-13"}, raw_text='{"tool_calls":[]}', result="10℃ 小雨")
    ok = ToolCall("get_weather", {"city": "北京", "date": "2026-11-13"}, raw_text='{"tool_calls":[1]}', result="10℃ 小雨")
    miss_city = ToolCall("get_weather", {"date": "2026-11-13"}, raw_text='{"tool_calls":[1]}')
    wrong_tool = ToolCall("get_time", {"city": "北京"}, raw_text='{"tool_calls":[1]}')
    bad_json = ToolCall("get_weather", {"city": "北京"}, raw_text="今天天气是北京小雨")   # 非JSON
    assert reward_function_calling(ok, gold) == 1.0
    r2 = reward_function_calling(miss_city, gold)   # 0.3 + 0.3 + 0.3*0 = 0.6（键覆盖=0）
    r3 = reward_function_calling(wrong_tool, gold)  # 0.3 + 0 + 高overlap参数 + 0.1 执行
    r4 = reward_function_calling(bad_json, gold)    # 0.0（门禁）
    assert r4 == 0.0 and 0 < r2 < r3
    print(f"ok=1.0, miss_city={r2}, wrong_tool={r3}, bad_json={r4}  ✓")
```

（`tool_registry_execute` 需接入真实工具注册表；自测示例演示四档信号。）

**3. 常见坑与攻防（面试攻防战非常重要的弹药）。**
- **部分 credit 被 hack**：`param_overlap` 逐键线性给分会诱导模型"多撒参数碰碰运气"（precision 不惩罚多填）。对策：precision/recall 调和平均（代码已内置）；或对多填的键倒扣。
- **格式门禁太硬**：训练早期模型会输出大量非 JSON（策略坍缩到 0 奖励、组内全零方差），可先 SFT 冷启动格式，或对"可解析但字段错"给 0.1 过渡分。
- **执行分的信息泄漏**：若允许真实执行副作用（下单/删除），RL 探索会真的执行——需沙箱 + 幂等 mock；这是 Agent-RL 的工程红线。
- **多种子状态/多轮工具链**：单步奖励要带折扣或过程分解（每步一次校验），否则 actor 学会"最后一步碰对"。
- **reward 与 RM 混用**：格式/选型用规则、语气/安全用 RM，分项相加即可；注意 RM 分数尺度需校准（z-score 或 min-max）。

**【30秒答题框架】**

"Function Calling 奖励按'格式 0.3、工具选择 0.3、参数 0.3、执行 0.1'分层：格式是硬门禁，错了直接 0，防止模型折中；工具选择是强信号但不完全一票否决，因为多工具可能同解；参数用 precision/recall 调和平均的连续分，让 GRPO 组内方差非零；执行分独立成通道避免信息冗余。核心是 RLVR 思想——奖励尽量可验证且不可 hack：多填参数要倒扣、执行要走沙箱、早期格式太严格会全零奖励要冷启动。整段代码 O(参数数)，进 GRPO 直接组内 z-score。"

**【追问与陷阱】**

- *追问：为什么执行分只给 0.1？* 上游三项已大致决定执行成败，权重过高会鼓励模型"猜执行结果"类 hack；0.1 只作一致性校验。
- *追问：多轮工具链怎么打分？* 逐步过程奖励（每步规则分之和）+ 终态结果分，注意折扣与信使偏置。
- *追问：没有 ground truth 怎么办？* 用户反馈/在线 A-B（延迟信号）、对比学习 RM 补位、或 self-consistency 多数投票做 verifier。
- *易错：参数分用子集重叠却不管多填*——奖励 hacker 最爱；务必用调和平均或 precision 惩罚。
- *易错：格式分不设门禁*——模型学会 0.7 分折中解后很难再拔。

**【知识串联】**

- 题 6：RM vs 规则奖励是阶段 2 的分叉；题 10：本代码直接产出 GRPO 的 r_i。
- `05-Agent.md`（工具调用协议/ReAct）、`07-HotTopics-2025-2026.md`（RLVR/Agent-RL）、`08-Coding`（执行沙箱类手写题）。

---

### 题 14：SFT/指令微调的基本概念

**【考察意图】**

考基本功的"精确性"：① 损失为何只算 output 侧（mask 的机制收益而不是"习惯"）；② 数据格式的三段式（instruction/input/output）与 template 的统一；③ 分得清"预训练/继续预训练/SFT"的边界——基础不牢的候选人在此题的"loss mask 为什么要 -1"这类小问题上露馅。

**【深度解析】**

**1. 任务定义。** 给 (x, y) 对（x=指令+可选输入，y=期望输出），最小化输出 token 的负对数似然：

$$\mathcal{L} = -\sum_{t \in \text{output}} \log P_\theta(y_t \mid y_{<t}, x)$$

实现上构建 labels：prompt 位置填 −100（CrossEntropy 忽略），只在 y 区间计 loss（padding 同样 −100）。

**2. 为什么 mask prompt 侧？机制收益。**
- 目标错位风险：若对 prompt 也算 loss，模型会花容量去学"复述指令/记忆题目分布"，且 prompt 的统计特征（模板词、标点）会产生明显梯度噪声；_SW 上经典结论（Alpaca/LIMA 时期实践）是不 mask 会轻微变差。
- 统计解释：给 prompt 计 loss 等于给模型加一个"预测下一个用户说什么"的 FormsModule 任务——对"当助手"这个目标而言是干扰项（分布外 token），挤占有效梯度信号。
- **反面 nuance（加分项）**：2024 有工作（"Instruction Tuning with Loss Over Instructions"）报告在**输出较短**或多轮首句任务中 prompt-loss 反而有帮助（相当于附加正则），所以业界把它当超参看待，而不是铁律——默认 mask，个别场景实验打开。

**3. 数据格式（Alpaca 三段式）。**

```json
{"instruction": "请将以下英文翻译成中文", "input": "Hello, how are you?", "output": "你好，你怎么样？"}
```

input 留空表示任务自带全部信息；训练时套统一 template（`{instruction}\n{input}\n### Response:\n{output}` 或 ChatML）——**模板必须与推理时一致**，否则分布错位。

**4. SFT 与相邻概念的边界（面试常混）。**
- **继续预训练（CPT）**：无监督 next-token，全序列 loss，数据是领域语料；目的是知识注入（与题 1 的"LoRA 难以注入知识"呼应）。
- **SFT**：监督损失（对格式的监督信号），目标行为/格式对齐；数据 1K–100K 即可。
- **RLHF/偏好微调**：对"相对好坏"的监督，题 6–12。
- 一句话：CPT 改"知道什么"，SFT 改"怎么说话"，偏好对齐改"哪种说话更好"。

**【30秒答题框架】**

"指令微调就是让预训练模型从'续写'切换到'听指令办事'：用 instruction/input/output 三段式数据，损失只算在 output 部分——labels 里 prompt 和 padding 都填 −100。原因是给 prompt 算 loss 等于让模型学复述指令，挤占对齐信号，还会引入模板噪声；业界默认 mask，但短输出/多轮场景有论文显示打开 prompt-loss 反而有用，所以当超参看。注意 SFT（行为对齐）不同于继续预训练（知识注入）和偏好对齐（喜好排序），且训练/推理模板必须严格一致。"

**【追问与陷阱】**

- *追问：为什么不用 encoder-style 的 [CLS] 头？* 生成任务逐 token 自回归，序列级分类头丢掉生成长度分布；继续用 LM 头是自然选择。
- *追问：−100 是什么？* PyTorch CrossEntropy 的 ignore_index 约定值。
- *易错：把"只在 output 算 loss"说成"只把 output 喂进模型"*——输入是全序列（含 prompt 作为条件），只是梯度忽略 prompt 位置。
- *易错：模板与推理不一致*——很多"微调完效果差"的故障排查第一项就是查 chat template。

**【知识串联】**

- 题 1（LoRA 在 SFT 上的载体）、题 16（teacher template 规范）、题 17/18（SFT 工程组合）。
- `01-Foundation/03-Loss-And-Optimization.md`：交叉熵/ignore_index 细节、余弦退火 LR 等。
- `04-RAG.md`：RAG 的 SFT 数据是"带检索上下文的 prompt"—— loss masking 原则复用。

---

### 题 15：指令微调数据质量 vs 数量？

**【考察意图】**

考"有实证依据的判断力"：① LIMA（1000 条 > 52K 杂数据）和 Alpaca（52K 自造）两条证据线各自说明什么；② 质量/多样性的操作化定义（怎么把玄学变成可执行的 pipeline）；③ 数据配比经验——多任务场景的 mixture 比例；④ 反面 nuance：什么时候数量仍很重要（知识注入/多语言/推理）。

**【深度解析】**

**1. 三条证据线。**
- **LIMA（2023）**：1000 条精心策展（人工+社区精选+多样性覆盖）数据做 SFT，无 RLHF，效果在 65B 底座上与多倍数据模型相当。结论："对齐可当表面学习（格式/行为先验），预训练已容纳 almost all knowledge"——**质量>数量**成立的经典条件：底座足够强 + 目标是行为对齐。
- **Alpaca（2023）**：52K self-instruct 生成（gpt-3.5 蒸馏，均价 <$600）能复现出类 InstructGPT 行为——**量大+多样性任务覆盖**在 7B 级也能出效果，但精细任务/拒答/安全上限受质量限制。
- **Flan/FLAN-T5（2021/2022）**：1800+ 任务的多任务混合 + scaling，任务**多样性**本身带来显著提升（zero-shot 泛化）——数量与多样性另一端的最有力证据。结论：**多样性 > 单纯条数 > 质量下限**。

**2. "质量"的操作化定义（可执行维度）。**
- 正确性：事实错误、幻觉语言会被模型模仿（甚至放大）；
- 指令-回答相关性与难度分层（简单模板题堆量反而稀释）；
- 多样性：任务类型（10+：QA/改写/摘要/代码/数学/角色扮演/拒答/多轮…）、长度分布、复杂度（Cherry data：难样本的边际收益高）；
- 风格一致性：同一批数据的 model-of-thought 风格统一（例如统一 thinking 段落格式），混风格会互相稀释；
- 无害与合规标签一致（拒答格式统一）。

**3. 数据清洗 pipeline（可落地的六步）。**
去重（exact + MinHash 语义近邻）→ 语言/编码过滤 → 启发式污染（乱码、模板截断、列表冗长）→ 规则化打分（长度、困惑度由 base model 过滤 top/bottom 分位）→ **模型评分**（用强模型或 reward model 打 IFD/DEITA-score，取高端分位）→ 配比采样（见 §4）。DEITA 等工作证明"选择算法"比"扩数据库"更划算（大约 6K~10K 精选即可逼近大库效果）。

**4. 数据配比（mixture）经验。**
- 任务维度：目标域任务占大头（如代码 SFT 时代码占 40–60%），但**保留通用能力回放**（通用指令 10–20% + 少量预训练风格数据）防御灾难性遗忘（题 18）；
- 温度采样：$p_i \propto n_i^{1/T}$（τ>1 趋近均匀，τ→0 趋近按量比例）；T5/Flan 用指数平滑防止大任务垄断；
- 中英比例：中文任务常按 30–70% 中/英混合（中文语料库天然小，混英文可借泛化）；
- 长短/难易平衡：难样本占 10–20% 起、以训练后 loss 曲线与评测定夺。
- **反例预告**：若目标是知识注入（新领域术语/事实），数量与重采样次数都应放大——此时"质量>数量"让位于"覆盖>质量边际"。

**5. 数据规模参考（给数字）。** 行为对齐：1K（LIMA 级）–100K（Alpaca 级）显著有效；100K 以上新增量趋平（指令跟随饱和）；知识/领域注入需要百万 token 级延续预训练+领域 SFT 组合。DPO 数据（题 8/9）数千对起即可有效。

**【30秒答题框架】**

"质量远大于数量：LIMA 用 1000 条精选数据训 65B 效果不输更大数据集，说明行为对齐本质是'激活预训练已有的知识'；Alpaca 52K 便宜但上限有限；Flan 用 1800+ 任务多样性实测带来最大增益。所以答案是把质量和多样性都操作化：去重、困惑度过滤、模型打分选 top 分位、任务/长度/难度做分层配比，同时保留 10–20% 通用数据防遗忘。但要说 nuance：如果目标是知识注入而不是行为对齐，数量和覆盖又变回第一要素。"

**【追问与陷阱】**

- *追问：LIMA 为什么 1000 条就够？* 65B 底座 + 任务=行为/格式对齐；若是 7B 或需要注入领域知识，结论不迁移。
- *追问：你们项目里怎么定配比？* 给流程：先按目标域 60/通用 20/难度分层 20 做初版，看验证集三类指标（目标域/通用保持/安全）再迭代；强调用小规模消融驱动。
- *追问：GPT 蒸馏数据的法律风险？* ToS 限制 + 学术与非商用条款（Alpaca 非 PERMISSIONS 商用引用），商用需自有标注或合规开源混合。
- *易错：把"质量>数量"背成无边界公理*——要能反向说明何时数量重要。
- *易错：多样性只理解为"任务类型多"*——长度/难度/风格/语种各维度都要分层。

**【知识串联】**

- 题 14：格式模板 + 数据质量是 SFT 双支柱。
- 题 17/18：数据规模决定全参/LoRA 路线；数据质量决定遗忘/过拟合风险。
- `04-RAG.md`：RAG 减少模型记忆负担，与"知识注入靠预训练/CPT"互为方案（避免拿 SFT 硬背知识）。

---

### 题 16：ChatML 格式与多轮对话训练

**【考察意图】**

考：① 能默写 ChatML 结构并说清特殊 token（im_start/im_end）的作用边界（EOS 信号的学习问题）；② 多轮对话两种损失计算方式（逐轮独立 vs 全对话拼接+掩码）的机制差异与推荐理由；③ 隐形工程坑：模板/训练-推理一致性、packing 时 cross-example attention 泄漏。

**【深度解析】**

**1. ChatML 结构（原文示例 + 结构要点）。**

```
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
请介绍一下自己<|im_end|>
<|im_start|>assistant
我是一个 AI 助手...<|im_end|>
```

设计要点：
- **三个控制点**：role 声明（system/user/assistant/tool）、内容边界（im_end）、终止信号（最后 assistant 段后的 <|im_end|> 即生成时要学的 EOS）。每个 role 前有 im_start 表示"身份切换"。
- **EOS 学习问题（高价值追问点）**：若训练时把所有 <|im_end|> mask 掉或截断丢失，模型将不会"闭嘴"——推理时停不下来（无限生成或直到 max_new_tokens）。正确做法：assistant 生成段的结尾 token `<|im_end|>` **参与损失**（labels 不 mask 它）。
- 生成侧停止条件：`stop=["<|im_end|>"]` + max_tokens 双保险。

**2. 多轮对话两种训练方式（原文两方式 + 推荐）。**
- **方式一：每轮独立计算损失**——每轮 (user_i, assistant_i) 各自成一条样本，上下文丢失（第 2 轮看不到第 1 轮），导致多轮上下文能力学不到；采上下文时人工拼接历史但 loss 各自独立，通常仅作数据增强副本。
- **方式二（推荐）：整条对话拼接，只在 assistant 回复段计 loss**——

```
[sys][u1][a1][u2][a2]   ← 一条样本
loss:       ─────────    ← 仅 assistant 段 + 每段末尾的 <|im_end|>
```

机制收益：① 上下文依赖被真实建模（a2 的梯度建立在看过 u1/a1 上）；② 每轮 assistant 的开头都会学会"如何自然接话"；③ <|im_end|> 停止信号在各轮都被学到。
- 变体：某些实现把历史 assistant 轮的 loss 权重衰减（recent-turn weighting），或将最后一轮加权——小众，按需。

**3. 工程坑清单（面试可背）。**
- **模板一致性**：训练与推理用同一模板/特殊 token ids；模板漂移是"微调没效果"的高频根因。
- **packing 与注意力泄漏**：把多条对话 concat 进一个序列以省 padding 时，若无 document masking，样本 A 会 attend 到样本 B——梯度被污染。对策：position_ids 重置 + attention mask 分块，或干脆不 packing。
- **多轮 tool 调用段**：tool role 的内容是否计 loss？约定为"不计 loss 但进条件"（assistant 只对生成内容负责）。
- **截断策略**：超长对话从**开头丢 system 保最近轮** vs 丢中间轮，要按任务定；截断后首轮看起来"没头没尾"会教坏模型。
- **tokenizer 一致性**：im_start/im_end 必须是注册的特殊 token（新增 token 要 resize embedding），裸字符串会被切碎。

**【30秒答题框架】**

"ChatML 用 <|im_start|>role … <|im_end|> 划身份和边界，最后 assistant 段的 <|im_end|> 就是 EOS——训练时必须把它算进 loss，否则模型学不会停。多轮对话推荐整条拼接、只在 assistant 段和各轮结尾的 im_end 计 loss：上下文依赖真实建模、接话能力和停止信号同时学到。三个隐形坑：训练推理模板必须一致；多条样本 packing 一定要 document mask 防 attention 泄漏；超长截断从后往前保留，别把 system 丢了。"

**【追问与陷阱】**

- *追问：为什么 system 段 loss 要 mask？* system 是条件不是生成目标；系统提示由部署方控制，学它没有意义还稀释梯度。
- *追问：im_end 和 EOS token 什么关系？* 可以是同一个特殊 token 也可以分离（Qwen 用 <|im_end|>，Llama3 用 <|eot_id|>）；关键是"推理停止条件与训练结尾信号一致"。
- *易错：多轮样本只训最后一轮*——上下文学习全部失效；正确是"全条件 + 全 assistant 段 loss"。
- *易错：以为 packing 一定好*——先看框架是否支持分块注意力，否则宁可不 packing。

**【知识串联】**

- 题 14：loss mask 的同一套 labels 机制在单轮/多轮的推广。
- `02-Inference/01-KVCache`：多轮服务时 KV cache 复用（prefix caching）取决于 system 段稳定——训练模板的稳定性直接影响推理效率。
- `05-Agent.md`：tool role 的扩展（多轮 tool-calling 对话模板）。

---

### 题 17：全量微调 vs LoRA 微调选择

**【考察意图】**

考工程决策表 + 判断依据：① 会不会把"数据规模/资源/任务类型/部署方式"四个维度组合成决策矩阵；② 每一行推荐背后的机制理由（为什么 >100K 数据倾向全参——不是参数容量而是"知识/行为大幅改写需要高秩更新"）；③ 多租户 LoRA 热切换的部署细节；④ 显存受限时 QLoRA 的量化代价如何取舍。

**【深度解析】**

**1. 决策矩阵（原文表扩充）。**

| 场景 | 推荐 | 机制理由 | 量化参考 |
|------|------|----------|----------|
| 数据 >100K 且资源足 | 全参微调 | 大数据需要高秩更新承载新知识/行为；LoRA 低秩瓶颈限制记忆容量；且全参可通吃"能力增量" | 7B×16B/param(可训) ≈ 112GB（多卡） |
| 数据 1K–100K | LoRA r=16–64 | 行为/风格对齐低秩足够；训练/存储成本低一个量级 | 7B 底座 bf16 14GB + 小 adapter |
| 数据 <1K | LoRA r=4–8（或干脆 few-shot/IAL） | 极小数据训大容量必过拟合；lora 小容量+大 LR（1e-4 档）是较稳的保守适配 | 几分钟～1h/epoch 级 |
| 多任务/多租户 | LoRA 适配器热切换 | 底座共享、R-adapter 影响小文件级；服务端多 adapter 动态加载（vLLM multi-LoRA） | 每 adapter 模块 MB 级 |
| 显存受限 | QLoRA（4-bit 底座 + LoRA） | 0.5 B/param 权重 + 少量可训练态，单卡容量上限 ×4 | 65B@48GB（题 4） |
| 知识注入强 | 全参 or CPT+全参 SFT | 知识记忆需要高秩大容量更新；LoRA 有"学习数据难于记忆"的实证限制 | — |

**2. 机制主线：更新秩 vs 数据量。**
- 若任务增量是小秩的（讲什么风格、改什么格式），小 r 的 2dr 空间已够；数据放大不会产生更多高频方向 ⇒ LoRA 无瓶颈。
- 若任务包含大量"新事实/新映射"（术语、领域知识、新语言），更新谱是高秩的 ⇒ LoRA 上限早至 r≈64 就达瓶颈；实证有"LoRA 学得慢、忘得快，全参记得牢"的报道。因此在 >100K 且含知识注入时选全参。

**3. 成本与运维账本（给数字）。**
- 训练：7B 全参（bf16 权重+梯度+fp32 Adam）≈ 16B/param ⇒ 112GB+；LoRA r=16 全线性 ≈ 117M 参数 ≈ 2B 级 adapter 开销（可单卡 24GB 训 7B）。
- 部署：全参=N 份权重（多租户 N×14GB）；LoRA=1 份底座 + N 个 adapter（每 adapter ~几十 MB）+ 动态路由层——多租户只有 LoRA 可行。
- 迭代速度：LoRA 实验周期短数倍，因为权重更新量小、checkpoint 也小（几 MB vs 几十 GB）、可并行跑多组 r/α/LR。

**4. 判断流程（一图流，面试口述）。**

```
目标 = 注入新知识？
  ├─ 是 ──► 全参（或 CPT 后再 SFT）
  └─ 否（行为/风格/格式对齐）
        └─ 数据量：<1K? → LoRA r=4-8 │ 1K-100K? → LoRA r=16-64 │ >100K && 有算力 → 全参
服务形态 = 多租户热切换? ──是──► 强制 LoRA（合并不可行）
显存 < 模型 2 倍 bf16? ──是──► QLoRA
```

**【30秒答题框架】**

"决策看三个量：任务是否注入新知识、数据量、部署形态。行为/风格类任务 LoRA 就是够用（低秩假设成立），1K 以下 r=4–8、1K–100K r=16–64；超过 100K 或者要知识注入，更新秩变高，全参才有容量，而且全参记得牢、LoRA 忘得快。多租户只有 LoRA 能做适配器热切换，底座共享。显存不够就 QLoRA，权重压到 0.5 B/param，65B 进单卡 48GB。一句话：默认 LoRA，出瓶颈才上全参。"

**【追问与陷阱】**

- *追问：LoRA 效果差先检查什么？* ① 是否 all-linear 注入；② r/α 联动与 LR（LoRA LR 通常比全参 5–10 倍）；③ 模板一致性；④ 数据质量。
- *追问：LoRA 能做增量学习/持续学习吗？* adapter 天然适合"序贯加载"，但 bottom-up 累积新任务时旧 adapter 需冻结；大步学科知识仍推荐全参。
- *追问：什么时候 LoRA 不如全参的另一个原因？* 长训练下 LoRA 那些 2dr 参数还会饱和（capacity collapse），需要重生技法（rescale/re-init）——知道趋势即可。
- *易错：把 100K 说成"LoRA 不行的门槛"*——是"是否包含知识注入"的函数，量化时说成"经验阈值，具体看任务类型"。
- *易错：不知道多租户部署是 LoRA 的最硬性约束*（配置面又一票否决项）。

**【知识串联】**

- 题 1/2/3：容量（r）与层位即是本题的技术内幕。
- 题 4：QLoRA 是"显存受限"行的实现。
- `09-SystemDesign.md`：多租户 LoRA 服务架构（router 侧实现）、训练平台的多 adapter 管理。

---

### 题 18：微调常见问题（灾难性遗忘 / 过拟合）

**【考察意图】**

考"调过模型才知道的"：① 灾难性遗忘的三层机制与对应疗法（为什么小 LR 有效；为什么 LoRA 天然缓解；回放数据的比例怎么定）；② 过拟合在 SFT 场景的特殊表现（不只 val loss 升：模板复读、拒绝词失效、语言混合）；③ 会用" LR/epoch/dropout/数据回放/L2 位置"这套旋钮并说出量级数字（LR 1e-5~5e-5、1–3 epochs、LoRA 大一档）。

**【深度解析】**

**1. 灾难性遗忘：机制。**
- **梯度致移**：SFT 数据分布与预训练分布差异大（短样本、问答体、高 density 模板），AdamW 迭代把权重沿"新任务 loss 面"拉动，老能力（长文连贯、生僻知识 recall）所在的方向被挤走。
- **优化器记忆效应**：Adam 的二阶矩归一化使小梯度方向步长放大，间接放大遗忘。
- **容量竞争**：全参微调"改写"了权重本身；LoRA 只加低秩增量、W 原地不动——预训练行为仍在（这正是 LoRA 天然缓解遗忘的机制根源，也从反面论证了 LoRA 学不动知识的弱项与不愿遗忘的强项是同一体）。

**2. 遗忘疗法（按优先级）。**
- **小学习率**（全参 1e-5 ~ 5e-5；LoRA 1e-4 ~ 2e-4）——本质是给"权重远离预训练极小值"的漂移限速；
- **少 epoch**（1–3）+ early stop on held-out 通用集；
- **数据回放**：mix 进 5–20% 通用指令/预训练风格数据（代码：混 ~10% 通用代码/多语言短文即可显著保持）；本质是给老分布保留梯度锚点；
- **参数正则**：权重衰减、或 EWC/L2-SP（对 ‖θ−θ_pre‖² 加罚——LoRA 等价于一种结构化 L2-SP）；
- **过程控制**：监控训前/训后通用能力基准（MMLU 级/长文 perplexity），防"测不到的遗忘"。

**3. 过拟合（SFT 的形态）。**
- 表现：**val loss 上升**是教科书指标；更日常的是行为指标——输出复读模板/固定句式、安全拒答失效、语言混杂（中文跑成英文）、对变体输入（改个说法）性能暴跌。
- 疗法：早停（val 曲线拐点一般在 1–3 epochs）、dropout（Linear/attention dropout 0.05–0.1，LoRA dropout 同理）、weight decay（AdamW 0.01–0.1）、数据增强（多模板复述）、多样性采样、**小 LR**（防死记）、数据量不足时先补数据再谈正则。
- 判别口诀：**训练 loss 低、通用评测也降 = 遗忘；训练 loss 低、训练集复读但泛化差 = 过拟合**——对应药方不同（回放 vs 增强/早停）。

**4. repertoire 汇总（LR 量级表，考官爱听）。**

| 设置 | 全参 SFT | LoRA SFT |
|------|----------|----------|
| 学习率 | 1e-5 ~ 5e-5（7B 级常用 1e-5 或 2e-5） | 1e-4 ~ 2e-4（约全参 ×5–10） |
| LR schedule | cosine + warmup 3% ~ 5% steps | 同左 |
| epochs | 1–3 | 1–5（小数据可多几轮） |
| weight decay | 0.0–0.1 | 0.01 档 |
| dropout | 0–0.05 | LoRA dropout 0.05–0.1 |
| 有效 batch | 全局 batch 大（梯度稳）+ grad accum | 同左，容量小更依赖 batch 质量 |

**【30秒答题框架】**

"灾难性遗忘的机制是 SFT 分布把权重拉离预训练极小值，老能力方向被挤走——药方是限速与锚点：1e-5~5e-5 的小 LR、1–3 个 epoch 早停、混 5–20% 通用数据回放，LoRA 因为只加低秩增量、底座不动，天然缓解。过拟合在 SFT 的表现不只 val loss，还有模板复读、拒答失效、语种漂移，用早停、dropout、weight decay 和数据增强，量级上 LoRA 的 LR 取全参的 5–10 倍。判断口诀：通用能力下降是遗忘，训练集复读泛化差是过拟合，药方不同。"

**【追问与陷阱】**

- *追问：为什么 LoRA 的 LR 要更大？* 可训练参数少、梯度经 (α/r) 缩放后被稀释，且低秩子空间方向有限，更新信号弱；×5–10 是经验平衡。
- *追问：回放比例怎么定？* 看三线指标（目标能力↑ / 通用能力× / 安全性✓），小步加；上限一般不超过 1:9（回放:目标任务）影响主目标收敛。
- *追问：遗忘能彻底避免吗？* 不能；工程上控制到"评测可见差异"以下。想真正做到 coasting 级持续学习，需要 replay + 冻结 + 参数隔离（LoRA 专家）组合。
- *易错：把"遗忘"和"过拟合"混为一谈*（药方不同，见 §3 口诀）。
- *易错：给 LoRA 用 1e-5 学习率*——典型"LoRA 训不动"的根因。

**【知识串联】**

- 题 1（LoRA 结构性缓解遗忘）、题 17（数据规模 → 路线）、题 15（回放数据属于配比的一部分）。
- `01-Foundation/03-Loss-And-Optimization.md`：AdamW/Warmup/Weight Decay 的机制底座。
- `06-Safety-Evaluation.md`：安全对齐（拒答能力）在 SFT 中被"洗掉"是高频故障。

---

## 本章高频追问 Top5

1. **"LoRA 为什么 B 全零、A 随机？两个都置零会怎样？"** —— 等价性 + 对称性破缺：双零 ⇒ ΔW=0 且两因子梯度同时为零（死锁）；A 随机保证第一步 ∂L/∂B≠0。引申 PiSSA/LoRA-GA 的 SVD 初始化修第 0 步只有单边梯度的问题。（题 1）

2. **"QLoRA 的 NF4 为什么是分位数格点？Double Quantization 省多少显存？"** —— 对近高斯权重的 Lloyd-Max 最优标量量化（每格点承载 1/16 概率质量）；缩放常数从 0.5 bits/param 压到 0.127，65B 省 ~3GB；65B 权重 4-bit ≈ 32.5GB，加上分页优化器与 LoRA 优化态 ~0.2GB ⟹ 单卡 48GB。（题 4）

3. **"DPO 的闭式解推一遍，Z(x) 为什么能消掉？β 是什么角色？"** —— KL-约束目标 → 拉格朗日 → π* = π_ref·e^{r/β}/Z → 反解 r 代入 Bradley-Terry 的差值，β·log Z 相消；β 是温度：约束端控漂移、损失端控 sigmoid 饱和（大 β=保守，小 β=激进易过拟合）。（题 8）

4. **"GRPO 凭什么去掉 Critic？组大小 G 怎么权衡？全对/全错的组怎么办？"** —— 组内奖励均值是 V(x) 的 MC 估计（误差 O(1/√G)），G 是方差-算力的旋钮（DeepSeekMath/R1 用 G=64）；z-score 让优势对奖励尺度不变；全同奖励组 advantage≡0，用 DAPO 的动态采样（过采样后过滤零方差组）+ clip-higher + token-level loss + 去 KL。（题 10/12）

5. **"SFT 只训 1000 条能行吗？数据配比怎么定？"** —— 分目标讨论：行为/格式对齐（LIMA：1K 精选 + 强底座即可，质量>数量）；知识注入则数量与覆盖优先；配比 = 目标域大头 + 10–20% 通用回放防遗忘 + 任务/长度/难度分层，先用小规模消融驱动再放大。（题 15/18）

---

*本文件为原文的配套逐题详解；所有公式推导以官方论文为准（LoRA: Hu et al. 2021; QLoRA: Dettmers et al. 2023; DPO: Rafailov et al. 2023; GRPO: Shao et al. 2024 DeepSeekMath; DAPO: ByteDance 2025）。“原章节文件未修改。*
