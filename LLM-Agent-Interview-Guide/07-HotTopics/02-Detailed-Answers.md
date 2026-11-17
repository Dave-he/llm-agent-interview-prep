# 第七章 前沿热点 2025-2026 —— 逐题详解

> 输入文件：`07-HotTopics/01-Hot-Topics-2025-2026.md`（只读，未改动）。
> 研究依据：`/Users/hyx/agent-personal/personal-agent-research.md`（个人 AI Agent 系统调研报告 v1.1 事实核查修订版）。凡引用 2025-11 模型代际成绩/价格处，一律标注"厂商口径（研究文档表五）"；研究文档已修正 DeepSeek-V3.2 价格为 $0.28/$0.42（缓存命中 $0.028），与本文一致。公式与训练细节以官方论文（DeepSeek-V3 arXiv:2412.19437、DeepSeek-R1 arXiv:2501.12948、YaRN、Switch Transformer 等）为准。

## 本章考点分布

| 模块 | 题号 | 考频/难度 | 关联本仓库章节 |
|---|---|---|---|
| DeepSeek-V3 四大创新（MLA/MoE/FP8/MTP） | 题1 | ⭐⭐⭐ 高频大题 | 01-Foundation/01-Transformer、02-Inference/01-KVCache |
| DeepSeek-R1 纯 RL 涌现推理 | 题2 | ⭐⭐⭐ 高频大题 | 03-FineTuning/02-Alignment |
| MoE 原理 | 题3 | ⭐⭐⭐ 必考·双线题 | 01-Foundation/03-Loss、09-SystemDesign |
| MoE 训练挑战 | 题4 | ⭐⭐ 中频 | 01-Foundation/03-Loss-And-Optimization |
| o1/o3 vs DeepSeek-R1 | 题5 | ⭐⭐ 对比表题 | 03-FineTuning/02-Alignment |
| Test-Time Compute | 题6 | ⭐⭐⭐ 2025 新热点 | 02-Inference/02-Decoding |
| 长上下文扩展 | 题7 | ⭐⭐⭐ 必考·数学+工程双线 | 01-Foundation/01-Transformer（RoPE）、04-RAG |
| VLM 架构 | 题8 | ⭐⭐ 架构图题 | 01-Foundation/01-Transformer（ViT） |
| 2025-2026 趋势 | 题9 | ⭐⭐ 开放辨析题 | 05-Agent、09-SystemDesign、10-RealQuestions |

本章 9 题串起四条主线：**模型架构**（1/3/4/8）、**训练范式**（2/5/6）、**系统能力**（7）、**产业趋势**（9），是二面"热点自由问答"环节的核心弹药库。以下逐题按五段式【考察意图】【深度解析】【30秒答题框架】【追问与陷阱】【知识串联】详解。

### 题 1：DeepSeek-V3 的核心创新

**【考察意图】** 2024-12 以来的第一热点题。面试官要验证三层：(1) 你是否真读过机制，而非只会报名词；能否对四大创新各讲出"解决什么问题→怎么做→量化收益→代价"；(2) 数学功底：MLA 的低秩压缩与 absorption（吸收）技巧、解耦 RoPE 的原因；(3) 官方口径数字意识（671B/37B、14.8T tokens、2.788M H800 GPU 时）。

**【深度解析】**

背景：DeepSeek-V3（2024-12，MHA→MLA 的 V2延续）671B 总参 / 每 token 激活 37B，14.8T tokens，官方口径预训练总成本 2.788M H800 GPU 时（≈$5.58M，按 $2/GPU 时租金计，不含研究与失败实验）——比同质量闭源旗舰预训练低约一个数量级。四项创新互锁：

(1) **MLA 数学推导**。MHA 每层每 token 缓存 K、V 共 `2·n_h·d_h` 个数（V3 配置 n_h=128、d_h=56 → 14336 个）。MLA 低秩联合压缩：
- 压缩：`c_t = W_DKV h_t`（W_DKV ∈ R^{512×7168}），KV 被联合压到 512 维潜向量；
- 还原（多头）：`k_t^C = W_UK c_t`，`v_t^C = W_UV c_t`；
- RoPE 解耦路：`q_t^R = RoPE(W_QR h_t)`，`k_t^R = RoPE(W_KR h_t)`，各 64 维，承担相对位置；
- **absorption（吸收）**：由结合律 `(q_t^C)^T (W_UK c_s) = ((q_t^C)^T W_UK) c_s`，把上投影 W_UK 合并进 query 侧，推理时无需物化还原后的 K/V，注意力直接对 512 维潜向量算 → 缓存只剩 `c_t(512)+k_t^R(64)=576`/层/token，对比 MHA 的 2·128·56=14336，压缩约 25×（官方论文口径：KV Cache 较 MHA 减少 93.3%）。
- 为什么必须**解耦 RoPE**：RoPE 是位置相关的正交旋转 R_m；若直接对潜向量 RoPE，则 `k = R_m·W_UK·c` 中 R_m 卡在 W_UK 与 c 之间，无法用结合律吸收进 query 侧——因此单独留 64 维专用 RoPE 路，潜向量保持"无位置"才能被吸收。

(2) **DeepSeekMoE**：更细粒度专家 + 共享专家隔离。层输出 `h'_t = u_t + Σ共享 FFN_s(u_t) + Σ_{i∈TopK} g_{i,t}·FFN_r^{(i)}(u_t)`，`g_{i,t}=s_{i,t}·1[排名∈TopK]`，`s_{i,t}=softmax(u_t^T e_i)`。V3：256 路由专家+1 共享专家，每 token Top-8；共享专家吸收"所有 token 都要"的共性知识，避免在每个路由专家里重复存储；更细粒度切分（专家中间维 2048）在同等激活量下给出更细的知识分片。还配了**节点受限路由**（每 token ≤4 节点）压通信。

(3) **FP8 混合精度**（官方口径,首个 14.8T 级 FP8 预训练）：激活按 1×128 tile、权重按 128×128 block 分组定标缓解离群值；低精度累加后周期性（G=128）恢复到高精度累加，并用在线量化；位宽减半 → 显存与带宽均约减半，支撑 2.788M H800 GPU 时的低成本。

(4) **MTP**：主模型隐状态 h_t 与下一 token 嵌入拼接，送入单层 MTP 模块预测 `x_{t+2}`，与主预测 `x_{t+1}` 联合训练；推理可选保留做投机解码（论文口径：第二 token 接受率 85%–90%，吞吐约 1.8×）或直接丢弃、无部署负担。

数字锚点（官方口径）：V3 总参 671B、激活 37B、61 层、hidden 7168、128 头（d_h=56）、前 3 层为稠密 FFN、256 路由专家 + 1 共享专家、Top-8 路由、节点受限 ≤4。

**【30秒答题框架】**
"V3 四大创新。一，MLA：KV 压到 512 维潜向量 + 64 维解耦 RoPE，推理用吸收形式只需缓存 576/层/token，官方口径比 MHA 省 93.3% KV Cache；压缩靠低秩、RoPE 解耦是为了保住吸收技巧。二，DeepSeekMoE：256 路由+1 共享、Top-8 激活，671B 总参 37B 激活；共享专家去共性冗余。三，首次在 14.8T token 量级上 FP8 训练，靠细粒度分块与高精度累加保精度，官方口径全程 2.788M H800 GPU 时、约 557 万美元。四，MTP：一次多预测一个 token，官方口径接受率 85%–90%、解码约 1.8×。四点分别对应：省推理 KV、省激活算力、省训练带宽、提数据效率与解码速度。"

### 题 2：DeepSeek-R1 的推理能力

**【考察意图】** 验证你是否真懂"纯 RL 涌现推理"的机制链：为什么 GRPO 不需要 PPO 的 critic、为什么奖励不用神经 RM（可验证规则奖励防 reward hacking）、aha moment 说明什么（能力激发而非能力注入）、以及 R1-Zero→R1→蒸馏的四级闭环。二面标准：能白板写出 GRPO 目标函数并指出与 PPO 的两处结构差异。

**【深度解析】**

(0) R1 的 base 是 V3-Base；R1-Zero 是"无 SFT、直接 GRPO"的消融版，R1 是工程化完整版。任务定义在**可验证奖励**（Verifiable Reward）域：数学对答案、代码跑测试，奖励客观。

(1) **GRPO 推导**。PPO：`J_PPO = E[min(ρÂ, clip(ρ,1±ε)Â)] − β·KL(π_θ‖π_ref)`，优势 Â 由 GAE 计算，需要一个与策略同规模的**价值网络**（≈多一份几十 B 级模型的显存与一轮 backward）。GRPO 用**组相对基线**替掉 critic：每题采样 G 条回答 o_1..o_G，奖励 r_i：
- 优势：`Â_i = (r_i − mean(r)) / std(r)`；
- 目标：`J = E[ (1/G)Σ_i (1/|o_i|)Σ_t ( min(ρ_{i,t}Â_i, clip(ρ_{i,t},1−ε,1+ε)Â_i) − β·KL(π_θ‖π_ref) ) ]`，其中 `ρ_{i,t} = π_θ(o_{i,t}|q,o_{i,<t}) / π_θold(o_{i,t}|q,o_{i,<t})`；KL 用 k3 无偏估计 `D_KL ≈ π_ref/π_θ − log(π_ref/π_θ) − 1`。

与 PPO 的两处结构差异：去 critic（省显存与算力）；组内标准化让"同题多条回答"自带相对信用，适合结果奖励稀疏的场景。

(2) **奖励设计**：数学用规则匹配答案、代码用编译+测试用例；不走神经 RM——RM 的代理分数可被格式话术 hack（如押韵刷分），规则奖励几乎不可 hack。语言混杂问题靠加"语言一致性奖励"缓解。

(3) **R1-Zero 的涌现（官方论文口径）**：从 V3-Base 纯 GRPO 起步，response 平均长度随训练**持续增长**——模型学会用更多思考换准确率；AIME 2024 pass@1 从约 39.2% 升至约 71.0%；同时暴露可读性差与中英混杂两大问题。**aha moment**：论文训练中段自述 "Wait, wait, wait. That's an aha moment I can flag here..."——反思（验证、回溯、改解法）行为自发长出。要点：RL 是**激发**预训练中已存在的潜在能力，不是凭空创造。

(4) **R1 四级闭环**（修复 R1-Zero 短板）：a) 冷启动 SFT（官方口径数千条长 CoT 样本，修可读性）→ b) 推理 RL（规则奖励 + 语言一致性奖励）→ c) 拒绝采样 + SFT：官方口径 60 万推理 + 20 万通用 = 800k 条 → d) 全场景 RL（规则奖励 + 帮助/无害 RM）→ e) **蒸馏**：800k 样本直接 SFT 到 Qwen2.5/Llama3 系列。官方口径：R1-Distill-Qwen-32B AIME 2024 pass@1 72.6%，超 o1-mini 的 63.6%；小模型"蒸馏优于同规模从零 RL"。

(5) 性能锚点（官方论文口径，R1 vs o1-1217）：AIME 2024 79.8% vs 79.2%；MATH-500 97.3% vs 96.4%；GPQA Diamond 71.5% vs 75.7%；Codeforces 2029。

**【30秒答题框架】**
"R1 主线是纯 RL 涌现推理。GRPO 直接在 V3-Base 上 RL：组相对优势替代 PPO 的 critic，奖励只用数学答案匹配、代码测试这类可验证信号，几乎不能被 hack。训练中 response 长度自发增长，中段出现 aha moment——模型自己学会 wait、反思、回溯，AIME pass@1 从约 39 到 71（官方口径）。因为纯 RL 输出可读性差、语言混杂，R1 走四级闭环：冷启动 SFT → 推理 RL → 80 万条拒绝采样 SFT → 全场景 RL，最后蒸馏到 Qwen/Llama，Distill-Qwen-32B 超 o1-mini。一句话：可验证奖励 + 组相对 RL 激发长 CoT，再用 SFT/蒸馏工程化收口。"

**【追问与陷阱】**
- 追问 GRPO vs PPO 资源差异 → 少一个同规模 critic：省约一份骨干规模的显存与对应反向计算；追问 vs DPO → DPO 依赖静态偏好对、无 on-policy 探索，多步稀疏信用分配无能。
- 追问"为什么不奖励长度/格式" → 会 hack 出表演性长文本；R1 只在结果正确性与语言一致性上给信号。
- 追问"冷启动为什么必要" → 修可读性与语言混杂；R1-Zero 证明不修也能涨分但不可用。
- 追问"蒸馏为何优于小模型 RL" → 强教师样本是浓缩的探索轨迹；小模型自探索算力低效（官方口径）。
- 易错点：混淆 R1 / R1-Zero（有无冷启动 SFT）；把 aha moment 说成"学出新能力"。

**【知识串联】**
- 03-FineTuning/02-Alignment：RLHF→PPO→DPO 谱系，GRPO=去 critic 的 on-policy 变体；"可验证奖励"与 RLAIF 的分界。
- 本章题 5（与 o1 对比）、题 6（R1 长思考是 test-time scaling 的训练侧对应物）。
- 05-Agent + 研究文档第七章 #10/#11：过程分>能力分的评测趋势。
- 10-RealQuestions 面经 5（大模型算法）、面经 10（推理优化）有 GRPO/RL 变体题。

### 题 3：MoE 的原理

**【考察意图】** 双线题：算法线（gating 公式、Top-K 加权、两级稀疏）+ 工程线（稀疏激活的显存/算力账、Expert Parallelism 通信代价）。二面最好能现场写路由前向代码并标复杂度。

**【深度解析】**

核心思想：**每 token 只激活部分专家**——总参数决定知识容量上限，每 token 激活参数决定计算成本。FFN 占参数大头，MoE 把它换成 E 个专家 + Router。

公式（沿用原文件）：
- 门控：`G(x) = TopK(softmax(x·W_g))`——先对全部 E 个专家 softmax 打分，再取 Top-K；
- 输出：`y = Σ_{i∈TopK} G(x)_i · E_i(x)`——Top-K 之后用打分值直接加权；未选中专家零前向、零梯度（门控分除外）。
- 两级稀疏：token 级（每 token 只算 K 个专家）+ batch 级（token 间路由不同，专家都在干活的学分摊，GPU 利用率不塌）。

复杂度核算（E 专家、中间维 d_ff、模型维 d、序列 T、top-K=K）：
- 参数 O(E·d·d_ff)——**显存不省，算力省**（第一陷阱）；
- 前向浮点 O(T·K·d·d_ff)（同规模稠密为 O(T·E·d·d_ff)，激活比 K/E；V3 口径 37B/671B ≈ 5.5%);
- Router 打分 O(T·E·d)，相对很小；训练时通信 O(T·K·d) 的 all-to-all×2。

可运行参考实现（PyTorch，含 Switch 负载均衡辅助损失；已按 batch- flatten 思路写清）：
```python
import torch, torch.nn as nn, torch.nn.functional as F

class MoELayer(nn.Module):
    """Top-K MoE 层：router(softmax->topk) + 专家加权 + Switch aux loss。
    复杂度：参数 O(E*d*d_ff)；前向 O(T*K*d*d_ff)（K/E 稀疏比）；
    helper 循环是教学版，生产实现按专家分组做 all-to-all 见题4。"""
    def __init__(self, d_model=512, d_ff=2048, n_expert=8, top_k=2, aux_coef=0.01):
        super().__init__()
        self.E, self.K, self.alpha = n_expert, top_k, aux_coef
        self.wg = nn.Linear(d_model, n_expert, bias=False)                # Router: O(T*E*d)
        self.w1 = nn.Parameter(torch.randn(n_expert, d_model, d_ff) * 0.02)
        self.w2 = nn.Parameter(torch.randn(n_expert, d_ff, d_model) * 0.02)

    def expert_ffn(self, x, e):                                           # O(t_e*d*d_ff)
        return F.gelu(x @ self.w1[e]) @ self.w2[e]

    def forward(self, x):                      # x: [B, T, d] -> y: [B, T, d], aux
        B, T, d = x.shape; x2 = x.reshape(-1, d)                          # [BT, d]
        g = F.softmax(self.wg(x2).float(), dim=-1)                        # fp32 路由防数值溢出
        topv, topi = torch.topk(g, self.K, dim=-1)                        # Top-K（DeepSeek 系：K 后不归一化）
        flat_out = torch.zeros_like(x2)
        acc = torch.zeros_like(x2)                                        # Σ_k G(x)_k * E_k(x)
        for k in range(self.K):
            e = topi[:, k]                                                # 每 token 第 k 个专家
            h = torch.stack([self.expert_ffn(x2[i:i+1], e[i])             # 逐 token 调对应专家
                             for i in range(x2.shape[0])]).squeeze(1)     # 慢；向量化见下
            acc = acc + g.gather(1, e.unsqueeze(1)) * h                   # 门控值×专家输出
        flat_out = acc
        return flat_out.view(B, T, d), self.aux_loss(g, topi)

    def aux_loss(self, g, topi):                                          # Switch: alpha*E*Σ f_i*P_i
        f = torch.zeros(self.E, device=g.device)                          # 实际负载（硬选择）
        p = g.mean(0)                                                     # 平均路由概率（软）
        for e in range(self.E):
            f[e] = (topi == e).float().mean()
        return self.alpha * self.E * (f * p).sum()                        # 均衡时=alpha(最小)

# 向量化提示：生产实现用 index_add / one-hot 重排，把同专家 token 聚成批再 einsum；
# 或直接用 torch.stack([x @ w1[e] ...]) —— 上面逐 token 循环用于讲清"稀疏选择"语义。
moe = MoELayer(); x = torch.randn(2, 16, 512)
y, aux = moe(x)                # y: [2,16,512]，aux: 标量，加入总损失训练
```
（讲解版循环为 O(BT·K) 次单样本前向，仅说明语义；生产用"按专家分桶+矩阵乘"达满 GPU 利用率，复杂度仍 O(T·K·d·d_ff)。）

**【30秒答题框架】**
"MoE 的本质是条件计算：Router 对每个 token 对 E 个专家打分 softmax 取 Top-K，输出是这 K 个专家输出按门控值加权求和。总参数 O(E·d·d_ff) 决定知识容量，每 token 只激活 K 个专家，前向 O(T·K·d·d_ff)，激活比 K/E——DeepSeek-V3 是 671B 总参只激活 37B。两个第一反应的坑：一是显存不省算力省，所有专家权重都得常驻；二是 Top-K 是离散操作不可导，靠 softmax 概率与 aux loss 传梯度。所以训练三件事：负载均衡 aux loss、通信 all-to-all、路由稳定性。"

**【追问与陷阱】**
- 追问"为什么 Top-K 后不重新归一化" → 两种约定都存在：Switch 用 Top-1 硬选择（门控值=其 softmax 概率）；DeepSeek 系（V2 起）取 Top-K 后直接用 softmax 打分值加权、未选中路径记零。答的时候先说清楚约定，再讲论文出处差异，别混。
- 追问"K=1 行不行" → Switch 证明可行且最省算力，但表达下降、路由抖动更敏感。
- 追问"MoE 显存省不省" → 不省；追问"那收益在哪" → FLOPs 换容量 + 同算力下更优 loss 墙。
- 追问"Router 打分的梯度" → 离散 Top-K 不可导，门控梯度只沿被选中路径走，靠 softmax 概率间接更新未选中专家；已有 expert-choice（专家选 token）反向变体可天然均衡。
- 易错点：把"bag of experts 同质集成"与 MoE 混淆——MoE 的专家在梯度上被路由分化工序分化，是负反馈耦合不是简单平均。

**【知识串联】**
- 01-Foundation/01-Transformer 的 FFN/SwiGLU（专家即 FFN 变体）；01-Foundation/03-Loss-And-Optimization 的辅助损失与优化器交互。
- 本章题 1（DeepSeekMoE 的细粒度+共享专家 + aux-free 均衡）、题 4。
- 09-SystemDesign 分布式训练并行策略（EP 与 ZeRO/PP/TP 的组合矩阵）。
- 08-Coding/01-Coding-Problems 已有 Attention/KV 实现题，MoE 是同一风格的高频新生。

### 题 4：MoE 训练的挑战

**【考察意图】** 三件事讲到机制：负载均衡（aux loss 会推到最小值=alpha 吗？为什么 V3 转向 aux-free bias？）、通信（EP 的两次 all-to-all、为什么节点受限）、离散路由不稳定。二面重点：能从 f_i·P_i 的定义推出"均衡 ⇒ loss 最小"。

**【深度解析】**

(1) **负载均衡：数学推导（Switch 形式，与原文件公式一致）**。记 E 为专家数、T 为 batch token 量：
- `f_i = (1/T)Σ_t 1[router 选中 i]`（实际负载比例，硬选择，不可导），约束 Σf_i=1；
- `P_i = (1/T)Σ_t p_i(x_t)`（平均路由概率，软值，可导）；
- `L_aux = α·E·Σ_i f_i·P_i`。
推导下界：若完全均衡，f_i=P_i=1/E，则 `L_aux = α·E·E·(1/E·1/E) = α`，为最小值；若全部 token 挤到专家 1（最差），f_1=P_1=1，其余 0，则 `L_aux = α·E·1 = α·E`，是最优的 E 倍。乘积耦合的作用：让"软概率 P"去贴齐"硬负载 f"——哪个专家实际被用得多（f 大），路由器就被压低对它的打分，从而把 token 推去冷门专家。α 取小（Switch 口径 0.01），只在均衡决策上施加轻推力，避免干扰主损失。
- 代价与被淘汰原因：aux loss 直接改写路由梯度 → 主损失受损（被"平衡税"污染）；调 α 两难。

(2) **DeepSeek-V3 的 aux-loss-free 均衡（官方论文口径）**：给每个专家维护偏置 b_i，**只参与 Top-K 选择、不进入门控权重**：`s'_i = s_i + b_i` 仅用于排序取 K，`g_i` 仍用未加偏置的 s_i。每步按负载反馈调整：过载专家 `b_i ← b_i − γ`，欠载 `b_i ← b_i + γ`（γ 从 0.001 余弦退火到 0.0001）。效果：均衡通过"调控选择"而非"改梯度"实现，主损失不被污染。再配一个极小序列级辅助损失（α=0.0001）防单序列极端失衡，及"节点受限 Top-4 路由"。

(3) **通信开销**。Expert Parallelism：E 个专家划分到 P_ep 组设备。每层做两次 all-to-all：
- dispatch：把每 token 的（K 份隐向量）发送到其专家所在设备，量 O(T·K·d)；
- combine：把专家输出加权汇回原 token 所在设备，同量。
占用 whole training step 的通信面不小，所以 V3 用节点受限路由（≤4 节点，省跨机带宽）+ Dual MicroBatch 重叠：一条微批的 Attention/Dispatch 与另一条微批的 MoE/Combine 重叠执行。

(4) **训练不稳定**：路由离散性 → 同一 token 跨步路由漂移（专家"饿死/过热"）；早期路由未成熟时输出分布剧烈波动。业界手段：router 用 fp32 计算、路由输入加抖动/噪声、专家 dropless 训练（不丢 token，靠均衡把 drop 概率压到 0）以及 V3 的 bias 反馈机制。

**【30秒答题框架】**
"三大挑战。一，负载均衡：Switch 的 aux loss 是 α·E·Σ f_i·P_i，f 是硬负载、P 是软概率，均衡时两边都是 1/E、loss 收敛到 α；不均衡最坏放大 E 倍——但它直接污染主损失，V3 因此换成 aux-free bias：偏置只参与 Top-K 选择不进门控权重，过载就减一档 γ，从 0.001 退火到 0.0001，均衡不交税。二，通信：专家并行每层两次 all-to-all，token 的 K 份隐向量 dispatch+combine 各 O(T·K·d)，V3 用节点受限 Top-4 和双微批重叠压带宽。三，稳定性：离散路由不可导、专家会饿死过热，用 fp32 路由、热身、dropless 训练兜底。"

**【追问与陷阱】**
- 追问"α 调大会怎样" → 路由被强制摊平，专家失去专业化，主 loss 上升——"均衡 vs 专业"是超参两难，V3 bias 机制是 2024 的答案。
- 追问 expert-choice 是什么 → 专家挑 top token，天然负载平衡，但 token 侧覆盖不齐（有的 token 没专家挑）。
- 追问"一序列内极不均衡会怎样" → 单序列全走一专家 → 激活局部 OOM/劣化，故加序列级小 α。
- 陷阱：说"aux loss 大点更均衡更好"——均衡不是目的，主损失才是。
- 陷阱：把 all-to-all 与 all-reduce 混淆（后者 TP 梯度用；EP 是 token 分发型集合通信）。

**【知识串联】**
- 01-Foundation/03-Loss-And-Optimization：辅助损失设计原则与权值衰减/调度（γ 余弦退火同源）。
- 本章题 1（V3 的 DeepSeekMoE）、题 3。
- 09-SystemDesign/01-System-Design 的并行策略与 ZeRO：EP×DP×PP 的显存/通信矩阵。
- 02-Inference/01：MoE 推理算子的显存驻留（容量 wall）问题对照。

### 题 5：o1/o3 vs DeepSeek-R1 的区别

**【考察意图】** 对比表题考结构化思维：能否从"训练方法/开放性/推理呈现/成本"四维把差异说清，并指出 o1 是"隐式思考（思考被隐藏，对外只给答案）"而 R1 是"显式 CoT 可见可蒸馏"这一根本区别——它导致 R1 能被蒸馏开源，o1 不能。

**【深度解析】**

| 特性 | OpenAI o1/o3 | DeepSeek-R1 |
|---|---|---|
| 训练方法 | 未公开（官方口径：大规模 RL 验证链，o1 论文只给了 test-time scaling 与 hidden CoT）| 公开：GRPO + 可验证奖励 RL |
| 开源 | 闭源 API | MIT 权重开源（含蒸馏系列） |
| 推理方式 | 隐式思考（hidden chain，对外只呈现答案） | 显式 CoT（thinking token 直接可见） |
| 思考过程可见性 | 隐藏（不可蒸馏为数据） | 可见（可拒绝采样、可蒸馏） |
| 成本 | 高 | 低（官方口径 R1 API 价远低于 o1） |
| 验证窗口 | 大规模内部 RL 数据+评测器 | 规则奖励（数学/代码）+ 少量 RM |

深层解读（三条）：
1. **能力近似、形态不同**：官方口径两者在 AIME/MATH-500 几乎打平（79.8% vs 79.2%、97.3% vs 96.4%），但 R1 的铁三角是"公开配方+开源权重+低成本 API"，研究文档表五将其列为 2025-11 前后"开源权重四线追近闭源三强"的代表事件。
2. **显式 vs 隐式的生态意涵**：显式 CoT 让 R1 成为全行业蒸馏母体——DeepSeek-R1-Distill-Qwen-32B（72.6% AIME）超过 o1-mini（63.6%），学界出现 R1 或 R1-style 模型的复现潮；o1 的 hidden CoT 在相当一段时间内不可用于生成训练数据。
3. **可验证奖励 vs 隐式评测器**：R1 的 RL 信号几乎全是规则可验证；o1 官方未公开，但外界普遍推断其依赖大规模过程奖励/评审器。该差异也决定了 R1 的蒸馏路径（下一节追问）。

**【30秒答题框架】**
"四维对比。训练方法：o1 未公开，只知是大 RL；R1 全公开——GRPO+B 可验证奖励。开放性：o1 闭源 API；R1 MIT 开源、连蒸馏版全给。推理形态：o1 是隐式思考，对外只给答案；R1 的 CoT 显式可见。结果：R1 能力接近 o1（AIME 79.8 对 79.2 官方口径）但被整个社区蒸馏——R1-Distill-Qwen-32B 72.6% 超 o1-mini 的 63.6%。核心洞察：显式思考让能力可复制，这是 R1 对生态的结构性影响；o1 的壁垒在数据与评测器，R1 的壁垒在配方与成本。"

**【追问与陷阱】**
- 追问"o1 的 hidden CoT 为什么藏" → 一是防止蒸馏威胁自身，二是 hidden CoT 若直接可读会带来监督与奖励作弊倾向。
- 追问"哪个更强" → 分场景答：数学/代码竞赛类 R1 论文口径追平或超；通用写作/多语言 o1 更稳（R1 中文中英混杂曾是问题）。
- 追问"R1 之后为什么各家都能低成本复现 o1 式模型" → 可验证奖励域是通用配方：KL-GRPO + 规则奖励 + 冷启动/蒸馏。
- 陷阱：说"R1 是 o1 的开源版"——能力层可这样比喻，但训练配方/工程路线（无 PRM、无 MCTS）完全不同；R1 走的是结果奖励的纯 RL，不是过程奖励模型。
- 陷阱：价格数字——引用研究文档表五口径（R1 系列 API 人民币口径另有标注），别背旧闻。

**【知识串联】**
- 03-FineTuning/02-Alignment：RLHF/PPO/DPO/GRPO 全谱系。
- 本章题 2（R1 五段详解）、题 6（test-time compute 的对比面）。
- 研究文档表五：2025-11 代际成绩（Opus 4.5 80.9%/Codex-Max 77.9%/Gemini 3 Pro 76.2%，厂商口径）及开源追近叙事。
- 10-RealQuestions 多份面经以"o1 vs R1 差在哪"为引子引出 RL 推理题。

### 题 6：Test-Time Compute 是什么？

**【考察意图】** 是否跟上 2024-2025 的范式转移：把"推理期算力"当作与预训练/后训练并列的第三条 scaling 维度；能否讲清四类方法（Best-of-N / 长思考 / 自检验 / 搜索）的作用机制与开销曲线，并解释 o1/R1 是"把 TTS 内化成长 CoT"的自适应版本。

**【深度解析】**

定义：推理期给单个问题投入更多算力换取更高正确率。四条路线（原文件四分类展开）：

1. **Best-of-N / verifier 结合**：采样 N 条候选，用奖励模型/规则验证挑最优。数学结构：若单条通过率为 p 且候选独立，N 条全错概率 (1−p)^N → 成功概率 `1−(1−p)^N`；上限受 verifier 质量约束（RM 打分排名不准时 BoN 增益封顶）。compute-optimal 的结论（等价于 Snell 等 2024 官方口径）：均匀 Bo-N 浪费算力，把预算按"难题多采样、易题少采样"再分配可显著提升帕累托前沿。
2. **长思考（long CoT）**：让模型在标签后反复自我修辞性展开——R1 训练里 response 长度自发增长就是模型把 test-time 算力参数化为 token 数；Chinchilla 式的"预算—性能"换算从预训练 tokens 转移到"思考 tokens"。注意：长 CoT 增加的是**顺序算力**（serial），BoN 是**并行算力**（parallel），二者互补（不同错误模式的恢复途径）。
3. **自我验证 / 反思**：生成→检查→修正的循环（Reflexion 族）。有效条件是"验证比生成容易"（零知识的 self-check 只有当模型能可靠识别自己错误时才涨点，否则越反思越偏——过度思考现象）。
4. **搜索式策略**：Tree-of-Thought/MCTS 在推理树做扩展-评估-回溯；问题：需要可靠的值函数（PRM/verifier）分支评估，弱 verifier 下搜索易被"评分黑客"带偏。

量化肌理（面试可给的账）：难题上 BoN=64 相当于把有效通过率从 p 抬到 1−(1−p)^64，但 token 成本 ×64；而 o1/R1 式自适应长思考把"多想"变成条件行为：模型自己决定给难题分配几百到几万 token 的思考预算，成本-性能曲线更平滑。佐证：行业共识从"固定 N"走向"难度自适应的预算分配"（compute-optimal TTS 的工程化）。

**【30秒答题框架】**
"Test-time compute 是第三条 scaling 轴：预训练烧数据、后训练烧偏好，推理期烧算力换正确率。四类方法：BoN+ verifier、长 CoT、自我验证反思、树/搜索。关键数学：N 条独立候选成功率 1−(1−p)^N，但受 verifier 上限约束，所以 compute-optimal 分配优于均匀 BoN。o1/R1 的意义是把 test-time scaling 内化成模型自带的能力——R1 训练中 response 长度自发增长，模型学到'难题多想'的条件行为，等价于把 BoN 的 bag 窗口换成连续思考预算。落地注意：验证比生成容易时自反思才有效，弱 verifier 下搜索会被带偏。"

**【追问与陷阱】**
- 追问"BoN 与长思考谁更划算" → 并行 vs 串行：前者并行带宽高但同质采样相关性强、上限被 verifier 锁住；后者能产生与采样路径不同的新思路；结论：混合（先用长思考，残差错误用 BoN）。
- 追问"格式奖励会不会让长 CoT 表演" → 会；所以 R1 不奖励长度。
- 追问"如何在工程里做 TTS 预算" → 难度路由分类器 + per-request 预算上限 + 分级 verifier（规则→RM→人审）（可联动 09-SystemDesign 的推理服务）。
- 陷阱：把 TTS 说成"随机起跑多次取平均"——正确表述是算力-性能的帕累托曲线重构，不是 ensemble 求平均。
- 陷阱：忘给数字来源——举 1−(1−p)^N 请说明是独立假设下的理论形式；实际同模型采样相关，增益低于理论值。

**【知识串联】**
- 02-Inference/02-Decoding-Strategies：BoN/beam/采样三者关系；speculative decoding 的"验证制胜"同构。
- 本章题 2/5：R1 的长 CoT 是 TTS 内化；05-Agent 的 ToT/规划（题 17 同源）。
- 06-Safety-Evaluation：过度反思与 verifier 漏洞对应的安全面。
- 09-SystemDesign：预算控制 /难度路由/降级路径。

### 题 7：如何扩展模型的上下文窗口？

**【考察意图】** 三层递进：位置编码外推的数学（PI/NTK/YaRN 的插值域选择）、训练路线（分阶段渐进 + 长文本数据配比）、系统层（Ring/Striped Attention 的显存摊销）。满分要求能从 RoPE 频谱角度讲"为什么直觉插值毁局部、NTK 毁长程、YaRN 分频段混合"。

**【深度解析】**

(0) 问题本质：RoPE 第 j 对维度的旋转频率 `θ_j = b^{−2j/d}`（b=10000），波长 `λ_j = 2π/θ_j = 2π·b^{2j/d}`——从 2π（快档）到 `2π·b^{(d−2)/d}`（慢档）跨多个量级。外推失败的本质：**超出训练长度的相对位置对高频维度是"没见过"的相位组合**，注意力 logits 失真。

1. **PI（线性插值）**：把位置 m 压为 m/s（s=目标倍数）。效果：高频维度相位不再越界（全落在训练分布内），但**相邻 token 的相位差被等比例压缩 → 局部分辨率退化**（旋转一次的"距离感"变小），短程注意力变钝。
2. **NTK-aware（改基底不动位置）**：`b' = b·s^{d/(d−2)}`（等价于对频谱做非线性重排）。推导：改基后第 j 对波长 λ_j' = 2π·b'^{2j/d} = 2π·b^{2j/d}·s^{2j/(d−2)}，即 λ_j'/λ_j = s^{2j/(d−2)}——j=0（最快档）比值 1、j=d/2−1（最慢档）比值恰为 s：**低频（j 大）波长被抬得更多，高频几乎不动**。收益：局部分辨率保留；风险：中低频维度实质处于"外推"状态，从未见过的相位组合在长程仍会失真（YaRN 论文口径：NTK 式外推在中频段失效）。
3. **YaRN（分频段 + 注意力温标）**：按波长 r_j 分派：
   - 高频段（λ 明显小于训练长度 L）：给线性插值（保分布不越界）；
   - 低频段（λ 超过伸长后窗口）：不插值、保留原分辨率（NTK 风格）;
   - 中间段：ramp 函数线性混合γ（过渡带）；
   - 再乘注意力 softmax 温度 `~1/√(0.1·ln s + 1)` 补偿插值带来的 logits 熵放大。
   官方论文口径：YaRN 以约 0.06% 的预训练算力（数百步微调）即可把 RoPE 模型上下文从 4K 扩展并在 needle 检索上稳定到 128K 窗口；Qwen 系列的 128K 上下文（研究文档表五：Qwen3 主流 32K 原生/YaRN 128K，"256K 仅部分新型号"标注存疑）即 YaRN 一系工程化。
4. **训练路线**：长文本继续预训练（混入长依赖数据，Qwen 系官方口径 32K→128K 分阶段）+**渐进式**（先在 64K 学稳再上 128K/256K），配合数据课程（长文档、代码仓、多文档拼接）。
5. **系统层注意力优化**：
   - **Ring Attention**：序列切块分发到 P 个设备，每设备对自己的 N/P 个 query 与环形轮转来的 KV 块算部分注意力，同时通信与计算重叠——单设备注意力显存从 O(N²) 降到约 O(N²/P)，超长序列训练才可行（百万级 token）；
   - **Striped Attention**：把块在设备间错位（stripe）分配，修因果掩码导致的计算量不均衡（后半段 device 严重过载的负载倾斜）。
6. **部署配套**：MLA/GQA 稀疏化 KV（题1）、KV 分页（PagedAttention）、近似稀疏注意力（DeepSeek-V3.2 的 DSA，官方名称 DeepSeek Sparse Attention，官方口径 128K 上下文/低 API 价——研究文档表五）。

**【30秒答题框架】**
"三层。数学层：RoPE 每对维度频率 θ_j=b^{−2j/d}，外推失败是高频段相位出训练分布。PI 等比压缩位置，局部分辨率塌；NTK 改基 b·s^{d/(d−2)}，低频波长抬更多、局部保住但长程仍是外推；YaRN 分频段——高频插值、低频保留、中间 ramp，再加 0.1·ln s 温标修 logits 熵，官方口径 400 步换 4× 上下文。训练层：混长数据+渐进窗口（32K→128K）。系统层：Ring Attention 把 O(N²) 摊到 P 设备、Striped 修因果掩码负载不均，配上 MLA/DSA/PagedAttention 落地。最后一句：窗口≠能力——needle-in-haystack 通过≠长程推理，要自建长程评测。"

**【追问与陷阱】**
- 追问"YaRN 为什么还要温度因子" → 插值使 logits 尺度整体缩水，注意力熵变大；温度因子近似恢复熵（官方口径 0.1·ln(s) 形式）。
- 追问"为什么不用 ALiBi" → ALiBi 是直线距离惩罚，外推平滑但对相对位置的精细表达弱于 RoPE+全频率谱；RoPE 系生态成熟（旋转实现、YaRN 工具链）。
- 追问"512K/1M 窗口的真实收益" → 上下文瓶颈转移：KV Cache 显存、注意力 O(N²) 算力、**长上下文利用效率（context rot，研究文档考点 #6）**；对比 RAG：超长上下文抬高单请求成本，RAG 可控检索成本与可审计性更优（04-RAG）。
- 陷阱：说"YaRN=NTK 插值"——遗漏分频段与温标两处。
- 陷阱：混淆"训练时上下文"与"推理时窗口支持"——外推段的有效效率远低于训练窗口内，别用窗口上限当有效能力。
 
**【知识串联】**
- 01-Foundation/01-Transformer：RoPE 推导、位置编码对比。
- 本章题 1（MLA=窗口越大 KV 越省越多）、题 8（ViT 位置编码也受同问题）。
- 04-RAG：长上下文 vs RAG 的成本/可审计性权衡；Agentic RAG 多跳检索。
- 02-Inference/01-KVCache：PagedAttention/MQA/GQA 与长上下文的联合账本。

### 题 8：Vision-Language Model 架构

**【考察意图】** 三件套（Encoder→Projector→LLM）能否讲到组件选型理由：为什么用对比学习预训练的 ViT/SigLIP、投影层各形态（Linear/MLP/Q-Former）的取舍、视觉 token 数与 LLM 上下文的预算矛盾。进阶点：原生分辨率/动态分块与"任何模态统一"趋势。

**【深度解析】**

架构（原文件图展开）：

```
图像 → Vision Encoder (ViT/SigLIP) → Projector → ┌ LLM（与文本 embedding 同一序列）→ 文本输出
文本 → Tokenizer → Embedding ─────────────────────┘
```

(0) **Vision Encoder**：ViT 把图切 patch（如 14×14），经 [CLS]+patch tokens 出特征；训练用**对比学习**：CLIP（InfoNCE 双向对比）与 SigLIP（sigmoid 损失，去掉全批 softmax 配对需求，批内效率更高、大批次更稳）。SigLIP×LLaVA 式组合是当前开环主流底座（官方口径：LLaVA-1.5 用 CLIP，LLaVA-NeXT/InternVL/Qwen-VL 多转 SigLIP 族）。
(1) **Projector（对齐层）**三种形态：
- Linear/MLP：LLaVA 路线，直观高效——LLM 本身多模态要求数据规模，简单投影即可；
- Q-Former（BLIP-2）：可学习 query 把千级 patch 压到 32/64 token——省上下文，但信息瓶颈与即席训练复杂；
- Cross-attention 注入（Flamingo 门控 + gated xattn）：encoder 冻结、适配参数省，但 LLM 与视觉耦合深、社区后续减少。
取舍主线：**视觉 token 数 vs 信息保真**——分辨率升、token 数平方涨，多图/视频很快吃满窗口。缓解：pixel unshuffle/平均池化压 token（InternVL）、AnyRes 动态分块（LLaVA-NeXT：原生分辨率切块+缩略图全景）、Qwen-VL 的 NaViT 式原生任意长宽比。
(2) **LLM 侧**：文本 tokenizer 不变，视觉 token 直接进同一 embedding 空间；训练三段式——a) 对齐预训练（图文对，只训 projector），b) 指令微调（视觉指令数据），c) （可选）SFT/RLHF。开源代表（原文件列表）：LLaVA、Qwen-VL、InternVL；闭源代表 GPT-4V/4o（4o 为 natively multimodal 早训流）。
(3) 趋势锚点：多模态统一（一模型跑文本+图像+音频，研究文档趋势 #4——Qwen2.5-Omni/4o 式 any-to-any）；"视觉即工具"的 Agent 化（Computer Use 截屏-分析-点击循环，研究文档表三；本仓库 05-Agent 的 MCP 生态）。

**【30秒答题框架】**
"VLM 三件套：对比学习预训练的 ViT/SigLIP 出视觉特征，Projector 把视觉 token 映到 LLM 输入空间，与文本共用一条解码链。SigLIP 相比 CLIP 把 InfoNCE 换成 sigmoid 对，大批次更稳。投影层三选一：Linear/MLP 简单有效（LLaVA），Q-Former 用 query 压 token（省上下文但有瓶颈），Cross-attention 让 LLM 深耦合。核心账是视觉 token 预算：分辨率上 token 平方涨，用 AnyRes 动态分块或原生分辨率解决。训练三段：对齐、指令微调、偏好对齐。趋势是 any-to-any 统一模型与视觉接入 Agent 循环。"

**【追问与陷阱】**
- 追问"为什么不用端到端从头联合训练" → 视觉-文本配对数据量远小于文本，端到端联合预训练用图文对预算很难追平 LLM 语言能力；冻结 encoder + 组合式训练是算力约束下的最优解。
- 追问"vision token 数怎么算" → (H/14)·(W/14) 个 patch，224² 图 =256 token；672² = 2304——问 token 预算时能顺手算出。
- 追问"为什么 encoder 用 SigLIP 而不是监督分类头" → 特征要服务于"与语言空间对齐"，对比学习学出语义 embedding 而非离散类标。
- 陷阱：把 Q-Former 讲成必选项——当前主流开源系多数回到 MLP/Linear。
- 陷阱：把 GPT-4o 当"detector+LLM"——官方口径 4o 是原生多模态联合训练，无外挂编码器描述。

**【知识串联】**
- 01-Foundation/01-Transformer：ViT=patch+class token 的 Transformer，与文本同构；04-RAG 的多模态检索（CLIP 向量池）。
- 本章题 7：视觉 token 挤占长上下文窗口 → YaRN/MLA 同一个预算问题。
- 05-Agent/研究文档表三：Computer Use=VLM×Agent 的产品化（Skyvern/browser-use/Operator）。
- 10-RealQuestions 面经 2/面经 11 的多模态配置题。

### 题 9：2025-2026 趋势（原文件 6 条清单 + 模型代际对比）

**【考察意图】** 开放辨析题：能否把"现象清单"升级成"带证据与日期的产业叙事"，并对每条给出自己（而非复读）的判断。二面重点：趋势必须能落到"选型与设计含义"上；引用数字必须标注口径。

**【深度解析】**

逐条（结合研究文档事实核查叙事，2025-11 口径、厂商口径一律标注）：

1. **Agent 生态爆发——MCP 标准化、框架成熟**。时线（研究文档）：2024-10 Computer Use 公测、2024-11 MCP 开源 → 半年到一年内成为跨厂商事实标准（"Agent 的 USB-C"）；2025 年 Claude Code（2025-02 research preview）、Codex CLI（2025-04 开源、后 Rust 重写）、Gemini CLI（2025-06）等终端 Agent 密集落地；M×N 集成问题被压成各自实现一次 MCP。考点迁移：05-Agent 题 5（MCP 主机/客户端/服务端三层、tools/resources/prompts 原语）、多智能体取舍（研究文档表六：生产选 LangGraph、OpenAI 栈选 Agents SDK）。判断（个人观点位）：MCP 是协议层标准化，但记忆/上下文规范仍在演进，跨系统记忆标准未统一。
2. **推理模型持续提升**：R1 证明"可验证奖励+GRPO"可开源复刻 o1 后，这条线在 2025-11 代际成为旗舰标配——Kimi K2 Thinking（2025-11，更长上下文）、GPT-5.1-Codex-Max（官方口径 30+ 小时级连续自主任务）。判断：能力增量集中在"长时程可靠性+压缩式上下文管理"（compaction+checkpoint），而非单点 IQ。
3. **端侧部署**：开源权重 + vLLM/Ollama 使"个人本地 Agent"常规化（研究文档：本地候选 DeepSeek/Qwen/Kimi 三线）；核心矛盾是"个性化数据在本地隐私 vs 云端能力"，端侧小模型 + 本地记忆（mem0/Letta 自托管，研究文档表四）是 2026 主线叙事之一。
4. **多模态统一**：见题 8（any-to-any，4o/Omni 流）。
5. **合成数据**：两条证据链——R1 蒸馏出的 800k 样本反向赋能小模型（模型 outputs 成为新训练数据）、模拟环境（agent 轨迹）自助生成；风险：分布坍缩与 RM/verifier 自指（可联动 06-Safety-Evaluation）。
6. **World Models**：从"理解物理世界"到"Agent 训练环境"——可交互的仿真世界是长时程 RL 的奖励供给（Genie 系/游戏环境），与题 2 的可验证奖励同构：世界本身即 verifier。

**模型代际对比（2025-11，官方口径汇总，研究文档表五）**：

| 模型 | 上下文(约) | SWE-bench Verified | 价格（厂商口径） | 生态定位 |
|---|---|---|---|---|
| Claude Opus 4.5 | 200K | 80.9% | API 价较 4.1 的 $15/$75 下调至 $5/$25 | Agentic 编码判断第一梯队 |
| GPT-5.1 / Codex-Max | 400K(API) | 77.9% | 订阅生态 | 30+ 小时长自主任务 |
| Gemini 3 Pro | 1M | 76.2% | CLI/IDE 免费入口 | 超长上下文与多模态 |
| DeepSeek-V3.2 | 128K | — | $0.28/$0.42（缓存命中 $0.028，2025-09 起口径） | 开源+低价+DSA 稀疏注意力 |
| Qwen3 系列 | 32K 原生/YaRN 128K（256K 仅部分新型号，存疑） | — | 开源 | 中文与本地部署 |
| Kimi K2 (Thinking) | 128K（K2 Thinking 支持更长上下文） | — | 开源 | 超长文档/写作、MoE 万亿总参 |

补充轨迹锚点（研究文档）：SWE-bench 头部成绩 2023-10 个位数→2024 中 33%–49%→2025 年中约 70%→2025-11 76%–81%，两年量级跃迁；评测重心从能力分转向安全与过程分（AgentDojo/τ²-bench/OSWorld，研究文档第五章）。

**开源 vs 闭源五场景决策（把趋势落到选型）**：
- 纯编码/Agentic 编码：Opus 4.5（判断质量）或 Codex-Max（长任务+订阅便利）；证据：SWE-bench Verified 分数（厂商口径）。
- 长时程任务：Codex-Max 官方口径 30+ 小时连续自主 + 压缩/检查点机制；超长文档场景换 Gemini 3 Pro 的 1M 上下文。
- 超长上下文/多文档：Gemini 3 Pro 1M（官方口径）。
- 极致成本/高并发：DeepSeek-V3.2（$0.028 缓存命中文本价，厂商口径）——开源权重可自托管。
- 中文与本地化：Qwen3/Kimi K2（研究文档表二/表五）。
提示句：趋势判断的"防坑"——厂商宣传口径需打折（30 小时、数百万用户为厂商口径；研究文档附录把 5 处标注存疑）。

**【30秒答题框架】**
"六条趋势我给三条硬证据：一是 MCP 从 2024-11 开源到 2025 年变成跨厂商事实标准，Agent 生态从手写适配进协议化；二是推理模型从 R1 的可验证奖励配方变成 2025-11 旗舰标配，同期 Opus 4.5/Codex-Max/Gemini 3 Pro 全部主打 Agentic，SWE-bench Verified 两年从个位数到 76–81（厂商口径）；三是开源四线（DeepSeek/Qwen/Kimi+平价 API）把'个人本地 Agent'变成常规选项。选型口诀：编码看 Opus 4.5 或 Codex-Max、超长上下文 Gemini 1M、成本 DeepSeek-V3.2 缓存命中 $0.028、中文/本地 Qwen/Kimi——数字一律标注厂商口径，以官方文档为准。"

**【追问与陷阱】**
- 追问"MCP 为什么能赢" → 谜底三点：开放协议 + 三原语（tools/resources/prompts）+ 大厂首发带节奏；与 function calling 的关系：MCP 是分发标准、FC 是模型层调用机制，互补非替代（05-Agent 题 5）。
- 追问"多智能体什么时候用" → 并行探索/上下文隔离/异构分工；代价是成本倍增+失败级联+handoff 信息丢失（研究文档考点 #9）。
- 追问"端侧模型怎么选" → 隐私/延迟/成本 vs 能力折损；要点是"小模型+本地记忆+云端升级路由"。
- 陷阱：把 2025-11 的具体数字当长期事实——研究文档明示"迭代快、数字有效期短"。
- 陷阱：趋势题只报菜名不落选型——每条都要给"所以工程上该做什么"。

**【知识串联】**
- 05-Agent/01-Agent-Complete-Guide（MCP/多智能体/记忆）、09-SystemDesign（Agent 系统可靠性）、06-Safety-Evaluation（注入与评测）、10-RealQuestions 面经 4/6（Agent 平台/国际化）。
- 本章题 1/2/5/6/7 提供全部技术底层；研究文档表一~表六是工具/模型选型的主索引。

---

## 本章高频追问 Top5

1. **MLA 为什么必须解耦 RoPE？吸收技巧具体吸收了什么？**（题1）——高频最高追问，答不出吸收形式基本判二面挂；从 (q^C)^T·W_UK·c 的结合律讲起，位置矩阵卡在中间就吸收不了。
2. **GRPO 相比 PPO 省了什么、为什么组内归一化可行？何时失效？**（题2/5）——失效点：组内回答全对/全错（std→0，优势退化）时需要 G 足够大或加温度。
3. **MoE 显存不省算力省，那"大而稀"的真实收益与隐藏坑是什么？**（题3/4）——收益是同 FLOPs 下 loss 墙更低；坑是 EP 通信、aux loss 污染主损失、路由崩塌。
4. **上下文从 32K 扩到 1M，瓶颈转移到哪里？YaRN 之外还需要什么？**（题7）——瓶颈外移：KV 显存（MLA/DSA/Paging）、注意力算力（Ring）、长上下文利用效率（context rot）与"窗口≠能力"的评测。
5. **2025-11 三家旗舰爽点各是什么？给一个五场景选型决策树。**（题9）——Opus 4.5 判断力（80.9% SWE-bench Verified）/Codex-Max 长时程（77.9%、30+h）/Gemini 3 Pro 长上下文（76.2%、1M）/DeepSeek-V3.2 成本（$0.028 命中价）/Qwen-Kimi 中文本地——全部标注厂商口径（研究文档表五），并说明"数字有效期短，以官方文档为准"。

> 详解答口径备注：文中所有 2025-11 代际数字与价格来自研究文档表五（厂商口径）；DeepSeek 训练/机制数字（93.3%、671B/37B、14.8T、2.788M GPU 时、接受率 85%–90%、AIME 79.8% 等）来自官方论文口径；R1/Distill 数字（72.6% vs o1-mini 63.6%、四级流水线）同样以论文为准。研究文档附录的"存疑 5 处"提示：趋势类数字随时间失效，面试中先给机制再给数字。





