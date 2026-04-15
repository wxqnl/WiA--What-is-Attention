# DeltaNet

> 对应论文：`paper/DeltaNet-Parallelizable-Linear-Recurrence.pdf`（Yang et al., NeurIPS 2024）
>
> 核心问题：标准 Attention 的 $O(N^2)$ 复杂度让长序列训练代价高昂，而现有线性 Attention 的"只加不删"更新方式又导致记忆容量有限。DeltaNet 用 Delta 规则重写记忆更新逻辑，并通过 Householder 矩阵分解实现跨序列长度的并行训练。

---

## 1. 背景：为什么需要 DeltaNet

### 1.1 标准 Attention 的 $O(N^2)$ 瓶颈

在 Transformer 里，每个 token 都要和序列中所有其他 token 计算相关性。对于长度为 $N$ 的序列，这意味着要计算 $N \times N$ 的注意力矩阵：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

计算量是 $O(N^2 d)$，显存占用也是 $O(N^2)$。当序列长度从 2K 增长到 32K，计算量增长 256 倍。这在长文档处理、代码理解、长对话等场景下是真实的工程瓶颈。

除了计算量，推理时还有另一个问题：**KV Cache**。自回归生成时，每生成一个新 token，都需要把它的 Key 和 Value 存起来，以便后续 token 查询。KV Cache 的大小随序列长度线性增长，在长序列推理时会占用大量显存。

### 1.2 线性 Attention：把注意力改写成 RNN

**线性 Attention** 的核心思路是：把 softmax 替换掉，用一个特征映射 $\phi$ 来近似，从而把注意力改写成矩阵值隐藏状态的递推形式。

标准 softmax attention 在时刻 $t$ 的输出是：

$$
o_t = \sum_{i=1}^{t} \frac{\exp(k_i^\top q_t)}{\sum_{j=1}^{t} \exp(k_j^\top q_t)} v_i
$$

线性 Attention 把 $\exp(k_i^\top q_t)$ 替换为 $\phi(k_i)^\top \phi(q_t)$，于是分子可以提取公因子：

$$
o_t = \frac{\left(\sum_{i=1}^{t} v_i \phi(k_i)^\top\right) \phi(q_t)}{\left(\sum_{i=1}^{t} \phi(k_j)^\top\right) \phi(q_t)} = \frac{S_t \phi(q_t)}{z_t^\top \phi(q_t)}
$$

其中 $S_t = \sum_{i=1}^{t} v_i \phi(k_i)^\top \in \mathbb{R}^{d \times d}$ 是一个矩阵值隐藏状态，$z_t = \sum_{i=1}^{t} \phi(k_i) \in \mathbb{R}^d$。

这个隐藏状态 $S_t$ 可以递推更新：

$$
S_t = S_{t-1} + v_t k_t^\top
$$

（这里用恒等映射 $\phi(x) = x$ 简化，忽略分母归一化项。）

这样，线性 Attention 就变成了一个 **RNN**：推理时只需维护固定大小的矩阵 $S_t \in \mathbb{R}^{d \times d}$，不需要 KV Cache，推理复杂度降为 $O(1)$（每步只做矩阵更新）。

| 对比项 | 标准 Attention | 线性 Attention |
|:---|:---|:---|
| 训练复杂度 | $O(N^2 d)$ | $O(N d^2)$ |
| 推理 KV Cache | $O(N d)$，随序列增长 | $O(d^2)$，固定大小 |
| 推理每步复杂度 | $O(N d)$ | $O(d^2)$ |
| 长距离记忆能力 | 强（直接查询历史） | 弱（信息压缩进矩阵） |

### 1.3 线性 Attention 的致命弱点：只加不删

线性 Attention 的更新规则 $S_t = S_{t-1} + v_t k_t^\top$ 是**纯加法**的。每个新的 key-value 对都被直接加进隐藏状态，旧的关联永远不会被主动清除。

当序列长度 $L$ 远大于隐藏状态维度 $d$ 时，矩阵 $S_t$ 里存储的信息会越来越"拥挤"，新旧信息相互干扰，导致**关联记忆（associative recall）**能力下降——模型无法准确地"查出"之前存入的某个特定 key 对应的 value。

这正是 DeltaNet 要解决的问题。

---

## 2. 核心机制：Delta 规则

### 2.1 直觉：先擦后写

可以把线性 Attention 的隐藏状态 $S_t$ 理解成一块**黑板**，每个 key-value 对是一条笔记。

线性 Attention 的做法是：每来一条新笔记，直接写上去，旧笔记永远留着。黑板写满之后，新旧笔记叠在一起，根本看不清。

DeltaNet 的做法是：写新笔记之前，先**擦掉**黑板上和这个 key 相关的旧内容，再写入新内容。这样黑板上的信息始终是"最新版本"，不会被旧信息污染。

在数学上，这对应的是**Delta 规则**（也叫 Widrow-Hoff 学习规则），来自在线学习理论。

### 2.2 Delta 规则的推导

DeltaNet 的更新规则是：

$$
S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top
$$

其中 $\beta_t \in (0, 1)$ 是一个可学习的"写入强度"（由输入决定）。

拆开来看这个公式做了什么：

| 步骤 | 计算 | 含义 |
|:---|:---|:---|
| ① | $v_t^{\text{old}} = S_{t-1} k_t$ | 用当前 key 查询旧记忆，得到旧预测值 |
| ② | $v_t^{\text{new}} = \beta_t v_t + (1-\beta_t) v_t^{\text{old}}$ | 在新值和旧值之间插值，得到目标值 |
| ③ | $S_t = S_{t-1} - v_t^{\text{old}} k_t^\top + v_t^{\text{new}} k_t^\top$ | 先擦掉旧关联，再写入新关联 |

步骤③可以合并写成：

$$
S_t = S_{t-1} - \underbrace{v_t^{\text{old}} k_t^\top}_{\text{remove}} + \underbrace{v_t^{\text{new}} k_t^\top}_{\text{write}}
$$

当 $\beta_t = 1$ 时，$v_t^{\text{new}} = v_t$，旧关联被完全替换；当 $\beta_t = 0$ 时，记忆不变，$S_t = S_{t-1}$。

### 2.3 与在线学习的联系

Delta 规则有一个优雅的解释：它等价于对以下在线回归损失做一步梯度下降：

$$
\mathcal{L}_t(S) = \frac{1}{2} \|S k_t - v_t\|^2
$$

梯度是 $\nabla_{S_{t-1}} \mathcal{L}_t(S_{t-1}) = (S_{t-1} k_t - v_t) k_t^\top$，所以：

$$
S_t = S_{t-1} - \beta_t \nabla_{S_{t-1}} \mathcal{L}_t(S_{t-1}) = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top
$$

这意味着 DeltaNet 的每一步更新都在**最小化"用当前 key 查询记忆得到的预测值与真实 value 之间的误差"**。记忆矩阵 $S_t$ 是一个不断被在线优化的"联想记忆"。

相比之下，线性 Attention 对应的是最小化负内积损失 $\mathcal{L}_t = -\langle S k_t, v_t \rangle$，其梯度是常数（不随预测误差变化），这解释了为什么线性 Attention 在关联记忆任务上表现更差。

### 2.4 输出计算

DeltaNet 的输出和线性 Attention 完全相同：

$$
o_t = S_t q_t
$$

用更新后的记忆矩阵直接乘以 query 向量，读出当前时刻的输出。


---

## 3. 并行化：跨序列长度的训练加速

### 3.1 问题：递推天然串行

DeltaNet 的递推公式 $S_t = S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ 是一个**矩阵值 RNN**。

注意这里的更新不是简单的逐元素操作（Hadamard 积），而是矩阵乘法：$S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top)$。这意味着无法直接用 parallel scan（并行前缀扫描）来加速，因为关联算子的代价是 $O(d^2)$ 而不是 $O(d)$。

原始 DeltaNet 论文（Schlag et al., 2021）只有串行实现，无法在现代 GPU 上高效训练。这篇论文（Yang et al., NeurIPS 2024）的核心贡献就是解决这个问题。

### 3.2 关键观察：Householder 矩阵分解

论文的核心洞察是：DeltaNet 的隐藏状态 $S_t$ 可以写成一个**纯加法**的形式：

$$
S_t = \sum_{i=1}^{t} u_i k_i^\top
$$

其中 $u_i$ 是一个"伪 value 向量"，定义为：

$$
u_t = \beta_t \left(v_t - \sum_{i=1}^{t-1} u_i (k_i^\top k_t)\right)
$$

这个形式和线性 Attention 的 $S_t = \sum_{i=1}^{t} v_i k_i^\top$ 完全一样，只是把 $v_i$ 替换成了 $u_i$。

一旦有了所有的 $u_i$（堆叠成矩阵 $\mathbf{U} \in \mathbb{R}^{L \times d}$），输出就可以用线性 Attention 的并行形式计算：

$$
\mathbf{O} = (\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}) \mathbf{U}
$$

其中 $\mathbf{M}$ 是因果掩码矩阵。

问题转化为：如何高效地计算所有 $u_i$，而不需要逐步实例化 $d \times d$ 的隐藏状态矩阵？

论文证明，$u_t$ 的计算等价于对 $v_t$ 应用一系列 **Householder 变换**（广义 Householder 矩阵是形如 $\mathbf{I} - \beta k k^\top$ 的秩一更新矩阵），而这些变换的乘积可以用 **WY 表示**（Bischof & Van Loan, 1987）紧凑地存储在 $O(d)$ 内存中，无需显式实例化 $d \times d$ 矩阵。

### 3.3 分块并行形式（Chunkwise Parallel Form）

实际训练中，论文采用**分块并行（chunkwise parallel）**策略，在完全并行和完全递推之间取得平衡。

把长度为 $L$ 的序列切成 $L/C$ 个大小为 $C$ 的块（chunk）。对于第 $t$ 个块：

**块间递推**（chunk-to-chunk，串行但步数少）：

$$
S_{[t+1]} = S_{[t]} + (\mathbf{U}_{[t]} - \mathbf{W}_{[t]} S_{[t]}^\top) \mathbf{K}_{[t]}
$$

**块内并行**（intra-chunk，完全并行）：

$$
\mathbf{O}_{[t]} = \mathbf{Q}_{[t]} S_{[t]}^\top + (\mathbf{Q}_{[t]} \mathbf{K}_{[t]}^\top \odot \mathbf{M}) (\mathbf{U}_{[t]} - \mathbf{W}_{[t]} S_{[t]}^\top)
$$

其中 $\mathbf{W}_{[t]}$ 和 $\mathbf{U}_{[t]}$ 是块内的辅助矩阵，通过以下公式计算（利用 UT 变换将递推改写为矩阵运算）：

$$
\mathbf{T}_{[t]} = \left(\mathbf{I} + \text{tril}(\text{diag}(\beta_{[t]}) \mathbf{K}_{[t]} \mathbf{K}_{[t]}^\top, -1)\right)^{-1} \text{diag}(\beta_{[t]})
$$

$$
\mathbf{W}_{[t]} = \mathbf{T}_{[t]} \mathbf{K}_{[t]}, \quad \mathbf{U}_{[t]} = \mathbf{T}_{[t]} \mathbf{V}_{[t]}
$$

这里 $\mathbf{T}_{[t]}$ 是一个下三角矩阵，可以用前向替换（forward substitution）高效求逆，避免了 $O(C^3)$ 的矩阵求逆代价。

分块并行形式的复杂度是 $O(LCd + Ld^2)$，步数是 $O(L/C)$。当 $C = L$ 时退化为完全并行形式，当 $C = 1$ 时退化为纯递推形式。实践中 $C$ 取 64 或 128，在次二次（subquadratic）复杂度下实现高 GPU 利用率。

### 3.4 训练并行 vs 推理效率

这是 DeltaNet 最重要的设计权衡：

| 阶段 | 使用形式 | 复杂度 | 特点 |
|:---|:---|:---|:---|
| 训练 | 分块并行形式 | $O(LCd + Ld^2)$ | 高 GPU 利用率，tensor core 友好 |
| 推理（prefill） | 分块并行形式 | 同上 | 批量处理输入序列 |
| 推理（decode） | 纯递推形式 | $O(d^2)$ 每步 | 固定大小状态，无 KV Cache |

训练时用并行形式，推理时用递推形式——这和 RWKV、GLA、Mamba 等线性模型的共同策略一致。DeltaNet 的贡献在于，它是第一个让**带矩阵乘法更新的 RNN**也能享受这种"训练并行、推理递推"优势的模型。


---

## 4. 完整架构：DeltaNet Transformer

### 4.1 整体结构

DeltaNet Transformer 基本沿用 LLaMA 架构，把每个 Transformer Block 中的 Multi-Head Self-Attention 替换为 DeltaNet 层：

```
输入 x
  │
  ├─→ RMSNorm → DeltaNet 层 → + x → RMSNorm → SwiGLU FFN → + x → 输出
```

DeltaNet 层内部的数据流：

```
输入 x
  │
  ├─→ Linear → k（key）
  ├─→ Linear → v（value）
  ├─→ Linear → q（query）
  ├─→ Linear → β（写入强度，经 sigmoid）
  │
  ├─→ Conv1d（短卷积，捕捉局部信息）
  │
  ├─→ Delta Rule 递推/并行计算 → S_t
  │
  └─→ o_t = S_t q_t → RMSNorm → Linear → 输出
```

### 4.2 特征映射与归一化

DeltaNet 对 key 和 query 向量做了特殊处理：

$$
k_t = \frac{\text{SiLU}(\mathbf{W}_K x_t)}{\|\text{SiLU}(\mathbf{W}_K x_t)\|_2}, \quad q_t = \frac{\text{SiLU}(\mathbf{W}_Q x_t)}{\|\text{SiLU}(\mathbf{W}_Q x_t)\|_2}
$$

使用 $L_2$ 归一化（而非原始论文的 $L_1$ 归一化）有一个直观的几何意义：当 $\beta_t = 1$ 时，更新矩阵 $\mathbf{I} - k_t k_t^\top$ 是一个真正的投影矩阵，它把 $k_t$ 方向的信息完全擦除，同时保留其余 $d-1$ 个方向的信息。这实现了更精准的"定向遗忘"。

写入强度 $\beta_t$ 由输入决定：

$$
\beta_t = \sigma(\mathbf{W}_\beta x_t) \in (0, 1)
$$

其中 $\sigma$ 是 sigmoid 函数。$\beta_t$ 接近 1 时，旧关联被强力覆盖；$\beta_t$ 接近 0 时，记忆几乎不变。

### 4.3 混合架构（Hybrid Models）

论文还实验了两种混合架构，将 DeltaNet 层与标准 softmax attention 层结合：

**滑动窗口混合（Sliding Window Hybrid）**：每隔一层插入一个滑动窗口 attention 层（局部精确注意力），其余层用 DeltaNet。这弥补了线性 attention 在局部精确匹配上的不足。

**全局 Attention 混合（Global Attention Hybrid）**：只在第 2 层和第 $N/2 + 1$ 层用全局 softmax attention，其余层用 DeltaNet。仅用 2 层全局 attention 就能显著提升性能。

实验结果显示，这两种混合架构都能超过纯 Transformer++ 基线，说明 DeltaNet 和 softmax attention 的能力是互补的。

---

## 5. 与其他模型的对比

### 5.1 核心对比表

| 模型 | 隐藏状态更新规则 | 训练复杂度 | 推理复杂度 | 关联记忆能力 |
|:---|:---|:---|:---|:---|
| Transformer | 无隐藏状态（直接查询） | $O(N^2 d)$ | $O(N d)$（KV Cache） | 强（精确查询） |
| 线性 Attention | $S_t = S_{t-1} + v_t k_t^\top$ | $O(N d^2)$ | $O(d^2)$ | 弱（只加不删） |
| DeltaNet | $S_t = S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ | $O(N d^2)$ | $O(d^2)$ | 中强（先擦后写） |
| Mamba (SSM) | $S_t = \text{diag}(\alpha_t) S_{t-1} + v_t k_t^\top$ | $O(N d)$ | $O(d^2)$ | 弱（逐元素衰减） |
| GLA | $S_t = S_{t-1} \odot G_t + v_t k_t^\top$ | $O(N d^2)$ | $O(d^2)$ | 中（门控衰减） |
| RWKV | $S_t = e^{-w} S_{t-1} + v_t k_t^\top$ | $O(N d^2)$ | $O(d^2)$ | 弱（固定指数衰减） |

### 5.2 关键区别：更新算子的结构

从"关联 RNN"的统一视角来看，这些模型的区别在于隐藏状态更新中的矩阵 $M_t$：

$$
S_t = S_{t-1} \bullet M_t + v_t k_t^\top
$$

- **Mamba / GLA / RWKV**：$M_t$ 是对角矩阵（逐元素操作），可以用 parallel scan 高效并行，但表达能力有限。
- **DeltaNet**：$M_t = \mathbf{I} - \beta_t k_t k_t^\top$ 是秩一更新矩阵（Householder 矩阵），表达能力更强，但需要特殊的并行化技巧。
- **标准矩阵乘法**：$M_t$ 是任意矩阵，表达能力最强，但每步代价 $O(d^2 n)$，无法高效并行。

DeltaNet 在"表达能力"和"可并行性"之间找到了一个新的平衡点。

### 5.3 实验结果摘要

在 1.3B 参数、100B tokens 的语言建模实验中（Wikitext perplexity，越低越好）：

| 模型 | Wiki PPL ↓ | 零样本平均 ↑ | 状态大小 |
|:---|:---|:---|:---|
| Transformer++ | 16.85 | 50.9 | N/A |
| Mamba | 17.06 | 50.0 | 64x |
| GLA | 17.22 | 50.6 | 256x |
| DeltaNet | 16.87 | 49.5 | 128x |
| DeltaNet + 滑动窗口 | 16.56 | 52.1 | N/A |
| DeltaNet + 全局 Attn | 16.55 | 51.8 | N/A |

DeltaNet 在困惑度上接近 Transformer++，在关联记忆任务（SWDE、SQuAD、FDA）上显著优于 GLA 和 Mamba。混合架构则全面超越 Transformer++ 基线。


---

## 6. 常见混淆问题

**Q：DeltaNet 和标准 Attention 都能做关联记忆，区别在哪？**

标准 Attention 是"精确查询"：给定 query，直接在所有历史 key-value 对中找最相关的，不压缩信息。DeltaNet 是"压缩记忆"：把所有历史信息压缩进一个固定大小的矩阵 $S_t \in \mathbb{R}^{d \times d}$，查询时从这个矩阵里读出。前者记忆容量随序列增长，后者固定。DeltaNet 的优势是推理时内存固定，劣势是记忆容量有上限。

**Q：$\beta_t$ 是超参数还是可学习的？**

$\beta_t$ 是**输入相关的**，由当前 token 的表示通过一个线性层加 sigmoid 计算得到：$\beta_t = \sigma(\mathbf{W}_\beta x_t)$。它不是固定的超参数，而是模型根据当前输入动态决定"这次要写多强"。这是 DeltaNet 相比固定衰减模型（如 RWKV）的优势之一。

**Q：DeltaNet 的训练速度比 Transformer 快还是慢？**

在相同序列长度下，DeltaNet 的训练速度接近 GLA，比 Mamba 慢（因为 Mamba 的状态更新是逐元素的，更简单）。但对于长序列（如 16K tokens），DeltaNet 比标准 Transformer 快得多，因为它避免了 $O(N^2)$ 的注意力矩阵计算。论文图 6 显示，在 H100 上训练 1.3B 模型时，DeltaNet 的吞吐量接近 GLA，显著快于 Mamba。

**Q：为什么用 $L_2$ 归一化而不是 $L_1$？**

$L_2$ 归一化后，$\|k_t\|_2 = 1$，使得 $\mathbf{I} - \beta_t k_t k_t^\top$ 的特征值为 $1$（$d-1$ 个方向）和 $1 - \beta_t$（$k_t$ 方向）。这保证了转移矩阵的谱范数不超过 1，训练更稳定。同时，当 $\beta_t = 1$ 时，$k_t$ 方向的信息被完全投影掉，几何意义更清晰。

**Q：DeltaNet 能处理任意长度的序列吗？**

推理时可以，因为隐藏状态 $S_t$ 大小固定（$d \times d$），不随序列增长。但论文发现 DeltaNet 的**长度泛化**能力有限——在比训练序列更长的序列上性能会下降。这可能是因为 DeltaNet 缺少显式的衰减因子（GLA 和 RetNet 有），导致模型难以学到"随时间遗忘"的行为。加入门控项（如 Gated DeltaNet）可以改善这一问题。

**Q：DeltaNet 和 TTT（Test-Time Training）有什么关系？**

两者都可以理解为"在推理时对隐藏状态做梯度更新"。DeltaNet 的 Delta 规则等价于对在线回归损失做一步 SGD，而 TTT 系列工作（如 TTT-Linear、Titans）则更明确地把这个过程框架化为测试时的梯度下降。DeltaNet 可以看作 TTT 思想的一个早期、高效的实例化。

---

## 7. 完整 PyTorch 实现

下面的代码实现了 DeltaNet 的核心逻辑，包括递推形式（用于推理）和简化的并行形式（用于理解原理）。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 工具函数
# ============================================================

def l2_normalize(x, dim=-1, eps=1e-6):
    """对向量做 L2 归一化，使 ||x||_2 = 1"""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ============================================================
# Delta 规则：递推形式（用于推理，每次处理一个 token）
# ============================================================

def delta_rule_recurrent(q, k, v, beta):
    """
    DeltaNet 的递推实现，适合自回归推理。

    参数：
        q:    (B, T, d)  query 向量（已 L2 归一化）
        k:    (B, T, d)  key 向量（已 L2 归一化）
        v:    (B, T, d)  value 向量
        beta: (B, T, 1)  写入强度，sigmoid 输出，范围 (0, 1)

    返回：
        o:    (B, T, d)  输出向量
    """
    B, T, d = q.shape
    # 初始化矩阵值隐藏状态为零矩阵
    S = torch.zeros(B, d, d, device=q.device, dtype=q.dtype)
    outputs = []

    for t in range(T):
        qt = q[:, t, :]        # (B, d)，当前 query
        kt = k[:, t, :]        # (B, d)，当前 key
        vt = v[:, t, :]        # (B, d)，当前 value
        bt = beta[:, t, :]     # (B, 1)，当前写入强度

        # ① 用当前 key 查询旧记忆，得到旧预测值
        # S: (B, d, d)，kt: (B, d) → v_old: (B, d)
        v_old = torch.einsum('bdn,bn->bd', S, kt)

        # ② 计算目标 value：在新值和旧值之间插值
        # bt * vt + (1 - bt) * v_old
        v_new = bt * vt + (1 - bt) * v_old

        # ③ 更新记忆矩阵：先擦掉旧关联，再写入新关联
        # S = S - v_old ⊗ kt + v_new ⊗ kt = S + (v_new - v_old) ⊗ kt
        delta = v_new - v_old  # (B, d)，需要修正的"误差"
        # 外积：delta ⊗ kt，形状 (B, d, d)
        S = S + torch.einsum('bd,bn->bdn', delta, kt)

        # ④ 读出当前时刻的输出：o_t = S_t q_t
        ot = torch.einsum('bdn,bn->bd', S, qt)
        outputs.append(ot)

    return torch.stack(outputs, dim=1)  # (B, T, d)


# ============================================================
# Delta 规则：简化并行形式（用于理解原理，非最优实现）
# ============================================================

def delta_rule_parallel_naive(q, k, v, beta):
    """
    DeltaNet 的简化并行实现（非硬件最优，仅用于理解）。

    核心思路：
    1. 计算"伪 value"矩阵 U，其中 u_i 是修正后的写入向量
    2. 用线性 Attention 的并行公式计算输出：O = (QK^T ⊙ M) U

    参数：
        q:    (B, T, d)
        k:    (B, T, d)
        v:    (B, T, d)
        beta: (B, T, 1)

    返回：
        o:    (B, T, d)
    """
    B, T, d = q.shape

    # 构造因果掩码（下三角矩阵，包含对角线）
    # mask[i][j] = 1 if j <= i else 0
    mask = torch.tril(torch.ones(T, T, device=q.device))  # (T, T)

    # 计算注意力分数矩阵（未归一化）
    # A[i][j] = k_j^T q_i，表示位置 j 对位置 i 的贡献
    A = torch.bmm(q, k.transpose(1, 2))  # (B, T, T)
    A = A * mask.unsqueeze(0)             # 应用因果掩码

    # 计算伪 value 矩阵 U
    # 这里用简化的迭代方式，实际论文用 UT 变换实现 O(1) 步并行
    # u_t = beta_t * (v_t - sum_{i<t} u_i * (k_i^T k_t))
    U = torch.zeros_like(v)  # (B, T, d)
    KK = torch.bmm(k, k.transpose(1, 2))  # (B, T, T)，k_i^T k_j

    for t in range(T):
        # 累积之前所有 u_i 对当前位置的贡献
        correction = torch.zeros(B, d, device=q.device)
        for i in range(t):
            # u_i * (k_i^T k_t)
            correction = correction + U[:, i, :] * KK[:, i, t].unsqueeze(-1)
        # u_t = beta_t * (v_t - correction)
        U[:, t, :] = beta[:, t, :] * (v[:, t, :] - correction)

    # 用线性 Attention 并行公式计算输出
    # O = (QK^T ⊙ M) U，其中 M 是因果掩码
    O = torch.bmm(A, U)  # (B, T, d)
    return O


# ============================================================
# DeltaNet 注意力层（完整模块）
# ============================================================

class DeltaNetAttention(nn.Module):
    """
    DeltaNet 注意力层，替代 Transformer 中的 Multi-Head Self-Attention。

    主要改动：
    - 用 Delta 规则更新矩阵值隐藏状态，而非 softmax attention
    - 增加 beta 门控，控制每步的写入强度
    - key/query 做 L2 归一化，保证转移矩阵谱范数 ≤ 1
    - 加入短卷积（depthwise conv1d），捕捉局部位置信息
    """

    def __init__(self, d_model, n_heads, conv_kernel=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.d_model = d_model

        # Q、K、V 投影
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)

        # beta 投影：输出标量写入强度，每个头独立
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)

        # 短卷积：在 Q/K/V 投影后加一个 depthwise conv1d
        # 用于捕捉局部 token 间的相对位置信息
        self.conv_q = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_model, bias=False
        )
        self.conv_k = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_model, bias=False
        )

        # 输出投影前的归一化（稳定训练）
        self.out_norm = nn.RMSNorm(d_model)
        # 输出投影
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, use_recurrent=False):
        """
        参数：
            x:             (B, T, d_model)  输入序列
            use_recurrent: bool，True 时用递推形式（推理），False 时用并行形式（训练）

        返回：
            output: (B, T, d_model)
        """
        B, T, _ = x.shape
        h = self.n_heads
        d = self.head_dim

        # ① 线性投影得到 Q、K、V
        q = self.wq(x)  # (B, T, d_model)
        k = self.wk(x)
        v = self.wv(x)

        # ② 短卷积：增强局部位置感知
        # Conv1d 期望输入形状 (B, C, T)，需要转置
        q = self.conv_q(q.transpose(1, 2))[:, :, :T].transpose(1, 2)
        k = self.conv_k(k.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # ③ SiLU 激活 + L2 归一化（key 和 query）
        q = l2_normalize(F.silu(q))  # (B, T, d_model)
        k = l2_normalize(F.silu(k))

        # ④ 计算写入强度 beta，sigmoid 保证在 (0, 1) 范围内
        beta = torch.sigmoid(self.w_beta(x))  # (B, T, n_heads)

        # ⑤ 拆分多头：(B, T, d_model) → (B*h, T, d)
        # 把多头展开到 batch 维度，方便后续计算
        q = q.view(B, T, h, d).permute(0, 2, 1, 3).reshape(B * h, T, d)
        k = k.view(B, T, h, d).permute(0, 2, 1, 3).reshape(B * h, T, d)
        v = v.view(B, T, h, d).permute(0, 2, 1, 3).reshape(B * h, T, d)
        # beta 扩展到 head_dim 维度
        beta = beta.permute(0, 2, 1).reshape(B * h, T, 1)  # (B*h, T, 1)

        # ⑥ Delta 规则计算（训练用并行形式，推理用递推形式）
        if use_recurrent:
            out = delta_rule_recurrent(q, k, v, beta)
        else:
            out = delta_rule_parallel_naive(q, k, v, beta)
        # out: (B*h, T, d)

        # ⑦ 合并多头：(B*h, T, d) → (B, T, d_model)
        out = out.reshape(B, h, T, d).permute(0, 2, 1, 3).reshape(B, T, -1)

        # ⑧ 输出归一化 + 线性投影
        out = self.out_norm(out)
        return self.wo(out)


# ============================================================
# DeltaNet Transformer Block
# ============================================================

class DeltaNetBlock(nn.Module):
    """
    单个 DeltaNet Transformer Block。
    结构：RMSNorm → DeltaNet → 残差 → RMSNorm → SwiGLU FFN → 残差
    """

    def __init__(self, d_model, n_heads, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = DeltaNetAttention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model)

        # SwiGLU FFN：两个并行线性层，一个做门控
        ffn_dim = int(d_model * ffn_mult * 2 / 3)  # SwiGLU 的标准维度缩放
        self.ffn_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.ffn_up   = nn.Linear(d_model, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x, use_recurrent=False):
        # DeltaNet 子层（Pre-Norm + 残差）
        x = x + self.attn(self.norm1(x), use_recurrent=use_recurrent)

        # SwiGLU FFN 子层（Pre-Norm + 残差）
        h = self.norm2(x)
        # SwiGLU：gate 分支做 SiLU 激活，与 up 分支逐元素相乘
        x = x + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x


# ============================================================
# 简单测试
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, d_model, n_heads = 2, 16, 64, 4

    block = DeltaNetBlock(d_model, n_heads)
    x = torch.randn(B, T, d_model)

    # 训练模式：并行形式
    out_parallel = block(x, use_recurrent=False)
    print(f"并行形式输出形状: {out_parallel.shape}")  # (2, 16, 64)

    # 推理模式：递推形式
    out_recurrent = block(x, use_recurrent=True)
    print(f"递推形式输出形状: {out_recurrent.shape}")  # (2, 16, 64)

    # 验证两种形式的输出是否接近（简化实现有数值误差）
    diff = (out_parallel - out_recurrent).abs().mean().item()
    print(f"并行 vs 递推平均误差: {diff:.6f}")
```


---

## 8. 读完这篇之后，你应该能回答这些问题

- 标准 Attention 的 $O(N^2)$ 复杂度来自哪里？KV Cache 是什么，为什么会随序列增长？
- 线性 Attention 如何把注意力改写成 RNN？隐藏状态 $S_t$ 的更新规则是什么？
- 线性 Attention 的"只加不删"问题具体指什么？为什么会导致关联记忆能力下降？
- Delta 规则的更新公式 $S_t = S_{t-1} - \beta_t(S_{t-1}k_t - v_t)k_t^\top$ 中，每一项的含义是什么？"先擦后写"体现在哪里？
- 为什么 DeltaNet 的递推形式无法直接用 parallel scan 并行化？Householder 矩阵分解如何解决这个问题？
- 分块并行形式（chunkwise parallel form）是如何在完全并行和完全递推之间取得平衡的？块大小 $C$ 如何影响计算效率？
- DeltaNet 训练时用并行形式、推理时用递推形式，这两种形式的计算复杂度分别是多少？
- $\beta_t$ 是如何计算的？当 $\beta_t = 1$ 和 $\beta_t = 0$ 时，DeltaNet 的行为分别是什么？
- 为什么对 key 和 query 做 $L_2$ 归一化，而不是 $L_1$？这在几何上意味着什么？
- DeltaNet 在哪类任务上比 GLA 和 Mamba 更有优势？在哪类任务上仍然不如标准 Transformer？

---

## 参考资料

- 原始论文：`paper/DeltaNet-Parallelizable-Linear-Recurrence.pdf`（Yang et al., NeurIPS 2024）
- DeltaNet 原始工作：Schlag et al., "Linear Transformers Are Secretly Fast Weight Programmers", ICML 2021
- FlashLinearAttention 库：https://github.com/fla-org/flash-linear-attention
- WY 表示（Householder 矩阵乘积的紧凑表示）：Bischof & Van Loan, "The WY Representation for Products of Householder Matrices", 1987
- 相关文章：`Attention/GLA.md`、`Attention/RWKV.md`、`Attention/Mamba.md`
