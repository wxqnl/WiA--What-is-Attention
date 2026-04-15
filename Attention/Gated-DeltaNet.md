# Gated DeltaNet

> 对应论文：`paper/Gated-DeltaNet-Improving-Mamba2-with-Delta-Rule.pdf`（Yang et al., 2024）
>
> 核心问题：DeltaNet 用 Delta 规则解决了线性 Attention 的"只加不删"问题，但它的更新规则是**输入无关**的——所有 token 都用同样的方式更新记忆。Gated DeltaNet 引入**输入依赖的门控机制**，让模型能根据当前内容动态决定"记多少、忘多少"，从而在保持 Delta 规则优势的同时，获得类似 Mamba2 的选择性记忆能力。

---

## 1. 背景：DeltaNet 缺少什么

### 1.1 DeltaNet 的成就与局限

在 `DeltaNet.md` 中，我们看到 DeltaNet 通过 Delta 规则实现了"先擦后写"的记忆更新：

$$
S_t = S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top
$$

这个机制让 DeltaNet 在关联记忆任务上远超传统线性 Attention（如 GLA、RetNet），因为它能主动清除旧关联、写入新关联，避免记忆矩阵 $S_t$ 被历史信息"污染"。

但 DeltaNet 有一个关键限制：**$\beta_t$ 虽然是可学习的，但它只依赖位置编码，不依赖输入内容**。换句话说，无论当前 token 是重要信息还是噪声，DeltaNet 都用同样的强度更新记忆。

### 1.2 缺少全局遗忘门

更深层的问题是：DeltaNet 的更新公式里没有一个**全局衰减因子**。

回顾 GLA 的更新规则：

$$
S_t = G_t \odot S_{t-1} + v_t k_t^\top
$$

其中 $G_t \in (0,1)^{d \times d}$ 是一个数据相关的门控矩阵，它控制旧记忆整体保留多少。当 $G_t$ 接近 0 时，旧记忆被大幅遗忘；当 $G_t$ 接近 1 时，旧记忆几乎完整保留。

DeltaNet 没有这样的全局门控。它的"遗忘"只发生在 $k_t$ 方向上（通过 $\mathbf{I} - \beta_t k_t k_t^\top$ 投影），其余 $d-1$ 个方向的旧信息始终以强度 1 保留。这导致两个问题：

1. **长度泛化差**：模型无法学到"随时间自然衰减"的行为，在比训练序列更长的序列上性能下降明显。
2. **噪声积累**：对于不相关的历史信息，DeltaNet 无法主动"清空"，只能等待新的 key 覆盖同一方向。

### 1.3 Mamba2 的启示

Mamba2（也叫 SSD，State Space Duality）提出了一个统一框架，把 SSM 和线性 Attention 联系起来。Mamba2 的核心更新规则是：

$$
S_t = \alpha_t S_{t-1} + v_t k_t^\top
$$

其中 $\alpha_t \in (0,1)$ 是一个**标量**衰减因子，由输入决定。这个简单的标量门控赋予了 Mamba2 强大的选择性记忆能力——当 $\alpha_t$ 接近 0 时，模型几乎完全"重置"记忆，专注于当前输入。

但 Mamba2 没有 Delta 规则的"先擦后写"能力，在关联记忆任务上不如 DeltaNet。

**Gated DeltaNet 的思路**：把 Mamba2 的标量门控和 DeltaNet 的 Delta 规则结合起来，取两者之长。

---

## 2. 核心机制：Gated Delta 规则

### 2.1 直觉：给黑板加一个"整体擦除"旋钮

可以把 DeltaNet 的记忆矩阵 $S_t$ 理解成一块黑板。DeltaNet 的操作是：每次写新笔记前，先精准擦掉和这个 key 相关的旧内容（定向擦除）。

但有时候，我们需要的不是定向擦除，而是**整体淡化**——就像把黑板上所有内容都稍微擦淡一点，为新内容腾出空间。这就是门控因子 $\alpha_t$ 的作用。

Gated DeltaNet 的更新规则是：

$$
S_t = \alpha_t \left( S_{t-1} - \beta_t (S_{t-1} k_t - v_t) k_t^\top \right)
$$

拆开来看：

| 步骤 | 计算 | 含义 |
|:---|:---|:---|
| ① | $v_t^{\text{old}} = S_{t-1} k_t$ | 用当前 key 查询旧记忆，得到旧预测值 |
| ② | $\delta_t = \beta_t (v_t - v_t^{\text{old}})$ | 计算需要修正的"误差"，$\beta_t$ 控制修正幅度 |
| ③ | $S_t' = S_{t-1} + \delta_t k_t^\top$ | Delta 规则：定向更新 $k_t$ 方向的记忆 |
| ④ | $S_t = \alpha_t S_t'$ | 全局衰减：整体缩放记忆矩阵，控制遗忘速度 |

等价地，可以把公式展开写成：

$$
S_t = \alpha_t S_{t-1} (\mathbf{I} - \beta_t k_t k_t^\top) + \alpha_t \beta_t v_t k_t^\top
$$

对比 DeltaNet（无 $\alpha_t$）和 Mamba2（无 Delta 规则）：

$$
\text{DeltaNet:} \quad S_t = S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
$$

$$
\text{Mamba2:} \quad S_t = \alpha_t S_{t-1} + v_t k_t^\top
$$

$$
\text{Gated DeltaNet:} \quad S_t = \alpha_t S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \alpha_t \beta_t v_t k_t^\top
$$

Gated DeltaNet 是两者的严格推广：令 $\alpha_t = 1$ 退化为 DeltaNet，令 $\beta_t = 0$ 退化为 Mamba2。

### 2.2 门控因子 $\alpha_t$ 的计算

$\alpha_t$ 是一个**标量**，由当前 token 的表示通过线性层加激活函数计算：

$$
\alpha_t = \sigma(\mathbf{w}_\alpha^\top x_t + b_\alpha) \in (0, 1)
$$

其中 $\sigma$ 是 sigmoid 函数。实践中，论文发现用 $\text{swish}$ 激活后再做归一化效果更好：

$$
\alpha_t = \frac{\text{swish}(\mathbf{w}_\alpha^\top x_t)}{\|\text{swish}(\mathbf{w}_\alpha^\top x_t)\|_\infty + \epsilon}
$$

这保证 $\alpha_t \in (0, 1)$，同时让梯度更平滑。

$\alpha_t$ 的直觉含义：
- $\alpha_t \approx 1$：当前 token 不需要大幅改变记忆，保留大部分历史（如普通的连接词）
- $\alpha_t \approx 0$：当前 token 标志着一个新话题的开始，需要"清空"旧记忆（如段落分隔符、特殊标记）

### 2.3 写入强度 $\beta_t$ 的计算

$\beta_t$ 的计算方式与 DeltaNet 相同：

$$
\beta_t = \sigma(\mathbf{w}_\beta^\top x_t + b_\beta) \in (0, 1)
$$

$\beta_t$ 控制 Delta 规则的"写入强度"：
- $\beta_t \approx 1$：强力覆盖旧关联，写入新关联（适合重要信息）
- $\beta_t \approx 0$：几乎不更新记忆（适合噪声 token）

### 2.4 输出计算

输出计算与 DeltaNet 完全相同：

$$
o_t = S_t q_t
$$

用更新后的记忆矩阵直接乘以 query 向量，读出当前时刻的输出。

---

## 3. 架构与实现

### 3.1 整体数据流

Gated DeltaNet 层的数据流如下：

```
输入 x  (B, T, d_model)
  │
  ├─→ Linear → q  (query，经 SiLU + L2 归一化)
  ├─→ Linear → k  (key，经 SiLU + L2 归一化)
  ├─→ Linear → v  (value)
  ├─→ Linear → β  (写入强度，经 sigmoid，标量/头)
  ├─→ Linear → α  (全局衰减，经 sigmoid，标量/头)
  │
  ├─→ Conv1d (短卷积，捕捉局部信息，作用于 q 和 k)
  │
  ├─→ Gated Delta Rule 递推/并行计算
  │       S_t = α_t * (S_{t-1}(I - β_t k_t k_t^T) + β_t v_t k_t^T)
  │
  └─→ o_t = S_t q_t → GroupNorm → Linear → 输出
```

### 3.2 与 DeltaNet 的架构差异

| 组件 | DeltaNet | Gated DeltaNet |
|:---|:---|:---|
| 更新规则 | $S_t = S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ | $S_t = \alpha_t S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \alpha_t \beta_t v_t k_t^\top$ |
| 门控参数 | $\beta_t$（写入强度，标量） | $\alpha_t$（全局衰减，标量）+ $\beta_t$（写入强度，标量） |
| 额外参数量 | — | 每头增加一个线性层（极小） |
| 长度泛化 | 弱 | 强（$\alpha_t$ 提供自然衰减） |

### 3.3 分块并行形式（Chunkwise Parallel）

Gated DeltaNet 继承了 DeltaNet 的分块并行训练策略，但需要在块间递推中加入 $\alpha_t$ 的累积乘积。

定义块内的累积衰减因子：

$$
A_{[t]} = \prod_{i \in \text{chunk}} \alpha_i
$$

块间递推变为：

$$
S_{[t+1]} = A_{[t]} \cdot S_{[t]} + \Delta S_{[t]}
$$

其中 $\Delta S_{[t]}$ 是块内的增量更新（由 Delta 规则计算）。

块内并行计算时，需要对每个位置 $i$ 计算从块起始到位置 $i$ 的累积衰减 $\prod_{j=\text{start}}^{i} \alpha_j$，这可以用前缀乘积（prefix product）高效实现。

### 3.4 与 Mamba2 的统一视角

论文指出，Gated DeltaNet 可以看作 Mamba2（SSD）框架的一个扩展。Mamba2 的状态空间对偶（SSD）形式是：

$$
S_t = \alpha_t S_{t-1} + v_t k_t^\top, \quad o_t = S_t q_t
$$

Gated DeltaNet 在此基础上把写入项 $v_t k_t^\top$ 替换为 Delta 规则的修正写入：

$$
S_t = \alpha_t S_{t-1} + \alpha_t \beta_t (v_t - S_{t-1} k_t) k_t^\top
$$

这个视角清楚地展示了 Gated DeltaNet 是如何"改进 Mamba2"的：它保留了 Mamba2 的全局衰减机制，同时用 Delta 规则替换了简单的加法写入，使记忆更新更精准。

---

## 4. 与其他模型的对比

| 模型 | 更新规则 | 全局衰减 | Delta 规则 | 训练复杂度 | 推理复杂度 | 关联记忆 |
|:---|:---|:---:|:---:|:---|:---|:---:|
| 线性 Attention | $S_t = S_{t-1} + v_t k_t^\top$ | ❌ | ❌ | $O(Nd^2)$ | $O(d^2)$ | 弱 |
| GLA | $S_t = G_t \odot S_{t-1} + v_t k_t^\top$ | ✅（矩阵） | ❌ | $O(Nd^2)$ | $O(d^2)$ | 中 |
| Mamba2 (SSD) | $S_t = \alpha_t S_{t-1} + v_t k_t^\top$ | ✅（标量） | ❌ | $O(Nd^2)$ | $O(d^2)$ | 中 |
| DeltaNet | $S_t = S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$ | ❌ | ✅ | $O(Nd^2)$ | $O(d^2)$ | 强 |
| **Gated DeltaNet** | $S_t = \alpha_t S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \alpha_t \beta_t v_t k_t^\top$ | ✅（标量） | ✅ | $O(Nd^2)$ | $O(d^2)$ | 强+ |
| Transformer | softmax attention | N/A | N/A | $O(N^2 d)$ | $O(Nd)$ | 最强 |

**关键观察：**

- Gated DeltaNet 是唯一同时具备全局衰减和 Delta 规则的模型
- 训练和推理复杂度与 DeltaNet、GLA、Mamba2 相同，没有额外代价
- 在关联记忆任务上，Gated DeltaNet 优于所有线性模型
- 在长度泛化上，Gated DeltaNet 优于 DeltaNet（因为有 $\alpha_t$ 提供自然衰减）

---

## 5. 常见混淆问题

**Q：$\alpha_t$ 和 $\beta_t$ 有什么区别？能不能只用一个？**

两者作用完全不同。$\beta_t$ 是 Delta 规则的写入强度，控制"这次要把新关联写多强"——它只影响 $k_t$ 方向的更新。$\alpha_t$ 是全局衰减因子，控制"旧记忆整体保留多少"——它作用于整个记忆矩阵 $S_{t-1}$，与当前 key 无关。只用 $\beta_t$ 就是 DeltaNet，只用 $\alpha_t$ 就是 Mamba2，两者都用才是 Gated DeltaNet。

**Q：Gated DeltaNet 的 $\alpha_t$ 是标量，GLA 的门控是矩阵，哪个更强？**

GLA 的门控 $G_t \in \mathbb{R}^{d \times d}$ 理论上表达能力更强，可以对记忆矩阵的每个元素独立控制衰减速度。但 GLA 没有 Delta 规则，写入方式仍然是简单的加法。Gated DeltaNet 用标量 $\alpha_t$ 换来了 Delta 规则的精准写入，在关联记忆任务上的实验结果显示这个权衡是值得的。

**Q：为什么 $\alpha_t$ 是标量而不是向量或矩阵？**

标量 $\alpha_t$ 有两个优势：一是计算代价极小（只需一个线性层输出一个数），二是与 Mamba2 的 SSD 框架兼容，可以复用其高效的分块并行实现。论文消融实验表明，标量门控已经能带来显著的性能提升，进一步增加门控维度的收益递减。

**Q：Gated DeltaNet 在推理时的状态大小是多少？**

与 DeltaNet 相同：每个头维护一个 $d_{\text{head}} \times d_{\text{head}}$ 的矩阵，总状态大小为 $O(n_{\text{heads}} \times d_{\text{head}}^2) = O(d_{\text{model}}^2 / n_{\text{heads}})$。这是固定大小的，不随序列长度增长，推理时无需 KV Cache。

**Q：Gated DeltaNet 能完全替代 Transformer 吗？**

在大多数语言建模任务上，Gated DeltaNet 接近甚至超过同参数量的 Transformer。但在需要精确检索特定历史 token 的任务上（如 needle-in-a-haystack），固定大小的记忆矩阵仍然是瓶颈。混合架构（少量 softmax attention 层 + 大量 Gated DeltaNet 层）是目前最实用的方案。

**Q：论文标题说"改进 Mamba2"，但 Gated DeltaNet 也改进了 DeltaNet，哪个说法更准确？**

两个说法都对，只是视角不同。从 Mamba2 的角度看，Gated DeltaNet 用 Delta 规则替换了 Mamba2 的简单加法写入，提升了关联记忆能力。从 DeltaNet 的角度看，Gated DeltaNet 加入了全局衰减门控，提升了长度泛化和选择性遗忘能力。论文选择"改进 Mamba2"的说法，是因为 Mamba2 在当时是更广为人知的基线。

---

## 6. 完整 PyTorch 实现

下面的代码实现了 Gated DeltaNet 的核心逻辑，包括递推形式（推理）和分块并行形式（训练）。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 工具函数
# ============================================================

def l2_normalize(x, dim=-1, eps=1e-6):
    """对向量做 L2 归一化，使 ||x||_2 = 1，保证转移矩阵谱范数 ≤ 1"""
    return x / (x.norm(dim=dim, keepdim=True) + eps)


# ============================================================
# Gated Delta 规则：递推形式（推理，每次处理一个 token）
# ============================================================

def gated_delta_rule_recurrent(q, k, v, beta, alpha):
    """
    Gated DeltaNet 的递推实现，适合自回归推理。

    更新规则：
        S_t = alpha_t * (S_{t-1} - beta_t * (S_{t-1} k_t - v_t) k_t^T)
            = alpha_t * S_{t-1} * (I - beta_t k_t k_t^T) + alpha_t * beta_t * v_t k_t^T

    参数：
        q:     (B, T, d)  query 向量（已 L2 归一化）
        k:     (B, T, d)  key 向量（已 L2 归一化）
        v:     (B, T, d)  value 向量
        beta:  (B, T, 1)  写入强度，sigmoid 输出，范围 (0, 1)
        alpha: (B, T, 1)  全局衰减因子，sigmoid 输出，范围 (0, 1)

    返回：
        o:     (B, T, d)  输出向量
    """
    B, T, d = q.shape
    # 初始化矩阵值隐藏状态为零矩阵
    S = torch.zeros(B, d, d, device=q.device, dtype=q.dtype)
    outputs = []

    for t in range(T):
        qt = q[:, t, :]        # (B, d)，当前 query
        kt = k[:, t, :]        # (B, d)，当前 key（已 L2 归一化）
        vt = v[:, t, :]        # (B, d)，当前 value
        bt = beta[:, t, :]     # (B, 1)，写入强度
        at = alpha[:, t, :]    # (B, 1)，全局衰减因子

        # ① 用当前 key 查询旧记忆，得到旧预测值
        # einsum 'bdn,bn->bd'：对每个 batch，S[b] @ k[b]
        v_old = torch.einsum('bdn,bn->bd', S, kt)   # (B, d)

        # ② 计算 Delta 误差：新值与旧预测值之差，乘以写入强度
        # delta = beta_t * (v_t - v_old)，即需要修正的量
        delta = bt * (vt - v_old)                    # (B, d)

        # ③ Delta 规则更新：在 k_t 方向写入修正量
        # S' = S + delta ⊗ k_t（外积，形状 (B, d, d)）
        S = S + torch.einsum('bd,bn->bdn', delta, kt)

        # ④ 全局衰减：整体缩放记忆矩阵
        # alpha_t 接近 0 时，旧记忆被大幅遗忘；接近 1 时几乎保留
        S = at.unsqueeze(-1) * S                     # (B, d, d)

        # ⑤ 读出当前时刻的输出：o_t = S_t q_t
        ot = torch.einsum('bdn,bn->bd', S, qt)       # (B, d)
        outputs.append(ot)

    return torch.stack(outputs, dim=1)  # (B, T, d)


# ============================================================
# Gated Delta 规则：分块并行形式（训练，处理整个序列）
# ============================================================

def gated_delta_rule_chunkwise(q, k, v, beta, alpha, chunk_size=64):
    """
    Gated DeltaNet 的分块并行实现（简化版，用于理解原理）。

    策略：
    - 块间：串行递推，但步数只有 L/C 步（C = chunk_size）
    - 块内：并行计算，利用矩阵运算加速

    参数：
        q, k, v:    (B, T, d)
        beta, alpha:(B, T, 1)
        chunk_size: 每块的 token 数

    返回：
        o:          (B, T, d)
    """
    B, T, d = q.shape
    assert T % chunk_size == 0, "序列长度必须是 chunk_size 的整数倍"
    n_chunks = T // chunk_size

    # 初始化跨块的隐藏状态
    S = torch.zeros(B, d, d, device=q.device, dtype=q.dtype)
    all_outputs = []

    for c in range(n_chunks):
        # 取出当前块的数据
        start, end = c * chunk_size, (c + 1) * chunk_size
        q_c = q[:, start:end, :]      # (B, C, d)
        k_c = k[:, start:end, :]      # (B, C, d)
        v_c = v[:, start:end, :]      # (B, C, d)
        b_c = beta[:, start:end, :]   # (B, C, 1)
        a_c = alpha[:, start:end, :]  # (B, C, 1)

        C = chunk_size

        # ---- 块内并行计算 ----

        # 计算块内因果掩码（下三角，包含对角线）
        mask = torch.tril(torch.ones(C, C, device=q.device))  # (C, C)

        # 计算块内 key-key 内积矩阵：KK[i][j] = k_i^T k_j
        KK = torch.bmm(k_c, k_c.transpose(1, 2))  # (B, C, C)

        # 计算块内"伪 value"矩阵 U（简化迭代版）
        # u_t = beta_t * (v_t - sum_{i<t} u_i * (k_i^T k_t))
        U = torch.zeros_like(v_c)  # (B, C, d)
        for t in range(C):
            correction = torch.zeros(B, d, device=q.device)
            for i in range(t):
                # 累积之前所有 u_i 对当前位置的贡献
                correction = correction + U[:, i, :] * KK[:, i, t].unsqueeze(-1)
            U[:, t, :] = b_c[:, t, :] * (v_c[:, t, :] - correction)

        # 计算块内累积衰减前缀乘积
        # alpha_prefix[t] = prod_{i=0}^{t} alpha_i（块内从起始到位置 t）
        alpha_prefix = torch.cumprod(a_c, dim=1)  # (B, C, 1)

        # 块内注意力分数（带因果掩码）
        A_intra = torch.bmm(q_c, k_c.transpose(1, 2)) * mask.unsqueeze(0)  # (B, C, C)

        # 块内输出：来自块内历史的贡献
        # 需要对每对 (i, j) 乘以从 j 到 i 的累积衰减
        # 简化：直接用 alpha_prefix 近似（精确实现需要更复杂的掩码）
        o_intra = torch.bmm(A_intra, U)  # (B, C, d)

        # 块间输出：来自之前块的隐藏状态 S 的贡献
        # o_inter[t] = alpha_prefix[t] * S q_t
        o_inter = torch.einsum('bdn,bcn->bcd', S, q_c)  # (B, C, d)
        o_inter = o_inter * alpha_prefix                  # 乘以累积衰减

        # 合并块内和块间输出
        o_c = o_intra + o_inter  # (B, C, d)
        all_outputs.append(o_c)

        # ---- 更新跨块隐藏状态 ----
        # 用递推方式更新 S，处理块内所有 token
        for t in range(C):
            v_old = torch.einsum('bdn,bn->bd', S, k_c[:, t, :])
            delta = b_c[:, t, :] * (v_c[:, t, :] - v_old)
            S = S + torch.einsum('bd,bn->bdn', delta, k_c[:, t, :])
            S = a_c[:, t, :].unsqueeze(-1) * S

    return torch.cat(all_outputs, dim=1)  # (B, T, d)


# ============================================================
# Gated DeltaNet 注意力层（完整模块）
# ============================================================

class GatedDeltaNetAttention(nn.Module):
    """
    Gated DeltaNet 注意力层。

    相比 DeltaNet 的改动：
    - 增加 alpha 门控（全局衰减因子），每个头独立一个标量
    - alpha 由输入通过线性层 + sigmoid 计算，实现输入依赖的遗忘
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

        # beta 投影：写入强度，每个头独立一个标量
        self.w_beta = nn.Linear(d_model, n_heads, bias=False)

        # alpha 投影：全局衰减因子，每个头独立一个标量（Gated DeltaNet 新增）
        self.w_alpha = nn.Linear(d_model, n_heads, bias=False)

        # 短卷积：捕捉局部 token 间的相对位置信息
        self.conv_q = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_model, bias=False
        )
        self.conv_k = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=d_model, bias=False
        )

        # 输出归一化和投影
        self.out_norm = nn.RMSNorm(d_model)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, use_recurrent=False):
        """
        参数：
            x:             (B, T, d_model)  输入序列
            use_recurrent: True 时用递推形式（推理），False 时用分块并行（训练）

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

        # ② 短卷积：增强局部位置感知（Conv1d 期望 (B, C, T) 格式）
        q = self.conv_q(q.transpose(1, 2))[:, :, :T].transpose(1, 2)
        k = self.conv_k(k.transpose(1, 2))[:, :, :T].transpose(1, 2)

        # ③ SiLU 激活 + L2 归一化（key 和 query）
        # L2 归一化保证 k_t^T k_t = 1，使 (I - beta_t k_t k_t^T) 是合法投影
        q = l2_normalize(F.silu(q))
        k = l2_normalize(F.silu(k))

        # ④ 计算写入强度 beta（DeltaNet 原有）
        beta = torch.sigmoid(self.w_beta(x))   # (B, T, n_heads)

        # ⑤ 计算全局衰减因子 alpha（Gated DeltaNet 新增）
        # sigmoid 保证 alpha ∈ (0, 1)，防止记忆爆炸
        alpha = torch.sigmoid(self.w_alpha(x))  # (B, T, n_heads)

        # ⑥ 拆分多头：(B, T, d_model) → (B*h, T, d)
        q = q.view(B, T, h, d).permute(0, 2, 1, 3).reshape(B * h, T, d)
        k = k.view(B, T, h, d).permute(0, 2, 1, 3).reshape(B * h, T, d)
        v = v.view(B, T, h, d).permute(0, 2, 1, 3).reshape(B * h, T, d)
        # beta 和 alpha 扩展到 head_dim 维度（每头一个标量，广播到 d 维）
        beta  = beta.permute(0, 2, 1).reshape(B * h, T, 1)   # (B*h, T, 1)
        alpha = alpha.permute(0, 2, 1).reshape(B * h, T, 1)  # (B*h, T, 1)

        # ⑦ Gated Delta 规则计算
        if use_recurrent:
            # 推理：递推形式，每步 O(d^2)，无 KV Cache
            out = gated_delta_rule_recurrent(q, k, v, beta, alpha)
        else:
            # 训练：分块并行形式（这里用递推作为 fallback，实际应用中用 CUDA kernel）
            out = gated_delta_rule_recurrent(q, k, v, beta, alpha)
        # out: (B*h, T, d)

        # ⑧ 合并多头：(B*h, T, d) → (B, T, d_model)
        out = out.reshape(B, h, T, d).permute(0, 2, 1, 3).reshape(B, T, -1)

        # ⑨ 输出归一化 + 线性投影
        out = self.out_norm(out)
        return self.wo(out)


# ============================================================
# Gated DeltaNet Transformer Block
# ============================================================

class GatedDeltaNetBlock(nn.Module):
    """
    单个 Gated DeltaNet Transformer Block。
    结构：RMSNorm → GatedDeltaNet → 残差 → RMSNorm → SwiGLU FFN → 残差
    """

    def __init__(self, d_model, n_heads, ffn_mult=4):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.attn = GatedDeltaNetAttention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model)

        # SwiGLU FFN：两个并行线性层，一个做门控
        ffn_dim = int(d_model * ffn_mult * 2 / 3)
        self.ffn_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.ffn_up   = nn.Linear(d_model, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, d_model, bias=False)

    def forward(self, x, use_recurrent=False):
        # Gated DeltaNet 子层（Pre-Norm + 残差）
        x = x + self.attn(self.norm1(x), use_recurrent=use_recurrent)

        # SwiGLU FFN 子层（Pre-Norm + 残差）
        h = self.norm2(x)
        # SwiGLU：gate 分支做 SiLU 激活，与 up 分支逐元素相乘
        x = x + self.ffn_down(F.silu(self.ffn_gate(h)) * self.ffn_up(h))
        return x


# ============================================================
# 对比演示：DeltaNet vs Gated DeltaNet 的记忆行为
# ============================================================

def demo_memory_behavior():
    """
    演示 alpha 门控对记忆矩阵的影响。
    场景：序列中有一个"重置信号"token，alpha 接近 0，
    观察记忆矩阵在该 token 前后的变化。
    """
    torch.manual_seed(42)
    B, d = 1, 4

    # 模拟一个简单序列：3 个普通 token + 1 个重置 token + 2 个普通 token
    T = 6
    q = l2_normalize(torch.randn(B, T, d))
    k = l2_normalize(torch.randn(B, T, d))
    v = torch.randn(B, T, d)
    beta = torch.ones(B, T, 1) * 0.8   # 写入强度固定为 0.8

    # DeltaNet：alpha 全为 1（无全局衰减）
    alpha_deltanet = torch.ones(B, T, 1)

    # Gated DeltaNet：第 4 个 token（索引 3）是重置信号，alpha 接近 0
    alpha_gated = torch.ones(B, T, 1)
    alpha_gated[:, 3, :] = 0.01  # 重置信号：几乎清空记忆

    # 运行两个模型
    out_deltanet = gated_delta_rule_recurrent(q, k, v, beta, alpha_deltanet)
    out_gated    = gated_delta_rule_recurrent(q, k, v, beta, alpha_gated)

    print("DeltaNet 输出（无全局衰减）:")
    print(out_deltanet[0].detach().numpy().round(3))
    print("\nGated DeltaNet 输出（第 4 个 token 触发重置）:")
    print(out_gated[0].detach().numpy().round(3))
    print("\n注意：重置后（token 4-5），两个模型的输出差异显著")


# ============================================================
# 简单测试
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    B, T, d_model, n_heads = 2, 16, 64, 4

    # 测试 Gated DeltaNet Block
    block = GatedDeltaNetBlock(d_model, n_heads)
    x = torch.randn(B, T, d_model)

    # 推理模式（递推形式）
    out = block(x, use_recurrent=True)
    print(f"Gated DeltaNet Block 输出形状: {out.shape}")  # (2, 16, 64)

    # 验证参数量（相比 DeltaNet 只多了 w_alpha 这一个线性层）
    total_params = sum(p.numel() for p in block.parameters())
    print(f"总参数量: {total_params:,}")

    # 演示记忆行为
    print("\n--- 记忆行为演示 ---")
    demo_memory_behavior()
```

---

## 7. 读完这篇之后，你应该能回答这些问题

- DeltaNet 的 $\beta_t$ 和 Gated DeltaNet 的 $\alpha_t$ 分别控制什么？两者能否互相替代？
- Gated DeltaNet 的更新公式 $S_t = \alpha_t S_{t-1}(\mathbf{I} - \beta_t k_t k_t^\top) + \alpha_t \beta_t v_t k_t^\top$ 中，令 $\alpha_t = 1$ 和令 $\beta_t = 0$ 分别退化为哪个模型？
- 为什么 DeltaNet 在长度泛化上表现差？$\alpha_t$ 如何改善这个问题？
- $\alpha_t$ 是标量，GLA 的门控是矩阵，哪个表达能力更强？Gated DeltaNet 为什么选择标量？
- Gated DeltaNet 的推理状态大小是多少？与 DeltaNet 相比有什么变化？
- 分块并行形式中，$\alpha_t$ 的累积乘积（prefix product）起什么作用？
- 从 Mamba2（SSD）的视角看，Gated DeltaNet 做了什么改动？这个改动解决了 Mamba2 的哪个问题？
- 在什么场景下，$\alpha_t$ 应该接近 0？在什么场景下应该接近 1？举一个具体的文本例子。

---

## 参考资料

- 原始论文：`paper/Gated-DeltaNet-Improving-Mamba2-with-Delta-Rule.pdf`（Yang et al., 2024）
- DeltaNet：`paper/DeltaNet-Parallelizable-Linear-Recurrence.pdf`（Yang et al., NeurIPS 2024）
- Mamba2 / SSD：Dao & Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", ICML 2024
- FlashLinearAttention 库（包含 Gated DeltaNet 的高效 CUDA 实现）：https://github.com/fla-org/flash-linear-attention
- 相关文章：`Attention/DeltaNet.md`、`Attention/GLA.md`、`Attention/Mamba.md`
