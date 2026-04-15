# Qwen 3.5

> 对应论文：暂无官方技术报告，架构信息来自 [HuggingFace 代码实现](https://huggingface.co/Qwen/Qwen3.5-397B-A17B) 和 [GitHub 技术分析](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)
>
> 核心问题：标准 Transformer 的 $O(N^2)$ 注意力在长上下文场景下代价过高，但纯线性注意力又会在关联检索任务上退化。Qwen 3.5 的解法是用**混合架构**——75% 的层使用 Gated DeltaNet（线性注意力，$O(1)$ 推理），25% 的层使用标准 GQA（全注意力，强检索能力），再搭配 MoE 控制激活参数量。

---

## 1. 背景：为什么需要混合架构

### 1.1 纯 Transformer 的长上下文困境

在 `Transformer.md` 中我们知道，标准 **softmax attention** 的复杂度是 $O(N^2 d)$，其中 $N$ 是序列长度。当上下文窗口从 4K 扩展到 128K 时，计算量和显存占用增长 1024 倍。

LLaMA 4 用 **MLA**（多头潜在注意力）压缩 KV Cache 来缓解推理压力，但 MLA 并没有改变注意力的 $O(N^2)$ 计算本质——它只是让每次注意力计算的 KV 更小。

### 1.2 纯线性注意力的问题

在 `DeltaNet.md` 和 `Gated-DeltaNet.md` 中我们看到，线性注意力（包括 Gated DeltaNet）把复杂度降到 $O(N d^2)$，推理时每步只需 $O(d^2)$——与序列长度无关。

但线性注意力有一个根本性的代价：**它用固定大小的状态矩阵 $S \in \mathbb{R}^{d_k \times d_v}$ 压缩整个序列的信息**。在"大海捞针"式的精确检索任务上（比如"第 37 页提到的那个数字是多少？"），固定大小的状态矩阵很难比得上能直接看到所有 KV 对的全注意力。

### 1.3 Qwen 3.5 的解法：混合 + MoE

Qwen 3.5 的思路很直接：

> 既然全注意力和线性注意力各有优劣，为什么不把它们混在一起？

具体来说，Qwen 3.5 做了两件事：

1. **混合注意力层**：75% 的层用 Gated DeltaNet（线性注意力），25% 的层用标准 GQA（全注意力）
2. **稀疏 MoE**：MLP 层使用 Mixture-of-Experts，397B 总参数中只激活 17B

两者的作用不同：
- 混合注意力解决的是**序列维度的计算瓶颈**
- MoE 解决的是**参数维度的计算瓶颈**

---

## 2. 核心机制：Gated DeltaNet 层

Qwen 3.5 的线性注意力层使用的是 Gated DeltaNet（详见 `Gated-DeltaNet.md`），这里做一个快速回顾，重点说明 Qwen 3.5 实现中的具体细节。

### 2.1 更新规则回顾

Gated DeltaNet 维护一个固定大小的状态矩阵 $S_t$，每一步用**门控衰减 + Delta 规则**更新：

$$
S_t = g_t \cdot S_{t-1} + \tilde{k}_t \otimes \bigl[\beta_t (v_t - g_t \cdot S_{t-1}^\top \tilde{k}_t)\bigr]
$$

其中：
- $g_t = \exp(\alpha_t) \in (0, 1]$：**全局衰减门**，控制旧记忆的保留程度
- $\beta_t = \sigma(b_t) \in (0, 1)$：**更新强度**，控制新信息的写入幅度
- $\tilde{k}_t, \tilde{q}_t$：$L_2$ 归一化后的 Key 和 Query
- $S_{t-1}^\top \tilde{k}_t$：当前状态对 Key 的预测值

读出操作：

$$
o_t = \frac{\tilde{q}_t^\top S_t}{\sqrt{d_k}}
$$

> 可以把 Gated DeltaNet 的记忆管理理解成**整理笔记**。$g_t$ 决定"旧笔记保留多少"（时间越久，可能越不重要），$\beta_t$ 决定"新内容写多少"，而 Delta 规则 $(v_t - \text{旧预测})$ 只写入"和已有知识不一致的部分"——避免重复记录。

### 2.2 Qwen 3.5 的具体配置

根据 HuggingFace 代码，Qwen 3.5 的 Gated DeltaNet 层有如下设计：

**投影方式**：输入 $x$ 经过两个线性投影：

$$
[q, k, v, z] = \text{Linear}_{qkvz}(x), \quad [b, a] = \text{Linear}_{ba}(x)
$$

其中 $z$ 是输出门，$b$ 用来计算 $\beta$，$a$ 用来计算衰减门 $g$。

**因果卷积**：Q、K、V 在进入 Delta 规则之前，会先通过一个**因果 1D 卷积**（kernel size = 4）：

$$
[q, k, v] = \text{SiLU}(\text{CausalConv1D}([q, k, v]))
$$

这一步提供**局部上下文混合**，替代了线性注意力层中不使用的 RoPE。直觉上，因果卷积让每个 token 能"看到"最近 4 个邻居的信息。

**头分组（类似 GQA）**：Qwen 3.5 的线性注意力层使用不对称的 K/V 头数：

| 参数 | 值 |
|:---|:---:|
| `num_k_heads` | 16 |
| `num_v_heads` | 32 |
| `key_head_dim` | 128 |
| `value_head_dim` | 128 |

K 有 16 个头，V 有 32 个头。K 头通过 `repeat_interleave` 扩展到和 V 一样的头数。这种设计和 GQA 类似——更少的 K 头意味着更小的状态矩阵 $S$。

**门控计算**：

$$
\beta_t = \sigma(b_t), \quad g_t = -\exp(A_{\log}) \cdot \text{softplus}(a_t + \text{dt\_bias})
$$

其中 $A_{\log} \in \mathbb{R}^{H}$ 是可学习的 per-head 衰减率（初始化在 $[0, 16]$ 的均匀分布），$\text{dt\_bias} \in \mathbb{R}^{H}$ 是可学习的时间步偏置。

**输出门控**：最终输出经过门控 RMSNorm：

$$
\text{output} = \text{RMSNorm}(o_t) \odot \text{SiLU}(z_t)
$$

$z_t$ 来自最初的 $qkvz$ 投影，提供额外的输出过滤。

### 2.3 推理 vs 训练的不同算法

和所有线性注意力模型一样，Qwen 3.5 在推理和训练时使用不同的算法：

**推理（自回归生成）**：逐 token 递推更新状态矩阵 $S_t$，每步只需 $O(d_k \cdot d_v)$ 的计算——与序列长度无关。

**训练（prefill）**：使用**分块并行算法**（chunkwise parallel），把序列按 chunk size = 64 分块，块内并行计算，块间通过状态矩阵传递。复杂度 $O(N \cdot C \cdot d^2)$，其中 $C = 64$ 是 chunk 大小。

---

## 3. 混合架构：层交替模式

### 3.1 层排列方式

Qwen 3.5 的层配置由 `layer_types` 参数控制。默认模式是每隔 3 个 Gated DeltaNet 层，插入 1 个全注意力层：

```
[L, L, L, F, L, L, L, F, L, L, L, F, ...]
```

其中 L = Gated DeltaNet（线性注意力），F = 标准 GQA 全注意力。

也就是说：**75% 线性注意力 + 25% 全注意力**。

> 可以把这种混合架构理解成**考试复习策略**。平时的大部分时间（75%）你在用"快速浏览"的方式处理信息（线性注意力——快但不精确），每隔一段时间（25%）你会"精读"一遍之前的内容（全注意力——慢但精确），确保没有遗漏。两种方式交替使用，既保证了效率，又保证了关键信息不丢失。

### 3.2 为什么是 3:1 的比例

这个比例是一个工程权衡：

- **线性层太多**（比如 7:1）：检索能力下降，"大海捞针"任务变差
- **全注意力太多**（比如 1:1）：计算优势消失，长序列推理仍然很慢
- **3:1 刚好**：在全注意力基准测试上几乎不掉点，同时推理时 75% 的层不需要 KV Cache

从另一个角度看：全注意力层只需要处理 25% 的层，这意味着 KV Cache 只需存储 25% 层的 KV 对。相比于全部 32 层都用全注意力，KV Cache 减少了约 75%。

### 3.3 两种层的对比

| 特性 | Gated DeltaNet 层 | 全注意力层（GQA） |
|:---|:---:|:---:|
| 复杂度 | $O(N d^2)$ | $O(N^2 d)$ |
| 推理状态 | 固定矩阵 $S \in \mathbb{R}^{d_k \times d_v}$ | KV Cache（随序列增长） |
| 位置编码 | 因果卷积（局部） | RoPE（全局） |
| 归一化 | $L_2$ norm on Q, K | $\sqrt{d_k}$ 缩放 |
| 长处 | 长序列推理效率 | 精确关联检索 |
| 弱点 | 固定状态容量 | 计算和显存开销大 |

### 3.4 混合缓存结构

由于两种层有不同的状态类型，Qwen 3.5 使用**异构缓存** `Qwen3NextDynamicCache`：

**全注意力层**（标准 KV Cache，随序列增长）：

$$
\text{key\_cache}[l] \in \mathbb{R}^{B \times H_{kv} \times T \times d}
$$

$$
\text{value\_cache}[l] \in \mathbb{R}^{B \times H_{kv} \times T \times d}
$$

**线性注意力层**（固定大小状态）：

$$
\text{recurrent\_states}[l] \in \mathbb{R}^{B \times H \times d_k \times d_v} \quad \text{(固定大小)}
$$

$$
\text{conv\_states}[l] \in \mathbb{R}^{B \times d_{conv} \times 4} \quad \text{(固定大小)}
$$

以 Qwen3.5-9B（32 层，24 线性 + 8 全注意力）为例，线性层的 recurrent states 占用：

$$
24 \times B \times 32 \times 128 \times 128 \times 2 \text{ bytes (fp16)} \approx 24 \times 1 \text{ MB} = 24 \text{ MB}
$$

这是一个**与序列长度无关的固定开销**。

---

## 4. MoE：控制激活参数量

### 4.1 为什么需要 MoE

混合注意力解决了序列维度的瓶颈，但模型仍然需要大量参数来存储知识。如果直接把模型做到 397B 参数，每次前向传播都要处理所有参数，计算代价不可接受。

**Mixture-of-Experts (MoE)** 的思路是：把 MLP 层拆分成多个"专家"（expert），每个 token 只激活其中少数几个。

### 4.2 Qwen 3.5 的 MoE 配置

Qwen 3.5-397B-A17B 的关键参数：

| 参数 | 值 |
|:---|:---:|
| 总参数 | 397B |
| 激活参数 | 17B |
| 激活比例 | ~4.3% |
| 上下文长度 | 262K |

激活比例 4.3% 意味着每个 token 只使用 4.3% 的 MLP 参数。这让模型拥有大参数量的知识容量，同时保持推理时的计算成本可控。

### 4.3 MoE 路由机制

每个 token 经过一个**路由器（router）**，计算它与每个专家的匹配分数：

$$
\text{expert\_id} = \text{TopK}(\text{softmax}(W_r \cdot x_t), k)
$$

$$
\text{output} = \sum_{i \in \text{TopK}} \text{softmax}(W_r \cdot x_t)_i \cdot \text{Expert}_i(x_t)
$$

其中 $W_r$ 是路由器的权重矩阵，TopK 选出匹配分数最高的 $k$ 个专家，然后按 softmax 分数加权求和。每个专家是一个独立的 SwiGLU MLP。

---

## 5. 完整架构总览

### 5.1 数据流

一个 token 在 Qwen 3.5 中经过多个 Transformer block，每层包含：

```
┌─────────────────────────────────────────┐
│  Transformer Block (layer i)            │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ Token Mixing (注意力层)           │  │
│  │  · layer_type = "linear":         │  │
│  │    Gated DeltaNet (75% 的层)      │  │
│  │  · layer_type = "full_attention": │  │
│  │    标准 GQA + RoPE (25% 的层)     │  │
│  └───────────────────────────────────┘  │
│              ↓ 残差连接                  │
│  ┌───────────────────────────────────┐  │
│  │ Channel Mixing (MLP 层)           │  │
│  │  · MoE: Router → Top-K Experts    │  │
│  │  · Expert = SwiGLU MLP           │  │
│  └───────────────────────────────────┘  │
│              ↓ 残差连接                  │
└─────────────────────────────────────────┘

层排列: [L, L, L, F, L, L, L, F, ...]
         L = Gated DeltaNet (线性注意力)
         F = Full Attention (GQA + RoPE)
```

### 5.2 推理时的内存优势

对于长度为 $T$ 的序列，传统全注意力模型的 KV Cache 大小为：

$$
\text{KV Cache}_{\text{full}} = 2 \times L \times B \times H_{kv} \times T \times d \times 2 \text{ bytes}
$$

Qwen 3.5 的缓存大小：

$$
\text{KV Cache}_{\text{hybrid}} = 2 \times \frac{L}{4} \times B \times H_{kv} \times T \times d \times 2 \text{ bytes}
$$

加上线性层的固定开销（与 $T$ 无关）。长序列时，KV Cache 减少约 75%。

---

## 6. 与其他模型的对比

### 6.1 注意力机制对比

| 模型 | 注意力类型 | 推理状态 | 上下文长度 |
|:---|:---:|:---:|:---:|
| LLaMA 3 | GQA（全注意力） | KV Cache（线性增长） | 128K |
| LLaMA 4 | MLA（低秩压缩） | 压缩 KV Cache | 128K |
| Qwen 3 | GQA（全注意力） | KV Cache（线性增长） | 32K |
| **Qwen 3.5** | **混合：Gated DeltaNet + GQA** | **75% 固定 + 25% KV Cache** | **262K** |
| GLM-5 | MLA + 稀疏注意力 | 压缩 KV Cache | 128K |
| MiniMax M2.5 | MHA（全注意力） | KV Cache（线性增长） | 1M |
| Kimi K2.5 | MLA | 压缩 KV Cache | 128K |

### 6.2 MoE 配置对比

| 模型 | 总参数 | 激活参数 | 激活比例 |
|:---|:---:|:---:|:---:|
| LLaMA 3 70B | 70B | 70B | 100%（Dense） |
| Qwen 3 235B | 235B | 22B | 9.4% |
| **Qwen 3.5** | **397B** | **17B** | **4.3%** |
| DeepSeek-V3 | 671B | 37B | 5.5% |
| Kimi K2.5 | ~1T | ~14B | ~1.4% |

Qwen 3.5 的激活比例 4.3% 在主流大模型中处于较低水平，意味着推理时每个 token 的计算量接近一个 17B dense 模型。

### 6.3 注意力架构的设计光谱

把各模型的注意力策略放在一个光谱上：

```
纯全注意力 ←————————————————————→ 纯线性注意力
   MiniMax M2.5          Qwen 3.5           Mamba
   LLaMA 3               (3:1 混合)       RWKV-6
   Qwen 3
                              ↑
                          当前最优平衡点？
```

Qwen 3.5 的 3:1 混合比例是目前在"精确检索"和"推理效率"之间的一个新探索点。

---

## 7. 常见混淆问题

### Q1：Gated DeltaNet 层和全注意力层处理的是同一个序列吗？

是的。两者处理的是完全相同的输入 token 序列。区别在于**如何计算 token 之间的关系**：Gated DeltaNet 用固定大小的状态矩阵压缩历史信息，全注意力则让每个 token 直接看到所有其他 token 的 KV 对。

### Q2：为什么线性注意力层不用 RoPE？

因为线性注意力层用**因果卷积**（kernel size = 4）来获取局部位置信息。因果卷积提供的是一个滑动窗口的局部感受野，足以捕捉相邻 token 的位置关系。而 RoPE 的作用是让注意力分数反映相对位置——这在 softmax attention 中很重要（因为要区分不同距离的 token 的权重），但在 Gated DeltaNet 中，位置信息通过门控和 Delta 规则以不同方式编码。

### Q3：75% 的层用线性注意力，检索能力真的够吗？

实验表明，只要每隔几层有一次全注意力层（25%），检索性能就可以和全注意力模型持平。原因是全注意力层充当了"全局信息汇聚点"——它们看到的 KV 对包含前面线性层积累的所有信息。线性层则负责高效地传递这些信息。

### Q4：MoE 的专家是按什么维度划分的？

MoE 在 MLP 层实现，每个专家是一个完整的 SwiGLU MLP。路由器根据 token 的隐藏状态动态选择专家，不同类型的文本（代码、数学、自然语言、多语言）倾向于被路由到不同的专家组合。这意味着模型可以"专门化"而不用增加每次推理的计算量。

### Q5：Qwen 3.5 和 Qwen 3-Next 是什么关系？

Qwen 3-Next（2025 年 9 月）是 Qwen 团队对混合注意力架构的**预研版本**。Qwen 3.5（2026 年 2 月）是在 Qwen 3-Next 基础上大规模验证后的正式产品。两者使用相同的 Gated DeltaNet + GQA 混合架构，但 Qwen 3.5 的规模更大（397B vs 较小规模），并增加了原生多模态能力和更长的上下文窗口。

### Q6：状态矩阵 $S$ 的大小够用吗？$128 \times 128$ 能存多少信息？

每个头的状态矩阵是 $d_k \times d_v = 128 \times 128 = 16384$ 个 float。32 个头共有 $32 \times 16384 \approx 0.5M$ 个参数。这个容量对于存储"关键信息的压缩摘要"是足够的，但它不能像全注意力那样精确回溯每一个 token。这就是为什么需要穿插全注意力层来做精确检索。

---

## 8. 完整模块代码

下面是一个简化但完整的 Qwen 3.5 混合注意力层实现，包含 Gated DeltaNet 层和全注意力层的交替。

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedDeltaNetLayer(nn.Module):
    """Qwen 3.5 的 Gated DeltaNet 线性注意力层（推理用递推形式）"""

    def __init__(self, hidden_size=4096, num_k_heads=16, num_v_heads=32,
                 key_head_dim=128, value_head_dim=128):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim

        key_dim = num_k_heads * key_head_dim       # 16 * 128 = 2048
        value_dim = num_v_heads * value_head_dim    # 32 * 128 = 4096

        # ① QKV + 输出门 z 的联合投影
        self.in_proj_qkvz = nn.Linear(
            hidden_size, key_dim * 2 + value_dim * 2, bias=False
        )
        # ② 门控参数 β 和 α 的投影
        self.in_proj_ba = nn.Linear(hidden_size, num_v_heads * 2, bias=False)

        # ③ 可学习的全局衰减参数（per-head），初始化在 [0, 16]
        self.A_log = nn.Parameter(torch.randn(num_v_heads) * 4 + 8)
        # ④ 可学习的时间步偏置
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads))

        self.norm = nn.RMSNorm(value_head_dim, eps=1e-6)
        self.out_proj = nn.Linear(value_dim, hidden_size, bias=False)

    def _compute_gates(self, ba):
        """从投影结果计算 β（更新强度）和 g 的 log 值（衰减门）"""
        b, a = ba.chunk(2, dim=-1)                    # 各 (B, T, H)
        beta = torch.sigmoid(b)                       # 更新强度 ∈ (0, 1)
        # 衰减门：负指数确保 g ∈ (0, 1]，是输入依赖的
        g = -torch.exp(self.A_log) * F.softplus(a + self.dt_bias)
        return beta, g

    def forward_recurrent(self, x, state=None):
        """
        自回归推理：逐 token 递推更新状态矩阵
        x: (B, 1, hidden_size)
        state: (B, H_v, d_k, d_v) 或 None
        返回: output (B, 1, hidden), new_state (B, H_v, d_k, d_v)
        """
        B = x.size(0)
        key_dim = self.num_k_heads * self.key_head_dim
        value_dim = self.num_v_heads * self.value_head_dim

        # ① 投影
        qkvz = self.in_proj_qkvz(x)                  # (B, 1, key_dim*2 + value_dim*2)
        ba = self.in_proj_ba(x)                       # (B, 1, H_v * 2)
        beta, g = self._compute_gates(ba)             # (B, 1, H_v)

        # ② 拆分 Q, K, V, Z
        q = qkvz[:, :, :key_dim]
        k = qkvz[:, :, key_dim:key_dim * 2]
        v = qkvz[:, :, key_dim * 2:key_dim * 2 + value_dim]
        z = qkvz[:, :, key_dim * 2 + value_dim:]     # 输出门

        # ③ Reshape 成多头格式
        q = q.view(B, 1, self.num_k_heads, self.key_head_dim)
        k = k.view(B, 1, self.num_k_heads, self.key_head_dim)
        v = v.view(B, 1, self.num_v_heads, self.value_head_dim)
        z = z.view(B, 1, self.num_v_heads, self.value_head_dim)

        # K 头数 < V 头数，扩展 K 以匹配 V（类似 GQA）
        n_rep = self.num_v_heads // self.num_k_heads
        k = k.repeat_interleave(n_rep, dim=2)         # (B, 1, H_v, d_k)

        # ④ L2 归一化 Q 和 K（替代 softmax 归一化）
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # ⑤ 初始化状态矩阵
        if state is None:
            state = torch.zeros(
                B, self.num_v_heads, self.key_head_dim, self.value_head_dim,
                device=x.device, dtype=x.dtype
            )

        # ⑥ Gated Delta 规则递推更新
        g_exp = torch.exp(g).squeeze(1)               # (B, H_v)
        beta_sq = beta.squeeze(1)                     # (B, H_v)
        k_t = k.squeeze(1)                            # (B, H_v, d_k)
        v_t = v.squeeze(1)                            # (B, H_v, d_v)
        q_t = q.squeeze(1)                            # (B, H_v, d_k)

        # 步骤 a：衰减旧状态
        state = state * g_exp.unsqueeze(-1).unsqueeze(-1)  # (B, H, d_k, d_v) * (B, H, 1, 1)

        # 步骤 b：检索当前状态对 key 的预测
        retrieved = torch.einsum('bhkv,bhk->bhv', state, k_t)  # (B, H_v, d_v)

        # 步骤 c：Delta = 新值与旧预测的误差，乘以更新强度
        delta = (v_t - retrieved) * beta_sq.unsqueeze(-1)       # (B, H_v, d_v)

        # 步骤 d：用外积写入纠偏
        state = state + torch.einsum('bhk,bhv->bhkv', k_t, delta)

        # 步骤 e：读出
        output = torch.einsum('bhkv,bhk->bhv', state, q_t)
        output = output / math.sqrt(self.key_head_dim)          # 缩放

        # ⑦ 门控 RMSNorm + 输出投影
        z_sq = z.squeeze(1)                                     # (B, H_v, d_v)
        output = self.norm(output) * F.silu(z_sq)               # 门控归一化
        output = self.out_proj(
            output.reshape(B, self.num_v_heads * self.value_head_dim).unsqueeze(1)
        )

        return output, state


class FullAttentionLayer(nn.Module):
    """Qwen 3.5 的全注意力层：标准 GQA + RoPE"""

    def __init__(self, hidden_size=4096, num_q_heads=32,
                 num_kv_heads=8, head_dim=128):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

    def forward(self, x, rope_fn=None, mask=None):
        """
        x: (B, T, hidden_size)
        rope_fn: RoPE 旋转位置编码函数
        """
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.num_q_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)

        # 施加 RoPE 旋转位置编码（全注意力层使用 RoPE）
        if rope_fn is not None:
            q, k = rope_fn(q, k)

        # GQA: 扩展 KV 头以匹配 Q 头数
        n_rep = self.num_q_heads // self.num_kv_heads
        k = k.repeat_interleave(n_rep, dim=2)
        v = v.repeat_interleave(n_rep, dim=2)

        # 转置为 (B, H, T, d) 方便矩阵乘法
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 标准缩放点积注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask                     # 因果掩码：-inf → softmax 后为 0
        weights = F.softmax(scores, dim=-1)
        output = torch.matmul(weights, v)              # (B, H, T, d)

        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(output)


class Qwen35Block(nn.Module):
    """单个 Transformer Block：注意力层 + MoE MLP 层"""

    def __init__(self, hidden_size, layer_type="linear"):
        super().__init__()
        self.layer_type = layer_type
        self.attn_norm = nn.RMSNorm(hidden_size)
        self.ffn_norm = nn.RMSNorm(hidden_size)

        if layer_type == "full_attention":
            self.attn = FullAttentionLayer(hidden_size)
        else:
            self.attn = GatedDeltaNetLayer(hidden_size)

        # 简化：实际使用 MoE SwiGLU，这里用单层 MLP 替代
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
        )

    def forward(self, x, rope_fn=None, mask=None, delta_state=None):
        # 注意力子层（带残差）
        h = self.attn_norm(x)
        if self.layer_type == "full_attention":
            x = x + self.attn(h, rope_fn=rope_fn, mask=mask)
        else:
            out, delta_state = self.attn.forward_recurrent(h, state=delta_state)
            x = x + out

        # MLP 子层（带残差）
        x = x + self.ffn(self.ffn_norm(x))
        return x, delta_state


class Qwen35Model(nn.Module):
    """简化的 Qwen 3.5 混合架构"""

    def __init__(self, hidden_size=4096, num_layers=32,
                 full_attention_interval=4):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            # 每 4 层插入一个全注意力层，其余用 Gated DeltaNet
            if (i + 1) % full_attention_interval == 0:
                self.layers.append(
                    Qwen35Block(hidden_size, layer_type="full_attention")
                )
            else:
                self.layers.append(
                    Qwen35Block(hidden_size, layer_type="linear")
                )

    def forward(self, x, rope_fn=None, mask=None):
        """
        x: (B, T, hidden_size)
        """
        delta_state = None
        for layer in self.layers:
            x, delta_state = layer(x, rope_fn=rope_fn, mask=mask,
                                   delta_state=delta_state)
        return x
```

---

## 9. 读完这篇之后，你应该能回答这些问题

1. Qwen 3.5 为什么不全部使用 Gated DeltaNet，而是要混入 25% 的全注意力层？如果全用线性注意力会怎样？
2. Gated DeltaNet 层里的因果卷积（kernel=4）起什么作用？为什么不直接用 RoPE？
3. 状态矩阵 $S \in \mathbb{R}^{d_k \times d_v}$ 在推理时的大小是多少？和全注意力层的 KV Cache 相比有什么优势？
4. Qwen 3.5 的 `Qwen3NextDynamicCache` 里，线性注意力层和全注意力层分别存了什么？它们的大小如何随序列长度变化？
5. MoE 的路由器是怎么决定每个 token 激活哪些专家的？397B 总参数只激活 17B 是怎么做到的？
6. Gated DeltaNet 更新公式中的 $g_t$（衰减门）和 $\beta_t$（更新强度）分别控制什么？它们都是输入依赖的吗？
7. Qwen 3.5 的 3:1 混合比例和 MiniMax M2.5 的纯 MHA 相比，各有什么优劣？在什么场景下你会更倾向其中一种？
8. 为什么 Qwen 3.5 的线性注意力层使用不对称的 K/V 头数（16 vs 32）？这和 GQA 的设计思想有什么联系？

---

## 参考资料

- [Qwen3.5 模型权重](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)
- [HuggingFace Qwen3.5 实现代码](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5)
- [Qwen3.5 Gated DeltaNet 技术分析 (GitHub Gist)](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)
- [Nobody Agrees on Attention Anymore (HuggingFace Blog)](https://huggingface.co/blog/mlabonne/qwen35)
- Gated DeltaNet 论文：Yang et al., "Gated Delta Networks: Improving Mamba2 with Delta Rule", ICLR 2025
- DeltaNet 论文：Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", NeurIPS 2024
- 本项目相关文章：`DeltaNet.md`、`Gated-DeltaNet.md`、`Qwen.md`
