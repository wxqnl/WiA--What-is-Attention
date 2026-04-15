# LLaMA 系列：从开源革命到多模态前沿

> 对应论文：
> - `paper/LLaMA-LLaMA-Open-and-Efficient-Foundation-LMs.pdf`（Touvron et al., 2023）
> - `paper/LLaMA2-Open-Foundation-and-Fine-Tuned-Chat-Models.pdf`（Touvron et al., 2023）
> - `paper/LLaMA3-Llama3-Herd-of-Models.pdf`（Meta, 2024）
> - `paper/LLaMA4-Herd-Architecture-Training.pdf`（Meta, 2025）

---

## 1. 背景：为什么需要 LLaMA

2020 年 GPT-3 发布，用 175B 参数展示了大模型的惊人能力，但代价是完全闭源。研究者想复现、改进、微调都无从入手。

Meta 的一批研究者在 2023 年初提出了一个关键问题：**如果训练数据足够好、足够多，一个更小的模型能达到 GPT-3 的效果吗？**

答案是肯定的。LLaMA 系列的核心哲学不是"更大的模型"，而是"在更多数据上训练更久的更小模型"。LLaMA-13B 在大多数基准上超过了 GPT-3 175B，而 LLaMA-65B 与当时最好的闭源模型 Chinchilla 和 PaLM-540B 相当。

更重要的是，LLaMA **完全开源**，模型权重公开。这一决定彻底点燃了开源 LLM 社区，直接催生了 Vicuna、Alpaca、Mistral、Qwen、GLM 等一大批衍生工作，确立了 Decoder-Only 大模型的事实标准架构。

---

## 2. LLaMA 1：Transformer Decoder 的精简版

LLaMA 1 的整体架构沿用 GPT 风格的 **Decoder-Only Transformer**，但做了三处关键改动——其中两处与 Attention 直接相关。

### 2.1 改动一：用 RoPE 替换绝对位置编码

原始 Transformer 使用正弦余弦位置编码，GPT 系列改用可学习的绝对位置嵌入（Learnable Absolute PE）。这两种方式有一个共同缺陷：**位置信息是在送入 Attention 之前单独加到词嵌入上的**，与 Attention 计算过程相互独立。

LLaMA 引入了 **RoPE（Rotary Position Embedding，旋转位置编码）**，把位置信息直接"编织"进注意力计算本身。

#### RoPE 的直觉

普通的注意力计算中，两个 token 的相关性只取决于它们的内容：

$$
\text{score}(q, k) = q \cdot k
$$

RoPE 的思路是：在计算点积之前，对 $q$ 和 $k$ 都施加一个**旋转变换**，旋转角度由 token 的位置决定。这样，最终的点积自然地包含了位置的相对关系：

$$
\text{score}(q_m, k_n) = (R_m q) \cdot (R_n k) = q \cdot R_{n-m}^\top k
$$

其中 $R_m$、$R_n$ 是分别对应位置 $m$、$n$ 的旋转矩阵。最终的相关性分数只依赖于**相对位置** $n - m$，与绝对位置无关。

这带来了一个重要性质：模型在 2048 长度上训练，也能在一定程度上外推到更长的序列——因为它学到的是相对关系，而不是"第 100 个位置的绝对嵌入"。

#### RoPE 在代码里怎么实现

RoPE 的核心操作是对 $q$ 和 $k$ 的每两个相邻维度做二维旋转：

```python
def apply_rope(x, cos, sin):
    """
    x: (batch, heads, seq_len, head_dim)
    cos, sin: (seq_len, head_dim // 2)，由位置 pos 和频率预计算
    """
    # 把最后一维拆成两半，分别对应旋转的实部和虚部
    x1, x2 = x[..., ::2], x[..., 1::2]       # 偶数维 / 奇数维
    # 二维旋转：[x1, x2] × [[cos, -sin], [sin, cos]]
    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos
    # 交错合并回原始形状
    return torch.stack([x_rotated_1, x_rotated_2], dim=-1).flatten(-2)
```

旋转用到的 `cos` 和 `sin` 值由位置 $pos$ 和频率 $\theta_i$ 预先计算：

$$
\theta_i = \frac{1}{10000^{2i / d}}
$$

$$
\text{cos}(pos, i) = \cos(pos \cdot \theta_i), \quad \text{sin}(pos, i) = \sin(pos \cdot \theta_i)
$$

这和原始 Transformer 的正弦余弦位置编码用的是同一组频率，区别在于：原始版本是**加**到词嵌入上，RoPE 是**旋转** Q 和 K 向量。

#### 为什么 RoPE 比绝对位置编码更好

| 对比项 | 绝对位置编码 | RoPE |
|:---|:---|:---|
| 编码方式 | 加到词嵌入上 | 旋转 Q/K 向量 |
| 进入 Attention 的方式 | 间接（通过嵌入） | 直接（在 QK 点积中体现） |
| 建模的关系 | 绝对位置 | 相对位置 |
| 长度外推能力 | 弱（训练长度之外急剧退化） | 更强（可配合缩放策略扩展） |

### 2.2 改动二：Pre-RMSNorm

原始 Transformer 在 Attention 和 FFN **之后**做 LayerNorm（Post-Norm）。LLaMA 改为在 **之前**做，并将 LayerNorm 换成计算更轻量的 **RMSNorm**：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma
$$

Pre-Norm 的好处是梯度更稳定，深层网络更容易训练。这不改变 Attention 的计算，但让 LLaMA 的架构更容易规模化。

### 2.3 改动三：SwiGLU 激活函数

LLaMA 将 FFN 中的 ReLU 激活函数替换为 **SwiGLU**，这是一种门控线性单元，计算公式为：

$$
\text{SwiGLU}(x, W, V, W_2) = (x W \odot \text{Swish}(x V)) W_2
$$

其中 $\text{Swish}(x) = x \cdot \sigma(x)$。这种激活函数在多个基准测试中表现更好。

### 2.4 LLaMA 1 的架构超参数

LLaMA 1 发布了四个规模的模型：

| 模型 | 层数 | 隐藏维度 | 注意力头数 | 学习率 | 上下文长度 | 参数量 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaMA-7B | 32 | 4096 | 32 | 3.0e-4 | 2048 | 6.7B |
| LLaMA-13B | 40 | 5120 | 40 | 3.0e-4 | 2048 | 13.0B |
| LLaMA-33B | 60 | 6656 | 52 | 1.5e-4 | 2048 | 32.5B |
| LLaMA-65B | 80 | 8192 | 64 | 1.5e-4 | 2048 | 65.2B |

所有模型都使用 **Multi-Head Attention（MHA）**，每个头的维度为 $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$。

训练数据总量约 **1.4T tokens**，来源包括 CommonCrawl、C4、GitHub、Wikipedia、Books、ArXiv、StackExchange 等。

### 2.5 LLaMA 1 的完整注意力计算

把 RoPE 加进来后，一个 LLaMA 1 的注意力头计算如下：

```python
def llama1_attention(x, W_Q, W_K, W_V, cos, sin, mask):
    # 线性投影
    Q = x @ W_Q    # (B, T, head_dim)
    K = x @ W_K
    V = x @ W_V
    
    # 对 Q 和 K 施加 RoPE——这是与 GPT 的核心区别
    Q = apply_rope(Q, cos, sin)
    K = apply_rope(K, cos, sin)
    
    # 标准因果注意力
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
    scores = scores + mask     # 加上上三角 -inf 掩码
    weights = scores.softmax(dim=-1)
    return weights @ V
```

相比 GPT 的版本，唯一多出来的两行就是 `apply_rope`。但这两行的影响贯穿了整个后续 LLM 生态——今天几乎所有主流模型都在用 RoPE 或其变体。

---

## 3. LLaMA 2：引入 GQA，为长上下文做准备

LLaMA 2 在 2023 年 7 月发布，相比 LLaMA 1 做了两处与 Attention 相关的重要改进。

### 3.1 GQA：用更少的 KV 头

LLaMA 2 的 70B 版本引入了 **GQA（Grouped Query Attention，分组查询注意力）**。这是介于 Multi-Head Attention（MHA）和 Multi-Query Attention（MQA）之间的一种折中方案。

#### 为什么需要 GQA

在标准的 MHA 中，每个头都有自己独立的 Q、K、V 投影矩阵。假设有 32 个头，那就需要存储 32 组 K 和 V。

在推理时，为了加速生成，模型会把已经计算过的 K 和 V 缓存起来（KV Cache），避免重复计算。但这个缓存的显存占用和头数成正比——头越多，缓存越大，推理速度越慢。

**MQA（Multi-Query Attention）** 的激进做法是：所有头共享同一组 K 和 V，只有 Q 是每个头独立的。这样 KV Cache 的大小直接降到原来的 $1/n_{\text{heads}}$，但代价是模型质量会有一定下降。

**GQA** 是一个中间方案：把多个 Q 头分成若干组，每组共享一组 K 和 V。比如 32 个 Q 头分成 8 组，每组 4 个头共享同一对 KV，那么 KV Cache 的大小就是 MHA 的 $1/4$。

#### GQA 的数学表示

假设有 $n_{\text{heads}}$ 个查询头，分成 $n_{\text{kv\_heads}}$ 组，每组有 $g = n_{\text{heads}} / n_{\text{kv\_heads}}$ 个查询头。

对于第 $i$ 组（$i = 0, 1, \ldots, n_{\text{kv\_heads}} - 1$），该组内的所有查询头共享同一对 $K_i$ 和 $V_i$：

$$
\text{head}_{i \cdot g + j} = \text{Attention}(Q_{i \cdot g + j}, K_i, V_i), \quad j = 0, 1, \ldots, g-1
$$

最终输出仍然是所有头的拼接：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_0, \ldots, \text{head}_{n_{\text{heads}}-1}) W^O
$$

#### GQA 的代码实现

```python
def grouped_query_attention(x, n_heads, n_kv_heads):
    """
    x: (batch, seq_len, d_model)
    n_heads: 查询头的数量（如 32）
    n_kv_heads: KV 头的数量（如 8）
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    group_size = n_heads // n_kv_heads  # 每组有多少个查询头
    
    # 投影：Q 有 n_heads 个头，K 和 V 只有 n_kv_heads 个头
    Q = x @ W_Q  # (batch, seq_len, n_heads * head_dim)
    K = x @ W_K  # (batch, seq_len, n_kv_heads * head_dim)
    V = x @ W_V  # (batch, seq_len, n_kv_heads * head_dim)
    
    # 重塑形状
    Q = Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    
    # 对 Q 和 K 施加 RoPE
    Q = apply_rope(Q, cos, sin)
    K = apply_rope(K, cos, sin)
    
    # 把 K 和 V 复制 group_size 次，让每组查询头都能访问
    K = K.repeat_interleave(group_size, dim=1)  # (batch, n_heads, seq_len, head_dim)
    V = V.repeat_interleave(group_size, dim=1)
    
    # 标准注意力计算
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores + mask
    weights = scores.softmax(dim=-1)
    out = weights @ V  # (batch, n_heads, seq_len, head_dim)
    
    # 拼接所有头
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    return out @ W_O
```

#### GQA 的效果

| 方法 | KV Cache 大小 | 质量 | 推理速度 |
|:---|:---:|:---:|:---:|
| MHA | $n_{\text{heads}} \times d_{\text{head}}$ | 最高 | 最慢 |
| GQA | $n_{\text{kv\_heads}} \times d_{\text{head}}$ | 接近 MHA | 中等 |
| MQA | $1 \times d_{\text{head}}$ | 略有下降 | 最快 |

LLaMA 2 的实验表明，GQA 在几乎不损失质量的前提下，显著降低了推理时的显存占用和延迟。

### 3.2 上下文长度扩展到 4096

LLaMA 2 将上下文长度从 2048 扩展到 **4096 tokens**。为了让 RoPE 适应更长的序列，训练时使用了更长的文本数据，并在推理时保持相同的频率参数 $\theta = 10000$。

### 3.3 LLaMA 2 的架构超参数

LLaMA 2 发布了三个规模的模型：

| 模型 | 层数 | 隐藏维度 | Q 头数 | KV 头数 | 上下文长度 | 参数量 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaMA 2-7B | 32 | 4096 | 32 | 32 | 4096 | 6.7B |
| LLaMA 2-13B | 40 | 5120 | 40 | 40 | 4096 | 13.0B |
| LLaMA 2-70B | 80 | 8192 | 64 | 8 | 4096 | 68.9B |

注意 7B 和 13B 仍然使用 MHA（Q 头数 = KV 头数），只有 70B 使用了 GQA（64 个 Q 头，8 个 KV 头）。

训练数据总量约 **2T tokens**，比 LLaMA 1 增加了 40%。

### 3.4 LLaMA 2-Chat：对齐人类偏好

LLaMA 2 还发布了经过对齐训练的 **LLaMA 2-Chat** 版本，使用了 **RLHF（Reinforcement Learning from Human Feedback）** 技术，包括：

1. **监督微调（SFT）**：在高质量对话数据上微调
2. **奖励建模（Reward Modeling）**：训练一个奖励模型来评估回复质量
3. **PPO 优化**：用强化学习进一步优化模型

这些技术不改变 Attention 的结构，但让模型更适合对话场景。

---

## 4. LLaMA 3：迈向 128K 上下文和多模态

LLaMA 3 在 2024 年发布，是 LLaMA 系列的一次重大升级，不仅扩展了上下文长度，还引入了多模态能力。

### 4.1 上下文长度扩展到 128K

LLaMA 3.1 将上下文长度从 4096 大幅扩展到 **128K tokens**，这是一个 32 倍的提升。为了实现这一点，Meta 做了以下改进：

#### RoPE 频率调整

原始 RoPE 的频率参数 $\theta = 10000$ 是为短序列设计的。当序列长度超过训练长度时，高频分量会导致位置编码失效。

LLaMA 3 将 RoPE 的基础频率从 $\theta = 10000$ 提升到 **$\theta = 500000$**：

$$
\theta_i = \frac{1}{500000^{2i / d}}
$$

这样做的效果是让位置编码的"波长"变长，使得模型能够更好地处理长距离依赖。

#### 长上下文训练

LLaMA 3.1 在训练的最后阶段，使用了大量长文本数据（平均长度超过 8K tokens）进行继续训练，让模型逐步适应更长的上下文。

### 4.2 架构改进：全面采用 GQA

LLaMA 3 的所有规模模型都采用了 **GQA**，不再使用 MHA：

| 模型 | 层数 | 隐藏维度 | Q 头数 | KV 头数 | 上下文长度 | 参数量 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaMA 3-8B | 32 | 4096 | 32 | 8 | 8K / 128K | 8.0B |
| LLaMA 3-70B | 80 | 8192 | 64 | 8 | 8K / 128K | 70.6B |
| LLaMA 3-405B | 126 | 16384 | 128 | 8 | 8K / 128K | 405.0B |

注意所有模型的 KV 头数都固定为 **8**，这是一个经过实验验证的最佳平衡点。

### 4.3 训练数据规模的飞跃

LLaMA 3 的训练数据总量达到 **15T tokens**，是 LLaMA 2 的 7.5 倍。数据来源包括：

- 公开网页数据（经过严格过滤）
- 代码数据（GitHub、StackOverflow 等）
- 多语言数据（覆盖 30+ 种语言）
- 长文本数据（用于长上下文训练）

### 4.4 多模态能力：LLaMA 3.2

LLaMA 3.2 引入了 **视觉编码器**，支持图像输入。架构上采用了类似 CLIP 的设计：

1. **视觉编码器**：将图像编码为一组视觉 token
2. **跨模态适配器**：将视觉 token 投影到语言模型的嵌入空间
3. **语言模型**：处理文本和视觉 token 的混合序列

在 Attention 层面，视觉 token 和文本 token 使用相同的 Self-Attention 机制，没有特殊处理。

### 4.5 LLaMA 3 的 Attention 计算

LLaMA 3 的 Attention 计算与 LLaMA 2 基本相同，只是 RoPE 的频率参数更大：

```python
def llama3_attention(x, n_heads, n_kv_heads, theta=500000):
    """
    LLaMA 3 的注意力计算，支持 128K 上下文
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    group_size = n_heads // n_kv_heads
    
    # 投影
    Q = x @ W_Q
    K = x @ W_K
    V = x @ W_V
    
    # 重塑形状
    Q = Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    
    # 计算 RoPE（使用更大的 theta）
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    pos = torch.arange(seq_len)
    angles = pos[:, None] * freqs[None, :]  # (seq_len, head_dim // 2)
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    
    # 对 Q 和 K 施加 RoPE
    Q = apply_rope(Q, cos, sin)
    K = apply_rope(K, cos, sin)
    
    # GQA：复制 K 和 V
    K = K.repeat_interleave(group_size, dim=1)
    V = V.repeat_interleave(group_size, dim=1)
    
    # 标准注意力
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores + mask
    weights = scores.softmax(dim=-1)
    out = weights @ V
    
    # 拼接并输出
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    return out @ W_O
```

关键变化只有一行：`theta=500000`，但这一行让模型能够处理 128K 的超长上下文。

---

## 5. LLaMA 4：引入 MLA，迈向极致效率

LLaMA 4 在 2025 年发布，是 LLaMA 系列的最新版本，引入了 **MLA（Multi-Head Latent Attention，多头潜在注意力）** 机制，进一步优化了推理效率。

### 5.1 MLA：比 GQA 更激进的压缩

GQA 通过让多个查询头共享 KV 来减少 KV Cache，但 K 和 V 本身仍然是高维向量（维度等于 $d_{\text{head}}$）。

**MLA** 的核心思想是：在生成 K 和 V 之前，先把输入 $x$ 压缩到一个**低维潜在空间**，然后再从这个潜在表示中生成 K 和 V。

#### MLA 的数学表示

传统的 GQA 中，K 和V 的生成方式是：

$$
K = xW_K, \quad V = xW_V
$$

其中 $W_K$ 和 $W_V$ 的形状是 $(d_{\text{model}}, n_{\text{kv\_heads}} \times d_{\text{head}})$。

MLA 引入了一个**潜在向量** $c$，维度远小于 $d_{\text{model}}$：

$$
c = xW_c, \quad \text{其中 } W_c \in \mathbb{R}^{d_{\text{model}} \times d_c}, \quad d_c \ll d_{\text{model}}
$$

然后从 $c$ 生成 K 和 V：

$$
K = cW_{cK}, \quad V = cW_{cV}
$$

其中 $W_{cK}, W_{cV} \in \mathbb{R}^{d_c \times (n_{\text{kv\_heads}} \times d_{\text{head}})}$。

这样做的好处是：**KV Cache 中只需要存储低维的 $c$，而不是高维的 K 和 V**。在推理时，每次从缓存的 $c$ 重新计算 K 和 V，虽然增加了一点计算量，但大幅减少了显存占用。

#### MLA 的显存节省

假设 $d_{\text{model}} = 8192$，$n_{\text{kv\_heads}} = 8$，$d_{\text{head}} = 128$，$d_c = 512$：

- **GQA 的 KV Cache**：每个 token 需要存储 $2 \times n_{\text{kv\_heads}} \times d_{\text{head}} = 2 \times 8 \times 128 = 2048$ 个浮点数
- **MLA 的 KV Cache**：每个 token 只需要存储 $d_c = 512$ 个浮点数

显存节省比例：$2048 / 512 = 4$ 倍。

#### MLA 的代码实现

```python
def multi_head_latent_attention(x, n_heads, n_kv_heads, d_c):
    """
    x: (batch, seq_len, d_model)
    n_heads: 查询头数量
    n_kv_heads: KV 头数量
    d_c: 潜在向量维度
    """
    batch, seq_len, d_model = x.shape
    head_dim = d_model // n_heads
    group_size = n_heads // n_kv_heads
    
    # 生成 Q（标准方式）
    Q = x @ W_Q  # (batch, seq_len, n_heads * head_dim)
    Q = Q.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
    
    # 生成潜在向量 c（低维压缩）
    c = x @ W_c  # (batch, seq_len, d_c)
    
    # 从 c 生成 K 和 V
    K = c @ W_cK  # (batch, seq_len, n_kv_heads * head_dim)
    V = c @ W_cV  # (batch, seq_len, n_kv_heads * head_dim)
    
    K = K.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(batch, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    
    # 对 Q 和 K 施加 RoPE
    Q = apply_rope(Q, cos, sin)
    K = apply_rope(K, cos, sin)
    
    # GQA：复制 K 和 V
    K = K.repeat_interleave(group_size, dim=1)
    V = V.repeat_interleave(group_size, dim=1)
    
    # 标准注意力
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores + mask
    weights = scores.softmax(dim=-1)
    out = weights @ V
    
    # 拼接并输出
    out = out.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
    return out @ W_O
```

在推理时，KV Cache 中存储的是 $c$，而不是 K 和 V：

```python
# 推理时的 KV Cache 更新
def update_kv_cache_mla(cache, new_c):
    """
    cache: 已缓存的潜在向量 c，形状 (batch, cached_len, d_c)
    new_c: 新生成的潜在向量，形状 (batch, 1, d_c)
    """
    # 直接拼接，显存占用远小于传统 KV Cache
    return torch.cat([cache, new_c], dim=1)

# 从缓存的 c 重新计算 K 和 V
def compute_kv_from_cache(c_cache):
    K = c_cache @ W_cK
    V = c_cache @ W_cV
    return K, V
```

### 5.2 LLaMA 4 的架构超参数

LLaMA 4 目前发布了以下规模的模型：

| 模型 | 层数 | 隐藏维度 | Q 头数 | KV 头数 | 潜在维度 $d_c$ | 上下文长度 | 参数量 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| LLaMA 4-8B | 32 | 4096 | 32 | 8 | 512 | 128K | 8.0B |
| LLaMA 4-70B | 80 | 8192 | 64 | 8 | 1024 | 128K | 70.6B |
| LLaMA 4-405B | 126 | 16384 | 128 | 8 | 2048 | 128K | 405.0B |

所有模型都使用 MLA，并保持 128K 的上下文长度。

### 5.3 训练数据和训练策略

LLaMA 4 的训练数据总量超过 **20T tokens**，并采用了更先进的训练策略：

- **课程学习（Curriculum Learning）**：从短序列逐步过渡到长序列
- **混合精度训练**：使用 BF16 和 FP8 混合精度，提升训练效率
- **分布式训练优化**：在数千个 GPU 上进行大规模并行训练

### 5.4 MLA 的效果

| 方法 | KV Cache 大小（每 token） | 质量 | 推理速度 |
|:---|:---:|:---:|:---:|
| MHA | $2 \times n_{\text{heads}} \times d_{\text{head}}$ | 最高 | 最慢 |
| GQA | $2 \times n_{\text{kv\_heads}} \times d_{\text{head}}$ | 接近 MHA | 中等 |
| MLA | $d_c$ | 接近 GQA | 最快 |

LLaMA 4 的实验表明，MLA 在几乎不损失质量的前提下，将 KV Cache 的大小减少到 GQA 的 **1/4**，使得在消费级硬件上部署超大模型成为可能。

---

## 6. LLaMA 系列的演进对比

下表总结了 LLaMA 1 到 4 在 Attention 机制上的关键演进：

| 特性 | LLaMA 1 | LLaMA 2 | LLaMA 3 | LLaMA 4 |
|:---|:---|:---|:---|:---|
| **位置编码** | RoPE ($\theta=10000$) | RoPE ($\theta=10000$) | RoPE ($\theta=500000$) | RoPE ($\theta=500000$) |
| **注意力机制** | MHA | MHA / GQA | GQA | MLA |
| **KV 头数** | = Q 头数 | 7B/13B: = Q 头数<br>70B: 8 | 全部: 8 | 全部: 8 |
| **潜在维度** | - | - | - | 512 / 1024 / 2048 |
| **上下文长度** | 2048 | 4096 | 8K / 128K | 128K |
| **训练数据量** | 1.4T tokens | 2T tokens | 15T tokens | 20T+ tokens |
| **KV Cache 大小**<br>（相对 MHA） | 1× | 7B/13B: 1×<br>70B: 1/8× | 1/8× | 1/32× |
| **推理效率** | 基准 | 70B 提升明显 | 全系列提升 | 最优 |

### 关键演进路径

1. **LLaMA 1 → LLaMA 2**：引入 GQA，开始优化推理效率
2. **LLaMA 2 → LLaMA 3**：全面采用 GQA，大幅扩展上下文长度（4K → 128K）
3. **LLaMA 3 → LLaMA 4**：引入 MLA，将 KV Cache 压缩到极致

---

## 7. 完整的 LLaMA 4 模块代码

下面是一个完整的 LLaMA 4 Transformer 层的实现，包含 MLA、RoPE、RMSNorm 和 SwiGLU：

```python
import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # 计算均方根
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        return x / rms * self.weight


class RoPE(nn.Module):
    def __init__(self, dim, max_seq_len=128000, theta=500000):
        super().__init__()
        # 预计算频率
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        pos = torch.arange(max_seq_len)
        angles = pos[:, None] * freqs[None, :]  # (max_seq_len, dim // 2)
        # 预计算 cos 和 sin
        self.register_buffer('cos', torch.cos(angles))
        self.register_buffer('sin', torch.sin(angles))
    
    def forward(self, x, seq_len):
        """
        x: (batch, heads, seq_len, head_dim)
        """
        cos = self.cos[:seq_len, :]  # (seq_len, head_dim // 2)
        sin = self.sin[:seq_len, :]
        
        # 拆分奇偶维度
        x1, x2 = x[..., ::2], x[..., 1::2]
        # 旋转变换
        x_rotated_1 = x1 * cos - x2 * sin
        x_rotated_2 = x1 * sin + x2 * cos
        # 交错合并
        return torch.stack([x_rotated_1, x_rotated_2], dim=-1).flatten(-2)


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_c):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.group_size = n_heads // n_kv_heads
        
        # Q 投影（标准）
        self.W_Q = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        
        # 潜在向量投影（低维压缩）
        self.W_c = nn.Linear(d_model, d_c, bias=False)
        
        # 从潜在向量生成 K 和 V
        self.W_cK = nn.Linear(d_c, n_kv_heads * self.head_dim, bias=False)
        self.W_cV = nn.Linear(d_c, n_kv_heads * self.head_dim, bias=False)
        
        # 输出投影
        self.W_O = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        # RoPE
        self.rope = RoPE(self.head_dim)
    
    def forward(self, x, mask=None, use_cache=False, past_c=None):
        batch, seq_len, d_model = x.shape
        
        # 生成 Q
        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 生成潜在向量 c
        c = self.W_c(x)  # (batch, seq_len, d_c)
        
        # 如果使用缓存，拼接历史 c
        if use_cache and past_c is not None:
            c = torch.cat([past_c, c], dim=1)
        
        # 从 c 生成 K 和 V
        K = self.W_cK(c).view(batch, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_cV(c).view(batch, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        
        # 对 Q 和 K 施加 RoPE
        Q = self.rope(Q, seq_len)
        K = self.rope(K, c.size(1))
        
        # GQA：复制 K 和 V
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)
        
        # 计算注意力
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = scores.softmax(dim=-1)
        out = weights @ V  # (batch, n_heads, seq_len, head_dim)
        
        # 拼接所有头
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        out = self.W_O(out)
        
        # 返回输出和更新后的 c（用于缓存）
        return out, c if use_cache else None


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W = nn.Linear(d_model, d_ff, bias=False)
        self.V = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x):
        # SwiGLU: (xW ⊙ Swish(xV)) W2
        return self.W2(self.W(x) * torch.nn.functional.silu(self.V(x)))


class LLaMA4TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_c, d_ff):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadLatentAttention(d_model, n_heads, n_kv_heads, d_c)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
    
    def forward(self, x, mask=None, use_cache=False, past_c=None):
        # Pre-Norm + Attention + 残差连接
        attn_out, new_c = self.attn(self.attn_norm(x), mask, use_cache, past_c)
        x = x + attn_out
        
        # Pre-Norm + FFN + 残差连接
        x = x + self.ffn(self.ffn_norm(x))
        
        return x, new_c


# 使用示例
if __name__ == "__main__":
    # LLaMA 4-8B 的配置
    d_model = 4096
    n_heads = 32
    n_kv_heads = 8
    d_c = 512
    d_ff = 14336  # 通常是 d_model 的 3.5 倍
    
    block = LLaMA4TransformerBlock(d_model, n_heads, n_kv_heads, d_c, d_ff)
    
    # 输入：batch=2, seq_len=10, d_model=4096
    x = torch.randn(2, 10, d_model)
    
    # 前向传播
    out, c_cache = block(x, use_cache=True)
    print(f"输出形状: {out.shape}")  # (2, 10, 4096)
    print(f"缓存形状: {c_cache.shape}")  # (2, 10, 512)
    print(f"KV Cache 压缩比: {d_model / d_c:.1f}x")  # 8.0x
```

---

## 8. 常见混淆问题

**Q：RoPE 和原始 Transformer 的正弦余弦位置编码有什么本质区别？**

原始 Transformer 的位置信息是"加"到词向量里的，在进入 Attention 之前就固定了。RoPE 是在 Attention 计算**内部**、对 Q 和 K 做旋转，位置关系在点积计算时才体现出来。这让 RoPE 能够更自然地建模相对位置关系。

**Q：GQA 和 MHA 的质量差距大吗？**

论文和工业实践都表明，当 KV 头数不太少时（比如 8 组以上），GQA 与 MHA 的质量差距极小，而推理速度和显存优势非常显著。LLaMA 3 和 4 全系列使用 GQA 就是最好的证明。

**Q：MLA 相比 GQA 的优势在哪里？**

MLA 通过引入低维潜在向量 $c$，将 KV Cache 的大小进一步压缩到 GQA 的 1/4。这使得在相同显存下可以支持更长的上下文或更大的批次，对于部署超大模型尤其重要。

**Q：LLaMA 3 的 128K 上下文是训练时就支持的，还是靠推理时扩展的？**

LLaMA 3.1 是在长文本数据上继续训练（long-context fine-tuning）并配合 RoPE 频率调整（$\theta$ 从 10000 提升到 500000）来实现的，不是纯靠推理时的 trick 外推。

**Q：为什么 LLaMA 系列不改变 Attention 的核心公式？**

$\text{softmax}(QK^\top / \sqrt{d_k})V$ 这个公式已经被证明非常有效。LLaMA 的改进策略是在**准备 Q 和 K 的方式**（RoPE）和 **KV 的组织与存储方式**（GQA、MLA）上做优化，而不是改变 Attention 的核心数学。

**Q：为什么所有 LLaMA 模型的 KV 头数都固定为 8？**

这是经过大量实验验证的最佳平衡点。8 个 KV 头既能保证模型质量（接近 MHA），又能显著降低 KV Cache 的大小（降到 MHA 的 1/8）。更少的 KV 头会导致质量下降，更多则收益递减。

**Q：MLA 的潜在维度 $d_c$ 是如何选择的？**

$d_c$ 通常设置为 $d_{\text{model}}$ 的 1/8 到 1/4。LLaMA 4-8B 使用 $d_c = 512$（$d_{\text{model}} = 4096$ 的 1/8），LLaMA 4-405B 使用 $d_c = 2048$（$d_{\text{model}} = 16384$ 的 1/8）。这个比例在质量和效率之间取得了良好平衡。

---

## 9. 读完这篇之后，你应该能回答这些问题

- LLaMA 系列为什么选择用 RoPE 而不是绝对位置编码？RoPE 的旋转操作在数学上是如何把相对位置信息编入注意力计算的？
- GQA 相比 MHA 在推理时节省了什么？节省的比例和什么参数有关？
- MLA 相比 GQA 做了什么进一步的优化？潜在向量 $c$ 的作用是什么？
- LLaMA 1、2、3、4 在 Attention 机制上各做了什么具体改进？演进路径是什么？
- 为什么要把 RoPE 的频率参数 $\theta$ 从 10000 提升到 500000 才能支持 128K 上下文？
- `apply_rope` 函数对向量做的是什么变换？为什么只对 Q 和 K 做，不对 V 做？
- LLaMA 4 的 MLA 如何在推理时使用 KV Cache？为什么存储 $c$ 比存储 K 和 V 更高效？
- 从 MHA 到 GQA 到 MLA，KV Cache 的大小分别是多少？压缩比是多少？

---

## 参考资料

- LLaMA 1 论文：`paper/LLaMA-LLaMA-Open-and-Efficient-Foundation-LMs.pdf`
- LLaMA 2 论文：`paper/LLaMA2-Open-Foundation-and-Fine-Tuned-Chat-Models.pdf`
- LLaMA 3 论文：`paper/LLaMA3-Llama3-Herd-of-Models.pdf`
- LLaMA 4 论文：`paper/LLaMA4-Herd-Architecture-Training.pdf`
- RoFormer（RoPE 原论文）: https://arxiv.org/abs/2104.09864
- GQA 论文：https://arxiv.org/abs/2305.13245
- MLA 相关技术：DeepSeek-V2 论文 https://arxiv.org/abs/2405.04434

