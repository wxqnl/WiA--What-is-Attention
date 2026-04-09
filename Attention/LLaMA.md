# LLaMA 系列

> 对应论文：
> - `paper/LLaMA-LLaMA-Open-and-Efficient-Foundation-LMs.pdf`（Touvron et al., 2023）
> - `paper/LLaMA2-Open-Foundation-and-Fine-Tuned-Chat-Models.pdf`（Touvron et al., 2023）
> - LLaMA 3：[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)（Meta, 2024）

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

### 2.2 改动二：Pre-RMSNorm（非 Attention 改动，但影响训练稳定性）

原始 Transformer 在 Attention 和 FFN **之后**做 LayerNorm（Post-Norm）。LLaMA 改为在 **之前**做，并将 LayerNorm 换成计算更轻量的 **RMSNorm**：

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \cdot \gamma
$$

Pre-Norm 的好处是梯度更稳定，深层网络更容易训练。这不改变 Attention 的计算，但让 LLaMA 的架构更容易规模化。

### 2.3 LLaMA 1 的完整注意力计算

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

LLaMA 2 在 LLaMA 1 的基础上做了两处与 Attention 相关的改进：

### 3.1 GQA：用更少的 KV 头

LLaMA 2 的 70B 版本引入了 **GQA（Grouped Query Attention，分组查询注意力）**。

标准 Multi-Head Attention（MHA）中，每个 Query 头都有一组专属的 K 和 V，假设有 $h$ 个头，就需要 $h$ 组 KV。在推理时，这些 KV 需要全部缓存起来（即 **KV Cache**），显存开销随着序列长度线性增长。

GQA 的做法是：把 $h$ 个 Query 头分成若干**组**，每组共享同一对 K 和 V。例如 32 个 Query 头分 8 组，每组 4 个 Query 头共享 1 对 KV，KV 的数量就从 32 组降到了 8 组。

$$
\text{head}_i = \text{Attention}(Q_i W_i^Q,\; K_{\lfloor i / g \rfloor} W^K,\; V_{\lfloor i / g \rfloor} W^V)
$$

其中 $g$ 是每组的 Query 头数，$\lfloor i / g \rfloor$ 表示第 $i$ 个 Query 头对应第几组 KV。

GQA 将 KV Cache 的大小压缩为原来的 $1/g$，推理速度和显存占用大幅改善，质量损失几乎可以忽略。

> GQA 的详细原理和完整代码见 → [`GQA.md`](GQA.md)

### 3.2 上下文长度：2K → 4K

LLaMA 2 把训练时的上下文长度从 2048 扩展到 4096。这看起来只是一个参数变化，但背后需要确保 RoPE 的频率设置在 4K 长度下仍然能稳定区分位置，不会出现"旋转角度撞车"的问题。

LLaMA 2 沿用了 LLaMA 1 的 RoPE 频率设置（$\theta = 10000$），在 4K 长度内表现良好。

---

## 4. LLaMA 3：GQA 全面普及，上下文扩展到 128K

LLaMA 3 在 Attention 上的主要进展有两点：

### 4.1 全系列采用 GQA

LLaMA 2 只在 70B 版本中用 GQA，LLaMA 3 把 GQA 推广到**所有规模**（8B、70B、405B）。这反映了 GQA 已经被充分验证为"无损压缩"：不需要为小模型保留 MHA。

LLaMA 3.1（405B）的具体配置：

| 参数 | 数值 |
|:---|:---:|
| 总层数 | 126 |
| Query 头数 | 128 |
| KV 头数 | 8 |
| 头维度 | 128 |
| 每组 Query 头数 | 16 |

每 16 个 Query 头共享 1 对 KV，KV Cache 压缩为 MHA 的 $1/16$。

### 4.2 RoPE 频率调整，支持 128K 上下文

从 4K 扩展到 128K，RoPE 的频率参数 $\theta$ 需要相应调整。LLaMA 3 把 $\theta$ 从 $10000$ 提升到 $500000$，使得低频维度（对应长程位置关系）的旋转角度更密，能够在更长的序列上保持足够的分辨率。

直觉理解：$\theta$ 越大，最低频的旋转完成一圈所需的 token 数越多，也就能"看得更远"而不混淆。

---

## 5. 三个版本的架构演化对比

| | LLaMA 1 | LLaMA 2 | LLaMA 3 |
|:---|:---:|:---:|:---:|
| 基础架构 | Decoder-Only | Decoder-Only | Decoder-Only |
| 位置编码 | RoPE（$\theta=10000$） | RoPE（$\theta=10000$） | RoPE（$\theta=500000$） |
| Attention 类型 | MHA | MHA（7B/13B）/ GQA（70B） | GQA（全系列） |
| 上下文长度 | 2K | 4K | 8K–128K |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm | Pre-RMSNorm |
| 激活函数 | SwiGLU | SwiGLU | SwiGLU |
| 参数规模 | 7B–65B | 7B–70B | 8B–405B |

---

## 6. 完整模块伪代码

```python
def llama_attention(x, layer_idx, config):
    B, T, d = x.shape
    head_dim = d // config.n_heads

    # 线性投影：Q 有 n_heads 个头，KV 只有 n_kv_heads 个头（GQA）
    Q = x @ W_Q   # (B, T, n_heads * head_dim)
    K = x @ W_K   # (B, T, n_kv_heads * head_dim)
    V = x @ W_V   # (B, T, n_kv_heads * head_dim)

    # reshape 为多头形式
    Q = Q.view(B, T, config.n_heads, head_dim).transpose(1, 2)
    K = K.view(B, T, config.n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(B, T, config.n_kv_heads, head_dim).transpose(1, 2)

    # 施加 RoPE：只对 Q 和 K，不对 V
    Q = apply_rope(Q, cos[T], sin[T])
    K = apply_rope(K, cos[T], sin[T])

    # GQA：将 KV 重复扩展，对齐 Q 的头数
    # 每 (n_heads // n_kv_heads) 个 Q 头共享同一对 KV
    n_rep = config.n_heads // config.n_kv_heads
    K = K.repeat_interleave(n_rep, dim=1)   # (B, n_heads, T, head_dim)
    V = V.repeat_interleave(n_rep, dim=1)

    # 标准 Scaled Dot-Product Attention + 因果掩码
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim)
    scores = scores + causal_mask[:T, :T]
    weights = scores.softmax(dim=-1)
    out = weights @ V   # (B, n_heads, T, head_dim)

    # 合并多头，过输出投影
    out = out.transpose(1, 2).contiguous().view(B, T, d)
    return out @ W_O
```

---

## 7. 初学者常见混淆

**Q：LLaMA 的 RoPE 和原始 Transformer 的位置编码有什么本质区别？**

原始 Transformer 的位置信息是"加"到词向量里的，在进入 Attention 之前就固定了。RoPE 是在 Attention 计算**内部**、对 Q 和 K 做旋转，位置关系在点积计算时才体现出来。

**Q：GQA 和 MHA 的质量差距大吗？**

论文和工业实践都表明，当 KV 头数不太少时（比如 8 组以上），GQA 与 MHA 的质量差距极小，而推理速度和显存优势非常显著。LLaMA 3 405B 全系列使用 GQA 就是最好的证明。

**Q：LLaMA 3 的 128K 上下文是训练时就支持的，还是靠推理时扩展的？**

LLaMA 3.1 是在长文本数据上继续训练（long-context fine-tuning）并配合 RoPE 频率调整来实现的，不是纯靠推理时的 trick 外推。

**Q：LLaMA 系列是否改动了 Attention 的计算公式本身？**

没有。$\text{softmax}(QK^\top / \sqrt{d_k})V$ 这个公式一字未动。LLaMA 的改动是在**准备 Q 和 K 的方式**（RoPE）和**KV 的组织方式**（GQA）上，而不是 Attention 的核心数学。

---

## 8. 读完这篇之后，你应该能回答这些问题

- LLaMA 系列为什么选择用 RoPE 而不是绝对位置编码？RoPE 的旋转操作在数学上是如何把相对位置信息编入注意力计算的？
- GQA 相比 MHA 在推理时节省了什么？节省的比例和什么参数有关？
- LLaMA 1、2、3 在 Attention 机制上各做了什么具体改进？
- 为什么要把 RoPE 的频率参数 $\theta$ 从 10000 提升到 500000 才能支持 128K 上下文？
- `apply_rope` 函数对向量做的是什么变换？为什么只对 Q 和 K 做，不对 V 做？

---

## 参考资料

- `paper/LLaMA-LLaMA-Open-and-Efficient-Foundation-LMs.pdf`
- `paper/LLaMA2-Open-Foundation-and-Fine-Tuned-Chat-Models.pdf`
- The Llama 3 Herd of Models: https://arxiv.org/abs/2407.21783
- RoFormer（RoPE 原论文）: https://arxiv.org/abs/2104.09864
