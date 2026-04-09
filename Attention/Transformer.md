# Transformer

> 对应论文：`paper/Transformer-Attention-Is-All-You-Need.pdf`（Vaswani et al., 2017）

---

## 1. 从 RNN 到 Transformer：为什么我们需要一种新架构

在 Transformer 出现之前，NLP 任务的主角是循环神经网络（RNN）以及它的改良版 LSTM。它们之所以流行，是因为文本天然是序列，而 RNN 的设计就是为序列量身打造的——它会逐个读入每个 token，同时维护一个"隐藏状态"来携带历史信息，直到读完整个句子。

但 RNN 有两个几乎无法回避的结构性缺陷：

**第一，无法并行计算。** RNN 必须先读完第 1 个 token，才能计算第 2 个，再才能计算第 3 个……这种串行的依赖关系意味着，即便你有再多的 GPU 核心，也只能让它们排队等待，极大地浪费了现代硬件的并行计算能力。

**第二，长距离依赖难以捕捉。** 句子里两个相距很远的词之间的关系，需要通过隐藏状态一步一步"传递"过来。路越长，信息衰减越严重。LSTM 通过门控机制做了一定程度的改善，但在非常长的序列上仍然力不从心。

Vaswani 等人在 2017 年提出 Transformer，核心想法很简洁：**既然信息在 RNN 里只能沿时间步一点点流动，为什么不让每个 token 直接去"看"整个句子中和自己最相关的部分呢？**

这个"看"的机制，就是注意力（Attention）。

---

## 2. 注意力机制：从直觉到公式

### 2.1 一个让人一下就懂的类比

设想你在图书馆查资料，手边有一本字典：

```
{
    "苹果": 10,
    "香蕉":  5,
    "椅子":  2
}
```

字典里，键（Key）是词条，值（Value）是对应的数据。如果你要查"苹果"，直接精确匹配就好了，很简单。

但假如你要查的是"水果"呢？字典里没有这个词条。你只能根据"水果"这个查询（Query）和每个键的相关性，给它们分别打一个权重：

```
"苹果" → 0.6
"香蕉" → 0.4
"椅子" → 0.0
```

然后用这组权重对所有值做加权求和：

$$
\text{结果} = 0.6 \times 10 + 0.4 \times 5 + 0.0 \times 2 = 8
$$

这就是注意力机制的核心逻辑：

- **Query（查询）**：你想找的是什么
- **Key（键）**：每个候选项上有哪些可供匹配的特征
- **Value（值）**：每个候选项真正要被读出的内容

**注意力 = 用 Query 与所有 Key 计算相关性 → 得到权重 → 用权重对 Value 加权求和**

### 2.2 用向量表示相关性

真实的注意力机制里，Query、Key、Value 都是向量，不是字符串。那怎么衡量两个向量的相关性？

一个自然的选择是**点积**。如果两个向量方向接近，点积就大；方向相反，点积就小；互相正交，点积接近零。这和我们对"相关性"的直觉是吻合的。

所以，给定一个 Query 向量 $q$ 和一组 Key 向量（堆叠成矩阵 $K$），我们可以一次性计算 Query 与所有 Key 的相关性：

$$
\text{相关性分数} = qK^\top
$$

### 2.3 用 Softmax 把分数变成权重

光有相关性分数还不够，我们需要把它转化成"权重"——所有权重加起来为 1，才能做有意义的加权求和。

Softmax 正是做这件事的：

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

它会把任意一组实数映射成一个概率分布：每个值都在 $(0, 1)$ 之间，且所有值之和为 1。分数越高的位置，最终获得的权重就越大。

### 2.4 完整的注意力公式

把上面几步串起来：计算相关性 → Softmax 得到权重 → 对 Value 加权求和，就得到了注意力机制的基本公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}(QK^\top)\, V
$$

这里 $Q$、$K$、$V$ 都是矩阵，对应一个序列中所有位置的 Query、Key、Value 向量的集合，允许我们一次性并行计算所有位置的注意力结果。

### 2.5 为什么要除以 $\sqrt{d_k}$

上面的公式还差最后一步优化。当向量维度 $d_k$ 较大时，$QK^\top$ 的每个元素是 $d_k$ 个数的乘积之和，数值会变得很大。把很大的数送进 Softmax，会导致某几个位置的权重接近 1，其余位置接近 0——权重分布极端尖锐，梯度几乎为零，模型很难训练。

解决方法很简单：除以 $\sqrt{d_k}$ 做一个缩放，把数值压回到合理范围：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

这就是论文中的 **Scaled Dot-Product Attention**，也是今天所有 Transformer 变体的基础公式。

拆开来看，它做了三件事：

| 步骤 | 计算 | 含义 |
|:---|:---|:---|
| ① | $QK^\top / \sqrt{d_k}$ | 计算每对位置之间的相关性，并缩放 |
| ② | $\text{softmax}(\cdots)$ | 把相关性转成概率权重 |
| ③ | $\cdots V$ | 按权重把 Value 信息汇总 |

对应的 PyTorch 实现如下：

```python
import math
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    # ① 计算相关性分数并缩放
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果有掩码，加到分数上（-inf 位置在 softmax 后会变成 0）
    if mask is not None:
        scores = scores + mask
    # ② Softmax 得到注意力权重
    weights = scores.softmax(dim=-1)
    # ③ 对 Value 加权求和
    return torch.matmul(weights, value), weights
```

---

## 3. Self-Attention：让句子内部互相"看"

### 3.1 什么是 Self-Attention

在经典注意力机制里，Query 和 Key/Value 可以来自两个不同的序列——比如在机器翻译里，Query 来自目标语言，Key/Value 来自源语言。

但 Transformer 的 Encoder 用的是一种变体——**自注意力（Self-Attention）**：**Q、K、V 全部来自同一个输入序列**，只是通过三组不同的参数矩阵 $W_Q$、$W_K$、$W_V$ 分别投影得到：

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

这意味着，句子里的每个 token 都会去"查询"同一句话里的所有其他 token，从而学习到：

- 主语和谓语之间的依赖
- 代词和它所指代的名词之间的关联
- 修饰语和被修饰词之间的搭配
- 跨越多个词的长距离关系

举个经典例子：

> *The animal didn't cross the street because **it** was too tired.*

当模型更新 **it** 的表示时，Self-Attention 会让它更多地关注 **animal** 和 **tired**，而不是 **street**。这正是语言中"指代关系"的建模。

在代码里，Self-Attention 的实现非常直白——只需把同一个张量传给 Q、K、V 三个位置：

```python
# x 是输入序列的表示
output, weights = scaled_dot_product_attention(x, x, x)
```

### 3.2 Masked Self-Attention：生成任务里不能"偷看"未来

在**生成任务**中，模型预测第 $t$ 个 token 时，只能看到位置 $1, 2, \ldots, t-1$ 的历史信息，不能提前看到第 $t+1$ 及之后的 token——否则就是"作弊"。

但如果每次预测都要串行推进，Transformer 的并行优势就完全消失了。

解决方案是 **Causal Mask（因果掩码）**，也叫上三角掩码。

具体做法：在训练时，把整个目标序列一次性送入，但在计算注意力分数时，给矩阵中所有"未来位置"加上 $-\infty$：

$$
\text{scores}[i][j] = \begin{cases} \text{原始分数} & j \leq i \\ -\infty & j > i \end{cases}
$$

由于 $\text{softmax}(-\infty) = 0$，那些未来位置的权重会被完全置零，模型实际上就看不到未来的信息了。同时，整个序列是并行计算的，训练效率不受影响。

```python
# 生成上三角掩码（1, seq_len, seq_len）
mask = torch.full((1, seq_len, seq_len), float("-inf"))
mask = torch.triu(mask, diagonal=1)  # 保留上三角，其余置零

# 应用掩码
scores = scores + mask
weights = scores.softmax(dim=-1)
```

这就是 Decoder 中 **Masked Multi-Head Self-Attention** 的核心逻辑。

---

## 4. Multi-Head Attention：同时学习多种关系

### 4.1 单个注意力头的局限

一次注意力计算，只能学到输入序列中的一种关联模式。但语言中的依赖关系是多种多样的：

- 有的关系是语法层面的（主谓一致）
- 有的关系是语义层面的（词义消歧）
- 有的关系是指代层面的（代词 → 先行词）
- 有的关系是局部的（相邻词搭配），有的是全局的（长距离依赖）

用单个注意力头同时拟合这一切，显然力不从心。

### 4.2 多头注意力的做法

**多头注意力（Multi-Head Attention）** 的思路很简单：把隐藏维度切分成 $h$ 份，让 $h$ 个"头"各自独立地做注意力计算，最后把所有头的结果拼接起来，再经过一个线性层整合。

公式如下：

$$
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
$$

每个头都有自己独立的参数矩阵 $W_i^Q$、$W_i^K$、$W_i^V$，因此可以各自学到不同的注意力模式。论文作者通过可视化也验证了这一点——不同的头确实倾向于捕捉不同类型的语言关系。

多头注意力的高效实现并不是真的跑 $h$ 次独立的注意力计算，而是通过矩阵重塑（reshape）来实现并行计算：

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape

        # 线性投影，然后拆成多头：(B, T, d_model) -> (B, n_heads, T, head_dim)
        Q = self.wq(q).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wk(k).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.wv(v).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 每个头独立做注意力计算
        out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # 拼接所有头的输出：(B, n_heads, T, head_dim) -> (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)
```

---

## 5. 位置编码：给 Transformer 注入顺序感

### 5.1 Attention 天生不知道顺序

注意力机制在计算相关性时，只看 Query 和 Key 的内容，**完全不知道它们在序列中的位置**。这意味着，对于下面两个句子：

> *狗 咬 了 人*
> *人 咬 了 狗*

如果不加额外处理，Transformer 看到的只是同一组词的不同排列，但在数学上这两个输入几乎没有区别。然而这两句话的意思显然天差地别。

### 5.2 正弦余弦位置编码

原始论文的解决方案是：在把词向量送入 Transformer 之前，先和一个**位置编码**向量相加，把位置信息"注入"到表示中：

$$
\text{输入} = \text{词嵌入} + \text{位置编码}
$$

原论文使用了基于正弦和余弦函数的位置编码，对于位置 $\text{pos}$、维度 $i$：

$$
PE(\text{pos},\, 2i) = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE(\text{pos},\, 2i+1) = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)
$$

你不必一开始就记住这个公式的细节，只需要理解它背后的两个关键性质：

1. **每个位置都有唯一的编码**：不同位置的 PE 向量不同，模型可以区分它们。
2. **相对位置可以被推算**：正弦/余弦函数的周期性使得 $PE(\text{pos} + k)$ 可以用 $PE(\text{pos})$ 线性表示，模型因此有机会学到相对位置关系。

---

## 6. Transformer 的完整结构

原始 Transformer 采用经典的 **Encoder-Decoder** 架构，专为"读入一段文本、生成另一段文本"的任务（如机器翻译）设计。

### 6.1 Encoder：把输入序列编码成上下文表示

Encoder 由若干个相同的 Block 堆叠而成，每个 Block 包含两个子层：

1. **Multi-Head Self-Attention**：让序列里的每个位置都能关注到所有其他位置。
2. **Feed-Forward Network（FFN）**：对每个位置的表示做独立的非线性变换。

这两个子层都配备了**残差连接（Residual Connection）**和**层归一化（Layer Normalization）**：

$$
x = \text{LayerNorm}(x + \text{MultiHeadSelfAttention}(x))
$$

$$
x = \text{LayerNorm}(x + \text{FFN}(x))
$$

残差连接的作用是让梯度在深层网络中更顺畅地流动，避免训练时的梯度消失问题。层归一化则稳定了每层的激活值分布，加快收敛。

Encoder 一层 Block 的数据流可以这样理解：

```
输入表示 X
  │
  ├─→ Multi-Head Self-Attention  ──→ + X ──→ LayerNorm ──→ X'
  │                                                           │
  └─────────────────────────────────────────────────────────┘
                                                             │
  ├─→ Feed-Forward Network       ──→ + X' ──→ LayerNorm ──→ 输出
```

多层堆叠之后，每个 token 的表示就不再只是"它自己"的嵌入，而是"吸收了整句上下文信息之后的自己"。

### 6.2 Decoder：逐步生成输出序列

Decoder 同样由若干个 Block 堆叠而成，但每个 Block 包含**三个**子层：

1. **Masked Multi-Head Self-Attention**：对已生成的历史 token 做自注意力，同时用因果掩码屏蔽未来位置。
2. **Cross-Attention（交叉注意力）**：让 Decoder 的当前状态去"查询" Encoder 的输出。
3. **Feed-Forward Network**：同 Encoder。

其中，Cross-Attention 是 Encoder 和 Decoder 之间的桥梁：

- **Query** 来自 Decoder 的当前状态（我生成到哪了）
- **Key 和 Value** 来自 Encoder 的输出（原始输入是什么）

通过这个机制，Decoder 在每一步生成时，既能参考自己已经生成的历史，又能随时回看整个输入序列，决定"现在应该重点参考输入的哪个部分"。

### 6.3 整体架构一览

```
输入序列
    │
  词嵌入 + 位置编码
    │
  ┌─────────────────┐
  │   Encoder Block │ × N
  │  ┌─────────────┐│
  │  │ MH Self-Att ││
  │  │ Add & Norm  ││
  │  │    FFN      ││
  │  │ Add & Norm  ││
  │  └─────────────┘│
  └────────┬────────┘
           │ Encoder 输出 (K, V)
           │
  ┌────────▼────────┐     输出序列（移位）
  │   Decoder Block │ × N ←── 词嵌入 + 位置编码
  │  ┌─────────────┐│
  │  │Masked MH SA ││
  │  │ Add & Norm  ││
  │  │ Cross-Att   ││  ← 使用 Encoder 输出作为 K, V
  │  │ Add & Norm  ││
  │  │    FFN      ││
  │  │ Add & Norm  ││
  │  └─────────────┘│
  └────────┬────────┘
           │
        线性层 + Softmax
           │
        预测下一个 token
```

---

## 7. 为什么 Transformer 能替代 RNN 成为基础架构

Transformer 真正改变整个 NLP 领域，不只是因为它的公式优雅，而是它同时解决了 RNN 的两个核心痛点：

| 对比项 | RNN/LSTM | Transformer |
|:---|:---|:---|
| 并行计算 | 必须串行，GPU 利用率低 | Q、K、V 矩阵运算可完全并行 |
| 长距离依赖 | 信息需逐步传递，容易衰减 | 任意两个位置之间只需一次 Attention |
| 序列长度限制 | 过长时性能显著下降 | 仅受显存限制（后续改进持续扩展） |
| 可扩展性 | 难以扩展到极大规模 | 支持大规模参数 + 数据的 scaling |

正是因为这些优势，Transformer 成为了 GPT、BERT、T5、LLaMA、Qwen、Gemma 等几乎所有现代大语言模型的共同基础架构。

---

## 8. 初学者最容易混淆的几个问题

**Q：Attention 是"只选一个词"吗？**

不是。Attention 是对序列中**所有位置**都分配了权重，只是权重有大有小。它始终在做"加权求和"，而不是"挑选"。

**Q：Self-Attention 和 Causal（Masked）Attention 是同一个东西吗？**

不是。Self-Attention 只是说 Q、K、V 都来自同一序列。是否加 Causal Mask 是独立的选择：Encoder 用的是普通 Self-Attention（可以看到整个序列）；Decoder 的第一个子层用的是 Masked Self-Attention（只能看到历史部分）。

**Q：Q、K、V 是人工标注的三份数据吗？**

不是。它们是**同一个输入**通过三组不同的可学习线性层（参数矩阵）投影出来的。本质上是同一信息的三种不同"视角"。

**Q：多头注意力只是把注意力重复做了 $h$ 次吗？**

更准确的理解是：$h$ 个头在**不同的子空间**里各自学习，每个头都有独立的参数，因此会学到不同的依赖关系。这是质量上的提升，不只是数量上的重复。

**Q：位置编码加到词向量上，会不会破坏词义信息？**

不会。位置编码只是一个**固定的偏移量**，语义信息依然保留在词嵌入里。模型通过训练可以学会如何把两种信息分别利用。

---

## 9. 完整模块伪代码

下面的伪代码把本文涉及的核心模块串联起来，可以作为理解整体结构的参考：

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / sqrt(d_k)
    if mask is not None:
        scores = scores + mask          # -inf 位置在 softmax 后变为 0
    weights = softmax(scores, dim=-1)
    return weights @ V


def multi_head_attention(Q, K, V, mask=None):
    # 分别投影到 n_heads 个子空间
    Q = split_heads(linear_q(Q))       # (B, n_heads, T, head_dim)
    K = split_heads(linear_k(K))
    V = split_heads(linear_v(V))
    # 每个头独立计算注意力
    out = scaled_dot_product_attention(Q, K, V, mask)
    # 拼接后再过一个线性层
    return linear_o(merge_heads(out))


def encoder_block(x):
    x = layer_norm(x + multi_head_attention(x, x, x))          # Self-Attention
    x = layer_norm(x + feed_forward(x))                         # FFN
    return x


def decoder_block(x, encoder_output):
    causal_mask = make_causal_mask(x)
    x = layer_norm(x + multi_head_attention(x, x, x, causal_mask))         # Masked Self-Attention
    x = layer_norm(x + multi_head_attention(x, encoder_output, encoder_output))  # Cross-Attention
    x = layer_norm(x + feed_forward(x))                                     # FFN
    return x


def transformer(src, tgt):
    src = token_embedding(src) + positional_encoding(src)
    tgt = token_embedding(tgt) + positional_encoding(tgt)

    for _ in range(N):
        src = encoder_block(src)

    for _ in range(N):
        tgt = decoder_block(tgt, src)

    return linear(tgt)   # 映射到词表，预测下一个 token
```

---

## 10. 读完这篇之后，你应该能回答这些问题

- Transformer 用什么机制解决了 RNN 的串行计算和长距离依赖问题？
- Q、K、V 分别代表什么？它们是从哪里来的？
- 注意力公式 $\text{softmax}(QK^\top / \sqrt{d_k})\,V$ 中每一步在做什么？为什么要除以 $\sqrt{d_k}$？
- Self-Attention、Masked Self-Attention、Cross-Attention 分别用在哪里？有什么区别？
- 为什么 Transformer 必须引入位置编码？原始论文用的是什么方法？
- Multi-Head Attention 相比单头注意力的优势是什么？
- Encoder 和 Decoder 的结构分别是什么？各有几个子层？

---

## 参考资料

- 原始论文：`paper/Transformer-Attention-Is-All-You-Need.pdf`
- Happy-LLM 第二章：https://datawhalechina.github.io/happy-llm/#/./chapter2/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Transformer%E6%9E%B6%E6%9E%84
