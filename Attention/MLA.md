# MLA：Multi-Head Latent Attention

> 对应论文：`paper/DeepSeek-V2-MLA-MoE.pdf`
> DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model，DeepSeek，2024
> https://arxiv.org/abs/2405.04434

---

## 1. 背景：GQA 之后，KV Cache 还能压缩多少？

GQA 把 KV Cache 从 $h$ 组压缩到 $g$ 组（$g \ll h$），已经是一个很大的进步。但 DeepSeek 团队问了一个更激进的问题：

**KV Cache 的本质是"存 K 和 V 向量"。那能不能不存完整的 K/V，而是存一个更小的"压缩表示"，需要时再解压？**

这就是 **MLA（Multi-Head Latent Attention，多头潜在注意力）** 的核心想法。

回顾一下 KV Cache 的规模：对于一个有 $h$ 个头、每头维度 $d_k$ 的模型，每个 token 每层需要存储：

$$
\text{MHA 每 token 每层 KV Cache} = 2 \times h \times d_k
$$

GQA 把它降到 $2 \times g \times d_k$（$g < h$）。MLA 的目标是进一步降到：

$$
\text{MLA 每 token 每层 KV Cache} = d_c \ll 2 \times h \times d_k
$$

其中 $d_c$ 是一个远小于 $2hd_k$ 的**潜在向量**（latent vector）维度。DeepSeek-V2 中，$d_c = 512$，而 $2 \times h \times d_k = 2 \times 128 \times 128 = 32768$，压缩比高达 **93%**。

---

## 2. MLA 的核心想法：低秩投影

### 2.1 一个关键观察

标准注意力里，Q、K、V 都是从输入 $x$ 经过线性投影得来的：

$$
K = x W^K, \quad V = x W^V
$$

$W^K$ 和 $W^V$ 的维度都是 $[d_\text{model},\; h \times d_k]$。

这里有一个矩阵分解的视角：一个 $[d_\text{model},\; h \times d_k]$ 的矩阵可以被分解为两个矩阵的乘积：

$$
W^K = W^{DK} \cdot W^{UK}, \quad \text{其中 } W^{DK} \in \mathbb{R}^{d_\text{model} \times d_c},\; W^{UK} \in \mathbb{R}^{d_c \times (h \times d_k)}
$$

这就是低秩分解（Low-Rank Decomposition）。如果 $d_c \ll d_\text{model}$ 且 $d_c \ll h \times d_k$，这个分解会让参数减少，但同时引入了一个"瓶颈"——**所有信息都必须先经过 $d_c$ 维的压缩表示**。

MLA 利用这个结构，把**压缩后的 $d_c$ 维向量**存入 KV Cache，而不是完整的 K/V。

### 2.2 MLA 的具体计算流程

MLA 引入一个中间变量 $c^{KV}$，称为**KV 潜在向量（KV latent vector）**：

**第一步：Down-projection，把输入压缩成潜在向量**

$$
c^{KV} = x W^{DKV}
$$

其中 $W^{DKV} \in \mathbb{R}^{d_\text{model} \times d_c}$ 是下投影矩阵，$c^{KV} \in \mathbb{R}^{d_c}$。

**这个 $c^{KV}$ 就是 MLA 缓存的全部内容。**

**第二步：Up-projection，从潜在向量还原 K 和 V**

$$
K = c^{KV} W^{UK}, \quad V = c^{KV} W^{UV}
$$

其中 $W^{UK} \in \mathbb{R}^{d_c \times (h \times d_k)}$，$W^{UV} \in \mathbb{R}^{d_c \times (h \times d_v)}$。

**第三步：正常做多头注意力**

Q 的计算也有类似的压缩-解压结构（但 Q 不需要缓存），然后按标准方式计算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

整个流程可以用下图表示：

```
输入 x
  │
  ├──→ W^{DQ} ──→ c^Q（Q 的潜在向量）──→ W^{UQ} ──→ Q
  │                                               │
  └──→ W^{DKV} ──→ c^{KV}（缓存这个！）──→ W^{UK} ──→ K
                                           └──→ W^{UV} ──→ V
                                                         │
                                         softmax(QKᵀ/√d) × V ──→ 输出
```

---

## 3. 为什么这样做能大幅压缩 KV Cache？

标准 MHA（或 GQA）缓存的是**每个头的完整 K 和 V**，每个 token 需要存 $h \times d_k + h \times d_v$ 个数（或者 GQA 是 $g \times d_k + g \times d_v$）。

MLA 缓存的是**潜在向量 $c^{KV}$**，每个 token 只需存 $d_c$ 个数。

DeepSeek-V2 的具体数字：

| 量 | MHA | GQA（g=8） | MLA |
|:---|:---:|:---:|:---:|
| 每 token 每层 KV Cache | $2 \times 128 \times 128 = 32768$ | $2 \times 8 \times 128 = 2048$ | $512$ |
| 相对 MHA 的压缩比 | 1× | 16× | **64×** |

MLA 的压缩比比 GQA 还要再压缩 4 倍（在 DeepSeek-V2 的配置下）。

---

## 4. MLA 的 RoPE 问题：一个需要特别处理的细节

RoPE 要求对 K 和 V 施加旋转，且旋转角度依赖于 **token 的位置**。问题是：

**如果我们缓存的是压缩后的 $c^{KV}$，而不是完整的 K，那 RoPE 应该加在哪里？**

如果在 up-projection 之后加 RoPE，则位置信息已经融入了 K，但 $c^{KV}$ 本身不含位置信息——每次使用时都需要重新做 up-projection 再加 RoPE，这就失去了"直接复用缓存的 K"的好处。

MLA 的解法是**解耦 RoPE**：

- 把 K 分成两部分：**不含 RoPE 的内容部分**（从 $c^{KV}$ 解压来的）和**含 RoPE 的位置部分**（单独计算）
- 缓存时，两部分分别缓存
- Q 同样做类似的解耦

具体来说：

$$
K_i = [c^{KV} W^{UK}_i;\; \text{RoPE}(x W^{KR})]
$$

方括号里是拼接（concatenation）。左边是从潜在向量解压出来的内容键（不含位置信息），右边是带 RoPE 的位置键（直接从原始输入计算）。

这样，缓存里存的是 $c^{KV}$（内容部分）和 $\text{RoPE}(x W^{KR})$（位置部分），两者都比完整 K 小得多。

---

## 5. 代码框架（简化版）

```python
class MultiHeadLatentAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_c, head_dim):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.d_c = d_c

        # Q 的压缩-解压（Q 不缓存，所以可以合并）
        self.W_DQ = torch.nn.Linear(d_model, d_c, bias=False)
        self.W_UQ = torch.nn.Linear(d_c, n_heads * head_dim, bias=False)
        # Q 的 RoPE 部分（解耦）
        self.W_QR = torch.nn.Linear(d_model, n_heads * head_dim, bias=False)

        # KV 的下投影：这是缓存的对象
        self.W_DKV = torch.nn.Linear(d_model, d_c, bias=False)
        # KV 的上投影：推理时从缓存中还原
        self.W_UK = torch.nn.Linear(d_c, n_heads * head_dim, bias=False)
        self.W_UV = torch.nn.Linear(d_c, n_heads * head_dim, bias=False)
        # K 的 RoPE 部分（解耦）
        self.W_KR = torch.nn.Linear(d_model, head_dim, bias=False)

        self.W_O = torch.nn.Linear(n_heads * head_dim, d_model, bias=False)

    def forward(self, x, cos, sin, mask=None, kv_cache=None):
        B, T, _ = x.shape

        # ① Q 计算：压缩 → 解压 → 加 RoPE（不缓存）
        c_Q = self.W_DQ(x)
        Q_content = self.W_UQ(c_Q).view(B, T, self.n_heads, self.head_dim)
        Q_rope = apply_rope(self.W_QR(x).view(B, T, self.n_heads, self.head_dim), cos, sin)
        Q = Q_content + Q_rope    # 内容 + 位置，拼接或相加（实现细节略有差异）

        # ② KV 潜在向量：这是缓存的核心
        c_KV = self.W_DKV(x)     # (B, T, d_c)  ← 只缓存这个

        # ③ 从潜在向量还原 K/V（推理时从缓存中取 c_KV 并 up-project）
        K_content = self.W_UK(c_KV).view(B, T, self.n_heads, self.head_dim)
        V = self.W_UV(c_KV).view(B, T, self.n_heads, self.head_dim)
        K_rope = apply_rope(self.W_KR(x).unsqueeze(2).expand(-1,-1,self.n_heads,-1), cos, sin)
        K = K_content + K_rope    # 内容键 + 位置键

        # ④ 标准注意力计算
        Q, K, V = [t.transpose(1, 2) for t in [Q, K, V]]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        out = scores.softmax(dim=-1) @ V
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out)
```

> 注：上面是简化示意，实际 DeepSeek-V2 的实现更精细，尤其是 RoPE 解耦的部分有多种处理方式。

---

## 6. MHA / GQA / MLA 的全面对比

| | MHA | GQA | MLA |
|:---|:---:|:---:|:---:|
| KV Cache 内容 | 完整 K/V（h 组） | 部分 K/V（g 组） | 潜在向量 $c^{KV}$（1 个） |
| 每 token 每层缓存量 | $2hd_k$ | $2gd_k$ | $d_c$（+ RoPE 部分） |
| 参数量 | 标准 | 略少于 MHA | 多了上/下投影矩阵 |
| 表达能力 | 最强 | 接近 MHA | 理论上等价（充分大 $d_c$） |
| 实现复杂度 | 简单 | 简单 | **复杂**（尤其是 RoPE 解耦） |
| 典型使用者 | 早期 GPT、BERT | LLaMA 2/3、Qwen | DeepSeek-V2/V3 |

---

## 7. MLA 真的比 GQA 更好吗？

从 KV Cache 压缩比来看，MLA 确实更激进。但 MLA 也有自己的代价：

1. **推理时的额外计算**：每次生成时，需要从缓存的 $c^{KV}$ 做 up-projection 还原 K 和 V，这比 GQA 的直接缓存多了计算步骤。

2. **实现复杂度高**：RoPE 解耦让代码更难写，也更难优化（FlashAttention 等内核需要额外适配）。

3. **吸收 up-projection 的技巧**：推理时，可以把 $W^{UK}$ 和后续 $W^O$ 合并，或与注意力后的投影合并，避免显式地还原完整 K/V。这是一个工程优化，但增加了理解难度。

总体来说，MLA 在**显存极度紧张、需要服务超长序列**的场景下优势明显，是 DeepSeek 团队针对推理成本优化的一次系统性创新。

---

## 8. 读完这篇之后，你应该能回答这些问题

- MLA 和 GQA 的出发点相同，都是压缩 KV Cache。它们的压缩方式有什么本质区别？
- MLA 缓存的是什么？每个 token 每层需要存多少数据？
- 低秩投影（Down-projection → Up-projection）在 MLA 里起什么作用？为什么能压缩 KV Cache？
- MLA 中 RoPE 为什么需要"解耦"？不解耦会发生什么问题？
- MLA 相比 GQA 的代价是什么？为什么不是所有模型都用 MLA？

---

## 参考资料

- `paper/DeepSeek-V2-MLA-MoE.pdf`
- DeepSeek-V2 论文：https://arxiv.org/abs/2405.04434
- DeepSeek-V3 技术报告：https://arxiv.org/abs/2412.19437
