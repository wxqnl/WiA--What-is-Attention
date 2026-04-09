# MQA：Multi-Query Attention

> 对应论文：`paper/MQA-Multi-Query-Attention.pdf`
> Fast Transformer Decoding: One Write-Head is All You Need，Shazeer，2019
> https://arxiv.org/abs/1911.02727

---

## 1. 背景：推理时的 KV Cache 瓶颈

训练大模型的时候，我们把整个序列一次性送进去并行计算，效率很高。但**推理**（生成）阶段完全是另一回事。

模型生成第 $t$ 个 token 时，必须把前 $t-1$ 个 token 的历史信息都考虑进来。如果每次都从头重算所有历史 token 的 K 和 V，计算量会随着序列长度线性增长，速度越来越慢。

工程上的解决方案叫 **KV Cache**：把每个层、每个头计算出来的 K 和 V 向量缓存在内存里，下次生成新 token 时直接拿来用，不重复计算。

这个方案非常有效，但有一个代价：**KV Cache 占用的显存随序列长度和头数线性增长**。

具体来说，标准 Multi-Head Attention（MHA）的 KV Cache 大小是：

$$
\text{KV Cache 大小} = 2 \times L \times h \times d_k \times \text{数据类型字节数}
$$

其中 $L$ 是序列长度，$h$ 是头数，$d_k$ 是每头维度。对于一个 70B 参数的模型，批量推理时 KV Cache 可能占用数十 GB 显存，严重限制了批大小（batch size），进而限制了吞吐量。

**问题的根源是什么？** 每个 Query 头都有一组独立的 K 和 V。但推理时，K 和 V 才是缓存的主要对象，Query 是即时计算的——为什么 K 和 V 必须有 $h$ 组？

---

## 2. MQA 的核心想法：一组 KV 服务所有 Query 头

**Multi-Query Attention（MQA）** 的思路简单直接：**保留 $h$ 个 Query 头，但所有 Query 头共享同一组 K 和 V**。

类比一下：想象一个图书馆有 32 位读者（Query 头），他们可以分别带着不同的问题（Query）来查资料，但图书馆只有一套索引系统（K）和一套藏书（V）。每位读者用自己的问题去检索同一套资料，得到的答案各不相同（因为检索侧重点不同），但底层的资料是共享的。

数学上，MHA 的每个头是：

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
$$

每个头有独立的 $W_i^K$ 和 $W_i^V$，产生独立的 K 和 V。

MQA 改为：

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W^K,\; V W^V)
$$

**所有 $h$ 个头共用同一个 $W^K$ 和 $W^V$**，K 和 V 只有 1 组。

---

## 3. KV Cache 压缩了多少

MHA 中 KV Cache 有 $h$ 组；MQA 中只有 **1 组**。

KV Cache 压缩比 = $h$（头数）。对于一个 32 头的模型，MQA 将 KV Cache 压缩到原来的 $1/32$。

这意味着：
- 同等显存下，batch size 可以大幅提升
- 推理延迟降低（更少的内存带宽消耗）
- 长上下文场景下的实际可用性大幅提升

---

## 4. 代码实现

```python
import torch
import math

class MultiQueryAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # 每个 Query 头有独立的投影矩阵
        self.W_Q = torch.nn.Linear(d_model, d_model, bias=False)
        # K 和 V 只有一组——这是 MQA 的核心
        self.W_K = torch.nn.Linear(d_model, self.head_dim, bias=False)
        self.W_V = torch.nn.Linear(d_model, self.head_dim, bias=False)
        self.W_O = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, d = x.shape

        # Q：(B, n_heads, T, head_dim)
        Q = self.W_Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # K 和 V：只有一组，(B, 1, T, head_dim)
        # unsqueeze(1) 在 heads 维度插入 1，后续广播到所有 Query 头
        K = self.W_K(x).unsqueeze(1)   # (B, 1, T, head_dim)
        V = self.W_V(x).unsqueeze(1)   # (B, 1, T, head_dim)

        # 注意力计算：Q 是 (B, n_heads, T, d)，K/V 是 (B, 1, T, d)
        # PyTorch 广播会自动将 K/V 扩展到 n_heads 份
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = scores.softmax(dim=-1)
        out = weights @ V   # (B, n_heads, T, head_dim)

        # 合并多头
        out = out.transpose(1, 2).contiguous().view(B, T, d)
        return self.W_O(out)
```

关键点：K 和 V 只经过一次线性投影（输出维度是 `head_dim` 而不是 `d_model`），然后通过广播机制参与所有 Query 头的计算，**不需要显式复制 $h$ 份**。

---

## 5. MQA 的代价：质量有所损失

MQA 不是免费的午餐。用单组 KV 服务所有 Query 头，意味着不同头在提取特征时失去了独立性——它们看到的 K 和 V 是完全一样的，差异只来自 Query。

实验上，MQA 相比 MHA 在质量上有一定损失，尤其是在模型规模较小或任务复杂度较高时损失更明显。

另一个工程问题是：已有的 MHA 模型**无法直接转换为 MQA**，必须重新训练（或者用 Uptraining 近似，但效果有折扣）。

这两个问题催生了 MQA 的改良版本——GQA。

---

## 6. MHA / MQA / GQA 三者关系

```
Multi-Head Attention (MHA)
  每个 Query 头有独立的 KV
  h 组 KV
      │
      ├── 极端化 ──→ Multi-Query Attention (MQA)
      │              所有 Query 头共享 1 组 KV
      │              KV Cache 压缩 h 倍，质量有损失
      │
      └── 折中 ────→ Grouped Query Attention (GQA)
                     Query 头分组，每组共享 1 组 KV
                     KV Cache 压缩 g 倍（g 为每组头数）
                     质量与效率的平衡
```

MQA 是 GQA 在 $g = h$（每组只有一个 KV）时的特殊情况；MHA 是 GQA 在 $g = 1$（每个 Query 头独占一个 KV）时的特殊情况。

> GQA 的详细介绍见 → [`GQA.md`](GQA.md)

---

## 7. 谁在用 MQA

| 模型 | 使用情况 |
|:---|:---|
| PaLM | 全系列使用 MQA |
| Falcon（40B/180B） | 使用 MQA |
| Mistral 7B | 使用 GQA（而非 MQA） |
| LLaMA 2（70B） | 使用 GQA |
| LLaMA 3（全系列） | 使用 GQA |

MQA 在早期（2019–2022）被部分大模型采用，2023 年后 GQA 更流行，因为它在质量和效率之间找到了更好的平衡点。

---

## 8. 读完这篇之后，你应该能回答这些问题

- 为什么推理阶段需要 KV Cache？如果没有 KV Cache，推理的计算复杂度是多少？
- MQA 和 MHA 的区别在哪里？MQA 的 KV Cache 压缩比是多少？
- MQA 在代码里怎么实现 K 和 V 的共享？是复制 $h$ 份还是通过广播？
- MQA 的质量损失来自哪里？为什么会有损失？
- MQA 和 GQA 是什么关系？GQA 是如何解决 MQA 的缺陷的？

---

## 参考资料

- 原始论文：`paper/MQA-Multi-Query-Attention.pdf`
- https://arxiv.org/abs/1911.02727
