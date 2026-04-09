# GQA：Grouped Query Attention

> 对应论文：`paper/GQA-Grouped-Query-Attention.pdf`
> GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints，Ainslie et al.，2023
> https://arxiv.org/abs/2305.13245

---

## 1. 背景：MQA 的质量损失能否弥补？

上一篇 MQA 讲到，把所有 Query 头的 KV 合并成一组，能把 KV Cache 压缩 $h$ 倍，但代价是质量有所下降——所有头共享同一套特征提取，灵活性损失了。

能不能找一个中间方案：**既不像 MHA 那样 $h$ 组 KV 全部独立，也不像 MQA 那样极端地压缩到 1 组，而是取一个适中的数目**？

这就是 **GQA（Grouped Query Attention，分组查询注意力）** 的出发点。

---

## 2. GQA 的核心思想：Query 头分组，每组共享一对 KV

GQA 把 $h$ 个 Query 头分成 $g$ 组，每组包含 $h/g$ 个 Query 头，**每组共享同一对 K 和 V**。

用一个具体的例子来说：假设有 8 个 Query 头，分成 4 组，每组 2 个 Query 头：

```
Query 头:  Q1  Q2 | Q3  Q4 | Q5  Q6 | Q7  Q8
                 ↓        ↓        ↓        ↓
KV 组:      KV1      KV2      KV3      KV4
```

Q1 和 Q2 共享 KV1，Q3 和 Q4 共享 KV2，以此类推。

这样，KV Cache 的大小从 MHA 的 $h$ 组变成了 $g$ 组（$g < h$），压缩比是 $h/g$。

三种方案的关系用一张图表示：

```
MHA（g = h）       GQA（1 < g < h）    MQA（g = 1）

Q1─KV1             Q1─┐               Q1─┐
Q2─KV2             Q2─┤─KV1           Q2─┤
Q3─KV3             Q3─┐               Q3─┤
Q4─KV4             Q4─┤─KV2           Q4─┤─KV
Q5─KV5             Q5─┐               Q5─┤
Q6─KV6             Q6─┤─KV3           Q6─┤
Q7─KV7             Q7─┐               Q7─┤
Q8─KV8             Q8─┘─KV4           Q8─┘

每头独立 KV         4 组 KV             1 组 KV
KV Cache 最大       KV Cache 折中        KV Cache 最小
质量最好            质量接近 MHA          质量有损失
```

---

## 3. 从公式看 GQA

MHA 中第 $i$ 个头：

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
$$

GQA 中第 $i$ 个头（属于第 $\lfloor i \cdot g / h \rfloor$ 组）：

$$
\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_{\lfloor i \cdot g / h \rfloor}^K,\; V W_{\lfloor i \cdot g / h \rfloor}^V)
$$

也就是说，同一组内的所有 Query 头共用同一个 $W^K$ 和 $W^V$，但 $W^Q$ 仍然各自独立。

---

## 4. 代码实现

```python
import torch
import math

class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads 必须是 n_kv_heads 的整数倍"
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads   # 每组的 Query 头数
        self.head_dim = d_model // n_heads

        # Q 的投影：n_heads 个独立头
        self.W_Q = torch.nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        # K/V 的投影：只有 n_kv_heads 组
        self.W_K = torch.nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_V = torch.nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_O = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, _ = x.shape

        # Q: (B, n_heads, T, head_dim)
        Q = self.W_Q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # K/V: (B, n_kv_heads, T, head_dim)
        K = self.W_K(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 将 K/V 扩展到 n_heads：每组 KV 重复 n_rep 次，对齐 Q 的头数
        # repeat_interleave 会把 [KV1, KV2] → [KV1, KV1, KV2, KV2]（n_rep=2 时）
        K = K.repeat_interleave(self.n_rep, dim=1)   # (B, n_heads, T, head_dim)
        V = V.repeat_interleave(self.n_rep, dim=1)

        # 标准 Scaled Dot-Product Attention
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        weights = scores.softmax(dim=-1)
        out = weights @ V   # (B, n_heads, T, head_dim)

        # 合并多头，过输出投影
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_O(out)
```

关键参数：
- `n_heads = 32`，`n_kv_heads = 8`：这是 LLaMA 2 70B 的配置，`n_rep = 4`，每 4 个 Query 头共享 1 对 KV
- `n_heads = 128`，`n_kv_heads = 8`：LLaMA 3.1 405B 的配置，`n_rep = 16`

---

## 5. 从 MHA 检查点转化为 GQA：Uptraining

GQA 论文还提出了一个实用技巧：如果已经有一个训练好的 MHA 模型，怎么把它转化成 GQA 而不从头训练？

方法是 **Mean Pooling**：对于同一组内的所有 KV 头，把它们的参数矩阵做均值：

$$
W_{\text{group}}^K = \frac{1}{h/g} \sum_{i \in \text{group}} W_i^K
$$

然后用少量数据继续训练（Uptraining），让模型适应新的 GQA 结构。论文发现只需用原始训练数据的 5% 做 Uptraining，就能恢复到接近 MHA 的质量。

这解决了 MQA 的一个痛点：不必从头训练，已有模型可以被"转换"。

---

## 6. GQA 在实际模型中的配置

| 模型 | n_heads | n_kv_heads | n_rep | KV Cache 压缩比 |
|:---|:---:|:---:|:---:|:---:|
| LLaMA 2 7B（MHA） | 32 | 32 | 1 | 1× |
| LLaMA 2 70B | 64 | 8 | 8 | 8× |
| LLaMA 3 8B | 32 | 8 | 4 | 4× |
| LLaMA 3.1 405B | 128 | 8 | 16 | 16× |
| Mistral 7B | 32 | 8 | 4 | 4× |
| Qwen2.5 72B | 64 | 8 | 8 | 8× |

可以观察到，`n_kv_heads = 8` 似乎是一个工业上比较流行的选择，在压缩比和质量之间取得了不错的平衡。

---

## 7. 为什么 GQA 的质量损失比 MQA 小得多？

回到直觉：MQA 里所有 32 个 Query 头共享 1 对 KV，等于是说"图书馆里 32 位读者只能看同一套资料"，差异化完全靠 Query 侧。

GQA 里每 4 个 Query 头共享 1 对 KV（以 n_rep=4 为例），每组之间的 KV 是不同的。这意味着：
- **组间**：不同组的 KV 可以学到不同类型的特征（句法关系 vs 语义关系 vs 代词指代……）
- **组内**：同组的 4 个 Query 头用相同的 KV，但通过不同的 Query 投影去"问不同的问题"

这种设计保留了 MHA 的大部分表达能力，同时把 KV 数量压缩到了可接受的范围。

---

## 8. 常见混淆

**Q：GQA 是把 K 和 V 复制了 n_rep 份存进 KV Cache 吗？**

不是。**KV Cache 里只存 n_kv_heads 组 KV**，是在实际做注意力计算的时候（`repeat_interleave`）才临时扩展到 n_heads 份。复制只是为了让矩阵乘法的维度对齐，不影响存储量。

**Q：GQA 的 Q 还是每头独立的吗？**

是的。每个 Query 头有独立的 $W_i^Q$，Query 侧的多样性完全保留，只有 KV 侧做了合并。

**Q：n_kv_heads 越少越好吗？**

不是。n_kv_heads 越少，KV Cache 越小，但质量损失也越大。n_kv_heads = 1 就是 MQA，质量损失明显。实践中 8 组是一个常见的折中点。

---

## 9. 读完这篇之后，你应该能回答这些问题

- GQA 和 MHA、MQA 的关系是什么？用一句话说清楚 GQA 做了什么。
- 在代码里，GQA 是怎么实现"n_kv_heads 组 KV 服务 n_heads 个 Query 头"的？`repeat_interleave` 在这里起什么作用？
- KV Cache 存的是 n_heads 份还是 n_kv_heads 份？
- 已有 MHA 模型怎么转化为 GQA？Uptraining 的核心步骤是什么？
- 为什么 GQA 的质量损失比 MQA 小？从特征提取的角度解释一下。

---

## 参考资料

- 原始论文：`paper/GQA-Grouped-Query-Attention.pdf`
- https://arxiv.org/abs/2305.13245
