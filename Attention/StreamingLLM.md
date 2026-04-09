# StreamingLLM：用 Attention Sink 实现无限流式推理

> 对应论文：`paper/StreamingLLM-Efficient-Streaming-with-Attention-Sinks.pdf`
> Efficient Streaming Language Models with Attention Sinks，Xiao et al.，2023
> https://arxiv.org/abs/2309.17453

---

## 1. 背景：KV Cache 撑爆显存怎么办

读完 MQA、GQA、MLA，你已经知道推理时最大的显存杀手之一就是 **KV Cache**——模型为了避免重复计算，会把每一层每一个 token 对应的 Key 和 Value 都缓存在显存里。

KV Cache 的大小和序列长度正比增长：

$$
\text{KV Cache 大小} \propto L \times n_{\text{layers}} \times n_{\text{heads}} \times d_k
$$

其中 $L$ 是当前已生成的序列长度。对一个 7B 的模型来说，跑几千个 token 问题不大，但如果你想让它做**长达几万 token 的实时对话、文档续写、或者持续性的流式推理**，KV Cache 会线性增长直到撑爆显存，模型被迫停止。

最朴素的解决思路是：**只保留最近的 $W$ 个 token 的 KV Cache，超出窗口的直接扔掉**。这样 KV Cache 大小固定为 $W$，不再增长。

但这个方法有一个致命问题，导致它几乎无法直接用。

---

## 2. 问题：朴素滑动窗口为什么会崩

### 2.1 直接观察现象

实验表明，如果你直接把窗口外的 token 从 KV Cache 里删掉，模型的输出质量会**断崖式下跌**，甚至生成完全乱码。这让人困惑——毕竟已经有足够的近邻上下文，为什么模型会崩？

更奇怪的是：哪怕你只删掉**第一个 token**，结果也会彻底崩溃。

### 2.2 Attention 权重的异常模式

论文作者可视化了各层的注意力权重矩阵，发现了一个规律性的异常：

**不管当前输入是什么内容，几乎每个 token 在所有层都会给序列中最初的几个 token 分配极高的注意力分数**——即使那些 token 在语义上和当前内容毫无关系。

这种现象被命名为 **Attention Sink（注意力汇聚）**：最初的几个 token 就像一个"水槽"，吸收了大量本不属于它们的注意力权重。

### 2.3 为什么会有 Attention Sink

可以这样理解：softmax 有一个结构性特点，它必须让所有位置的权重之和等于 1，即使某个 token 在当前时刻并不想关注任何历史位置。

$$
\sum_{j} \text{softmax}\!\left(\frac{q_i k_j^\top}{\sqrt{d_k}}\right) = 1 \quad \text{（对任意 } i \text{ 成立）}
$$

当"没有特别需要关注的位置"时，这 1 份注意力总得有个地方去。模型在训练中慢慢学到，**把多余的注意力"倾倒"到初始 token 上是最安全的选择**——初始 token 在每个上下文里都稳定存在，把权重分给它不会引入错误信息，只是相当于"空转"。

这是一种隐式的约定，而不是初始 token 真的存储了重要信息。

### 2.4 于是，删掉第一个 token 为什么会崩

原来，模型在训练中建立了一个**隐性依赖**：初始 token 充当了"权重垃圾桶"的角色，保证了 softmax 的归一化在每一步都能正常工作。

一旦把这个"垃圾桶"删掉，多余的注意力权重没有地方可去，softmax 的分布被迫重新分配，整个模型的激活值开始出现异常，进而影响所有后续层，最终输出崩溃。

---

## 3. StreamingLLM 的解决方案

### 3.1 核心思路：保留 Sink token + 滑动窗口

既然问题的根源是"初始 token 被模型用作注意力垃圾桶，删不掉"，那就**不要删它们**。

StreamingLLM 的 KV Cache 策略是：

```
[  Sink tokens  |        滑动窗口（最近的 W 个 token）        ]
   保留最初 4 个               超出窗口的旧 token 被丢弃
```

具体来说：
- 永远保留最初的 $s$ 个 token 的 KV（论文默认 $s = 4$，称为 **sink tokens**）
- 再保留最近的 $W$ 个 token 的 KV 作为局部上下文
- 中间部分（距离超出 $W$ 的历史 token）丢弃

KV Cache 的总大小被固定在 $s + W$，不再随序列长度增长，因此可以跑任意长的序列。

可以把这个结构想象成一个博物馆：展厅入口（sink tokens）永远保留——不是因为入口最重要，而是因为它是整栋建筑的参照系；展厅内部只展示最新的 $W$ 件展品，旧展品轮换出去。

### 3.2 位置编码的难题

这里有一个隐患：**滑动窗口中的 token 的位置编码怎么处理？**

假设当前序列已经生成了 1000 个 token，窗口保留最后 100 个（位置 901 到 1000）和最初 4 个（位置 1 到 4）。当新的 token 来做注意力计算时，它需要和窗口内的 KV 做点积——问题是，如果用原始的绝对位置编码，位置 901 和位置 1 之间差了 900，而新 token 的位置编码只认识"自己是第几位"，不认识"它和窗口内 token 的相对距离"。

**错误方案**：保留 KV 的同时保留原始位置编码 → 位置编码不连续，注意力计算错乱。

**StreamingLLM 的方案**：使用 **RoPE** 或类似的相对位置编码，并且在加入窗口时**重新编码位置**。具体来说，每次做注意力时，不使用 token 的原始绝对位置，而是**按照它在当前 KV Cache 中的顺序重新赋予位置编号**：

```
KV Cache 内容：[sink_0, sink_1, sink_2, sink_3, window_901, window_902, ..., window_1000]
重新编号为：   [   0,      1,      2,      3,       4,          5,     ...,      103   ]
```

sink tokens 永远占据位置 0–3，窗口内容按顺序紧接着编号。这样位置编码保持连续，避免了跨越式的位置跳跃。

这个重新编号的代价是：位置编码不再反映 token 在原始长序列中的真实位置，但模型在推理时至少能看到一个**合理连续的位置序列**，足以维持正常输出。

---

## 4. KV Cache 的管理逻辑

### 4.1 滑动窗口的数据结构

每生成一个新 token，KV Cache 的更新过程如下：

1. 计算新 token 的 K 和 V，追加到 KV Cache
2. 如果 KV Cache 总长度超过 $s + W$，把中间最旧的窗口部分弹出
3. Sink tokens 永远不被弹出

用伪代码表示：

```python
class StreamingKVCache:
    def __init__(self, n_sink, window_size):
        self.n_sink = n_sink        # sink token 数量，默认 4
        self.window_size = window_size  # 滑动窗口大小
        self.sink_keys = []         # sink tokens 的 KV，永不丢弃
        self.sink_values = []
        self.window_keys = []       # 滑动窗口的 KV
        self.window_values = []

    def update(self, new_key, new_value, position):
        """新 token 到来时更新 KV Cache"""
        if position < self.n_sink:
            # 前 n_sink 个 token 进入 sink 区，永久保留
            self.sink_keys.append(new_key)
            self.sink_values.append(new_value)
        else:
            # 之后的 token 进入滑动窗口
            self.window_keys.append(new_key)
            self.window_values.append(new_value)
            # 窗口超过上限，弹出最旧的一个
            if len(self.window_keys) > self.window_size:
                self.window_keys.pop(0)
                self.window_values.pop(0)

    def get(self):
        """取出当前 KV Cache 用于注意力计算（sink + 窗口拼接）"""
        keys = self.sink_keys + self.window_keys
        values = self.sink_values + self.window_values
        # 重新按顺序编号位置（0, 1, 2, ..., len-1）
        positions = list(range(len(keys)))
        return keys, values, positions
```

### 4.2 注意力计算的变化

正常的推理注意力计算：

```python
def generate_next_token(x, kv_cache, position):
    # 计算当前 token 的 Q, K, V
    q = x @ W_Q      # (1, head_dim)
    k = x @ W_K
    v = x @ W_V

    # 更新 KV Cache
    kv_cache.update(k, v, position)

    # 取出当前 KV Cache（sink + 窗口），并获得重新编号的位置
    keys, values, positions = kv_cache.get()

    # 用重新编号的位置对 q 和 keys 施加 RoPE
    q = apply_rope(q, position=len(positions) - 1)  # q 在新序列中的位置
    keys = [apply_rope(k_i, position=pos) for k_i, pos in zip(keys, positions)]

    # 标准注意力计算
    scores = q @ stack(keys).T / sqrt(head_dim)
    weights = softmax(scores)
    return weights @ stack(values)
```

关键在于 `apply_rope` 使用的是**重新编号后的位置**，而不是 token 在原始序列中的全局位置。

---

## 5. 与其他方案的对比

| 方案 | KV Cache 大小 | 长序列能力 | 记忆方式 | 复杂度 |
|:---|:---:|:---:|:---|:---:|
| 完整 KV Cache | $O(L)$，随长度增长 | 受显存限制 | 精确保留所有历史 | 低（无需特殊处理） |
| 朴素滑动窗口 | $O(W)$，固定 | 理论无限，但实际崩溃 | 只记最近 $W$ 个 | 低，但不可用 |
| **StreamingLLM** | $O(s + W)$，固定 | **实际可用的无限流式** | sink + 最近 $W$ 个 | 低，工程简单 |
| H2O / SnapKV | $O(W)$，固定 | 有限（窗口内） | 按语义重要性选择 | 中（需要额外计算重要性分数） |
| MLA | $O(L \cdot d_c)$，增长但极小 | 受显存限制 | 压缩后精确保留所有历史 | 中（需要压缩/解压） |

StreamingLLM 的核心优势是**实现简单、不需要重新训练现有模型**。作者在原论文中验证，对已有的 LLaMA、MPT、Falcon 等模型，只需修改推理时的 KV Cache 管理逻辑，不改动任何权重，StreamingLLM 就能正常工作。

H2O 和 SnapKV 走的是"按重要性选 token"的思路，能更充分地利用有限的 KV Cache 空间，但需要额外计算每个 token 的重要性分数，且在 attention sink 问题上也需要特别处理。

---

## 6. StreamingLLM 的局限性

这里有一个容易被忽视的关键区别：

**StreamingLLM 实现的是"无限流式推理"，不是"无限长上下文理解"。**

| 能力 | StreamingLLM 是否支持 |
|:---|:---:|
| 连续生成超过显存限制的长文本 | ✅ 支持 |
| 在生成过程中引用任意位置的历史内容 | ❌ 不支持 |
| 记住 5000 个 token 之前说的内容 | ❌ 会忘记 |
| 语义一致的超长对话 | ⚠️ 仅限最近窗口内的内容 |

一旦某个 token 离开滑动窗口，它就真的消失了——模型无法"回忆"它。这和人类的短期记忆很像：你可以持续对话，但对话内容会随时间淡出记忆，而不是永久保留。

如果你需要的是"真正理解并随时检索任意长度历史的能力"，StreamingLLM 解决不了这个问题，需要考虑 RAG（检索增强生成）或者专为长上下文设计的架构。

---

## 7. 初学者常见混淆

**Q：Attention Sink 是模型的设计，还是无意学到的行为？**

是训练中**自发形成**的行为，不是人工设计的。模型发现"把多余注意力堆在初始 token 上"是一个低风险的策略，于是在训练过程中形成了这个习惯。原始 Transformer 论文里完全没有提到这个现象，它是后来研究者做可解释性分析时才发现的。

**Q：为什么是最初的几个 token，而不是随机哪几个 token？**

初始 token 的特殊性在于它们**总是存在**——不管对话内容怎么变，它们永远在序列的最前面，位置稳定。模型需要一个稳定的"垃圾桶"，初始 token 是最合适的候选。有些论文还发现，BOS（序列起始符）token 特别容易成为 attention sink。

**Q：sink tokens 的数量 $s = 4$ 是怎么定的？**

通过实验确定的经验值。论文测试了不同的 $s$ 值，发现从 $s = 1$ 到 $s = 4$ 效果逐步提升，$s = 4$ 之后继续增加收益递减。这也侧面说明，不只是第一个 token 在发挥 sink 的作用，而是前几个 token 共同承担了这个角色。

**Q：StreamingLLM 需要重新训练模型吗？**

标准版本**不需要**，直接修改推理时的 KV Cache 管理逻辑即可。但论文也提出了一个改进方向：在训练时在每条数据前添加一个固定的"sink token"（类似 `[sink]` 这样的专用 token），让模型学会把注意力集中倾倒在这个专用位置，从而效果更稳定。这需要重新训练，但改动很小。

---

## 8. 读完这篇之后，你应该能回答这些问题

- 为什么标准推理的 KV Cache 大小会随序列长度无限增长？这会带来什么实际问题？
- "朴素滑动窗口"方案为什么会失效？即使保留了足够多的近邻上下文，为什么删掉第一个 token 还是会崩？
- 什么是 Attention Sink？为什么模型会自发地形成把注意力"堆在"初始 token 的习惯？
- StreamingLLM 的 KV Cache 结构是什么？sink tokens 和滑动窗口各自扮演什么角色？
- 为什么在 StreamingLLM 中需要对 KV Cache 内的 token 重新进行位置编码？如果不重新编号会发生什么？
- StreamingLLM 和 H2O/SnapKV 的核心区别是什么？各自适合什么场景？
- "无限流式推理"和"无限长上下文理解"有什么本质区别？StreamingLLM 实现的是哪一个？

---

## 参考资料

- 原始论文：`paper/StreamingLLM-Efficient-Streaming-with-Attention-Sinks.pdf`
- https://arxiv.org/abs/2309.17453
- KV Cache 基础：见本项目 [`MQA.md`](MQA.md)、[`GQA.md`](GQA.md)
- H2O：`paper/H2O-Heavy-Hitter-Oracle-KV-Cache.pdf`
- SnapKV：`paper/SnapKV-LLM-Knows-What-You-Are-Looking-For.pdf`
