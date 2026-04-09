# BERT

> 对应论文：`paper/BERT-Pre-training-Deep-Bidirectional-Transformers.pdf`（Devlin et al., 2018）
> 论文链接：https://arxiv.org/abs/1810.04805

---

## 1. 背景：GPT 能做的事，BERT 为什么不直接用？

读到这里，你已经知道了 GPT：Decoder-Only、Causal Masked Attention、自左向右生成。这个设计天然适合"续写"任务，但有一个代价——**每个 token 只能看见它左边的上下文**。

对于很多 NLP 任务，这个限制并不合理。

考虑情感分类任务：判断"这部电影虽然节奏慢，但结局让人意外地感动"是正面还是负面评价时，你需要同时看"节奏慢"和"感动"这两个信息，而不是从左到右一个词一个词地推进。

再考虑命名实体识别、问答、阅读理解——这些任务都需要模型**同时参考整句话的双侧上下文**，而不是只能往左看。

BERT（**B**idirectional **E**ncoder **R**epresentations from **T**ransformers）的核心想法就是：**把 Causal Mask 去掉**，让每个 token 可以自由地关注整个序列中的所有位置——无论左边还是右边。

---

## 2. BERT 的 Attention：全双向，没有掩码

### 2.1 与 GPT 的对比

BERT 和 GPT 的 Attention 公式是一模一样的：

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

唯一的区别在于**是否加 Causal Mask**：

| | GPT（Decoder） | BERT（Encoder） |
|:---|:---:|:---:|
| Attention 类型 | Causal Masked Self-Attention | **双向** Self-Attention |
| 每个位置能看到 | 自己和左边 | **整个序列** |
| 用途 | 自回归生成 | 表示学习、理解任务 |
| 训练目标 | 预测下一个 token | 预测被遮盖的 token（MLM） |

BERT 只有 Encoder，没有 Decoder。它不生成文本，而是生成一个**对整句话有充分理解的表示**，再交给下游任务的分类头使用。

### 2.2 为什么去掉 Causal Mask 之后就不能做语言建模了？

这是一个值得想清楚的问题。

GPT 用 Causal Mask 是因为它的训练目标是：**给定前 $t-1$ 个 token，预测第 $t$ 个**。如果没有掩码，模型直接就能"看见"第 $t$ 个 token 本身，任务就退化为复制，什么都学不到。

BERT 如果也想用自回归语言模型目标，就必须加 Causal Mask，那就回到 GPT 的路线了。BERT 的选择是**换一个训练目标**，使得不加掩码也能有效学习。

---

## 3. 预训练目标：MLM 是如何让双向 Attention 变得有意义的

### 3.1 MLM：完形填空式训练

BERT 的核心预训练任务叫做 **MLM（Masked Language Model，遮盖语言模型）**。

做法很简单：随机选取输入序列中 15% 的 token，把它们替换成一个特殊符号 `[MASK]`，然后让模型预测这些位置原本是什么词。

例如：

```
原始：The cat sat on the mat .
输入：The [MASK] sat on the [MASK] .
目标：        cat              mat
```

这个任务的关键在于：要预测 `[MASK]` 的位置，模型**必须同时利用左侧和右侧的上下文**。预测第一个 `[MASK]` 时，"sat on the mat" 提供了右侧信息；预测第二个 `[MASK]` 时，"The cat sat on the" 提供了左侧信息。双向 Attention 因此得到了充分训练。

### 3.2 MLM 的细节设计

15% 的 token 被选中后，并不是全部都换成 `[MASK]`，而是：

- **80%** 替换为 `[MASK]`
- **10%** 替换为随机词（增加鲁棒性，防止模型只关注 `[MASK]` 位置）
- **10%** 保持不变（让模型也学习"原始 token 的表示是什么"）

这个设计防止了模型在微调阶段（没有 `[MASK]` token）表现退化。

### 3.3 NSP：第二个预训练目标（已被后续工作证明不必要）

BERT 原论文还包含了一个 **NSP（Next Sentence Prediction，下一句预测）** 任务：给模型两个句子，让它判断第二句是否是第一句的下一句。

后来 RoBERTa 等工作发现 NSP 其实并没有帮助，去掉反而更好。这里只需知道这个设计存在过，不必深究。

---

## 4. BERT 的输入：三种嵌入的叠加

BERT 的输入不只是词嵌入，而是三种嵌入的**逐元素相加**：

$$
\text{输入表示} = \text{Token Embedding} + \text{Segment Embedding} + \text{Position Embedding}
$$

```
[CLS] The  cat  sat  on  [SEP] The  mat  [SEP]
  ↓                         ↓
Token Embedding（词义）
  +
Segment Embedding（哪个句子：句 A = 0，句 B = 1）
  +
Position Embedding（位置：0, 1, 2, ...，可学习的绝对位置嵌入）
```

几个细节值得注意：

**`[CLS]` token**：每个输入序列的第一个位置固定放 `[CLS]`（Classification）。经过整个 Encoder 之后，`[CLS]` 的输出向量被当作**整句话的聚合表示**，用于句子级别的分类任务（情感分类、NLI 等）。

**`[SEP]` token**：用于分隔两个句子，配合 NSP 任务。

**可学习的绝对位置嵌入**：与 GPT-2 类似，BERT 不用正弦余弦公式，而是把位置嵌入作为参数直接学习。上限是 512 个位置，这也是 BERT 输入长度的上限。

---

## 5. BERT 的整体架构

BERT 是纯 **Encoder** 架构，由多个相同的 Transformer Encoder Block 堆叠而成：

```
输入序列（[CLS] + tokens + [SEP]）
    │
Token Embedding + Segment Embedding + Position Embedding
    │
┌────────────────────────┐
│   Transformer Encoder  │
│   Block  ×  L          │
│  ┌──────────────────┐  │
│  │ Multi-Head       │  │
│  │ Self-Attention   │  │  ← 全双向，无 Causal Mask
│  │ (no causal mask) │  │
│  │ Add & LayerNorm  │  │
│  │ Feed-Forward Net │  │
│  │ Add & LayerNorm  │  │
│  └──────────────────┘  │
└────────────────────────┘
    │
每个 token 位置的上下文表示
    │
  [CLS] 表示 ──→ 分类头（情感分析、NLI 等）
  token 表示 ──→ 序列标注头（NER、POS tagging 等）
  token 对   ──→ 问答头（起始/结束位置预测）
```

BERT-Base：12 层，768 维，12 个注意力头，110M 参数
BERT-Large：24 层，1024 维，16 个注意力头，340M 参数

---

## 6. 微调：一套预训练，适配所有下游任务

BERT 的设计哲学是"**预训练-微调（Pre-train & Fine-tune）**"：在大量无标注文本上预训练，学到通用表示；然后在特定任务的小规模标注数据上微调，快速适配。

微调时，只需在 BERT 输出上加一个轻量的任务头：

| 任务类型 | 输入 | 输出 |
|:---|:---|:---|
| 文本分类 | `[CLS]` 向量 | 全连接 → Softmax |
| 序列标注 | 每个 token 的向量 | 每位置全连接 → Softmax |
| 问答 | 问题 + 段落拼接 | 预测起始/结束 token 位置 |
| 句子对关系 | 两句话拼接 | `[CLS]` 向量 → 全连接 |

这套范式极大地降低了 NLP 任务的门槛——不再需要为每个任务设计专门的网络结构，只需微调 BERT 加一个简单头即可。

---

## 7. BERT vs GPT：两条路线的本质分歧

| | BERT | GPT |
|:---|:---:|:---:|
| 架构 | Encoder-Only | Decoder-Only |
| Attention | **双向**（全序列） | **单向**（仅左侧） |
| 训练目标 | MLM（完形填空） | CLM（预测下一词） |
| 核心能力 | **理解**（表示学习） | **生成**（自回归） |
| 代表应用 | 分类、NER、问答 | 对话、续写、代码生成 |
| 后续演化 | RoBERTa、ELECTRA | GPT-2/3/4、LLaMA |

一句话总结：**BERT 擅长"读懂"，GPT 擅长"写出"**。

随着 GPT-3 展示了超大模型的 few-shot 能力，以及 InstructGPT/ChatGPT 展示了对话能力，Decoder-Only 路线逐渐成为 LLM 主流。但 BERT 式双向 Encoder 在很多理解任务上依然是强基线，并且衍生了 BERT 的改进版（RoBERTa）和用于检索增强的 Bi-Encoder 等方向。

---

## 8. 读完这篇之后，你应该能回答这些问题

- BERT 和 GPT 的 Attention 机制有什么本质区别？仅仅是"加不加 Causal Mask"的区别吗？
- 为什么去掉 Causal Mask 之后不能再用"预测下一个 token"作为训练目标？BERT 用了什么替代目标？
- MLM 中 15% 的被选中 token 为什么不全部替换为 `[MASK]`？
- `[CLS]` token 在 BERT 里的作用是什么？为什么把它的输出当作整句表示是合理的？
- 如果你要用 BERT 做情感分类，整个微调流程是怎样的？

---

## 参考资料

- BERT 论文：https://arxiv.org/abs/1810.04805
- RoBERTa（证明 NSP 无用）：https://arxiv.org/abs/1907.11692
