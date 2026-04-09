# Attention 笔记索引

## 推荐阅读顺序

### 第一阶段：打好地基
1. [`Transformer.md`](Transformer.md) ← **从这里开始，不要跳**

### 第二阶段：大模型主线（Decoder-Only）
2. [`GPT.md`](GPT.md)
3. [`LLaMA.md`](LLaMA.md)（RoPE + GQA，开源大模型标准架构）
4. [`BERT.md`](BERT.md)（双向 Encoder，理解任务的另一条路）
5. [`Qwen.md`](Qwen.md)
6. [`GLM.md`](GLM.md)
7. [`Mistral.md`](Mistral.md)
8. [`Gemma.md`](Gemma.md)

### 第三阶段：Attention 结构优化
9. [`MQA.md`](MQA.md)（多查询注意力，KV Cache 压缩第一步）
10. [`GQA.md`](GQA.md)（分组查询注意力，MQA 与 MHA 的折中）
11. [`MLA.md`](MLA.md)（多头潜在注意力，DeepSeek 的低秩 KV 压缩）

### 第四阶段：稀疏与流式 Attention
12. [`StreamingLLM.md`](StreamingLLM.md)（Attention Sink 现象 + 无限流式推理）
13. [`NSA.md`](NSA.md)（原生稀疏 Attention，硬件对齐训练）

### 第五阶段：超越 Transformer
14. [`RWKV.md`](RWKV.md)
15. [`GLA.md`](GLA.md)
16. [`Mamba.md`](Mamba.md)（选择性状态空间，首个匹配 Transformer 的线性模型）

---

## 分类索引

### 标准 Transformer 路线
| 文章 | 核心主题 |
|:---|:---|
| Transformer | Multi-Head Attention、Encoder-Decoder、位置编码 |
| GPT | Decoder-Only、Causal Masked Attention、自回归生成 |
| LLaMA | RoPE 旋转位置编码、GQA、开源大模型标准架构 |
| BERT | 双向 Self-Attention、MLM 预训练、Encoder-Only |
| Qwen | RoPE、GQA、长上下文扩展 |
| GLM | 双向 Attention、2D 位置编码、空白填充预训练 |
| Mistral | Sliding Window Attention、GQA |
| Gemma | 交替全局/局部 Attention |

### Attention 结构优化路线
| 文章 | 核心主题 |
|:---|:---|
| MQA | 单 KV 头共享所有 Query，KV Cache 压缩 h 倍 |
| GQA | Query 头分组，每组共享 KV，质量与效率的折中 |
| MLA | 低秩 KV 压缩，KV Cache 压缩 93% |

### 稀疏与流式注意力路线
| 文章 | 核心主题 |
|:---|:---|
| StreamingLLM | Attention Sink 现象、固定大小 KV Cache 实现无限推理 |
| NSA | 原生稀疏 Attention、硬件对齐训练 |

### RNN / 线性注意力 / SSM 路线
| 文章 | 核心主题 |
|:---|:---|
| RWKV | 指数衰减线性 Attention、训练并行推理递推 |
| GLA | 门控线性 Attention、数据相关门控 |
| Mamba | 选择性状态空间模型、$\Delta$/B/C 输入相关、硬件感知扫描 |

---

## 每篇笔记的结构约定

所有笔记都遵循同一套写作标准（详见 `../skills/wia-writer/SKILL.md`）：

- 公式用 `$$` LaTeX，不用反引号
- 新术语首次出现时**加粗**
- 每个概念先说"为什么需要它"，再说"是什么"
- 代码段有行级注释
- 结尾有"读完后你应该能回答的问题"

---

## 计划中的笔记

以下内容尚未覆盖，按优先级排列：

**大模型主线**
- DeepSeek-V2 / V3
- T5（Encoder-Decoder 预训练范式）

**Attention 优化**
- RoPE / ALiBi（位置编码专题）
- FlashAttention 系列
- StreamingLLM 的配套：H2O、SnapKV

**线性 Attention & SSM**
- Mamba-2（统一 SSM 与 Attention 框架）
- RetNet
- S4（Mamba 的前身）
