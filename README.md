<div align="center">

# 🔍 What is Attention?

**从零理解 LLM 中 Attention 的演化脉络**

*一个专为初学者设计的系统性学习项目——不只告诉你公式是什么，而是告诉你它为什么存在。*

---

![License](https://img.shields.io/badge/license-MIT-blue)
![Language](https://img.shields.io/badge/language-中文-red)
![Status](https://img.shields.io/badge/status-持续更新-green)

</div>

---

## 这个项目是什么

学 LLM，绕不开 Attention。但市面上大多数资料要么停在 "softmax(QK^T/√d)V" 这个公式打住，要么直接扔给你几百页的论文。

这个项目想做的事情只有一件：**把 Attention 从零讲清楚**。

- 不只讲公式，先讲为什么需要它
- 不只讲一个模型，而是梳理整条演化脉络
- 不只给代码，而是给带注释的可读代码
- 不只给结论，而是给可以推导出结论的直觉

覆盖范围从 2017 年的原始 Transformer，一路追到 2025 年的 NSA、Mamba-3、MLA、Kimi Linear 等最新进展。

---

## 学习路线图

```
Transformer（必读，地基）
    │
    ├──► GPT ──► LLaMA ──► Qwen / GLM / Mistral / Gemma / DeepSeek / Phi
    │          （Decoder-Only 大模型主线）
    │
    ├──► BERT / T5
    │    （Encoder / Encoder-Decoder 预训练范式）
    │
    ├──► MQA / GQA / MLA ──► StreamingLLM / H2O / SnapKV / MInference / NSA
    │    （Attention 结构变体与稀疏优化）
    │
    ├──► RoPE / ALiBi / FlashAttention
    │    （位置编码 & 工程加速，补充阅读）
    │
    └──► Linear Attention ──► RWKV / GLA / RetNet / DeltaNet / GSA
                          ──► Mamba / Mamba-2 / Mamba-3 / TTT / Titans
         （线性 Attention & 状态空间模型）
```

**建议阅读顺序（从零开始）：**

| 阶段 | 内容 | 目标 |
|:---:|:---|:---|
| 第一步 | Transformer | 理解 Attention 的基本原理和完整架构 |
| 第二步 | GPT → LLaMA → Qwen | 理解 Decoder-Only 大模型如何从 Transformer 演变 |
| 第三步 | MQA / GQA / MLA | 理解 Attention 头部结构的演化与 KV Cache 优化 |
| 第四步 | StreamingLLM / NSA | 理解稀疏 Attention 如何解决长上下文问题 |
| 第五步 | RWKV / GLA / Mamba | 理解超越 O(n²) 的序列建模路线 |
| 扩展阅读 | RoPE / FlashAttention / BERT 等 | 位置编码、工程优化与预训练范式 |

---

## 覆盖范围

### 🏛️ 第一类：Transformer 基础 & 预训练范式

| 模型 / 方法 | 核心 Attention 贡献 | 论文 | 年份 | 笔记 |
|:---|:---|:---|:---:|:---:|
| **Transformer** | 提出 Multi-Head Scaled Dot-Product Attention，实现完全并行的序列处理 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | ✅ |
| **BERT** | 双向 Self-Attention + MLM 预训练，首次充分利用双侧上下文 | [BERT](https://arxiv.org/abs/1810.04805) | 2018 | ✅ |
| **GPT** | Decoder-Only Causal Masked Attention，统一预训练与生成范式 | [GPT](https://openai.com/research/language-unsupervised) | 2018 | ✅ |
| **GPT-3** | 175B 参数，验证 Causal Masked Attention 的 Few-Shot 涌现能力 | [GPT-3](https://arxiv.org/abs/2005.14165) | 2020 | ✅ |
| **T5** | Encoder-Decoder + 统一 Text-to-Text 框架，Cross-Attention 经典实现 | [T5](https://arxiv.org/abs/1910.10683) | 2019 | ✅ |
| **RoBERTa** | 优化 BERT 的 Masked Self-Attention 预训练策略 | [RoBERTa](https://arxiv.org/abs/1907.11692) | 2019 | ✅ |

---

### 🚀 第二类：主流开源大模型

| 模型 | 核心 Attention 改进 | 论文 / 报告 | 年份 | 笔记 |
|:---|:---|:---|:---:|:---:|
| **LLaMA** | Causal Self-Attention + RoPE，确立开源大模型标准架构 | [LLaMA](https://arxiv.org/abs/2302.13971) | 2023 | 📝 |
| **LLaMA 2** | 引入 GQA，大幅降低 KV Cache 显存 | [LLaMA 2](https://arxiv.org/abs/2307.09288) | 2023 | 📝 |
| **LLaMA 3** | 全面采用 GQA + RoPE，扩展长上下文 | [LLaMA 3](https://arxiv.org/abs/2407.21783) | 2024 | 📝 |
| **LLaMA 4** | iRoPE（交错 RoPE），部分层无位置编码，支持 1000 万 token 上下文 | [LLaMA 4](https://arxiv.org/abs/2601.11659) | 2025 | 📝 |
| **Qwen 2.5** | RoPE + GQA + 长上下文优化，支持 128K 上下文 | [Qwen2.5](https://arxiv.org/abs/2412.15115) | 2024 | ✅ |
| **Qwen 3** | 统一思考/非思考模式，标准 Attention 架构，覆盖 0.6B–235B | [Qwen3](https://arxiv.org/abs/2505.09388) | 2025 | 📝 |
| **Qwen 3.5** | Gated DeltaNet（线性 Attention）+ Gated Attention 混合 3:1，支持 1M 上下文 | [官方博客](https://qwen.ai/blog?id=qwen3.5) | 2026 | 📝 |
| **GLM** | 双向 Self-Attention + 2D 位置编码 + 自回归空白填充预训练 | [GLM-130B](https://arxiv.org/abs/2210.02414) | 2022 | ✅ |
| **Mistral 7B** | Sliding Window Attention (SWA) + GQA 组合 | [Mistral 7B](https://arxiv.org/abs/2310.06825) | 2023 | ✅ |
| **Gemma 2** | 交替全局 Attention 与局部滑动窗口 Attention | [Gemma 2](https://arxiv.org/abs/2408.15000) | 2024 | ✅ |
| **Gemma 3** | 局部窗口为主（25:1 比例），大幅降低长上下文 KV Cache | [Gemma 3](https://arxiv.org/abs/2503.19786) | 2025 | 📝 |
| **Gemma 4** | 局部窗口 + 全局 Attention 交替 5:1，Shared KV Cache，支持 256K 上下文 | [官方博客](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/) | 2026 | 📝 |
| **DeepSeek-V2** | 提出 MLA（Multi-Head Latent Attention），KV Cache 压缩 93% | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 2024 | 📝 |
| **DeepSeek-V3** | 改进 MLA + 无辅助损失 MoE 负载均衡 | [DeepSeek-V3](https://arxiv.org/abs/2412.19437) | 2024 | 📝 |
| **DeepSeek-V3.2** | MLA + DSA（稀疏选择 KV），Lightning Indexer 将复杂度降至 O(L·k) | [Technical Report](https://arxiv.org/abs/2512.02556) | 2025 | 📝 |
| **DeepSeek-R1** | 基于 DeepSeek-V3-Base（含 MLA），纯 RL 激发推理能力 | [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | 📝 |
| **Phi-4** | 标准 Dense Transformer，14B 参数依靠合成数据超越 GPT-4 | [Phi-4](https://arxiv.org/abs/2412.08905) | 2024 | 📝 |
| **Falcon** | 多查询注意力（MQA）早期大规模实践 | [Falcon](https://arxiv.org/abs/2311.16867) | 2023 | 📝 |

---

### 🧩 第三类：Attention 结构变体与稀疏优化

> 聚焦 **算法层面** 对 Attention 机制本身的改造：头部结构重设计、长上下文稀疏化、KV Cache 选择性保留。

#### 头部结构重设计

| 方法 | 解决的问题 | 论文 | 年份 | 笔记 |
|:---|:---|:---|:---:|:---:|
| **MQA**（多查询注意力） | 单 KV 头共享所有 Query 头，极致压缩 KV Cache | [MQA](https://arxiv.org/abs/1911.02727) | 2019 | 📝 |
| **GQA**（分组查询注意力） | MQA 与 MHA 之间的折中，兼顾质量与效率 | [GQA](https://arxiv.org/abs/2305.13245) | 2023 | 📝 |
| **MLA**（多头潜在注意力） | 低秩 KV 压缩至潜变量，KV Cache 压缩 93%，DeepSeek 提出 | [DeepSeek-V2](https://arxiv.org/abs/2405.04434) | 2024 | 📝 |

#### 稀疏 Attention 与长上下文

| 方法 | 核心思想 | 论文 | 年份 | 笔记 |
|:---|:---|:---|:---:|:---:|
| **Longformer** | 局部窗口 + 全局 token，O(n²) → O(n·w) | [Longformer](https://arxiv.org/abs/2004.08249) | 2020 | 📝 |
| **StreamingLLM** | 发现 Attention Sink 现象，保留初始 token + 滑动窗口实现无限流式推理 | [StreamingLLM](https://arxiv.org/abs/2309.17453) | 2023 | 📝 |
| **H2O** | 少量"重击者（Heavy Hitter）"token 主导注意力，动态 KV Cache 淘汰 | [H2O](https://arxiv.org/abs/2306.14048) | 2023 | 📝 |
| **Infini-Attention** | 局部 Attention + 线性 Attention 压缩记忆，单 block 实现无限上下文 | [Infini-Attention](https://arxiv.org/abs/2404.07143) | 2024 | 📝 |
| **SnapKV** | 用末尾观察窗口投票预测各头关注位置，生成前压缩 KV Cache | [SnapKV](https://arxiv.org/abs/2404.14469) | 2024 | 📝 |
| **Quest** | 对 KV Cache 分页，按 Query 估计每页关键性，仅加载 Top-K 页 | [Quest](https://arxiv.org/abs/2406.10774) | 2024 | 📝 |
| **MInference** | 发现 A-shape / Vertical-Slash / Block-Sparse 三种稀疏模式，prefill 加速 10× | [MInference](https://arxiv.org/abs/2407.02490) | 2024 | 📝 |
| **NSA**（原生稀疏注意力） | 硬件对齐的可端到端训练稀疏 Attention | [NSA](https://arxiv.org/abs/2502.11089) | 2025 | ✅ |
| **MiniCPM-SALA** | 25% 稀疏 Attention + 75% 线性 Attention 混合，支持百万 token 上下文 | [MiniCPM-SALA](https://arxiv.org/abs/2602.11761) | 2025 | 📝 |

---

### 🌀 第四类：线性 Attention & 状态空间模型

> 彻底重新设计序列建模机制，将复杂度从 O(n²) 降至 O(n)，同时尽可能保留 Transformer 的表达能力。

#### 线性 Attention 基础

| 方法 | 核心思想 | 论文 | 年份 | 笔记 |
|:---|:---|:---|:---:|:---:|
| **Linear Attention** | 改写 softmax Attention 使其可递推，O(n²) → O(n) | [Transformers are RNNs](https://arxiv.org/abs/2006.16236) | 2020 | 📝 |
| **RetNet** | 训练时并行、推理时递推，保留机制（Retention）替代 Attention | [RetNet](https://arxiv.org/abs/2307.08621) | 2023 | 📝 |
| **RWKV** | 指数衰减 RNN 式线性 Attention，训练并行、推理高效 | [RWKV](https://arxiv.org/abs/2305.13048) | 2023 | ✅ |
| **GLA**（门控线性注意力） | 线性 Attention + 数据相关门控，增强表达能力 | [GLA](https://arxiv.org/abs/2312.06635) | 2023 | ✅ |
| **HGRN / HGRN2** | 分层门控 RNN，HGRN2 通过外积扩展状态维度并获线性 Attention 解释 | [HGRN](https://arxiv.org/abs/2311.04823) / [HGRN2](https://arxiv.org/abs/2404.07904) | 2023/2024 | 📝 |
| **RWKV-6/7** | RWKV-6 引入动态循环机制；RWKV-7 引入广义 Delta 规则，具备状态跟踪能力 | [Eagle/Finch](https://arxiv.org/abs/2404.05892) / [RWKV-7](https://arxiv.org/abs/2503.14456) | 2024/2025 | 📝 |

#### Delta 规则与门控增强

| 方法 | 核心思想 | 论文 | 年份 | 笔记 |
|:---|:---|:---|:---:|:---:|
| **DeltaNet** | 用 Delta 规则（检索后更新）替代线性 Attention 累加，赋予纠错记忆能力 | [DeltaNet](https://arxiv.org/abs/2406.06484) | 2024 | 📝 |
| **Gated DeltaNet** | Mamba2 门控机制 + DeltaNet Delta 规则，自适应遗忘与精准更新协同 | [Gated DeltaNet](https://arxiv.org/abs/2412.06464) | 2024 | 📝 |
| **GSA**（门控槽位注意力） | 两层 GLA 经 softmax 连接，上下文感知读取与自适应遗忘兼顾 | [GSA](https://arxiv.org/abs/2409.07146) | 2024 | 📝 |
| **Kimi Linear** | Kimi Delta Attention（KDA）+ MLA 混合 MoE，首次线性 Attention 全面超越全注意力 | [Kimi Linear](https://arxiv.org/abs/2510.26692) | 2025 | 📝 |

#### 状态空间模型（SSM）

| 方法 | 核心思想 | 论文 | 年份 | 笔记 |
|:---|:---|:---|:---:|:---:|
| **S4** | 结构化状态空间 + HiPPO 矩阵，用快速卷积替代 Attention | [S4](https://arxiv.org/abs/2111.00396) | 2021 | 📝 |
| **Mamba** | 选择性 SSM，引入数据相关的状态转移，性能接近 Transformer | [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | 📝 |
| **Mamba-2** | 统一 SSM 与 Attention 框架，验证两者的数学等价关系 | [Mamba-2](https://arxiv.org/abs/2405.21060) | 2024 | 📝 |
| **Mamba-3** | 指数梯形离散化 + 复数值状态 + MIMO SSM，提升算术强度与状态追踪 | [Mamba-3](https://arxiv.org/abs/2603.15569) | 2025 | 📝 |
| **Jamba** | Mamba + 标准 Attention 混合架构，融合两者优势 | [Jamba](https://arxiv.org/abs/2403.19887) | 2024 | 📝 |
| **Hawk / Griffin** | Hawk（纯门控线性 RNN）+ Griffin（线性循环 + 局部 Attention 混合） | [Griffin](https://arxiv.org/abs/2402.19427) | 2024 | 📝 |
| **xLSTM** | 指数门控 + 矩阵记忆（mLSTM），将 LSTM 扩展至十亿参数级别 | [xLSTM](https://arxiv.org/abs/2405.04517) | 2024 | 📝 |
| **TTT** | 隐状态为可学习小模型，每条序列推理时在线更新，RNN 具备持续学习能力 | [TTT](https://arxiv.org/abs/2407.04620) | 2024 | 📝 |
| **Titans** | 神经长期记忆模块 + 局部 Attention + 持久记忆，支持 2M+ token 上下文 | [Titans](https://arxiv.org/abs/2501.00663) | 2025 | 📝 |
| **Hyena** | 参数化隐式卷积替代 Attention，无需显式 QK 计算 | [Hyena](https://arxiv.org/abs/2302.10866) | 2023 | 📝 |

---

### 🔧 第五类：位置编码 & 工程加速（补充阅读）

> 这一类不改变 Attention 的算法结构，而是从**位置表示**和**硬件实现**角度对 Transformer 做补充优化，是读懂工业级模型代码的必要背景知识。

| 方法 | 解决的问题 | 论文 | 年份 | 笔记 |
|:---|:---|:---|:---:|:---:|
| **RoPE**（旋转位置编码） | 将位置信息编码进 Attention 计算本身，外推性强，已成事实标准 | [RoFormer](https://arxiv.org/abs/2104.09864) | 2021 | 📝 |
| **ALiBi** | 用线性偏置替代位置嵌入，无需改动 Attention 结构即可外推长序列 | [ALiBi](https://arxiv.org/abs/2108.12409) | 2021 | 📝 |
| **FlashAttention** | IO 感知的分块 Attention，速度提升 2–4×，内存大幅降低 | [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | 📝 |
| **FlashAttention-2** | 改进分块策略与并行化，GPU 利用率进一步提升 | [FlashAttention-2](https://arxiv.org/abs/2307.08691) | 2023 | 📝 |
| **FlashAttention-3** | 针对 H100 架构优化，利用异步执行和低精度加速 | [FlashAttention-3](https://arxiv.org/abs/2407.08608) | 2024 | 📝 |

---

## 项目结构

```
WiA/
├── README.md              ← 你在这里
├── Attention/             ← 每个模型/方法的学习笔记
│   ├── README.md          ← 推荐阅读顺序索引
│   │
│   ├── ── 第一类：Transformer 基础 & 预训练范式 ──
│   ├── Transformer.md     ✅
│   ├── BERT.md            ✅
│   ├── GPT.md             ✅
│   ├── GPT-3.md           ✅
│   ├── T5.md              ✅
│   ├── RoBERTa.md         ✅
│   │
│   ├── ── 第二类：主流开源大模型 ──
│   ├── LLaMA.md           ✅
│   ├── Qwen.md            ✅
│   ├── GLM.md             ✅
│   ├── Mistral.md         ✅
│   ├── Gemma.md           ✅
│   │
│   ├── ── 第三类：Attention 结构变体与稀疏优化 ──
│   ├── MQA.md             ✅
│   ├── GQA.md             ✅
│   ├── MLA.md             ✅
│   ├── StreamingLLM.md    ✅
│   ├── NSA.md             ✅
│   │
│   ├── ── 第四类：线性 Attention & 状态空间模型 ──
│   ├── RWKV.md            ✅
│   ├── GLA.md             ✅
│   ├── Mamba.md           ✅
│   │
│   └── ── 第五类：位置编码 & 工程加速 ──
│       └── RoPE.md        ✅
├── paper/                 ← 对应的论文 PDF
│   └── README.md          ← 论文索引
└── skills/
    └── wia-writer/        ← 统一风格撰写新文章的 AI Skill
```

---

## 每篇笔记的结构

每篇笔记都遵循同一套结构：

1. **背景**：为什么需要它，解决了什么问题
2. **核心机制**：直觉 → 类比 → 公式 → 代码
3. **架构全貌**：数据流与组件关系
4. **与已有概念的对比**：表格梳理
5. **常见混淆**：初学者最容易卡住的地方
6. **你应该能回答的问题**：自测清单

---

## 本项目的取舍

专注 **Attention 算法主线**，刻意不展开：

- 训练配方与对齐（SFT、RLHF、DPO）
- 数据工程与清洗
- 推理系统与部署优化（vLLM、量化、蒸馏）
- 评测基准分析

---

## 参考资料

- Vaswani et al., *Attention Is All You Need*, 2017
- Happy-LLM：https://datawhalechina.github.io/happy-llm
- The Illustrated Transformer：https://jalammar.github.io/illustrated-transformer/
