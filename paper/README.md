# Paper Index

## 第一类：Transformer 基础 & 预训练范式

- `Transformer-Attention-Is-All-You-Need.pdf`
  Attention Is All You Need, Vaswani et al., 2017
  https://arxiv.org/abs/1706.03762

- `GPT-Improving-Language-Understanding-by-Generative-Pre-Training.pdf`
  Improving Language Understanding by Generative Pre-Training, Radford et al., 2018
  https://openai.com/research/language-unsupervised

- `LinearAttention-Transformers-are-RNNs.pdf`
  Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention, Katharopoulos et al., 2020
  https://arxiv.org/abs/2006.16236

---

## 第二类：主流开源大模型

- `LLaMA-LLaMA-Open-and-Efficient-Foundation-LMs.pdf`
  LLaMA: Open and Efficient Foundation Language Models, Touvron et al., 2023
  https://arxiv.org/abs/2302.13971

- `LLaMA2-Open-Foundation-and-Fine-Tuned-Chat-Models.pdf`
  Llama 2: Open Foundation and Fine-Tuned Chat Models, Touvron et al., 2023
  https://arxiv.org/abs/2307.09288

- `LLaMA4-Herd-Architecture-Training.pdf`
  The Llama 4 Herd, Meta, 2025
  https://arxiv.org/abs/2504.08696

- `Qwen-Qwen2.5-Technical-Report.pdf`
  Qwen2.5 Technical Report, Qwen Team, 2024
  https://arxiv.org/abs/2412.15115

- `Qwen3-Technical-Report.pdf`
  Qwen3 Technical Report, Qwen Team, 2025
  https://arxiv.org/abs/2505.09388

- Qwen3.5（无独立论文，官方技术博客）
  Qwen3.5: Towards Native Multimodal Agents, Qwen Team, 2026
  https://qwen.ai/blog?id=qwen3.5
  架构亮点：Gated DeltaNet（线性 Attention）+ Gated Attention 3:1 混合，397B/17B MoE，支持 1M token 上下文（API）

- `GLM-GLM-130B.pdf`
  GLM-130B: An Open Bilingual Pre-trained Model, Zeng et al., 2022
  https://arxiv.org/abs/2210.02414

- `Mistral-Mistral-7B.pdf`
  Mistral 7B, Jiang et al., 2023
  https://arxiv.org/abs/2310.06825

- `Gemma-Gemma-2-Technical-Report.pdf`
  Gemma 2: Improving Open Language Models at a Practical Size, Google DeepMind, 2024
  https://arxiv.org/abs/2408.15000

- `Gemma3-Technical-Report.pdf`
  Gemma 3 Technical Report, Google DeepMind, 2025
  https://arxiv.org/abs/2503.19786

- Gemma 4（无独立论文，官方博客 + Model Card）
  Gemma 4: Byte for byte, the most capable open models, Google DeepMind, 2026
  博客：https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/
  Model Card：https://ai.google.dev/gemma/docs/core/model_card_4
  架构亮点：局部滑动窗口（512/1024 tokens）+ 全局 Attention 5:1 交替，Shared KV Cache，支持 256K 上下文

- `DeepSeek-V2-MLA-MoE.pdf`
  DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model, DeepSeek, 2024
  https://arxiv.org/abs/2405.04434

- `DeepSeek-V3-Technical-Report.pdf`
  DeepSeek-V3 Technical Report, DeepSeek, 2024
  https://arxiv.org/abs/2412.19437

- `DeepSeek-V3.2-Technical-Report.pdf`
  DeepSeek-V3.2 Technical Report, DeepSeek, 2025
  https://arxiv.org/abs/2512.02556
  架构改动：在 MLA 基础上新增 DSA（DeepSeek Sparse Attention），Lightning Indexer 动态选取 Top-k KV，复杂度 O(L²) → O(L·k)

- `DeepSeek-R1-Incentivizing-Reasoning-via-RL.pdf`
  DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, DeepSeek, 2025
  https://arxiv.org/abs/2501.12948

- `Phi-4-Technical-Report.pdf`
  Phi-4 Technical Report, Microsoft, 2024
  https://arxiv.org/abs/2412.08905

---

## 第三类：Attention 结构变体与稀疏优化

### 头部结构重设计

- `MQA-Multi-Query-Attention.pdf`
  Fast Transformer Decoding: One Write-Head is All You Need, Shazeer, 2019
  https://arxiv.org/abs/1911.02727

- `GQA-Grouped-Query-Attention.pdf`
  GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints, Ainslie et al., 2023
  https://arxiv.org/abs/2305.13245

  注：MLA 论文见 DeepSeek-V2（第二类）

### 稀疏 Attention 与长上下文

- `Longformer-Long-Document-Transformer.pdf`
  Longformer: The Long-Document Transformer, Beltagy et al., 2020
  https://arxiv.org/abs/2004.08249

- `StreamingLLM-Efficient-Streaming-with-Attention-Sinks.pdf`
  Efficient Streaming Language Models with Attention Sinks, Xiao et al., 2023
  https://arxiv.org/abs/2309.17453

- `H2O-Heavy-Hitter-Oracle-KV-Cache.pdf`
  H2O: Heavy-Hitter Oracle for Efficient Generative Inference of LLMs, Zhang et al., 2023
  https://arxiv.org/abs/2306.14048

- `Infini-Attention-Infinite-Context-Transformers.pdf`
  Leave No Context Behind: Efficient Infinite Context Transformers with Infini-attention, Munkhdalai et al., 2024
  https://arxiv.org/abs/2404.07143

- `SnapKV-LLM-Knows-What-You-Are-Looking-For.pdf`
  SnapKV: LLM Knows What You Are Looking for Before Generation, Li et al., 2024
  https://arxiv.org/abs/2404.14469

- `Quest-Query-Aware-Sparsity-for-LLM-Inference.pdf`
  Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference, Tang et al., 2024
  https://arxiv.org/abs/2406.10774

- `MInference-Dynamic-Sparse-Attention-Long-Context.pdf`
  MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention, Jiang et al., 2024
  https://arxiv.org/abs/2407.02490

- `NSA-Native-Sparse-Attention.pdf`
  Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention, Yuan et al., 2025
  https://arxiv.org/abs/2502.11089

- `MiniCPM-SALA-Sparse-Linear-Attention-Hybrid.pdf`
  MiniCPM-SALA: Hybridizing Sparse and Linear Attention for Efficient Long-Context Modeling, 2025
  https://arxiv.org/abs/2602.11761

---

## 第四类：线性 Attention & 状态空间模型

### 线性 Attention 基础

- `RWKV-Reinventing-RNNs-for-the-Transformer-Era.pdf`
  RWKV: Reinventing RNNs for the Transformer Era, Peng et al., 2023
  https://arxiv.org/abs/2305.13048

- `GLA-Gated-Linear-Attention.pdf`
  Gated Linear Attention Transformers with Hardware-Efficient Training, Yang et al., 2023
  https://arxiv.org/abs/2312.06635

- `RetNet-Retentive-Network-Successor-to-Transformer.pdf`
  Retentive Network: A Successor to Transformer for Large Language Models, Sun et al., 2023
  https://arxiv.org/abs/2307.08621

- `HGRN-Hierarchically-Gated-Recurrent-Neural-Network.pdf`
  Hierarchically Gated Recurrent Neural Network for Sequence Modeling, Qin et al., 2023
  https://arxiv.org/abs/2311.04823

- `HGRN2-Gated-Linear-RNNs-with-State-Expansion.pdf`
  HGRN2: Gated Linear RNNs with State Expansion, Qin et al., 2024
  https://arxiv.org/abs/2404.07904

- `RWKV6-Eagle-and-Finch.pdf`
  Eagle and Finch: RWKV with Matrix-Valued States and Dynamic Recurrence, Peng et al., 2024
  https://arxiv.org/abs/2404.05892

- `RWKV7-Goose-Expressive-Dynamic-State-Evolution.pdf`
  RWKV-7 "Goose" with Expressive Dynamic State Evolution, Peng et al., 2025
  https://arxiv.org/abs/2503.14456

### Delta 规则与门控增强

- `DeltaNet-Parallelizable-Linear-Recurrence.pdf`
  Parallelizing Linear Transformers with the Delta Rule over Sequence Length, Yang et al., 2024
  https://arxiv.org/abs/2406.06484

- `Gated-DeltaNet-Improving-Mamba2-with-Delta-Rule.pdf`
  Gated Delta Networks: Improving Mamba2 with Delta Rule, Yang et al., 2024
  https://arxiv.org/abs/2412.06464

- `GSA-Gated-Slot-Attention.pdf`
  Gated Slot Attention for Efficient Linear-Time Sequence Modeling, Yang et al., 2024
  https://arxiv.org/abs/2409.07146

- `Kimi-Linear-Expressive-Efficient-Attention.pdf`
  Kimi Linear: An Expressive, Efficient Attention Architecture, Moonshot AI, 2025
  https://arxiv.org/abs/2510.26692

### 状态空间模型（SSM）

- `S4-Efficiently-Modeling-Long-Sequences-SSM.pdf`
  Efficiently Modeling Long Sequences with Structured State Spaces, Gu et al., 2021
  https://arxiv.org/abs/2111.00396

- `Mamba-Linear-Time-Sequence-Modeling-Selective-SSM.pdf`
  Mamba: Linear-Time Sequence Modeling with Selective State Spaces, Gu & Dao, 2023
  https://arxiv.org/abs/2312.00752

- `Mamba2-Transformers-are-SSMs.pdf`
  Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality, Dao & Gu, 2024
  https://arxiv.org/abs/2405.21060

- `Mamba3-Improved-Sequence-Modeling-SSM.pdf`
  Mamba-3: Improved Sequence Modeling using State Space Principles, 2025
  https://arxiv.org/abs/2603.15569

- `Jamba-Hybrid-Transformer-Mamba.pdf`
  Jamba: A Hybrid Transformer-Mamba Language Model, Lieber et al., 2024
  https://arxiv.org/abs/2403.19887

- `Griffin-Hawk-Gated-Linear-Recurrences-Local-Attention.pdf`
  Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models, De et al., 2024
  https://arxiv.org/abs/2402.19427

- `xLSTM-Extended-Long-Short-Term-Memory.pdf`
  xLSTM: Extended Long Short-Term Memory, Beck et al., 2024
  https://arxiv.org/abs/2405.04517

- `TTT-Learning-to-Learn-at-Test-Time.pdf`
  Learning to (Learn at Test Time): RNNs with Expressive Hidden States, Sun et al., 2024
  https://arxiv.org/abs/2407.04620

- `Titans-Learning-to-Memorize-at-Test-Time.pdf`
  Titans: Learning to Memorize at Test Time, Ali et al., 2025
  https://arxiv.org/abs/2501.00663

- `Hyena-Hierarchy-Learnable-Long-Range.pdf`
  Hyena Hierarchy: Towards Larger Convolutional Language Models, Poli et al., 2023
  https://arxiv.org/abs/2302.10866

---

## 第五类：位置编码 & 工程加速

- `RoPE-RoFormer-Rotary-Position-Embedding.pdf`
  RoFormer: Enhanced Transformer with Rotary Position Embedding, Su et al., 2021
  https://arxiv.org/abs/2104.09864

- `ALiBi-Attention-with-Linear-Biases.pdf`
  Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation, Press et al., 2021
  https://arxiv.org/abs/2108.12409

- `FlashAttention-Fast-Memory-Efficient-Exact-Attention.pdf`
  FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness, Dao et al., 2022
  https://arxiv.org/abs/2205.14135

- `FlashAttention2-Faster-Attention-Work-Partitioning.pdf`
  FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning, Dao, 2023
  https://arxiv.org/abs/2307.08691
