# Qwen 2.5 与 Qwen 3：从工程优化到架构创新

## 对应论文
- `paper/Qwen-Qwen2.5-Technical-Report.pdf` - Qwen Team, 2024
- `paper/Qwen3-Technical-Report.pdf` - Qwen Team, 2025

## 背景与动机

Qwen 系列是阿里巴巴推出的大语言模型家族，从 Qwen 2.5 到 Qwen 3 经历了从"工程优化"到"架构创新"的演进。

### Qwen 2.5 的设计目标
Qwen 2.5 延续了标准的 decoder-only Transformer 架构，重点在于：
- **长上下文能力**：通过 RoPE 和 YaRN 扩展技术，将上下文窗口从 32K 扩展到 128K
- **推理效率**：使用 GQA（Grouped Query Attention）降低 KV cache 显存占用
- **多语言与多模态**：在保持架构简洁的前提下，通过数据工程提升多语言和代码能力

### Qwen 3 的突破方向
Qwen 3 在 Qwen 2.5 的基础上引入了更激进的架构创新：
- **Mixture of Experts (MoE)**：通过稀疏激活降低计算成本，同时保持模型容量
- **更深的 GQA 优化**：进一步减少 KV heads 数量，在超长上下文场景下显著降低显存压力
- **Native Long Context**：原生支持 1M token 上下文窗口，无需后期扩展训练

---

## Attention 算法思路

### Qwen 2.5 的注意力机制

Qwen 2.5 采用标准的 **decoder-only causal self-attention**，核心流程为：

1. **线性投影**：将输入 $X \in \mathbb{R}^{L \times d}$ 投影为 Query、Key、Value
   $$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

2. **RoPE 位置编码**：在 Q 和 K 上应用旋转位置编码
   $$Q' = \text{RoPE}(Q), \quad K' = \text{RoPE}(K)$$

3. **Grouped Query Attention**：多个 query heads 共享少量 KV heads
   - Query heads: 28 (Qwen2.5-7B)
   - KV heads: 4
   - 每 7 个 query heads 共享 1 个 KV head

4. **因果注意力计算**：
   $$\text{Attention}(Q', K', V) = \text{softmax}\left(\frac{Q'K'^T}{\sqrt{d_k}} + M_{\text{causal}}\right)V$$
   其中 $M_{\text{causal}}$ 是下三角掩码，确保自回归生成。

### Qwen 3 的注意力创新

Qwen 3 在保留 Qwen 2.5 核心机制的基础上，引入了两大创新：

#### 1. 更激进的 GQA 配置
- **Qwen3-7B**: 28 query heads, **2 KV heads**（相比 Qwen2.5 减半）
- **Qwen3-72B**: 64 query heads, **8 KV heads**
- 在 1M 上下文场景下，KV cache 显存占用降低 50%

#### 2. Mixture of Experts (MoE) 架构
Qwen3-MoE 模型在每个 Transformer 层引入稀疏专家网络：

$$\text{FFN}_{\text{MoE}}(x) = \sum_{i=1}^{k} G(x)_i \cdot \text{Expert}_i(x)$$

其中：
- $G(x)$ 是 Top-k 门控网络，选择 $k=2$ 个专家
- 总共 64 个专家，每次只激活 2 个
- 激活参数量仅为 Dense 模型的 1/32

**关键设计**：
- **Shared Expert**：所有 token 都会经过 1 个共享专家，保证基础能力
- **Routing Strategy**：使用 auxiliary loss 平衡专家负载，避免专家坍缩
- **Attention 层保持 Dense**：MoE 只应用于 FFN，注意力层仍为全连接

---

## 核心机制详解

### 1. RoPE（Rotary Position Embedding）

RoPE 通过旋转矩阵将位置信息编码到 Q 和 K 中，公式为：

$$\text{RoPE}(x_m, m) = \begin{bmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

其中 $\theta = 10000^{-2i/d}$，$m$ 是位置索引。

**优势**：
- 相对位置信息自然融入点积：$q_m^T k_n = f(q, k, m-n)$
- 外推性好，适合长上下文扩展
- 无需额外的位置嵌入参数

### 2. YaRN（Yet another RoPE extensioN method）

Qwen 2.5 使用 YaRN 将上下文从 32K 扩展到 128K：

$$\theta'_i = \begin{cases}
\theta_i & \text{if } i < d \cdot \alpha \\
\theta_i / s & \text{if } i \geq d \cdot \alpha
\end{cases}$$

其中 $s$ 是扩展因子（通常为 4），$\alpha$ 是保留低频的比例。

**核心思想**：
- 低频维度（长距离依赖）保持不变
- 高频维度（局部依赖）进行插值
- 避免传统 PI（Position Interpolation）导致的高频信息丢失

### 3. Grouped Query Attention (GQA)

GQA 是 MQA（Multi-Query Attention）和 MHA（Multi-Head Attention）的折中方案：

| 方案 | Query Heads | KV Heads | KV Cache 大小 | 表达能力 |
|------|-------------|----------|---------------|----------|
| MHA  | $n_h$       | $n_h$    | $L \times n_h \times d_k$ | 最强 |
| GQA  | $n_h$       | $n_h / g$ | $L \times (n_h/g) \times d_k$ | 中等 |
| MQA  | $n_h$       | 1        | $L \times d_k$ | 较弱 |

**实现细节**：
```python
# Qwen2.5-7B: 28 query heads, 4 KV heads
num_query_heads = 28
num_kv_heads = 4
group_size = num_query_heads // num_kv_heads  # 7

# 将 KV 重复到每个 query group
K_repeated = K.repeat_interleave(group_size, dim=1)  # [batch, 28, seq, d_k]
V_repeated = V.repeat_interleave(group_size, dim=1)
```

---

## 架构对比

### Qwen 2.5 系列架构参数

| 模型 | 参数量 | Layers | Hidden Size | Query Heads | KV Heads | Context Length |
|------|--------|--------|-------------|-------------|----------|----------------|
| Qwen2.5-0.5B | 0.5B | 24 | 896 | 14 | 2 | 32K → 128K |
| Qwen2.5-1.5B | 1.5B | 28 | 1536 | 12 | 2 | 32K → 128K |
| Qwen2.5-3B | 3B | 36 | 2048 | 16 | 2 | 32K → 128K |
| Qwen2.5-7B | 7B | 28 | 3584 | 28 | 4 | 32K → 128K |
| Qwen2.5-14B | 14B | 48 | 5120 | 40 | 8 | 32K → 128K |
| Qwen2.5-32B | 32B | 64 | 5120 | 40 | 8 | 32K → 128K |
| Qwen2.5-72B | 72B | 80 | 8192 | 64 | 8 | 32K → 128K |

**关键特性**：
- **Vocabulary Size**: 151,936 tokens（支持多语言）
- **Activation Function**: SwiGLU
- **Normalization**: RMSNorm（pre-norm）
- **Training Data**: 18T tokens（Qwen2.5-72B）
- **Context Extension**: 使用 YaRN 从 32K 扩展到 128K

### Qwen 3 系列架构参数

| 模型 | 参数量 | 激活参数 | Layers | Hidden Size | Query Heads | KV Heads | Context Length |
|------|--------|----------|--------|-------------|-------------|----------|----------------|
| Qwen3-1.7B | 1.7B | 1.7B | 28 | 1536 | 12 | 2 | 32K |
| Qwen3-4B | 4B | 4B | 40 | 2048 | 16 | 2 | 32K |
| Qwen3-7B | 7B | 7B | 28 | 3584 | 28 | 2 | 32K |
| Qwen3-14B | 14B | 14B | 48 | 5120 | 40 | 8 | 32K |
| Qwen3-32B | 32B | 32B | 64 | 5120 | 40 | 8 | 32K |
| Qwen3-72B | 72B | 72B | 80 | 8192 | 64 | 8 | 32K |
| Qwen3-235B | 235B | 235B | 100 | 12288 | 96 | 12 | 32K |
| **Qwen3-MoE-16B** | **16B** | **2.4B** | 28 | 2048 | 16 | 2 | 32K |
| **Qwen3-MoE-57B** | **57B** | **8.7B** | 56 | 3584 | 28 | 4 | 32K |

**Qwen 3 的关键改进**：
- **更少的 KV Heads**：Qwen3-7B 从 4 个降到 2 个，显存占用减半
- **原生长上下文**：训练时即支持 32K，无需后期扩展
- **MoE 架构**：Qwen3-MoE-57B 拥有 57B 总参数，但每次只激活 8.7B
- **更大规模**：Qwen3-235B 是系列中最大的 Dense 模型

### Qwen 2.5 vs Qwen 3 对比

| 维度 | Qwen 2.5 | Qwen 3 |
|------|----------|--------|
| **架构类型** | Dense Transformer | Dense + MoE |
| **KV Heads（7B）** | 4 | 2 |
| **上下文窗口** | 32K → 128K（YaRN 扩展） | 32K（原生训练） |
| **最大模型** | 72B | 235B（Dense）+ 57B（MoE） |
| **训练数据** | 18T tokens | 20T+ tokens |
| **位置编码** | RoPE + YaRN | RoPE（原生长上下文） |
| **推理效率** | GQA 优化 | 更激进的 GQA + MoE 稀疏激活 |
| **多模态** | 需要额外适配 | 原生支持（Qwen3-VL） |

---

## 与 LLaMA 的对比

Qwen 和 LLaMA 都采用 decoder-only + RoPE + GQA 的主流架构，但在细节上有显著差异：

| 特性 | Qwen 2.5/3 | LLaMA 3.1 |
|------|------------|-----------|
| **Vocabulary Size** | 151,936 | 128,256 |
| **GQA 配置（7B）** | Qwen2.5: 28Q/4KV, Qwen3: 28Q/2KV | 32Q/8KV |
| **上下文扩展** | YaRN（Qwen2.5）, 原生（Qwen3） | RoPE scaling |
| **MoE 支持** | Qwen3-MoE | 无 |
| **多语言能力** | 强（中文优化） | 中等 |
| **训练数据** | 18T-20T | 15T+ |

**Qwen 的独特优势**：
1. **更激进的 KV 压缩**：Qwen3-7B 只用 2 个 KV heads，LLaMA3-8B 用 8 个
2. **MoE 架构**：Qwen3-MoE 提供稀疏激活选项，LLaMA 系列无 MoE 版本
3. **多语言词表**：15 万+ tokens 覆盖更多语言，中文效率更高
4. **原生长上下文**：Qwen3 训练时即支持 32K，无需后期扩展

---

## 常见问题

### Q1: 为什么 Qwen3 要减少 KV heads？
**A**: 在超长上下文场景下，KV cache 是显存瓶颈。以 Qwen3-7B 为例：
- 上下文长度 128K，KV heads 从 4 降到 2
- KV cache 显存占用：$2 \times 128K \times 2 \times 128 \times 2 = 131$ MB（FP16）
- 相比 Qwen2.5 节省 50% 显存，但实验表明性能损失小于 1%

### Q2: GQA 会损失多少性能？
**A**: 根据 Qwen 团队的消融实验：
- MHA → GQA（28Q/4KV）：困惑度上升 < 0.5%
- GQA（4KV）→ GQA（2KV）：困惑度上升 < 1%
- 但推理速度提升 30-50%（长上下文场景）

### Q3: YaRN 和 Position Interpolation 有什么区别？
**A**: 
- **PI（Position Interpolation）**：均匀压缩所有频率，$\theta'_i = \theta_i / s$
- **YaRN**：只压缩高频，保留低频，$\theta'_i = \theta_i / s$ (仅高频部分)
- YaRN 在 128K 上下文下困惑度比 PI 低 15-20%

### Q4: Qwen3-MoE 的专家是如何路由的？
**A**: 使用 Top-2 门控 + 辅助损失：
$$L_{\text{aux}} = \alpha \cdot \text{Var}(\text{load per expert})$$
确保每个专家的负载均衡，避免少数专家过载。

### Q5: Qwen 支持多长的上下文？
**A**:
- **Qwen2.5**: 训练 32K，通过 YaRN 扩展到 128K
- **Qwen3**: 原生训练 32K，理论上可扩展到 1M（需要额外微调）
- **实际推理**: 受限于显存，单卡 A100（80GB）约支持 64K-128K

---

## 完整代码实现

以下是 Qwen 2.5/3 的完整 Attention 实现，包含 RoPE、GQA 和 YaRN 扩展：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class RotaryEmbedding(nn.Module):
    """
    RoPE（Rotary Position Embedding）实现
    支持 YaRN 扩展用于长上下文
    """
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 32768,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        yarn_alpha: float = 1.0,  # YaRN 参数
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_factor = scaling_factor
        self.yarn_alpha = yarn_alpha
        
        # 计算频率：theta_i = base^(-2i/d)
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # YaRN 扩展：只对高频部分进行缩放
        if scaling_factor > 1.0:
            # 低频部分保持不变，高频部分除以 scaling_factor
            low_freq_wavelen = max_position_embeddings / scaling_factor
            high_freq_wavelen = max_position_embeddings / 2
            
            # 计算每个频率对应的波长
            wavelen = 2 * math.pi / inv_freq
            
            # 对高频部分进行插值
            inv_freq_mask = wavelen > low_freq_wavelen
            inv_freq = torch.where(
                inv_freq_mask,
                inv_freq / scaling_factor,  # 高频部分缩放
                inv_freq  # 低频部分保持
            )
        
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, num_heads, seq_len, head_dim]
            position_ids: [batch_size, seq_len]
        Returns:
            cos, sin: [batch_size, seq_len, head_dim]
        """
        # 计算位置编码：freqs = position_ids * inv_freq
        freqs = torch.einsum("bi,j->bij", position_ids.float(), self.inv_freq)
        # 拼接成完整维度：[batch, seq_len, dim]
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将输入的后半部分取负并与前半部分交换
    用于 RoPE 的旋转操作
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    应用 RoPE 到 Q 和 K
    
    Args:
        q, k: [batch, num_heads, seq_len, head_dim]
        cos, sin: [batch, seq_len, head_dim]
    """
    # 扩展 cos/sin 到 num_heads 维度
    cos = cos.unsqueeze(1)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1)
    
    # 应用旋转：q' = q * cos + rotate_half(q) * sin
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV heads 重复到 query heads 数量
    用于 Grouped Query Attention
    
    Args:
        hidden_states: [batch, num_kv_heads, seq_len, head_dim]
        n_rep: 重复次数（num_query_heads // num_kv_heads）
    Returns:
        [batch, num_query_heads, seq_len, head_dim]
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    
    # 重复 KV heads
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)

class QwenAttention(nn.Module):
    """
    Qwen 2.5/3 的 Attention 实现
    支持 RoPE、GQA、YaRN 扩展
    """
    def __init__(
        self,
        hidden_size: int = 3584,  # Qwen2.5-7B
        num_attention_heads: int = 28,
        num_key_value_heads: int = 4,  # Qwen2.5: 4, Qwen3: 2
        max_position_embeddings: int = 32768,
        rope_theta: float = 10000.0,
        rope_scaling: Optional[dict] = None,  # YaRN 配置
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = hidden_size // num_attention_heads
        
        # Q、K、V 投影
        self.q_proj = nn.Linear(hidden_size, num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_attention_heads * self.head_dim, hidden_size, bias=False)
        
        # RoPE 位置编码
        scaling_factor = 1.0
        yarn_alpha = 1.0
        if rope_scaling is not None:
            scaling_factor = rope_scaling.get("factor", 1.0)
            yarn_alpha = rope_scaling.get("alpha", 1.0)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
            scaling_factor=scaling_factor,
            yarn_alpha=yarn_alpha,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, seq_len, seq_len] 因果掩码
            position_ids: [batch_size, seq_len]
            past_key_value: (key_cache, value_cache) 用于推理加速
            use_cache: 是否返回 KV cache
        """
        batch_size, seq_len, _ = hidden_states.size()
        
        # 1. 线性投影得到 Q、K、V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # 重塑为多头形式：[batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # 2. 应用 RoPE 位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)
        
        cos, sin = self.rotary_emb(query_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # 3. 处理 KV cache（推理时使用）
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        past_key_value = (key_states, value_states) if use_cache else None
        
        # 4. Grouped Query Attention：重复 KV heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # 5. 计算注意力分数
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # 6. 应用因果掩码
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # 7. Softmax 归一化
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # 8. 加权求和
        attn_output = torch.matmul(attn_weights, value_states)
        
        # 9. 重塑并输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, past_key_value

# ============ 使用示例 ============

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """创建因果掩码（下三角）"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]

if __name__ == "__main__":
    # 配置参数（Qwen2.5-7B）
    batch_size = 2
    seq_len = 1024
    hidden_size = 3584
    num_heads = 28
    num_kv_heads = 4  # Qwen3-7B 改为 2
    
    # 初始化模型
    attention = QwenAttention(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        max_position_embeddings=32768,
        rope_theta=10000.0,
        rope_scaling={"factor": 4.0, "alpha": 1.0},  # YaRN 扩展到 128K
    )
    
    # 输入数据
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = create_causal_mask(seq_len, hidden_states.device)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    # 前向传播
    output, past_kv = attention(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    
    print(f"输入形状: {hidden_states.shape}")
    print(f"输出形状: {output.shape}")
    print(f"KV Cache 形状: {past_kv[0].shape}, {past_kv[1].shape}")
    
    # 计算 KV cache 显存占用（FP16）
    kv_cache_size = 2 * seq_len * num_kv_heads * (hidden_size // num_heads) * 2  # bytes
    print(f"KV Cache 显存: {kv_cache_size / 1024 / 1024:.2f} MB")
```

---

## 思考题

1. **为什么 Qwen3 选择 MoE 而不是继续扩大 Dense 模型？**
   - 提示：考虑训练成本、推理效率和模型容量的权衡

2. **如果将 Qwen3-7B 的 KV heads 从 2 降到 1（MQA），会有什么影响？**
   - 提示：从显存、性能和长上下文能力三个维度分析

3. **YaRN 为什么只缩放高频部分？低频部分代表什么信息？**
   - 提示：思考 RoPE 中不同频率对应的位置依赖范围

4. **Qwen3-MoE 的 Shared Expert 有什么作用？能否去掉？**
   - 提示：考虑专家坍缩和基础能力保证

5. **如何在有限显存下推理 128K 上下文的 Qwen2.5-7B？**
   - 提示：考虑 KV cache 量化、分块计算、offloading 等技术
