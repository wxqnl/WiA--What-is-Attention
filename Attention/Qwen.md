# Qwen

## 对应论文
- `paper/Qwen-Qwen2.5-Technical-Report.pdf`
- Qwen Team, 2024

## Attention 算法思路
Qwen 系列在公开技术报告里延续了 `decoder-only causal self-attention` 主干，但在工程化和长上下文能力上做了更完整的增强。  
基础流程仍然是：
- 对输入做线性投影得到 `Q、K、V`
- 在 `Q/K` 上施加 `RoPE`
- 使用因果掩码做自回归注意力

在代表性的 Qwen2.5 设计里，注意力层通常会结合：
- `RoPE`：让位置编码自然进入点积计算
- `GQA`：多个 query head 共享较少的 KV head，降低推理显存
- 长上下文扩展策略：在不改主干结构的情况下把上下文窗口拉长

## 核心改进
- 把绝对位置编码换成更适合长上下文外推的 `RoPE`。
- 用 `GQA` 减少 KV cache 的体积，加快推理。
- 在长上下文场景里保留标准 Transformer 的表达能力，同时控制成本。

## 简化伪代码
```python
def qwen_attention(X):
    Q = apply_rope(X @ W_Q)
    K = apply_rope(X @ W_K)
    V = X @ W_V

    K, V = repeat_kv_to_query_groups(K, V, num_query_heads, num_kv_heads)
    scores = (Q @ K.T) / sqrt(d_k)
    scores = scores + causal_mask(len(X))

    weights = softmax(scores, dim=-1)
    return weights @ V
```
