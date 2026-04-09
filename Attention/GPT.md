# GPT

## 对应论文
- `paper/GPT-Improving-Language-Understanding-by-Generative-Pre-Training.pdf`
- Radford et al., 2018

## Attention 算法思路
GPT 沿用了 Transformer 的注意力计算，但只保留 `decoder`，并引入 `causal mask`。  
这意味着第 `t` 个 token 只能看到 `1..t` 的历史位置，不能偷看未来 token。

因此，GPT 的注意力本质上是 `masked self-attention`：
- 仍然计算 `Q、K、V`
- 仍然使用 `softmax(QK^T / sqrt(d_k))`
- 但会把未来位置的分数强行设成 `-inf`

## 核心改进
- 从编码器-解码器结构简化为纯 decoder 结构，更适合自回归生成。
- 用因果掩码把注意力机制和语言建模目标对齐。
- 统一“预训练 + 下游迁移”的范式。

## 简化伪代码
```python
def masked_self_attention(X):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = (Q @ K.T) / sqrt(d_k)
    mask = upper_triangular_mask(len(X))  # future positions = -inf
    scores = scores + mask

    weights = softmax(scores, dim=-1)
    return weights @ V
```
