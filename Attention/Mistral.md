# Mistral

## 对应论文
- `paper/Mistral-Mistral-7B.pdf`
- Jiang et al., 2023

## Attention 算法思路
Mistral 7B 的核心 attention 改进是：
- `Sliding Window Attention (SWA)`
- `Grouped-Query Attention (GQA)`

SWA 的做法是：每个 token 不再看全序列，而只看最近一个窗口内的 token。  
单层虽然是局部感受野，但多层堆叠后，信息仍然可以逐层传播到更远处。

## 核心改进
- 把全局注意力的 `O(n^2)` 成本改成近似 `O(n * w)`，`w` 是窗口大小。
- 用 `GQA` 降低 KV cache 的大小和推理延迟。
- 配合滚动缓存机制，支持更长序列的流式推理。

## 简化伪代码
```python
def sliding_window_attention(X, window_size):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = (Q @ K.T) / sqrt(d_k)
    scores = scores + local_causal_mask(len(X), window_size)

    weights = softmax(scores, dim=-1)
    return weights @ V
```
