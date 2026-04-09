# NSA

## 对应论文
- `paper/NSA-Native-Sparse-Attention.pdf`
- Native Sparse Attention, 2025

## Attention 算法思路
NSA 的目标不是做“近似全注意力”，而是把稀疏注意力直接做成可训练、可部署、硬件友好的原生结构。  
它通常把注意力拆成三个互补分支：
- `local window`：保留局部细节
- `compressed tokens`：对远距离上下文做压缩摘要
- `selected blocks`：动态挑选最重要的远距离块做精细注意力

可以把它理解成“局部精看 + 远处先压缩看 + 再挑重点细看”。

## 核心改进
- 稀疏模式不是后处理近似，而是从训练阶段就原生存在。
- 用块级稀疏和压缩表示对齐 GPU/加速器的访存模式。
- 同时保留局部精度和远距离全局感知，适合超长上下文。

## 简化伪代码
```python
def native_sparse_attention(X):
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    local_out = local_window_attention(Q, K, V, window_size)
    Kc, Vc = compress_blocks(K, V, block_size)
    top_blocks = select_topk_blocks(Q, Kc, topk)
    sparse_out = attend_selected_blocks(Q, K, V, top_blocks)

    return fuse(local_out, sparse_out)
```
