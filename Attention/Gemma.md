# Gemma

## 对应论文
- `paper/Gemma-Gemma-2-Technical-Report.pdf`
- Gemma Team, 2024

## Attention 算法思路
Gemma 2 的注意力设计重点是 `global + local` 交替堆叠：
- 一部分层使用全局注意力
- 一部分层使用局部滑动窗口注意力

这样做的目标是兼顾：
- 全局信息整合能力
- 长序列下的计算效率

Gemma 2 还使用了 `GQA`，进一步减少推理时的 KV 存储和访存成本。

## 核心改进
- 用局部层承接高频、近邻依赖。
- 用全局层周期性做全局信息汇聚。
- 相比所有层都做 full attention，显著降低长上下文代价。

## 简化伪代码
```python
def gemma2_block(X, layer_id):
    if is_global_layer(layer_id):
        return full_attention(X)
    return sliding_window_attention(X, window_size=4096)


def gemma2_stack(X, num_layers):
    for layer_id in range(num_layers):
        X = gemma2_block(X, layer_id)
    return X
```
