# GLA

## 对应论文
- `paper/GLA-Gated-Linear-Attention.pdf`
- Yang et al., 2023/2024

## Attention 算法思路
GLA 的核心是 `linear attention + data-dependent gates`。  
普通线性注意力把 `softmax(QK^T)V` 改写为可递推的状态更新，但表达能力通常弱于 softmax attention。  
GLA 在这个基础上加入门控，让模型能动态决定：
- 历史信息保留多少
- 当前信息写入多少

因此它既能像 RNN 一样维护状态，又比朴素线性注意力更灵活。

## 核心改进
- 把二次复杂度注意力改成线性时间推理。
- 用门控提升线性注意力的表达能力和长度泛化能力。
- 配合硬件友好的 `FlashLinearAttention` 训练实现，提高吞吐。

## 简化伪代码
```python
def gla_step(x_t, state):
    q_t = phi(x_t @ W_Q)
    k_t = phi(x_t @ W_K)
    v_t = x_t @ W_V
    g_t = sigmoid(x_t @ W_G)

    state = g_t * state + outer(k_t, v_t)
    y_t = q_t @ state
    return y_t, state
```
