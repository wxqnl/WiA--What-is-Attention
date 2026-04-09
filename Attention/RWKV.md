# RWKV

## 对应论文
- `paper/RWKV-Reinventing-RNNs-for-the-Transformer-Era.pdf`
- Peng et al., 2023

## Attention 算法思路
RWKV 不是标准的 softmax attention。  
它试图保留 Transformer 训练时的并行性，同时在推理时退化成类似 RNN 的递推更新。

RWKV 的核心可以理解为：
- 用 `time-mix` 混合当前 token 和历史状态
- 用指数衰减的方式维护历史信息
- 用 `receptance` 门控决定当前 token 应该读取多少历史内容

因此它不需要保存完整的 KV cache，而是维护一个递推状态。

## 核心改进
- 把注意力读历史的过程改写成递推形式，推理成本近似线性。
- 保留训练时的并行实现，不完全回到传统 RNN。
- 在超长序列推理时，状态体积通常比标准注意力缓存更小。

## 简化伪代码
```python
def rwkv_step(x_t, state):
    k_t = x_t @ W_K
    v_t = x_t @ W_V
    r_t = sigmoid(x_t @ W_R)

    state.num = decay * state.num + exp(k_t) * v_t
    state.den = decay * state.den + exp(k_t)
    wkv_t = state.num / state.den

    y_t = (r_t * wkv_t) @ W_O
    return y_t, state
```
