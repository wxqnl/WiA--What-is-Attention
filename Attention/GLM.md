# GLM

## 对应论文
- `paper/GLM-GLM-130B.pdf`
- Zeng et al., 2022

## Attention 算法思路
GLM 不是简单做纯左到右生成，而是把 `autoregressive blank infilling` 做成统一训练目标。  
它的关键不是换掉注意力公式，而是设计一种更灵活的注意力 mask：
- 已知上下文部分可以双向可见
- 待生成 span 内部保持自回归顺序
- 模型通过空白片段补全来兼顾理解和生成

GLM 还引入了 `2D positional encoding`，把“token 在原句中的位置”和“token 在待生成块中的位置”同时编码进去。

## 核心改进
- 不再只做单向语言建模，而是把理解和生成统一到同一套框架。
- 用特殊 attention mask 支持 blank infilling。
- 用二维位置编码显式区分“原始位置”和“块内生成位置”。

## 简化伪代码
```python
def glm_attention(X, prefix_mask, generation_mask):
    Q = add_2d_position(X) @ W_Q
    K = add_2d_position(X) @ W_K
    V = X @ W_V

    scores = (Q @ K.T) / sqrt(d_k)
    scores = scores + build_glm_mask(prefix_mask, generation_mask)

    weights = softmax(scores, dim=-1)
    return weights @ V
```
