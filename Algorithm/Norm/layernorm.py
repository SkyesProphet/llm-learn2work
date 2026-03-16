"""
    层归一化 (Layer Normalization)
    与 Batch Normalization 不同，层归一化是在每个样本的特征维度上进行归一化，
    而不是在batch维度上进行归一化。适用于处理变长序列或小批量数据。
"""

import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    """
    归一化层：指的是对每一个样本的特征，进行均值为0，方差为1的归一化处理。
    LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta  # 标准化后 + 缩放平移
    """
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.gamma =nn.Parameter(torch.ones(d_model)) # 可学习的缩放参数，初始值为1
        self.beta = nn.Parameter(torch.zeros(d_model)) # 可学习的平移参数，初始值为0

    def forward(self, x):
        """ ---前向传播---
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]

        Returns: 
            归一化后的张量 [batch_size, seq_len, d_model]
        """
        # 计算特征维度的均值 [batch_size, seq_len, 1]
        mean = x.mean(dim = -1, keepdim = True)
        
        # 计算特征维度的方差 [batch_size, seq_len, 1]
        # torch.var 默认 unbiased = True (除以 N-1)，但在LayerNorm中通常使用无偏估计（除以 N），因此设置 unbiased = False
        var = x.var(dim = -1, keepdim = True, unbiased = False) 

        # 标准化： 0均值、单位方差
        x_norm = (x - mean ) / torch.sqrt(var + self.eps) 

        # 缩放和平移
        return x_norm * self.gamma + self.beta