"""
多头注意力：Muti-Head Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """多头注意力模块
    支持 self-attention and cross-attention
    - self-attention: q, k, v = x, x, x
    - cross-attention: q = x, k, v = memory
    """
    def __init__(self, d_model, num_heads, dropout = 0.1): 
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads" # 确保 d_model 可以被 num_heads 整除

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # 每个头被分到的维度

        # 定义线性层，将输入映射到 q, k, v
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model) # 输出线性层
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """ ---前向传播---

        Args:
            q: 查询张量，形状为 (batch_size, seq_len, d_model)
            k: 键张量，形状为 (batch_size, mem_seq_len, d_model)
            v: 值张量，形状为 (batch_size, mem_seq_len, d_model)
            mask: 可选的掩码张量，形状为 (batch_size, seq_len, mem_seq_len)

        Returns:
            output: 输出张量，形状为 (batch_size, seq_len, d_model) 与 x 一致
        """
        batch_size, seq_len, _ = q.size()

        # QKV线性变换并分头
        # （batch_size, seq_len, d_model) = (batch_size, seq_len, num_heads * head_dim)
        # -> (batch_size, seq_len, num_heads，head_dim) 
        # -> (batch_size, num_heads, seq_len, head_dim) 
        q = self.w_q(q).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        k = self.w_k(k).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)
        v = self.w_v(v).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,2)

        # 点积注意力分数 
        atten = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim) # (batch_size, num_heads, seq_len, mem_seq_len)

        # 注意力掩码
        if mask is not None:
            atten = atten.masked_fill(mask == 0, float('-inf')) # -1e9

        # 注意力权重
        atten = F.softmax(atten, dim = -1) # (batch_size, num_heads, seq_len, mem_seq_len)
        atten = self.dropout(atten)

        # 加权求和
        output = torch.matmul(atten, v) # (batch_size, num_heads, seq_len, head_dim)

        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # (batch_size, seq_len, d_model)

        output = self.w_o(output)
        
        return output