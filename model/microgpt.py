"""
200行实现一个微型GPT模型，包含数据准备、模型定义、训练循环和推理示例。
核心是一个简单的自动微分系统（Value类）和一个单层的Transformer
参考：https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95
"""

# datdasets 从url自动下载数据集
import math
import os
import random
random.seed(42)


if not os.path.exists('input.txt'):
    import urllib.request 
    names_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

# 读取数据集
docs = [ l.strip() for l in open('input.txt', 'r').read().split('\n') if l.strip()]
# 随即打乱
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Tokenizer 最简单的字符级分词
uchars = sorted(set(''.join(docs))) # 唯一字符合集
BOS = len(uchars) # 特殊标记，序列开始
vocab_size = len(uchars) + 1 #词表大小（字符数+1）
print(f"vocab size: {vocab_size}")
# 最终得到了一个包含 27 个字符的词汇表（26 个可能的小写字母，加上 BOS 标记的 1 个字符）

class Value:
    __slots__ = ('value', 'grad', '_children', '_local_grads') # 优化内存，指定固定属性
    # 初始化
    def __init__(self, data, children=(), local_grads=()):
        self.data =data     # 节点的实际数值
        self.grad = 0       # 梯度：损失对这个节点的导数（初始为0）
        self._children = children   # 这个节点依赖的子节点（比如a+b依赖a和b）
        self._local_grads = local_grads   # 这个节点对每个子节点的“局部导数”

    # 定义计算
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)  # 统一类型（比如2→Value(2)）
        return Value(self.data + other.data, (self, other), (1, 1))
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    
    def __pow__(self, other):
        return Value(self.data ** other, (self, ), other * self.data**(other - 1))
    
    def log(self):
        return Value(math.log(self.data), (self, ), (1/self.data, ))
    
    def exp(self):
        return Value(math.exp(self.data), (self, ), (math.exp(self.data), ))

    def relu(self):
        return Value(max(0, self.data), (self, ), (float(self.data > 0), ))
    
    def __neg__(self): return self * -1
    # 为了支持反向运算
    def __radd__(self, other): return self + other

    def __sub__(self, other): return self + (-other)

    def __rsub__(self, other): return other + (-self)

    def __rmul__(self,other): return self * other

    def __truediv__(self, other): return self * other**-1

    def __rtruediv__(self, other): return other *self**-1
    # 反向传播核心
    def backward(self):
        topo = []  # 拓扑排序，把所有依赖的节点按 “计算顺序” 排好
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children: build_topo(child)
                topo.append(v)
        build_topo(self)
        # 初始化根节点梯度（损失节点对自己的导数是1）
        self.grad = 1
        # 反向遍历，用链式法则算所有节点的梯度
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad # 子节点梯度 = 局部导数 × 父节点梯度

# 参数
n_embd = 16 # embedding dimension
n_head = 4  # number of attention heads
n_layer = 1 # number of layers
block_size = 16 # maximum sequence length
head_dim = n_embd // n_head # dimension of each head
# 矩阵初始化
# nout: 输出维度(矩阵行），nin: 输入维度(矩阵列)，std: 标准差（控制随机数的分布范围）
matrix = lambda nout, nin, std =0.08:[[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
# 初始化核心权重矩阵
# wte: token embedding matrix，wpe: position embedding matrix，lm_head: 语言模型头（输出层权重矩阵）
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    state_dict[f'layer_{i}.attn_wq'] = matrix(n_embd, n_embd) # q
    state_dict[f'layer_{i}.attn_wk'] = matrix(n_embd, n_embd) # k
    state_dict[f'layer_{i}.attn_wv'] = matrix(n_embd, n_embd) # v
    state_dict[f'layer_{i}.attn_wo'] = matrix(n_embd, n_embd) # 注意力输出整合矩阵
    state_dict[f'layer_{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd) # MLP第一层（升维）
    state_dict[f'layer_{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd) # MLP第二层（降维）

params = [p for mat in state_dict.values() for row in mat for p in row] # 把所有二维矩阵Value对象扁平化为一维列表
print(f"num_params: {len(params)}")

# 框架
# Define the model architecture: a stateless function mapping token sequence and parameters to logits over what comes next.
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
# 线性变换
def linear(x, w): #args: x: 输入向量，w: 权重矩阵
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

# 归一化
def softmax(logits):
    max_val = max(val.data for val in logits) # 取logits的最大值
    exps = [(val -max_val).exp() for val in logits]  # 每个logit减最大值后做指数运算
    total =sum(exps) # 所有指数的和
    return [e / total for e  in exps] # 归一化，总和为1（概率）

# 归一化层
def rmsnorm(x):
    ms =sum(xi * xi for xi in x) / len(x) # 计算均方值（mean square）
    scale = (ms + 1e-5) ** -0.5  # 计算缩放因子（1/√(均方值+小常数)）
    return [xi * scale for xi in x]  # 每个元素乘以缩放因子

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id] # 词嵌入：根据token_id取wte矩阵的对应行（16维）
    pos_emb = state_dict['wpe'][pos_id] # 位置嵌入：根据pos_id取wpe矩阵的对应行（16维）
    x = [t + p for t, p in zip(tok_emb,pos_emb)] # 合并：词向量+位置向量（保留顺序信息）
    x = rmsnorm(x) # 归一化，稳定初始向量

    for li in range(n_layer): # 遍历每一层
        # 1) Multi-Head Attention block
        x_residual = x # 保存残差：后续做残差连接
        x = rmsnorm(x) # 注意力层前的归一化
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        # 此处keys, values是传入的参数，会缓存所有历史tokens
        keys[li].append(k) # 缓存K：记录当前token的K，供后续token计算注意力用
        values[li].append(v) # 缓存V：记录当前token的V，供后续token计算注意力用
        x_attn = [] # 保存所有注意力头输出
        for h in range(n_head): # 遍历每个注意力头（n_head=4），并行计算注意力
            hs =h *head_dim # 每个头的起始位置（0,4,8,12
            q_h = q[hs : hs + head_dim] # 数组拆分：当前头的Q（4维）
            k_h = [ki[hs : hs +head_dim] for ki in keys[li]] # 拆分：所有历史token的K（每个4维）
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            
            # 计算注意力分数：Q和K的点积 ÷ √head_dim（缩放，防止分数太大）
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits) # 转成注意力权重（概率）：softmax后，权重和为1
            # 加权求和V：用注意力权重对所有V加权，得到当前头的输出（4维）
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out) # 拼接所有头的输出（4×4=16维）

        # 注意力输出整合：线性变换后，做残差连接
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a,b in zip(x, x_residual)]

        # 2) MLP block
        x_residual = x # 残差
        x = rmsnorm(x) # MLP层前的归一化
        x = linear(x, state_dict[f'layer{li}.mlp_fc1']) # 第一步：升维（16维→64维）
        x = [xi.relu() for xi in x] # ReLU activation
        x = linear(x, state_dict[f'layer{li}.mlp_fc2']) # 第二步：降维（64维→16维）
        x = [a + b for a, b in zip(x, x_residual)] # 残差连接

    logits = linear(x, state_dict['lm_head']) # 第三步：输出层（把16维向量转成词汇表大小的logits）
    return logits

# Training
# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)

num_steps = 1000  # epoch
for step in range(num_steps):

    # ------子模块1：数据准备-------
    doc =docs[step % len(docs)] # 1. 循环取文档：step%len(docs)保证循环用所有文档，不重复
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS] # 2. 分词：把文档转成token ID（BOS是特殊token
    n = min(block_size, len(tokens) - 1) # 3. 限制长度：不超过模型的最大序列长度block_size


    # ------ 子模块2：前向计算（让模型做题，算错误程度） ------
    keys,values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)] # 初始化每层的KV缓存
    losses = [] # 保存每个位置的损失
    for pos_id in range(n):
        token_id, target_id =tokens[pos_id], tokens[pos_id + 1] # 当前token ID（输入）和目标token ID (预测下一个)
        logits = gpt(token_id, pos_id, keys, values)  # 调用GPT前向函数，得到所有token的预测分数
        probs = softmax(logits) 
        loss_t = -probs[target_id].log() # 计算该位置的损失：负对数似然（预测越准，loss越小）
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # 整个文档的平均损失

    # ----- 子模块3：反向传播（算参数该怎么调整） -----
    loss.backward() # 调用Value类的backward，自动算所有params的grad

    # ----- 子模块4：Adam优化器更新参数（按梯度微调） -----
    lr_t = learning_rate * (1 -step / num_steps) # 学习率线性衰减：越训练到后期，步长越小，调整越精细
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad # 1. 更新一阶矩（m）：历史梯度的加权平均
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2 # 2. 更新二阶矩（v）：历史梯度平方的加权平均
        m_hat = m[i] / (1 - beta1 ** (step +1))
        v_hat = v[i] / (1 - beta2 ** (step +1)) # 3. 偏差修正（Adam的关键：初期m/v偏小，修正后更准确）
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam) # 4. 核心：更新参数的data值（往梯度反方向调，减小损失）
        p.grad = 0 # 5. 重置梯度（必须！否则下一轮梯度会累加）
    
    print(f"step {step + 1 :4d} / {num_steps : 4d} | loss: {loss.data: .4f}")


# Inference
temperature = 0.5
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20): # 20个例子
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([1 / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights = [p.data for p in probs])[0]
        if token_id == BOS: break
        sample.append(uchars[token_id])

    print(f"sample {sample_idx + 1 :2d}: {''.join(sample)}")
