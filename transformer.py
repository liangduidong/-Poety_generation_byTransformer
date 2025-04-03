import torch
import math
import torch.nn as nn

# 参数为句子长度
def get_subsequent_mask(size):
    # 1为batch维度
    mask_shape = (1, size, size)
    return 1 - torch.tril(torch.ones(mask_shape)).byte()


def get_padding_mask(x, padding_idx):
    # 扩展Q维度
    return (x == padding_idx).unsqueeze(1).byte()

# 此嵌入层
class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x)*math.sqrt(self.d_model)

# 位置编码
import numpy as np
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        """生成固定的 Positional Encoding"""
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # 不需要计算梯度
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# 注意力机制
def attention(Q, K, V, mask = None, dropout=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:   # mask = [32, 8, 50, 50] , scores = [32, 8, 49, 50]
        scores = scores.masked_fill(mask == 1, -float("inf"))
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, V), p_attn

# 多头注意力机制
class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.n_head = n_head

        # 三个线性变换，一个多头拼接后的线性变换
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.linear = nn.Linear(d_model, d_model, bias=False)
    def forward(self, Q, K, V, mask=None):
        residual = Q
        # 分头
        batch_size = Q.size(0)
        Q = self.W_Q(Q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.W_Q(V).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        # 计算注意力
        if mask is not None:
            mask = mask.unsqueeze(1)
        context, attn = attention(Q, K, V, mask, self.dropout)
        # 拼接
        concat = context.transpose(1, 2).reshape(batch_size, -1, self.n_head*self.d_k)
        output = self.linear(concat)
        # print(output + residual)
        return self.norm(output + residual)
# 前馈神经网络
class FeedforwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        x = self.relu(self.w1(x))
        x = self.dropout(self.w2(x))
        return self.norm(x + residual)

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadedAttention(d_model, n_head, dropout)
        self.fnn = FeedforwardNet(d_model, d_ff, dropout)
    def forward(self, x, mask=None):
        output = self.mha(x, x, x, mask)
        return self.fnn(output)
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
        super().__init__()
        self.emb = Embeddings(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList(  # 堆叠6次
            [EncoderLayer(d_model, n_head, d_ff, dropout) for i in range(N)]
        )
    def forward(self, x, mask=None):
        x = self.emb(x)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

# 生成器
class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.linear(x)


# 模型封装
class Transformer(nn.Module):
    def __init__(self, vocab_size,  d_model, n_head, d_ff, N=6, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_head, d_ff, N, dropout)
        self.generator = Generator(d_model, vocab_size)

    def forward(self, src_x, src_mask=None):
        output = self.encoder(src_x, src_mask)  # 掩码遮住
        return self.generator(output)

    def predict(self, src_x, src_mask=None):
        output = self.encoder(src_x, src_mask)
        return self.generator(output[:, -1, :])     # [BATCH,1,vocab_size]

    def generate(self, start_tokens, max_length=50, temperature=1.0):
        """
        生成诗歌
        :param start_tokens: 初始输入 token（List of int）
        :param max_length: 生成的最大长度
        :param temperature: 控制生成的多样性
        :return: 生成的 token 序列
        """
        self.eval()
        generated = start_tokens[:]
        input_tensor = torch.tensor([start_tokens], dtype=torch.long)  # [1, seq_len]

        with torch.no_grad():
            for _ in range(max_length):
                mask = get_padding_mask(input_tensor, padding_idx=0)  # 生成 padding mask
                logits = self.forward(input_tensor, mask)  # 计算 Transformer 输出
                logits = logits[:, -1, :] / temperature  # 取最后一个 token 的预测
                probs = torch.softmax(logits, dim=-1)  # 转换为概率分布
                next_token = torch.multinomial(probs, num_samples=1).item()  # 采样下一个 token

                generated.append(next_token)
                input_tensor = torch.tensor([generated], dtype=torch.long)  # 更新输入

                if next_token == 2:  # 2 代表 [END] token
                    break

        return generated  # 返回生成的 token 序列

def make_model(src_vocab_size, d_model, n_head, d_ff, N=6, dropout=0.1):
    model = Transformer(src_vocab_size,  d_model, n_head, d_ff, N, dropout)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

if __name__ == '__main__':
    src_vocab_size = 1000
    d_model = 512
    n_head = 8
    d_ff = 2048
    N = 6
    dropout = 0.1

    model = make_model(src_vocab_size, d_model, n_head, d_ff, N, dropout)
    # print(model)

    # 输入数据
    src_inputs = torch.tensor([
        [1, 2, 3],
        [4, 5, 0],
    ])
    src_mask = get_padding_mask(src_inputs, 0)

    tgt_inputs = torch.tensor([
        [1, 2, 3, 4],
        [4, 5, 0, 0],
    ])
    # 处理mask
    tgt_pad_mask = get_padding_mask(tgt_inputs, 0)
    subsequent_mask = get_subsequent_mask(4)
    tgt_mask = tgt_pad_mask | subsequent_mask
    print(subsequent_mask)

    predict = model(src_inputs, src_mask)
    print(predict.shape)


