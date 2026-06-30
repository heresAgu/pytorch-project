# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import numpy as np


# 构建RMS归一化的类
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        # dim: 归一化的特征维度
        # eps: 防止除零的小常数
        super(RMSNorm, self).__init__()
        self.eps = eps
        # 可学习的缩放参数
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: 输入张量 (..., dim)
        # 计算均方根（RMS Norm 是简化版的 Layer Norm，不需要均值和标准差）
        rms = torch.sqrt(torch.mean(x.float().pow(2), dim=-1, keepdim=True) + self.eps)
        x_normed = x.float() / rms
        # 恢复原始数据类型
        return self.weight * x_normed.to(x.dtype)


# 预计算旋转位置编码的cos和sin值
def precompute_freqs_cis(dim, max_seq_len, theta=10000.0):
    # dim: 注意力头的维度（head_dim）
    # max_seq_len: 最大序列长度
    # theta: 旋转频率基数
    # 计算每个维度的旋转频率 theta_k = 10000^(-2k/d)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # torch.arange(0, dim, 2)生成 [0, 2, 4, ..., dim-2]，共 dim//2个元素
    # 这里每个元素代表公式中的 2k（k 从0开始）

    # 计算每个位置的角度 m*theta_k
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)                 # (max_seq_len, dim//2)

    # 生成cos和sin值
    freqs_cos = torch.cos(freqs)                  # (max_seq_len, dim//2)
    freqs_sin = torch.sin(freqs)                  # (max_seq_len, dim//2)

    # 将cos和sin拼接成完整维度
    freqs_cos = torch.cat([freqs_cos, freqs_cos], dim=-1)  # (max_seq_len, dim)
    freqs_sin = torch.cat([freqs_sin, freqs_sin], dim=-1)  # (max_seq_len, dim)
    return freqs_cos, freqs_sin


# 旋转一半维度的辅助函数
def rotate_half(x):
    # x: (..., dim)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


# 应用旋转位置编码（支持start_pos偏移，用于KV Cache生成阶段）
def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin, start_pos=0):
    # xq: (batch_size, seq_len, n_heads, head_dim) 查询张量
    # xk: (batch_size, seq_len, n_kv_heads, head_dim) 键张量
    # freqs_cos, freqs_sin: (max_seq_len, head_dim) 预计算的cos/sin值
    # start_pos: 当前序列的起始位置偏移（生成阶段>0）
    # RoPE公式: x' = x*cos(m*theta) + rotate_half(x)*sin(m*theta)
    seq_len = xq.shape[1]
    # 根据start_pos偏移取对应位置的编码（Cache模式下只有1个新token）
    freqs_cos = freqs_cos[start_pos:start_pos+seq_len].unsqueeze(0).unsqueeze(2)
    freqs_sin = freqs_sin[start_pos:start_pos+seq_len].unsqueeze(0).unsqueeze(2)
    xq_out = xq * freqs_cos + rotate_half(xq) * freqs_sin
    xk_out = xk * freqs_cos + rotate_half(xk) * freqs_sin
    return xq_out, xk_out


# 构建因果掩码张量的函数
def causal_mask(seq_len, device='cpu', dtype=torch.float32):
    # seq_len: 序列长度
    # 生成上三角掩码，每个token只能看到自己和之前的token
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype), diagonal=1)
    return mask


# 实现克隆函数
def clones(module, N):
    # module: 要克隆的目标网络层
    # N: 克隆N个
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# 构建SwiGLU前馈全连接层的类
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, multiple_of=256):
        # d_model: 词嵌入维度
        # d_ff: 隐藏层维度
        # dropout: Dropout置零比率
        # multiple_of: 确保维度是multiple_of的倍数，优化计算
        super(FeedForward, self).__init__()

        # 将d_ff对齐到multiple_of的倍数  d_ff = hidden_dim 一般是4倍
        d_ff = ((d_ff + multiple_of - 1) // multiple_of) * multiple_of

        # SwiGLU需要三个权重矩阵
        self.w1 = nn.Linear(d_model, d_ff, bias=False)   # 门控线性层
        self.w2 = nn.Linear(d_ff, d_model, bias=False)   # 输出线性层
        self.w3 = nn.Linear(d_model, d_ff, bias=False)   # 值线性层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: 输入张量 (batch_size, seq_len, d_model)
        # SwiGLU公式: output = w2(silu(x@w1) * (x@w3))
        # 其中 silu(x) = x*sigma(x)，*表示逐元素相乘  silu就是Swish函数
        # 大致过程:
        # x是词向量 d_model比如512 x=(1,512)
        # w1是(512,2048),w3是(512,2048),w2是(2048,512)
        # x先通过w1,w3从512维升维到2048维
        # silu(x@w1)和(x@w3)逐元素相乘，相当于"门控"机制——一个控制信息量，一个控制信息本身
        # 最后再通过w2降维回512维
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


# 构建分组查询多头注意力机制的类（支持KV Cache）
class GroupedMultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, dropout=0.1):
        # d_model: 词嵌入维度
        # n_heads: 查询头的数量
        # n_kv_heads: 键值头的数量（GQA中，K和V共享相同的头数）
        # dropout: Dropout置零比率
        super(GroupedMultiQueryAttention, self).__init__()

        # 要确认一个事实：多头的数量需要整除词嵌入的维度
        # 每个多头只是分析一个词向量的一个部分，每个头长度要相同
        assert d_model % n_heads == 0

        self.n_heads = n_heads
        # GQA: KV头的数量默认为与Q头相同（即MHA），但可以更少
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        # 每个KV头对应多少个查询头
        self.n_rep = self.n_heads // self.n_kv_heads
        # 每个头的维度
        self.head_dim = d_model // n_heads

        # 定义Q、K、V的线性变换层（无偏置）
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        # 输出线性层
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, start_pos=0, kv_cache=None):
        # x: 输入张量 (batch_size, seq_len, d_model)
        # freqs_cos, freqs_sin: 预计算的RoPE编码
        # mask: 注意力掩码（因果掩码）
        # start_pos: 当前token在原始序列中的位置（用于RoPE，Cache模式>0）
        # kv_cache: (k_cache, v_cache) 来自之前token的K、V张量
        # 返回: (output, new_kv_cache)
        batch_size, seq_len, _ = x.shape

        # 1. 通过线性变换得到Q、K、V
        xq = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # 2. 对Q和K应用旋转位置编码（使用start_pos偏移）
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin, start_pos)

        # 3. KV Cache: 将新的K、V拼接到缓存后面
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            xk = torch.cat([k_cache, xk], dim=1)
            xv = torch.cat([v_cache, xv], dim=1)

        # 返回更新后的KV Cache（detach防止梯度传播到缓存）
        new_kv_cache = (xk.detach(), xv.detach())

        # 4. GQA: 重复KV头以匹配Q头的数量
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # 5. 转置为(batch, n_heads, seq_len, head_dim)做批量矩阵乘法
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 6. 计算注意力分数  scores = Q@K^T / sqrt(d_k)
        scores = torch.matmul(xq, xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 7. 应用掩码
        if mask is not None:
            scores = scores + mask

        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)

        # 8. 注意力加权求和
        output = torch.matmul(p_attn, xv)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.wo(output), new_kv_cache


# 构建Transformer解码器层的类（支持KV Cache）
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, dropout, multiple_of):
        # d_model: 词嵌入维度
        # n_heads: 查询头数
        # n_kv_heads: KV头数（GQA分组数）
        # d_ff: 前馈网络维度
        # dropout: Dropout置零比率
        # multiple_of: 前馈网络维度对齐基数
        super(TransformerBlock, self).__init__()

        # 分组查询多头注意力层（含RoPE + GQA + KV Cache）
        self.attention = GroupedMultiQueryAttention(d_model, n_heads, n_kv_heads, dropout)
        # SwiGLU前馈全连接层
        self.feed_forward = FeedForward(d_model, d_ff, dropout, multiple_of)

        # LLaMA采用预归一化（Pre-normalization），即在子层之前进行归一化
        # 与原始Transformer的Post-normalization不同
        self.attention_norm = RMSNorm(d_model)
        self.ffn_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, freqs_cos, freqs_sin, mask=None, start_pos=0, kv_cache=None):
        # x: 输入张量 (batch_size, seq_len, d_model)
        # freqs_cos, freqs_sin: 旋转位置编码
        # mask: 因果掩码
        # start_pos: 当前起始位置（用于KV Cache）
        # kv_cache: 当前层的KV缓存

        # 第一个子层: 预归一化 + 分组查询注意力 + 残差连接
        attn_out, new_kv_cache = self.attention(
            self.attention_norm(x), freqs_cos, freqs_sin,
            mask, start_pos, kv_cache
        )
        h = x + self.dropout(attn_out)

        # 第二个子层: 预归一化 + SwiGLU前馈全连接 + 残差连接
        out = h + self.dropout(self.feed_forward(self.ffn_norm(h)))
        return out, new_kv_cache


# 构建完整的LLaMA模型类（支持KV Cache）
class LLaMA(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, n_kv_heads,
                 d_ff, max_seq_len, dropout, multiple_of=256, rope_theta=10000.0):
        # vocab_size: 词表大小
        # d_model: 词嵌入维度
        # n_layers: 解码器堆叠的层数
        # n_heads: 注意力头数
        # n_kv_heads: KV头数（GQA）
        # d_ff: 前馈全连接层维度
        # max_seq_len: 最大序列长度
        # dropout: Dropout置零比率
        # multiple_of: 前馈网络维度对齐基数
        super(LLaMA, self).__init__()

        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        # 词嵌入层（Token Embedding）
        self.tok_embedding = nn.Embedding(vocab_size, d_model)

        # 使用clones函数克隆N个Transformer解码器层
        layer = TransformerBlock(d_model, n_heads, n_kv_heads, d_ff, dropout, multiple_of)
        self.layers = clones(layer, n_layers)

        # 最终的RMS归一化层
        self.norm = RMSNorm(d_model)

        # 输出线性层（将隐藏状态映射到词表大小）
        self.output = nn.Linear(d_model, vocab_size, bias=False)

        # 预计算RoPE的cos和sin值（预留两倍最大长度以支持KV Cache扩展）
        head_dim = d_model // n_heads
        freqs_cos, freqs_sin = precompute_freqs_cis(head_dim, max_seq_len * 2, rope_theta)
        self.rope_theta = rope_theta
        self.register_buffer('freqs_cos', freqs_cos)
        self.register_buffer('freqs_sin', freqs_sin)

    def forward(self, x, mask=None, start_pos=0, kv_caches=None):
        # x: 输入token IDs (batch_size, seq_len)
        # mask: 可选的注意力掩码
        # start_pos: 当前输入在完整序列中的起始位置（用于KV Cache逐token生成）
        # kv_caches: 每层对应的KV缓存列表，None或[(k_cache0, v_cache0), ...]
        # 返回: (logits, new_kv_caches)
        batch_size, seq_len = x.shape

        # 1. Token嵌入并缩放（与Transformer原论文不一致）
        # 注意：这里LLaMA没有采用和原本Transformer一致的获取张量后乘上sqrt(d_model) 因为直接用了RoPE，保留了长度
        h = self.tok_embedding(x)

        # 传入完整RoPE，让apply_rotary_emb根据start_pos自行切片
        freqs_cos = self.freqs_cos
        freqs_sin = self.freqs_sin

        # 生成因果掩码（仅在Prefill阶段且未提供掩码时）
        # Cache模式（kv_caches不为None）或单token输入时不生成掩码
        if mask is None and kv_caches is None and seq_len > 1:
            mask = causal_mask(seq_len, device=x.device, dtype=self.tok_embedding.weight.dtype)

        # 2. 逐层通过所有解码器层
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            h, new_kv_cache = layer(h, freqs_cos, freqs_sin, mask, start_pos, kv_cache)
            new_kv_caches.append(new_kv_cache)

        # 3. 最终归一化
        h = self.norm(h)

        # 4. 输出层（logits）
        logits = self.output(h)
        return logits, new_kv_caches

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):
        # idx: 初始token序列 (batch_size, seq_len)
        # max_new_tokens: 最大生成长度
        # temperature: 采样温度（>1更随机，<1更确定）
        # top_k: 只从概率最高的top_k个token中采样
        #
        # === 原理说明 ===
        # 不使用KV Cache时，每一步都要把全部序列重新过一遍模型，复杂度O(t^2)
        # 使用KV Cache时，只需要做两步：
        # Step 1 - Prefill: 用完整prompt过一次模型，得到所有层的KV Cache和第一个输出logit
        # Step 2 - Decode: 每次只计算最新1个token的K、V，拼接到Cache上，Q只与Cache做注意力，复杂度O(t)
        # === 实现 ===

        # ---------- Step 1: Prefill ----------
        # 用完整prompt初始化KV Cache
        logits, kv_caches = self(idx, start_pos=0, kv_caches=None)
        # 取最后一个位置的logits
        logits = logits[:, -1, :] / temperature

        # top-k采样
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, -1:]] = -float('Inf')

        # 采样第一个token
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

        # ---------- Step 2: Decode（逐token生成，使用KV Cache）----------
        for cur_pos in range(idx.shape[1] - 1, idx.shape[1] - 1 + max_new_tokens - 1):
            # 只取最新生成的1个token
            x = idx_next

            # 前向传播，传入KV Cache和当前位置
            logits, kv_caches = self(x, start_pos=cur_pos, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature

            # top-k采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def make_model(vocab_size=32000, d_model=4096, n_layers=32, n_heads=32,
               n_kv_heads=8, d_ff=11008, max_seq_len=2048, dropout=0.1, multiple_of=256, rope_theta=10000.0):
    # vocab_size: 词表大小（默认LLaMA-1的32K词表）
    # d_model: 词嵌入维度（默认LLaMA-1 7B的4096）
    # n_layers: 解码器堆叠层数（默认LLaMA-1 7B的32层）
    # n_heads: 注意力头数（默认LLaMA-1 7B的32头）
    # n_kv_heads: KV头数（GQA，默认LLaMA-2 7B的8个KV头）
    # d_ff: 前馈全连接层维度（默认LLaMA-1 7B的11008）
    # max_seq_len: 最大序列长度
    # dropout: Dropout置零比率
    # multiple_of: 前馈网络维度对齐基数
    model = LLaMA(vocab_size, d_model, n_layers, n_heads, n_kv_heads,
                  d_ff, max_seq_len, dropout, multiple_of, rope_theta)

    # 初始化所有参数
    # 对所有维度大于1的权重矩阵使用Xavier均匀初始化
    # 有助于梯度流动和训练稳定性
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


if __name__ == '__main__':
    # 小模型配置（用于测试）
    small_config = {
        'vocab_size': 5000,
        'd_model': 256,
        'n_layers': 4,
        'n_heads': 8,
        'n_kv_heads': 2,
        'd_ff': 1024,
        'max_seq_len': 128,
        'dropout': 0.1,
        'multiple_of': 64
    }

    # 构造模型
    model = make_model(**small_config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'llama小模型参数量: {total_params / 1e6:.2f}M')

    # 1. 前向传播测试（Prefill模式）
    x = torch.randint(0, small_config['vocab_size'], (2, 32))
    logits, kv_caches = model(x, start_pos=0, kv_caches=None)
    print(f'prefill 输入: {x.shape}, 输出: {logits.shape}, kv cache层数: {len(kv_caches)}')
    print(f'  每层k缓存形状: {kv_caches[0][0].shape}, v缓存形状: {kv_caches[0][1].shape}')

    # 2. 逐token生成测试（Decode模式，使用KV Cache）
    decode_x = torch.randint(0, small_config['vocab_size'], (2, 1))
    logits2, kv_caches2 = model(decode_x, start_pos=32, kv_caches=kv_caches)
    print(f'decode(带cache) 输入: {decode_x.shape}, 输出: {logits2.shape}')
    print(f'  拼接后k缓存形状: {kv_caches2[0][0].shape}（32+1=33 token）')

    # 3. generate测试（含自动Prefill+Decode）
    output = model.generate(x, max_new_tokens=10, temperature=0.8, top_k=50)
    print(f'生成后序列形状: {output.shape}')

