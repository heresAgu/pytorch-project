import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


# --------------------- 基础组件 ---------------------

def gelu(x):
    """GELU 激活函数 —— BERT 论文中使用 GELU 而非 ReLU
    GELU(x) = x * Φ(x), 其中 Φ(x) 是标准高斯分布的 CDF
    这里使用近似实现: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x^3)))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def clones(module, N):
    """克隆 N 个相同的网络层, 放入 ModuleList"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """层归一化 Layer Normalization (与 transformer_model.py 风格一致)"""
    def __init__(self, features, eps=1e-6):
        # features: 归一化维度的特征数, 即 d_model
        super(LayerNorm, self).__init__()
        # 可学习的缩放参数 (gain / weight), 初始化为全 1
        self.a_2 = nn.Parameter(torch.ones(features))
        # 可学习的偏移参数 (bias), 初始化为全 0
        self.b_2 = nn.Parameter(torch.zeros(features))
        # 防止除零的小常数
        self.eps = eps

    def forward(self, x):
        # 计算 x 在最后一维上的均值
        mean = x.mean(-1, keepdim=True)
        # 计算 x 在最后一维上的标准差
        std = x.std(-1, keepdim=True)
        # 归一化后做仿射变换: γ * (x-μ)/σ + β
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """子层连接: 残差连接 + Dropout + 层归一化 (Pre-LN 风格)
    与 transformer_model.py 中的 SublayerConnection 结构一致
    顺序: LayerNorm -> Sublayer -> Dropout -> + x (残差连接)
    """
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        # 层归一化
        self.norm = LayerNorm(size)
        # Dropout 层
        self.dropout = nn.Dropout(p=dropout)
        # 记录子层输入特征维度
        self.size = size

    def forward(self, x, sublayer):
        # Pre-LN: x + Dropout(sublayer(LayerNorm(x)))
        # 先归一化, 再经过子层 (注意力或FFN), Dropout 后残差连接
        return x + self.dropout(sublayer(self.norm(x)))


# --------------------- 注意力机制 ---------------------

def attention(query, key, value, mask=None, dropout=None):
    """缩放点积注意力 (Scaled Dot-Product Attention)
    与 transformer_model.py 中的 attention 函数完全一致
    """
    # d_k: 每个注意力头中 query/key 的维度
    d_k = query.size(-1)
    # scores = Q * K^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果提供了掩码张量, 将掩码位置填充为极小值
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 在最后一维上做 softmax, 得到注意力权重
    p_attn = F.softmax(scores, dim=-1)
    # 可选的 Dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 注意力权重与 value 相乘, 返回加权结果和注意力分布
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """多头注意力机制 (Multi-Head Attention)
    与 transformer_model.py 风格一致
    """
    def __init__(self, head, embedding_dim, dropout=0.1):
        # head: 注意力头的数量
        # embedding_dim: 词嵌入维度 (必须是 head 的整数倍)
        # dropout: Dropout 置零比率
        super(MultiHeadedAttention, self).__init__()
        assert embedding_dim % head == 0
        # 每个头的维度 = 总维度 / 头数
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim
        # Q, K, V 三个线性变换 + 输出线性变换, 共 4 个
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)
        # 注意力权重缓存 (用于可视化)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # mask 扩展维度: (batch, 1, seq_len) -> (batch, 1, 1, seq_len)
        # 此操作与 transformer_model.py 中的 unsqueeze(1) 逻辑一致
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # 1) 对 Q, K, V 分别做线性变换, 然后分头:
        #    从 (batch, seq_len, d_model) -> (batch, head, seq_len, d_k)
        query, key, value = [
            lin(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) 计算注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 将多头结果拼接: transpose + contiguous + view
        #    从 (batch, head, seq_len, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 4) 最后的输出线性层
        return self.linears[-1](x)


# --------------------- 前馈全连接层 ---------------------

class PositionWiseFeedForward(nn.Module):
    """逐位置前馈全连接网络 (Position-wise Feed-Forward Network)
    BERT 使用 GELU 激活函数 (原 Transformer 使用 ReLU)
    结构: Linear -> GELU -> Dropout -> Linear
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        # d_model: 词嵌入维度 (输入和最终输出维度)
        # d_ff:    中间隐藏层维度 (Feed-Forward 维度)
        # dropout: Dropout 置零比率
        super(PositionWiseFeedForward, self).__init__()
        # 第一层线性变换: d_model -> d_ff
        self.w_1 = nn.Linear(d_model, d_ff)
        # 第二层线性变换: d_ff -> d_model
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # 先升维 -> GELU -> Dropout -> 降维
        return self.w_2(self.dropout(gelu(self.w_1(x))))


# --------------------- BERT 嵌入层 ---------------------

class BertEmbedding(nn.Module):
    """BERT 嵌入层: Token 嵌入 + 位置嵌入 + 片段嵌入 (Segment Embedding)
    BERT 的嵌入是三部分的求和, 之后经过 LayerNorm 和 Dropout
    """
    def __init__(self, vocab_size, d_model, max_len=512, num_segments=2, dropout=0.1):
        # vocab_size:     词表大小
        # d_model:        词嵌入维度
        # max_len:        最大句子长度 (BERT 默认为 512)
        # num_segments:   片段类型数 (句子 A / 句子 B)
        # dropout:        Dropout 置零比率
        super(BertEmbedding, self).__init__()
        # Token 嵌入 (词嵌入) — 与 transformer_model.py Embedding 类风格一致
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # 位置嵌入 (BERT 使用可学习的位置嵌入, 而非正弦编码)
        self.position_embed = nn.Embedding(max_len, d_model)
        # 片段嵌入 (Token Type Embedding: 区分句子 A 和句子 B)
        self.segment_embed = nn.Embedding(num_segments, d_model)
        # 层归一化
        self.norm = LayerNorm(d_model)
        # Dropout
        self.dropout = nn.Dropout(p=dropout)

        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x, segment_ids=None):
        # x: 输入 token ids, 形状 (batch, seq_len)
        # segment_ids: 片段类型 ids, 形状 (batch, seq_len)
        seq_len = x.size(1)

        # 生成位置索引 [0, 1, 2, ..., seq_len-1], 形状 (1, seq_len)
        pos_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(x)

        # 如果没有提供 segment_ids, 默认全为 0 (单句场景)
        if segment_ids is None:
            segment_ids = torch.zeros_like(x)

        # Token 嵌入 (乘以 √d_model, 与 Embedding 类保持一致的缩放逻辑)
        token_embed = self.token_embed(x) * math.sqrt(self.d_model)
        # 位置嵌入
        position_embed = self.position_embed(pos_ids)
        # 片段嵌入
        segment_embed = self.segment_embed(segment_ids)

        # 三者求和 -> LayerNorm -> Dropout
        embeddings = token_embed + position_embed + segment_embed
        return self.dropout(self.norm(embeddings))


# --------------------- BERT 编码器层 ---------------------

class BertLayer(nn.Module):
    """BERT 编码器层: 多头自注意力 + 前馈全连接层
    结构与 transformer 中的 EncodeLayer 一致:
    SublayerConnection(MultiHeadedAttention) -> SublayerConnection(PositionWiseFeedForward)
    """
    def __init__(self, d_model, self_attn, feed_forward, dropout=0.1):
        # d_model:     词嵌入维度
        # self_attn:   多头自注意力实例
        # feed_forward:前馈全连接层实例
        # dropout:     Dropout 置零比率
        super(BertLayer, self).__init__()
        # 两个子层连接 (自注意力 + 前馈全连接)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = d_model

    def forward(self, x, mask=None):
        # 第一个子层: 多头自注意力 (Q=K=V=x, 即自注意力机制)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个子层: 前馈全连接网络
        x = self.sublayer[1](x, self.feed_forward)
        return x


class BertEncoder(nn.Module):
    """BERT 编码器: 堆叠 N 个 BertLayer
    对应 transformer 中的 Encoder 类
    """
    def __init__(self, layer, N):
        # layer: 一个 BertLayer 实例, 用于克隆
        # N:     编码器层数
        super(BertEncoder, self).__init__()
        # 克隆 N 个 BertLayer
        self.layers = clones(layer, N)
        # 最后一层 LayerNorm (BERT 在最后额外加一层 LayerNorm)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        # 逐层传递
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# --------------------- BERT 池化层 ---------------------

class BertPooler(nn.Module):
    """BERT 池化层: 取 [CLS] 位置的输出, 经过线性层 + Tanh
    用于 Next Sentence Prediction 等分类任务
    """
    def __init__(self, d_model):
        super(BertPooler, self).__init__()
        # 线性变换 + Tanh 激活
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        # 取序列第一个位置 ([CLS] token) 的输出
        cls_token = x[:, 0, :]       # (batch, d_model)
        # 线性层 + Tanh
        return self.activation(self.dense(cls_token))


# --------------------- BERT 预训练预测头 ---------------------

class BertPretrainingHeads(nn.Module):
    """BERT 预训练任务的预测头:
    1) Masked Language Model (MLM): 预测被 mask 的词
    2) Next Sentence Prediction (NSP): 预测两句话是否连续
    """
    def __init__(self, d_model, vocab_size):
        super(BertPretrainingHeads, self).__init__()
        # MLM 预测头: 线性层 + GELU + LayerNorm + 投影到词表
        self.mlm_dense = nn.Linear(d_model, d_model)
        self.mlm_activation = gelu
        self.mlm_norm = LayerNorm(d_model)
        self.mlm_decoder = nn.Linear(d_model, vocab_size)

        # NSP 预测头: 二分类 (IsNext / NotNext)
        self.nsp_dense = nn.Linear(d_model, 2)

    def forward(self, encoder_output, pooler_output):
        # encoder_output: 编码器输出, (batch, seq_len, d_model)
        # pooler_output:  池化层输出, (batch, d_model)

        # --- MLM 分支 ---
        mlm_out = self.mlm_dense(encoder_output)
        mlm_out = self.mlm_activation(mlm_out)
        mlm_out = self.mlm_norm(mlm_out)
        # 投影到词表大小, 得到每个位置在词表上的概率分布
        mlm_logits = self.mlm_decoder(mlm_out)

        # --- NSP 分支 ---
        nsp_logits = self.nsp_dense(pooler_output)

        return mlm_logits, nsp_logits


# --------------------- 完整的 BERT 模型 ---------------------

class BertModel(nn.Module):
    """完整的 BERT 模型 (BERT: Bidirectional Encoder Representations from Transformers)
    结构: BertEmbedding -> BertEncoder -> BertPooler
    """
    def __init__(self, embedding, encoder, pooler):
        # embedding: BertEmbedding 实例
        # encoder:   BertEncoder 实例
        # pooler:    BertPooler 实例
        super(BertModel, self).__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.pooler = pooler

    def forward(self, input_ids, segment_ids=None, attention_mask=None):
        # input_ids:     输入的 token ids, 形状 (batch, seq_len)
        # segment_ids:   片段类型 ids, 形状 (batch, seq_len), 默认 None
        # attention_mask:注意力掩码, 形状 (batch, seq_len), 1=有效token, 0=填充token

        # 1) 三部分嵌入求和
        embedding_out = self.embedding(input_ids, segment_ids)

        # 2) 扩展掩码维度: (batch, seq_len) -> (batch, 1, seq_len)
        #    后续 MHA.forward 会再 unsqueeze(1) 为 (batch, 1, 1, seq_len)
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1)  # (batch, 1, seq_len)
        else:
            extended_mask = None

        # 3) 经过 N 层编码器
        encoder_out = self.encoder(embedding_out, extended_mask)

        # 4) 池化 [CLS] token
        pooler_out = self.pooler(encoder_out)

        return encoder_out, pooler_out


# --------------------- 配置类 ---------------------

class BertConfig:
    """BERT 模型配置参数
    默认使用 BERT-base 配置 (L=12, H=768, A=12)
    """
    def __init__(self,
                 vocab_size=30522,      # 词表大小 (BERT-base uncased)
                 d_model=768,           # 嵌入维度 (Hidden size)
                 d_ff=3072,             # 前馈全连接层中间维度
                 num_layers=12,         # 编码器层数
                 num_heads=12,          # 注意力头数
                 max_len=512,           # 最大序列长度
                 num_segments=2,        # 片段类型数
                 dropout=0.1):          # Dropout 比率
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_len = max_len
        self.num_segments = num_segments
        self.dropout = dropout


# --------------------- 模型工厂函数 ---------------------

def make_bert(vocab_size=30522, d_model=768, d_ff=3072,
              num_layers=12, num_heads=12, max_len=512,
              num_segments=2, dropout=0.1):
    """创建 BERT 模型的工厂函数 (与 transformer_model.py 的 make_model 风格一致)

    参数:
        vocab_size:   词表大小 (默认 30522, 对应 BERT-base uncased)
        d_model:      词嵌入维度 (默认 768)
        d_ff:         前馈全连接层中间维度 (默认 3072)
        num_layers:   编码器堆叠层数 (默认 12)
        num_heads:    多头注意力头数 (默认 12)
        max_len:      最大序列长度 (默认 512)
        num_segments: 片段类型数 (默认 2: 句子A / 句子B)
        dropout:      Dropout 置零比率 (默认 0.1)

    返回:
        BertModel 实例
    """
    c = copy.deepcopy

    # 1) 创建多头注意力实例
    attn = MultiHeadedAttention(num_heads, d_model, dropout)

    # 2) 创建前馈全连接层实例
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)

    # 3) 构建 BERT 模型
    model = BertModel(
        # 嵌入层: Token + Position + Segment
        embedding=BertEmbedding(vocab_size, d_model, max_len, num_segments, dropout),
        # 编码器: 堆叠 N 层 BertLayer
        encoder=BertEncoder(
            BertLayer(d_model, c(attn), c(ff), dropout),
            N=num_layers
        ),
        # 池化层: 提取 [CLS] 表示
        pooler=BertPooler(d_model)
    )

    # 4) 初始化参数: 维度 > 1 的参数使用 Xavier 均匀初始化
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


def make_bert_pretraining(vocab_size=30522, d_model=768, d_ff=3072,
                          num_layers=12, num_heads=12, max_len=512,
                          num_segments=2, dropout=0.1):
    """创建带预训练预测头的 BERT 模型 (MLM + NSP)

    返回:
        (bert_model, pretraining_heads) 元组
    """
    bert = make_bert(vocab_size, d_model, d_ff, num_layers,
                     num_heads, max_len, num_segments, dropout)
    heads = BertPretrainingHeads(d_model, vocab_size)
    return bert, heads


# ========== 测试代码 (在直接运行时执行) ==========
if __name__ == "__main__":
    print("=== 测试 BERT 模型构建 ===")

    # 小型 BERT 配置 (便于测试)
    config = BertConfig(vocab_size=1000, d_model=128, d_ff=512,
                        num_layers=2, num_heads=4, max_len=64)
    print(f"BERT 配置: L={config.num_layers}, H={config.d_model}, A={config.num_heads}")

    # 创建模型
    model = make_bert(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_len=config.max_len,
        dropout=config.dropout
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 前向传播测试
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    attention_mask = torch.ones(batch_size, seq_len)

    encoder_out, pooler_out = model(input_ids, segment_ids, attention_mask)
    print(f"编码器输出形状: {encoder_out.shape}")  # (batch, seq_len, d_model)
    print(f"池化层输出形状: {pooler_out.shape}")    # (batch, d_model)

    # 测试预训练预测头
    bert_pt, heads = make_bert_pretraining(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
    )
    mlm_logits, nsp_logits = heads(encoder_out, pooler_out)
    print(f"MLM logits 形状: {mlm_logits.shape}")  # (batch, seq_len, vocab_size)
    print(f"NSP logits 形状: {nsp_logits.shape}")    # (batch, 2)

    print("\n[OK] BERT 模型构建并测试通过!")
