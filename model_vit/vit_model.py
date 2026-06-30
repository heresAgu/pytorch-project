import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


#构建LayerNorm类 实现规范化层
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        #features: 词嵌入维度
        #eps:防除零的极小值
        super(LayerNorm, self).__init__()
        #初始化缩放参数gamma和偏移参数beta
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x: (batch_size, seq_len, features)
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        #对最后一维做归一化
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


#实现克隆函数 因为在多头注意力机制中要用到多个结构相同的线性层
#需要使用clone函数 将他们一同初始化到一个网络层列表对象中
def clones(module, N):
    #module:代表要克隆的目标网络层
    #N:克隆N个
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


#实现注意力计算函数
def attention(query, key, value, mask=None, dropout=None):
    #q, k, v: 代表注意力的三个输入张量
    #mask：掩码张量
    #dropout：传入的Dropout示例对象
    d_k = query.size(-1)

    #注意力计算公式
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    #判断是否使用掩码张量
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


#构建子层连接结构 实现残差连接 + LayerNorm + Dropout
class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        #size:词嵌入维度
        #dropout：Dropout置零比率
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        #ViT使用的是Pre-LayerNorm 即先归一化再通过子层
        #x: 上一层的输入
        #sublayer: 该子层连接中要连接的子层函数
        return x + self.dropout(sublayer(self.norm(x)))


#实现多头注意力机制的类
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        #head:代表几个头的参数
        #embedding_dim :代表词嵌入的维度
        #dropout:进行Dropout操作时 置零的比率
        super(MultiHeadedAttention, self).__init__()
        #要确认一个事实：多头的数量需要整除词嵌入的维度 分而治之
        assert embedding_dim % head == 0

        #得到每个头获得的词向量的维度
        self.d_k = embedding_dim // head
        self.head = head
        self.embedding_dim = embedding_dim

        #获得线性层 要四个 分别给Q，K，V以及最终的输出线性层
        self.linears = clones(nn.Linear(embedding_dim, embedding_dim), 4)

        #初始化注意力张量
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        #q, k, v: 注意力的三个输入张量
        #mask：掩码张量
        if mask is not None:
            #多头注意力中 使用相同掩码 所以扩展维度
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        #1) 对Q、K、V分别做线性变换 然后切分成多头
        #query, key, value形状变化: (batch_size, seq_len, embedding_dim) -> (batch_size, head, seq_len, d_k)
        query, key, value = [
            l(x).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        #2) 计算注意力
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        #3) 合并所有头的输出
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.embedding_dim)

        #4) 通过最后的线性层得到最终输出
        return self.linears[-1](x)


#实现前馈全连接层的类
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        #d_model:词嵌入维度
        #d_ff:前馈全连接层中变换矩阵的维度
        #dropout:置零比率
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #ViT论文中使用GELU作为激活函数 相比ReLU更平滑
        #x: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))

"""---------------这里开始出现transformer中没有的结构----------------"""

#构建PatchEmbedding类 将图像切分为Patch并映射到嵌入空间
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim):
        #in_channels:输入图像的通道数（RGB=3）
        #patch_size:每个图像块的大小（如16x16）
        #embedding_dim:嵌入向量的维度
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size

        #使用Conv2d实现Patch Embedding
        #卷积核和步长都等于patch_size 一次前向得到所有Patch的嵌入
        #等价于：先切分Patch再展平最后做线性投影
        self.proj = nn.Conv2d(in_channels, embedding_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch_size, in_channels, height, width)
        #经过卷积: (batch_size, embedding_dim, h//patch_size, w//patch_size)
        x = self.proj(x)

        #展平最后两个维度: (batch_size, embedding_dim, num_patches)
        x = x.flatten(2)

        #转置为序列格式: (batch_size, num_patches, embedding_dim)
        x = x.transpose(1, 2)
        return x


#构建单个Transformer编码器块
class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, head, dropout):
        #d_model:嵌入维度
        #d_ff:前馈全连接层维度
        #head:多头注意力的头数
        #dropout:置零比率
        super(TransformerBlock, self).__init__()

        #多头自注意力子层
        self.self_attn = MultiHeadedAttention(head, d_model, dropout)

        #前馈全连接子层
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        #两个子层连接对象 分别连接注意力和前馈网络
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)

        #保存维度参数
        self.d_model = d_model

    def forward(self, x, mask=None):
        #x:当前层的输入张量
        #mask:掩码张量（ViT一般为None）

        #第一步 经过多头自注意力子层
        #self_attn的Q、K、V都是x 因为要做自注意力
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))

        #第二步 经过前馈全连接子层
        x = self.sublayer[1](x, self.feed_forward)

        return x


#构建编码器类 堆叠多个TransformerBlock
class Encoder(nn.Module):
    def __init__(self, layer, N):
        #layer:TransformerBlock实例
        #N:堆叠的层数
        super(Encoder, self).__init__()

        #克隆N个TransformerBlock
        self.layers = clones(layer, N)

        #最后的LayerNorm
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask=None):
        #顺序通过N个TransformerBlock
        for layer in self.layers:
            x = layer(x, mask)

        #最后经过LayerNorm
        return self.norm(x)


#构建ViT（Vision Transformer）完整模型类
class ViT(nn.Module):
    """
    ViT:将图像视为Patch序列 用标准Transformer Encoder处理
    架构：图像分块 -> Patch嵌入 -> 拼接[CLS]Token -> 加位置编码 -> Encoder -> 分类头
    """
    def __init__(self, image_size, patch_size, in_channels, num_classes,
                 embedding_dim, d_ff, head, num_layers, dropout=0.1):
        #image_size:输入图像的尺寸（高=宽）
        #patch_size:每个图像块的大小
        #in_channels:输入图像的通道数
        #num_classes:分类任务的类别数
        #embedding_dim:嵌入向量的维度
        #d_ff:前馈全连接层中间维度
        #head:多头注意力的头数
        #num_layers:Transformer Encoder堆叠层数
        #dropout:Dropout置零比率
        super(ViT, self).__init__()

        #验证图像尺寸能被Patch尺寸整除
        assert image_size % patch_size == 0
        #计算Patch的总数 例如224/16=14 14*14=196
        num_patches = (image_size // patch_size) ** 2

        #1) Patch Embedding：将图像分块并映射到嵌入空间
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embedding_dim)

        #2) Class Token：可学习的分类标记向量 类似BERT中的[CLS]
        #形状为 (1, 1, embedding_dim) 会在batch维度上扩展
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

        #3) Position Embedding：可学习的一维位置编码
        #形状为 (1, num_patches + 1, embedding_dim) 要为[CLS]Token留一个位置
        #ViT论文指出一维位置编码效果与二维相当 且更简单
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embedding_dim))

        #4) Dropout层 在位置编码之后使用
        self.dropout = nn.Dropout(p=dropout)

        #5) Transformer Encoder：堆叠多个TransformerBlock
        block = TransformerBlock(embedding_dim, d_ff, head, dropout)
        self.encoder = Encoder(block, num_layers)

        #6) 分类头（MLP Head）：接在[CLS]Token输出之后
        #一个LayerNorm + 一个线性层 映射到类别数
        self.head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, in_channels, height, width) 输入图像张量

        batch_size = x.size(0)

        #1) Patch Embedding: (batch_size, num_patches, embedding_dim)
        x = self.patch_embed(x)

        #2) 拼接Class Token到序列最前面
        #cls_tokens: (batch_size, 1, embedding_dim)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        #x: (batch_size, num_patches + 1, embedding_dim)
        x = torch.cat((cls_tokens, x), dim=1)

        #3) 添加位置编码 直接相加
        x = x + self.pos_embed
        x = self.dropout(x)

        #4) 通过Transformer Encoder
        x = self.encoder(x)

        #5) 取出[CLS]Token对应位置的输出 用于分类
        #x[:, 0]的形状: (batch_size, embedding_dim)
        x = x[:, 0]

        #6) 通过分类头得到最终分类结果
        x = self.head(x)

        return x


#构建make_vit函数 创建ViT模型实例
def make_vit(image_size=224, patch_size=16, in_channels=3, num_classes=1000,
             embedding_dim=768, d_ff=3072, head=12, num_layers=12, dropout=0.1):
    """
    image_size: 输入图像的尺寸 (默认224 标准ImageNet尺寸)
    patch_size: 每个图像块的大小 (默认16 ViT-B/16)
    in_channels: 输入图像的通道数 (RGB=3)
    num_classes: 分类任务的类别数 (默认1000 ImageNet)
    embedding_dim: 嵌入向量的维度 (默认768 ViT-Base)
    d_ff: 前馈全连接层中间维度 (默认3072 ViT-Base)
    head: 多头注意力的头数()默认12 ViT-Base)
    num_layers: Transformer Encoder堆叠层数(默认12 ViT-Base)
    dropout: Dropout置零比率(默认0.1)
    """
    model = ViT(image_size, patch_size, in_channels, num_classes,
                embedding_dim, d_ff, head, num_layers, dropout)

    #初始化整个模型的参数 判断参数的维度dim>1 将矩阵初始化成服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
