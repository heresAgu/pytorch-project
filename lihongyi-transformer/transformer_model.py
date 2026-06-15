import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import copy 

#参数
# embedding = nn.Embedding(10,3)
# input1 = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# print(embedding(input1))
#构建Embedding类来实现文本嵌入层
class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        #d_model：词嵌入的维度
        #vocab:词表大小
        super(Embedding,self).__init__()
        #定义Embedding层
        self.lut = nn.Embedding(vocab ,d_model)
        #将参数传入类中
        self.d_model = d_model
    
    def forward(self,x):
        # x: 输入进模型的文本 通过词汇映射后的数字张量
        return self.lut(x) * math.sqrt(self.d_model)


#构建位置编码器的类
class PositionalEncoding(nn.Module):
    def __init__(self, d_model ,dropout ,max_len = 5000):
        #d_model：词嵌入的维度
        #dropout :Dropout层的置零比率
        #max_len: 每个句子的最大长度
        super(PositionalEncoding,self).__init__()

        #实现Dropout
        self.dropout = nn.Dropout(p=dropout)

        #初始化一个位置编码矩阵 大小是max_len * d_model
        pe = torch.zeros(max_len ,d_model)
        
        #初始化一个绝对位置矩阵 max_len * 1
        position = torch.arange(0,max_len).unsqueeze(1)

        #定义一个变换矩阵div_term,跳跃式的初始化
        div_term = torch.exp(torch.arange(0,d_model,2) * -(math.log(10000.0) / d_model))

        #对变换矩阵进行奇偶数分别赋值
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        #将二维张量改为三维张量
        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self ,x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

#构建掩码张量的函数
def subsequent_mask(size):  
    #size : 代表掩码张量最后两个维度，形成一个方阵
    attn_shape = (1 , size , size)
    #使用np.ones()构建全一的张量 然后用np.triu()形成上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape),k=1).astype('uint8')
    #反转矩阵
    return torch.from_numpy(1-subsequent_mask)

def attention(query, key ,value , mask = None , dropout = None):
    #q, k, v: 代表注意力的三个输入张量
    #mask：掩码张量
    #dropout：传入的Dropout示例对象
    d_k = query.size(-1)

    #注意力计算公式
    scores = torch.matmul(query ,key.transpose(-2,-1)) / math.sqrt(d_k)

    #判断是否使用掩码张量
    if mask is not None:
        scores = scores.masked_fill(mask == 0,-1e9)
    p_attn = F.softmax(scores,dim = -1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    
    return torch.matmul(p_attn , value) ,p_attn

#实现克隆函数 因为在多头注意力机制中要用到多个结构相同的线性层
#需要使用clone函数 将他们一同初始化到一个网络层列表对象中
def clones(module,N):
    #module:代表要克隆的目标网络层
    #N:克隆N个
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

#实现多头注意力机制的类
class MultiHeadedAttention(nn.Module):
    def __init__(self, head , embedding_dim ,dropout = 0.1):
        #head:代表几个头的参数
        #embedding_dim :代表词嵌入的维度
        #dropout:进行Dropout操作时 置零的比率
        super(MultiHeadedAttention , self).__init__() 
        #要确认一个事实：多头的数量需要整除词嵌入的维度 分而治之
        assert embedding_dim % head == 0

        #得到每个头获得的词向量的维度
        self.d_k = embedding_dim // head

        self.head = head
        self.embedding_dim = embedding_dim

        #获得线性层 要四个 分别给Q，K，V以及最终的输出线性层
        self.linears = clones(nn.Linear(embedding_dim,embedding_dim) ,4)

        #初始化注意力张量
        self.attn = None

        #初始化Dropout对象
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self,query,key,value,mask = None):
        #query,key,value:代表注意力的三个输入张量 mask:掩码张量
        if mask is not None:
            #先将mask进行维度扩充 因为多头注意力机制需要n个头的mask
            mask = mask.unsqueeze(1)
        
        #得到batch_size
        batch_size = query.size(0)

        #使用zip将QKV与线性层结合 然后利用view和transpose拆分多头
        query , key, value = \
        [model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2)  
         for model ,x in zip(self.linears ,(query,key,value))]
        
        #将每个头拼在一起
        x, self.attn = attention(query,key,value,mask=mask,dropout=self.dropout)

        #将多个头拼接在一起（view前需要contiguous）
        #将多个头拼接在一起
        #使用transpose()交换维度 使用contiguous()转为连续 再用view()合并
        x = x.transpose(1,2).contiguous().view(batch_size,-1,self.head *self.d_k)

        #将x传入最后一个线性层 进行线性变换后输出
        return self.linears[-1](x)
    

# # 前馈全连接网络
# head = 8
# embedding_dim = 512
# dropout = 0.2

# # 层归一化
# embedding = Embedding(embedding_dim, 1000)
# dummy_input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
# embedded = embedding(dummy_input)
# pe = PositionalEncoding(embedding_dim, dropout)
# pe_result = pe(embedded)
# query = key = value = pe_result

# mask = torch.zeros(1,4,4)

# mha = MultiHeadedAttention(head, embedding_dim, dropout)
# mha_result = mha(query, key, value, mask)
# print(mha_result)
# print(mha_result.shape)


#构建子层连接结构
class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_model ,d_ff ,dropout=0.1):
        #d_model :词嵌入的维度
        #d_ff:前馈全连接网络中变换矩阵的维度 默认为2048
        super(PositionWiseFeedForward,self).__init__()
    
        #初始化两个全连接层
        self.w1 = nn.Linear(d_model,d_ff)
        self.w2 = nn.Linear(d_ff ,d_model)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self ,x):
        #x:上一层的张量
        #首先将x送入第一个线性层 再用relu激活 然后dropout 再进入第二个线性层
        #返回最后一层的计算结果
        return self.w2(self.dropout(F.relu(self.w1(x))))
    
    #计算Z的值
class LayerNorm(nn.Module):
    def __init__(self,features,eps = 1e-6):
        #features:词嵌入的维度
        #eps:防止分母为0的一个很小的数 默认为1e-6
        super(LayerNorm,self).__init__()

        #初始化两个参数a2和b2 形状与Z相同
        #使用nn.Parameter封装 使它们在训练时可以被更新

        self.a2 = nn.Parameter(torch.ones(features))
        self.b2 = nn.Parameter(torch.zeros(features))

        self.eps = eps

    def forward(self,x):
        #x:上一层的张量
        #首先将x进行规范化处理 然后送入子层函数处理
        mean = x.mean(-1 , keepdim = True)
        #处理结果进入dropout层 最后残差处理
        std = x.std(-1,keepdim = True)
        #返回Z加上输入x的值
        return self.a2 * (x-mean) / (std + self.eps) + self.b2
    
#构建子层连接结构
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout =0.1):
        #size:词嵌入的维度
        super(SublayerConnection,self).__init__()
        #初始化一个规范化层对象
        self.norm = LayerNorm(size)
        #初始化一个dropout对象
        self.dropout = nn.Dropout(p=dropout)
        self.size = size
    
    def forward(self ,x,sublayer):
        #x:上一层的张量
        #sublayer:该子层连接中子层函数
        #首先将x规范化 然后送入子层函数处理 处理结果进入dropout层 最后残差处理
        return x + self.dropout(sublayer(self.norm (x)))
    

#构建编码器层
class EncodeLayer(nn.Module):
    def __init__(self,size,self_attn ,feed_forward ,dropout):
        #self_attn:多头自注意力机制的对象
        #feed_forward:前馈全连接层的对象
        super(EncodeLayer,self).__init__()

        #初始化两个子层连接对象
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.size = size

        #编码器层中有2个子层连接结构 使用clone函数操作
        self.sublayer = clones(SublayerConnection(size,dropout),2)

    def forward(self ,x,mask):
        #首先让x经过第一个子层连接结构 内部包含多头自注意力机制子层
        #再让张量经过第二个子层连接结构 其中包含前馈全连接网络
        x = self.sublayer[0](x,lambda x: self.self_attn(x,x,x,mask))
        return self.sublayer[1](x,self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer ,N):
        super(Encoder ,self).__init__()
        #首先使用clones函数 克隆N个编码器 放置在self.layers中
        self.layers = clones(layer,N)
        #初始化一个规范化层 作用在编码器的最后面
        self.norm = LayerNorm(layer.size)

    def forward(self,x,mask):
        #让x经历N个编码器层的处理 最后再经过规范化层就可以输出了
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
        
#构建解码器层
class DecoderLayer(nn.Module):
    def __init__(self,size,self_attn , src_attn ,feed_forward ,dropout):
        #self_attn:多头自注意力机制的对象
        #src_attn:代表常规的自注意力机制对象

        super(DecoderLayer,self).__init__()

        #将参数传入类中
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.dropout = dropout

        #按照解码器层的结构图 使用clones函数克隆3个子层连接对象
        self.sublayer = clones(SublayerConnection(size ,dropout) ,3)

    def forward( self ,x ,memory, source_mask ,target_mask):
        #x:上一层的张量
        #memory:代表编码器的语义存储张量
        #source_mask:源数据的掩码张量
        #target_mask:目标数据的掩码张量
        m = memory

        #第一步让x经历第一个子层 多头子注意力机制的子层
        #采用target_mask 为了将解码时未来的信息进行遮蔽
        x =self.sublayer[0](x,lambda x: self.self_attn(x,x,x,target_mask))

        #第二步 x经历第二个子层 常规注意力机制的子层
        #采用source_mask 为了遮掩掉对结果无用的数据
        x =self.sublayer[1](x,lambda x: self.src_attn(x,m,m,source_mask))

        #第三步 x经历第三个子层 前馈全连接层
        return self.sublayer[2](x,self.feed_forward)
    
#构建解码器
class Decoder(nn.Module):
    def __init__(self, layer ,N):
        super(Decoder ,self).__init__()
        #首先使用clones函数 克隆N个解码器 放置在self.layers中
        self.layers = clones(layer,N)
        #初始化一个规范化层
        self.norm = LayerNorm(layer.size)

    def forward(self,x,memory,source_mask ,target_mask):
        #memory:编码器输出张量
        #source_mask:源数据的掩码张量
        #target_mask:目标数据的掩码张量
        for layer in self.layers:
            x = layer(x,memory,source_mask ,target_mask)
        return self.norm(x)


#构建Generator类
class Generator(nn.Module):
    def __init__(self,d_model ,vocab_size):
        super(Generator,self).__init__()
        #定义一个线性层 作用是完成网络输出维度的变换
        self.project = nn.Linear(d_model ,vocab_size)

    def forward(self,x):
        #首先将x送入线性层中 经过softmax函数的处理
        return F.log_softmax(self.project(x),dim=-1)
    

#构建编码器-解码器结构类
class EncoderDecoder(nn.Module):
    def __init__(self,encoder ,decoder ,source_embed ,target_embed ,generator):
        #encoder :代表编码器对象
        #decoder :代表解码器对象
        #target_embed:代表源数据的嵌入函数
        #target_embed:目标数据的嵌入函数
        #generator:输出部分类别生成器对象
        super(EncoderDecoder,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = source_embed
        self.tgt_embed = target_embed
        self.generator = generator

    def forward(self, source ,target ,source_mask ,target_mask):
        return self.decode(self.encode(source,source_mask),source_mask ,target ,target_mask)
    
    def encode(self,source ,source_mask):
        return self.encoder(self.src_embed(source),source_mask)

    def decode(self ,memory ,source_mask ,target ,target_mask):
        return self.decoder(self.tgt_embed(target),memory ,source_mask ,target_mask)
    
def make_model(source_vocab ,target_vocab ,N=6 ,d_model=512,d_ff=2048, head=8,dropout =0.1):
    #source_vocab：词汇总数
    #target_vocab：目标词汇总数
    #N：代表编码器和解码器堆叠的层数
    #d_model：词嵌入维度
    #d_ff：前馈全连接层中变换矩阵的维度
    #head：多头注意力的头数
    #dropout：置零比率
    c = copy.deepcopy

    #实例化一个多头注意力的类
    attn = MultiHeadedAttention(head ,d_model)

    #实例化一个前馈全连接层网络对象
    ff = PositionWiseFeedForward(d_model,d_ff ,dropout)

    #实例化一个位置编码器
    position = PositionalEncoding(d_model ,dropout)

    #实例化模型model 利用EncoderDecoder类
    #编码器中的结构里有2个子层 attention层和前馈全连接层
    #解码器中有3个子层 两个attention层和前馈全连接层
    model = EncoderDecoder(
        Encoder(EncodeLayer(d_model ,c(attn) ,c(ff) ,dropout),N),
        Decoder(DecoderLayer(d_model ,c(attn),c(attn),c(ff),dropout),N),
        nn.Sequential(Embedding(d_model,source_vocab),c(position)),
        nn.Sequential(Embedding(d_model,target_vocab),c(position)),
        Generator(d_model,target_vocab))
    
    #初始化整个模型的参数 判断参数的维度dim>1 将矩阵初始化成一个服从均匀分布的矩阵
    for p in model.parameters():
        if p.dim() >1:
            nn.init.xavier_uniform_(p)
    return model



