import torch, random, os

EN_WORDS = ['the','a','is','are','I','you','he','she','it','we','they',
            'eat','drink','see','like','love','have','make','go','come','take',
            'big','small','red','blue','good','bad','new','old','hot','cold',
            'apple','water','book','house','dog','cat','car','city','man','woman',
            'with','and','in','on','at','to','for','of','from','by']

ZH_WORDS = ['我','你','他','她','它','我们','他们',
            '吃','喝','看','喜欢','爱','有','做','去','来','拿',
            '大','小','红','蓝','好','坏','新','旧','热','冷',
            '苹果','水','书','房子','狗','猫','车','城市','男人','女人',
            '和','在','里','上','的','了','是','一个','不','很']

# 补齐词表到相同长度
while len(ZH_WORDS) < len(EN_WORDS):
    ZH_WORDS.append('某')

VOCAB_SIZE = len(EN_WORDS) + 3  # +3 for PAD(0), BOS(1), EOS(2)
assert len(EN_WORDS) == len(ZH_WORDS)
ID_MAX = len(EN_WORDS) + 2  # 最大有效ID (不含EOS)

ID_TO_EN = {i+3: w for i, w in enumerate(EN_WORDS)}
ID_TO_ZH = {i+3: w for i, w in enumerate(ZH_WORDS)}
ID_TO_EN[1] = '<BOS>'; ID_TO_EN[2] = '<EOS>'
ID_TO_ZH[1] = '<BOS>'; ID_TO_ZH[2] = '<EOS>'

def gen_sent(max_len=7):
    l = random.randint(2, max_len)
    ids = [random.randint(3, ID_MAX) for _ in range(l)]
    en = [ID_TO_EN[i] for i in ids]
    zh = [ID_TO_ZH[i] for i in ids]
    return ids, en, zh

def gen_dataset(n=5000, max_len=7):
    data = []
    for _ in range(n):
        ids, _, _ = gen_sent(max_len)
        src = torch.tensor([1] + ids + [2])       # BOS + ids + EOS
        tgt = torch.tensor([1] + ids[::-1] + [2]) # 反转序列
        data.append((src, tgt))
    return data

random.seed(42)
train = gen_dataset(5000)
val = gen_dataset(500)

import os; os.makedirs(r'E:\pytorch-project\data', exist_ok=True)
torch.save(train, r'E:\pytorch-project\data\train.pt')
torch.save(val,   r'E:\pytorch-project\data\val.pt')

# 展示
print('数据集: train=%d, val=%d, vocab=%d' % (len(train), len(val), VOCAB_SIZE))
ids, en, zh = gen_sent(5)
print('英文:', ' '.join(en))
print('中文:', ' '.join(zh))
print('任务: 输入序列 -> 反转序列')
src = train[0][0].tolist()
tgt = train[0][1].tolist()
print('源:', src)
print('目:', tgt)
print('成功!')
