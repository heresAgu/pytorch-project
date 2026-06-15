import torch, math, time, os, re
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import sys
sys.path.insert(0, r'E:\pytorch-project\lihongyi-transformer')
from transformer_model import make_model, subsequent_mask

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH = 64
EPOCHS = 5
D_MODEL = 128
D_FF = 512
N = 2
HEADS = 4
LR = 0.0005
MAX_LEN = 20
MIN_FREQ = 5

def tok_en(s):
    s = s.lower().strip()
    s = re.sub(r'([.,!?;:])', r' \1 ', s)
    return s.split()

def tok_zh(s):
    s = s.strip()
    return [c for c in s if '\u4e00' <= c <= '\u9fff' or c in '，。！？']

print('加载数据...')
pairs = []
with open(r'E:\pytorch-project\data\cmn.txt', 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            pairs.append((parts[0], parts[1]))

en_tok = [tok_en(p[0]) for p in pairs]
zh_tok = [tok_zh(p[1]) for p in pairs]

# 构建词表
PAD, BOS, EOS = 0, 1, 2
en_vocab = Counter(t for seq in en_tok for t in seq)
zh_vocab = Counter(t for seq in zh_tok for t in seq)

# 加入 <unk> 处理未登录词
en_i2w = [PAD, BOS, EOS, '<unk>'] + sorted([w for w, c in en_vocab.items() if c >= MIN_FREQ])
zh_i2w = [PAD, BOS, EOS, '<unk>'] + sorted([w for w, c in zh_vocab.items() if c >= MIN_FREQ])
en_w2i = {w:i for i,w in enumerate(en_i2w)}
zh_w2i = {w:i for i,w in enumerate(zh_i2w)}

print('英文词表:%d 中文词表:%d 数据:%d条' % (len(en_w2i), len(zh_w2i), len(pairs)))

# 数据集
class DS(Dataset):
    def __init__(self, d):
        self.d = d
    def __len__(self): return len(self.d)
    def __getitem__(self, i): return self.d[i]

def make_data(pairs, en_tok, zh_tok):
    data = []
    for (en, zh), e_tok, z_tok in zip(pairs, en_tok, zh_tok):
        e_ids = [en_w2i.get(t, 3) for t in e_tok[:MAX_LEN-2]]  # 3=<unk>
        z_ids = [zh_w2i.get(t, 3) for t in z_tok[:MAX_LEN-2]]
        data.append((torch.tensor([1]+e_ids+[2]), torch.tensor([1]+z_ids+[2])))
    return data

def collate(batch):
    pad = 0
    max_src = max(len(s) for s,_ in batch)
    max_tgt = max(len(t) for _,t in batch)
    src = torch.stack([torch.cat([s, torch.zeros(max_src-len(s), dtype=torch.long)]) for s,_ in batch])
    tin = torch.stack([torch.cat([t[:-1], torch.zeros(max_tgt-len(t)+1, dtype=torch.long)]) for _,t in batch])
    tout = torch.stack([torch.cat([t[1:], torch.zeros(max_tgt-len(t)+1, dtype=torch.long)]) for _,t in batch])
    return src.long(), tin.long(), tout.long()

split = int(len(pairs)*0.9)
train_loader = DataLoader(DS(make_data(pairs[:split], en_tok[:split], zh_tok[:split])),
                          BATCH, True, collate_fn=collate)
val_loader = DataLoader(DS(make_data(pairs[split:], en_tok[split:], zh_tok[split:])),
                        32, False, collate_fn=collate)

print('训练:%d 验证:%d' % (len(train_loader.dataset), len(val_loader.dataset)))

model = make_model(len(en_i2w), len(zh_i2w), N, D_MODEL, D_FF, HEADS, 0.1).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.NLLLoss(ignore_index=0)
print('参数:%d 设备:%s' % (sum(p.numel() for p in model.parameters()), DEVICE))
print()

for epoch in range(EPOCHS):
    model.train()
    ls, t0 = 0, time.time()
    for src, tin, tout in train_loader:
        src, tin, tout = src.to(DEVICE), tin.to(DEVICE), tout.to(DEVICE)
        sm = (src!=0).unsqueeze(1)
        tm = subsequent_mask(tin.size(1)).to(DEVICE)
        mem = model.encode(src, sm)
        pred = model.generator(model.decode(mem, sm, tin, tm))
        loss = crit(pred.reshape(-1, len(zh_i2w)), tout.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ls += loss.item()
    
    model.eval()
    vls = 0
    with torch.no_grad():
        for src, tin, tout in val_loader:
            src, tin, tout = src.to(DEVICE), tin.to(DEVICE), tout.to(DEVICE)
            sm = (src!=0).unsqueeze(1)
            tm = subsequent_mask(tin.size(1)).to(DEVICE)
            pred = model.generator(model.decode(model.encode(src,sm), sm, tin, tm))
            vls += crit(pred.reshape(-1,len(zh_i2w)), tout.reshape(-1)).item()
    
    print('Epoch %d | train=%.3f | val=%.3f | ppl=%.1f | %.1fs' % (
        epoch+1, ls/len(train_loader), vls/len(val_loader), math.exp(vls/len(val_loader)), time.time()-t0))

torch.save({'model':model.state_dict(),'en_i2w':en_i2w,'zh_i2w':zh_i2w,'d_model':D_MODEL,'d_ff':D_FF,'n':N,'heads':HEADS},
           r'E:\pytorch-project\lihongyi-transformer\translation_model.pt')
print('\n已保存!')
