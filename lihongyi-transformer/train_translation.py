import torch, math, time, os, glob
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.insert(0, r'E:\pytorch-project\lihongyi-transformer')
from transformer_model import make_model, subsequent_mask

DEVICE = 'cuda'
BATCH = 64
EPOCHS = 20
D_MODEL = 128
D_FF = 512
N = 2
HEADS = 4
LR = 0.0003
PAD, BOS, EOS = 0, 1, 2
CKPT_DIR = r'E:\pytorch-project\checkpoints'
os.makedirs(CKPT_DIR, exist_ok=True)

print('Loading preprocessed data...')
data = torch.load(r'E:\pytorch-project\data\preprocessed.pt')
train_en, train_zh = data['train_en'], data['train_zh']
val_en, val_zh = data['val_en'], data['val_zh']
en_i2w, zh_i2w = data['en_i2w'], data['zh_i2w']
print('Train: %d  Val: %d' % (len(train_en), len(val_en)))

class DS(Dataset):
    def __init__(self, en, zh):
        self.data = [(torch.tensor([BOS]+e+[EOS]), torch.tensor([BOS]+z+[EOS])) for e,z in zip(en,zh)]
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate(batch):
    max_src = max(len(s) for s,_ in batch)
    max_tgt = max(len(t) for _,t in batch)
    src = torch.stack([torch.cat([s, torch.zeros(max_src-len(s), dtype=torch.long)]) for s,_ in batch])
    tin = torch.stack([torch.cat([t[:-1], torch.zeros(max_tgt-len(t)+1, dtype=torch.long)]) for _,t in batch])
    tout = torch.stack([torch.cat([t[1:], torch.zeros(max_tgt-len(t)+1, dtype=torch.long)]) for _,t in batch])
    return src.long(), tin.long(), tout.long()

train_loader = DataLoader(DS(train_en, train_zh), BATCH, True, collate_fn=collate)
val_loader = DataLoader(DS(val_en, val_zh), BATCH, False, collate_fn=collate)

model = make_model(len(en_i2w), len(zh_i2w), N, D_MODEL, D_FF, HEADS, 0.1).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.NLLLoss(ignore_index=PAD)

# 检测是否有 checkpoint 可恢复
start_epoch = 0
ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, 'checkpoint_epoch_*.pt')),
                    key=lambda x: int(x.split('_')[-1].replace('.pt','')))
if ckpt_files:
    latest = ckpt_files[-1]
    start_epoch = int(latest.split('_')[-1].replace('.pt',''))
    ckpt = torch.load(latest)
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['optimizer'])
    print('Resumed from %s (epoch %d)' % (latest, start_epoch))

print('Params: %d' % sum(p.numel() for p in model.parameters()))
print()

for epoch in range(start_epoch, EPOCHS):
    model.train()
    ls, t0 = 0, time.time()
    for src, tin, tout in train_loader:
        src, tin, tout = src.to(DEVICE), tin.to(DEVICE), tout.to(DEVICE)
        sm = (src!=0).unsqueeze(1)
        tm = subsequent_mask(tin.size(1)).to(DEVICE)
        pred = model.generator(model.decode(model.encode(src,sm), sm, tin, tm))
        loss = crit(pred.reshape(-1,len(zh_i2w)), tout.reshape(-1))
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
    
    dt = time.time()-t0
    train_loss = ls/len(train_loader)
    val_loss = vls/len(val_loader)
    print('Epoch %2d/%d | train=%.3f | val=%.3f | ppl=%.1f | %.1fs' % (
        epoch+1, EPOCHS, train_loss, val_loss, math.exp(val_loss), dt))
    
    # 每轮保存 checkpoint
    torch.save({
        'epoch': epoch+1, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
        'train_loss': train_loss, 'val_loss': val_loss,
    }, os.path.join(CKPT_DIR, 'checkpoint_epoch_%d.pt' % (epoch+1)))

# 全部跑完 → 保存最终模型并清理旧 checkpoint
torch.save({'model':model.state_dict(),'en_i2w':en_i2w,'zh_i2w':zh_i2w,
            'd_model':D_MODEL,'d_ff':D_FF,'n':N,'heads':HEADS},
           r'E:\pytorch-project\lihongyi-transformer\translation_model.pt')
# 可选的: 删掉旧的 checkpoint 省空间
# for f in glob.glob(os.path.join(CKPT_DIR, 'checkpoint_epoch_*.pt')):
#     os.remove(f)
print('Done! Final model saved.')
