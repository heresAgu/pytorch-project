import torch, math, time, os
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

import sys
sys.path.insert(0, r'E:\pytorch-project\lihongyi-transformer')
from transformer_model import make_model, subsequent_mask

# 配置
VOCAB = 54
BATCH = 32
EPOCHS = 20
D_MODEL = 128
D_FF = 512
N = 3
HEADS = 4
LR = 0.0003
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载数据
data_dir = r'E:\pytorch-project\data'
train_data = torch.load(os.path.join(data_dir, 'train.pt'))
val_data = torch.load(os.path.join(data_dir, 'val.pt'))

# 补齐到相同长度（为了 batch 训练）
max_len = max(max(len(s), len(t)) for s, t in train_data + val_data)

def collate(batch):
    pad = 0
    src = []
    tgt_in = []
    tgt_out = []
    for s, t in batch:
        src.append(torch.cat([s, torch.zeros(max_len - len(s), dtype=torch.long)]))
        # t: [BOS, ids..., EOS]
        tgt_in.append(torch.cat([t[:-1], torch.zeros(max_len + 1 - len(t), dtype=torch.long)]))
        tgt_out.append(torch.cat([t[1:], torch.zeros(max_len + 1 - len(t), dtype=torch.long)]))
    return torch.stack(src).long(), torch.stack(tgt_in).long(), torch.stack(tgt_out).long()

train_loader = DataLoader(train_data, BATCH, True, collate_fn=collate)
val_loader = DataLoader(val_data, BATCH, False, collate_fn=collate)

# 模型
model = make_model(VOCAB, VOCAB, N, D_MODEL, D_FF, HEADS, 0.1).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
crit = nn.NLLLoss(ignore_index=0)

print('参数: %d' % sum(p.numel() for p in model.parameters()))
print('设备: %s' % DEVICE)
print('词表: %d, 序列最大长度: %d' % (VOCAB, max_len))
print()

for epoch in range(EPOCHS):
    model.train()
    loss_sum, t0 = 0, time.time()
    for src, tin, tout in train_loader:
        src, tin, tout = src.to(DEVICE), tin.to(DEVICE), tout.to(DEVICE)
        smask = (src != 0).unsqueeze(1)
        tmask = subsequent_mask(tin.size(1)).to(DEVICE)
        mem = model.encode(src, smask)
        dec = model.decode(mem, smask, tin, tmask)
        pred = model.generator(dec)
        loss = crit(pred.reshape(-1, VOCAB), tout.reshape(-1))
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_sum += loss.item()
    train_loss = loss_sum / len(train_loader)

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for src, tin, tout in val_loader:
            src, tin, tout = src.to(DEVICE), tin.to(DEVICE), tout.to(DEVICE)
            smask = (src != 0).unsqueeze(1)
            tmask = subsequent_mask(tin.size(1)).to(DEVICE)
            mem = model.encode(src, smask)
            dec = model.decode(mem, smask, tin, tmask)
            pred = model.generator(dec)
            loss = crit(pred.reshape(-1, VOCAB), tout.reshape(-1))
            val_loss += loss.item()
    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1:2d} | train loss={train_loss:.4f} | val loss={val_loss:.4f} | ppl={math.exp(val_loss):.1f} | time={time.time()-t0:.1f}s')

# 测试推理
model.eval()
test_src = torch.tensor([[1, 7, 3, 9, 5, 2]]).to(DEVICE)  # BOS + [7,3,9,5] + EOS
smask = (test_src != 0).unsqueeze(1)
mem = model.encode(test_src, smask)
ys = torch.ones(1, 1, dtype=torch.long).to(DEVICE)
for _ in range(12):
    tmask = subsequent_mask(ys.size(1)).to(DEVICE)
    dec = model.decode(mem, smask, ys, tmask)
    nw = model.generator(dec[:, -1]).argmax(-1)
    ys = torch.cat([ys, nw.unsqueeze(1)], 1)
    if nw.item() == 2: break

exp = test_src[0].flip(0).tolist()
out = ys[0, 1:].tolist()
print(f'\n测试:')
print(f'  输入: {test_src[0].tolist()}')
print(f'  输出: {out}')
print(f'  期望: {exp}')
print(f'  反转正确: {out == exp}')

torch.save(model.state_dict(), os.path.join(os.path.dirname(__file__), 'transformer_demo.pt'))
print('\n已保存到 transformer_demo.pt')
