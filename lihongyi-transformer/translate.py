import torch, re, sys
sys.path.insert(0, r'E:\pytorch-project\lihongyi-transformer')
from transformer_model import make_model, subsequent_mask

DEVICE = 'cuda'
print('Loading model...')
ckpt = torch.load(r'E:\pytorch-project\lihongyi-transformer\translation_model.pt', map_location=DEVICE)
en_i2w, zh_i2w = ckpt['en_i2w'], ckpt['zh_i2w']
en_w2i = {w:i for i,w in enumerate(en_i2w)}

model = make_model(len(en_i2w), len(zh_i2w), ckpt['n'], ckpt['d_model'], ckpt['d_ff'], ckpt['heads'], 0.1).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

def translate(en_text):
    s = re.sub(r'([.,!?;:])', r' \1 ', en_text.lower().strip())
    ids = [en_w2i.get(t, 3) for t in s.split()[:18]]
    src = torch.tensor([[1]+ids+[2]]).to(DEVICE)
    sm = (src!=0).unsqueeze(1)
    mem = model.encode(src, sm)
    ys = torch.ones(1,1,dtype=torch.long).to(DEVICE)
    for _ in range(30):
        tm = subsequent_mask(ys.size(1)).to(DEVICE)
        nw = model.generator(model.decode(mem, sm, ys, tm)[:,-1]).argmax(-1)
        ys = torch.cat([ys, nw.unsqueeze(1)], 1)
        if nw.item()==2: break
    # 跳过特殊标记 (0=PAD, 1=BOS, 2=EOS, 3=UNK)
    result = []
    for i in ys[0,1:-1].tolist():
        if i >= 4 and i < len(zh_i2w):
            result.append(str(zh_i2w[i]))
    return ''.join(result)

print('English -> Chinese Translator')
print('(输入 q 退出)')
print('='*40)

while True:
    try:
        s = input('EN: ').strip()
    except:
        break
    if not s:
        continue
    if s.lower() == 'q':
        break
    zh = translate(s)
    print('ZH:', zh)
    print()
