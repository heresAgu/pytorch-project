import torch, re, sys
sys.path.insert(0, r'E:\pytorch-project\lihongyi-transformer')
from transformer_model import make_model, subsequent_mask

DEVICE = 'cuda'

# 加载模型和词表
ckpt = torch.load(r'E:\pytorch-project\lihongyi-transformer\translation_model.pt')
en_i2w = ckpt['en_i2w']
zh_i2w = ckpt['zh_i2w']
en_w2i = {w:i for i,w in enumerate(en_i2w)}
zh_w2i = {w:i for i,w in enumerate(zh_i2w)}

model = make_model(len(en_i2w), len(zh_i2w), ckpt['n'], ckpt['d_model'], ckpt['d_ff'], ckpt['heads'], 0.1).to(DEVICE)
model.load_state_dict(ckpt['model'])
model.eval()

def tokenize(s):
    s = s.lower().strip()
    s = re.sub(r'([.,!?;:])', r' \1 ', s)
    return s.split()

def translate(en_text, max_len=20):
    tokens = tokenize(en_text)
    src_ids = [en_w2i.get(t, 3) for t in tokens[:max_len-2]]  # 3=<unk>
    src = torch.tensor([[1] + src_ids + [2]]).to(DEVICE)
    smask = (src != 0).unsqueeze(1)
    
    mem = model.encode(src, smask)
    ys = torch.ones(1, 1, dtype=torch.long).to(DEVICE)  # BOS
    for _ in range(30):
        tmask = subsequent_mask(ys.size(1)).to(DEVICE)
        dec_out = model.decode(mem, smask, ys, tmask)
        next_id = model.generator(dec_out[:, -1]).argmax(-1)
        ys = torch.cat([ys, next_id.unsqueeze(1)], 1)
        if next_id.item() == 2:
            break
    
    zh_ids = ys[0, 1:-1].tolist()  # 去掉 BOS 和 EOS
    zh_text = ''.join(zh_i2w[i] for i in zh_ids if i < len(zh_i2w))
    return zh_text

# 测试
tests = [
    'hi.',
    'how are you?',
    'i love you.',
    'what is your name?',
    'thank you.',
    'good morning.',
    'see you tomorrow.',
]

print('='*50)
print('英→中 翻译测试')
print('='*50)
for t in tests:
    zh = translate(t)
    print('%s -> %s' % (t, zh))

print()
print('自定义翻译:')
while True:
    s = input('输入英文: ').strip()
    if s.lower() in ('q', 'quit', 'exit'):
        break
    if not s:
        continue
    zh = translate(s)
    print('翻译:', zh)
    print()
