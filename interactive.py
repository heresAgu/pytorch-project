import torch, sys
sys.path.insert(0, r'E:\pytorch-project\lihongyi-transformer')
from transformer_model import make_model, subsequent_mask

VOCAB = 54
DEVICE = 'cuda'

model = make_model(VOCAB, VOCAB, 3, 128, 512, 4, 0.1).to(DEVICE)
model.load_state_dict(torch.load(r'E:\pytorch-project\lihongyi-transformer\transformer_demo.pt'))
model.eval()

if len(sys.argv) < 2:
    print('用法: python interactive.py <数字1> <数字2> ...')
    print('例子: python interactive.py 3 10 5')
    print('注意: 数字范围 3~53, 不要包含 0 1 2 (它们是特殊标记)')
    sys.exit(1)

ids = [int(x) for x in sys.argv[1:]]
# 检查范围
for x in ids:
    if x < 3 or x > 53:
        print('错误: %d 超出范围 (3~53), 0=pad 1=BOS 2=EOS 是保留标记' % x)
        sys.exit(1)

print('输入:', ' '.join(str(x) for x in ids))

src = torch.tensor([[1] + ids + [2]]).to(DEVICE)
smask = (src != 0).unsqueeze(1)
mem = model.encode(src, smask)

ys = torch.ones(1, 1, dtype=torch.long).to(DEVICE)
for _ in range(20):
    tmask = subsequent_mask(ys.size(1)).to(DEVICE)
    dec_out = model.decode(mem, smask, ys, tmask)
    next_id = model.generator(dec_out[:, -1]).argmax(-1)
    ys = torch.cat([ys, next_id.unsqueeze(1)], 1)
    if next_id.item() == 2:
        break

output = ys[0, 1:-1].tolist()
print('输出:', ' '.join(str(x) for x in output))
print('正确!' if output == ids[::-1] else '不匹配, 期望: ' + ' '.join(str(x) for x in ids[::-1]))
