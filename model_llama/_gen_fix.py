with open(r'E:\pytorch-project\model_llama\llama_model.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 给 generate 加 eos_token_id 参数，生成到 eos 就停
old = "    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):"
new = "    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):"
code = code.replace(old, new)

# 在 decode 循环里加 eos 判断
old_loop = "        # ---------- Step 2: Decode\xef\xbc\x88\xe9\x80\x90token\xe7\x94\x9f\xe6\x88\x90\xef\xbc\x8c\xe4\xbd\xbf\xe7\x94\xa8KV Cache\xef\xbc\x89----------\n"
new_loop = "        # ---------- Step 2: Decode ----------\n"
code = code.replace(old_loop, new_loop)

old_decode = """        for cur_pos in range(idx.shape[1] - 1, idx.shape[1] - 1 + max_new_tokens - 1):
            # \xe5\x8f\xaa\xe5\x8f\x96\xe6\x9c\x80\xe6\x96\xb0\xe7\x94\x9f\xe6\x88\x90\xe7\x9a\x841\xe4\xb8\xaatoken
            x = idx_next

            # \xe5\x89\x8d\xe5\x90\x91\xe4\xbc\xa0\xe6\x92\xad\xef\xbc\x8c\xe4\xbc\xa0\xe5\x85\xa5KV Cache\xe5\x92\x8c\xe5\xbd\x93\xe5\x89\x8d\xe4\xbd\x8d\xe7\xbd\xae
            logits, kv_caches = self(x, start_pos=cur_pos, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature

            # top-k\xe9\x87\x87\xe6\xa0\xb7
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)"""

new_decode = """        for cur_pos in range(idx.shape[1] - 1, idx.shape[1] - 1 + max_new_tokens - 1):
            x = idx_next
            logits, kv_caches = self(x, start_pos=cur_pos, kv_caches=kv_caches)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # \xe5\xa6\x82\xe6\x9e\x9c\xe7\x94\x9f\xe6\x88\x90\xe4\xba\x86eos\xef\xbc\x8c\xe6\x8f\x90\xe5\x89\x8d\xe5\x81\x9c\xe6\xad\xa2
            if eos_token_id is not None and idx_next.item() == eos_token_id:
                idx = torch.cat((idx, idx_next), dim=1)
                break

            idx = torch.cat((idx, idx_next), dim=1)"""

code = code.replace(old_decode, new_decode)

with open(r'E:\pytorch-project\model_llama\llama_model.py', 'w', encoding='utf-8') as f:
    f.write(code)
print("OK")
