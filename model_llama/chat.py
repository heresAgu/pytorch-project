# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from load_pretrained import load_pretrained
from transformers import AutoTokenizer

def main():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tinyllama_weights")
    if not os.path.exists(model_path):
        print("no tinyllama_weights, download first")
        return
    print("Loading TinyLlama (CUDA:" + str(torch.cuda.is_available()) + ")...")
    model = load_pretrained(model_path)
    device = next(model.parameters()).device
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.pad_token = tok.eos_token
    print("\n=== Chat ===\n")
    history = []
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input.strip():
            continue
        history.append({"role": "user", "content": user_input})
        prompt = ""
        for turn in history:
            r = turn["role"]
            prompt += "<|" + r + "|>\n" + turn["content"] + "</s>\n"
        prompt += "<|assistant|>\n"
        inputs = tok(prompt, return_tensors="pt")["input_ids"].to(device)
        out = model.generate(inputs, max_new_tokens=100, temperature=0.5, top_k=40, eos_token_id=tok.eos_token_id)
        reply = tok.decode(out[0], skip_special_tokens=True)
        reply = reply.split("<|assistant|>")[-1].strip()
        print("AI: " + reply)
        history.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    main()
