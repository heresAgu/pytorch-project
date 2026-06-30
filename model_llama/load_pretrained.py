"""
load_pretrained.py - 将HuggingFace上的LLaMA权重加载到我们的LLaMA模型中

支持的模型:
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (推荐入门，无需申请)
  - meta-llama/Meta-Llama-3.2-1B (需HF授权)
  - meta-llama/Llama-2-7b-hf (需HF授权)
  - meta-llama/Meta-Llama-3-8B (需HF授权)

用法:
  python load_pretrained.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import torch
import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_llama import make_model


# HuggingFace参数名 -> 我们的参数名 映射表
# {i} 会被替换为层索引
HF_KEY_MAP = {
    # 词嵌入
    'model.embed_tokens.weight':               'tok_embedding.weight',
    # 每层的归一化 + 注意力 + FFN
    'model.layers.{i}.input_layernorm.weight':     'layers.{i}.attention_norm.weight',
    'model.layers.{i}.self_attn.q_proj.weight':    'layers.{i}.attention.wq.weight',
    'model.layers.{i}.self_attn.k_proj.weight':    'layers.{i}.attention.wk.weight',
    'model.layers.{i}.self_attn.v_proj.weight':    'layers.{i}.attention.wv.weight',
    'model.layers.{i}.self_attn.o_proj.weight':    'layers.{i}.attention.wo.weight',
    'model.layers.{i}.post_attention_layernorm.weight': 'layers.{i}.ffn_norm.weight',
    'model.layers.{i}.mlp.gate_proj.weight':       'layers.{i}.feed_forward.w1.weight',
    'model.layers.{i}.mlp.up_proj.weight':         'layers.{i}.feed_forward.w3.weight',
    'model.layers.{i}.mlp.down_proj.weight':       'layers.{i}.feed_forward.w2.weight',
    # 最终归一化
    'model.norm.weight':                         'norm.weight',
    # 输出层
    'lm_head.weight':                            'output.weight',
}


# 各模型推荐配置（维度、层数、头数等）
MODEL_CONFIGS = {
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0': {
        'vocab_size': 32000, 'd_model': 2048, 'n_layers': 22,
        'n_heads': 32, 'n_kv_heads': 4, 'd_ff': 5632,
        'max_seq_len': 2048, 'rope_theta': 10000.0,
    },
    'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T': {
        'vocab_size': 32000, 'd_model': 2048, 'n_layers': 22,
        'n_heads': 32, 'n_kv_heads': 4, 'd_ff': 5632,
        'max_seq_len': 2048, 'rope_theta': 10000.0,
    },
    'meta-llama/Llama-2-7b-hf': {
        'vocab_size': 32000, 'd_model': 4096, 'n_layers': 32,
        'n_heads': 32, 'n_kv_heads': 32, 'd_ff': 11008,
        'max_seq_len': 4096, 'rope_theta': 10000.0,
    },
    'meta-llama/Meta-Llama-3-8B': {
        'vocab_size': 128256, 'd_model': 4096, 'n_layers': 32,
        'n_heads': 32, 'n_kv_heads': 8, 'd_ff': 14336,
        'max_seq_len': 8192, 'rope_theta': 500000.0,
    },
    'meta-llama/Meta-Llama-3.2-1B': {
        'vocab_size': 128256, 'd_model': 2048, 'n_layers': 16,
        'n_heads': 32, 'n_kv_heads': 8, 'd_ff': 8192,
        'max_seq_len': 8192, 'rope_theta': 500000.0,
    },
    'meta-llama/Meta-Llama-3.2-3B': {
        'vocab_size': 128256, 'd_model': 3072, 'n_layers': 28,
        'n_heads': 24, 'n_kv_heads': 8, 'd_ff': 8192,
        'max_seq_len': 8192, 'rope_theta': 500000.0,
    },
}


def build_hf_to_our_name_mapping(num_layers):
    """
    根据层数构建完整的 HF参数名 -> 我们的参数名 映射字典
    """
    mapping = {}
    for hf_key, our_key in HF_KEY_MAP.items():
        if '{i}' in hf_key:
            # 每层有独立的参数
            for i in range(num_layers):
                hf_name = hf_key.replace('{i}', str(i))
                our_name = our_key.replace('{i}', str(i))
                mapping[hf_name] = our_name
        else:
            mapping[hf_key] = our_key
    return mapping


def load_pretrained(model_name, device=None, torch_dtype=None):
    """
    从HuggingFace加载预训练LLaMA权重到我们的模型

    参数:
        model_name: HF模型名称（如 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'）
        device: 加载设备
        torch_dtype: 权重数据类型

    返回:
        model: 加载好权重的LLaMA模型
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'自动选择设备: {device}')
    if torch_dtype is None:
        torch_dtype = torch.float16 if device == 'cuda' else torch.float32

    try:
        from transformers import AutoModelForCausalLM
    except ImportError:
        raise ImportError(
            '需要安装transformers: pip install transformers'
        )

    print(f'[1/4] 从HuggingFace加载模型: {model_name}')
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    hf_state = hf_model.state_dict()
    hf_config = hf_model.config
    num_layers = hf_config.num_hidden_layers

    # 确定模型配置
    if model_name in MODEL_CONFIGS:
        cfg = MODEL_CONFIGS[model_name]
    else:
        # 自动从HF config推断
        cfg = {
            'vocab_size': hf_config.vocab_size,
            'd_model': hf_config.hidden_size,
            'n_layers': hf_config.num_hidden_layers,
            'n_heads': hf_config.num_attention_heads,
            'n_kv_heads': getattr(hf_config, 'num_key_value_heads',
                                   hf_config.num_attention_heads),
            'd_ff': hf_config.intermediate_size,
            'max_seq_len': getattr(hf_config, 'max_position_embeddings', 2048),
            'rope_theta': getattr(hf_config, 'rope_theta', 10000.0),
        }

    print(f'[2/4] 创建模型实例 (dim={cfg["d_model"]}, '
          f'layers={cfg["n_layers"]}, heads={cfg["n_heads"]}, '
          f'kv_heads={cfg["n_kv_heads"]})')

    model = make_model(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        n_heads=cfg['n_heads'],
        n_kv_heads=cfg['n_kv_heads'],
        d_ff=cfg['d_ff'],
        max_seq_len=cfg['max_seq_len'],
        dropout=0.0,  # 推理时dropout=0
        multiple_of=1,
        rope_theta=cfg['rope_theta'],
    )
    # 设置RMSNorm epsilon（从config读取）
    for module in model.modules():
        if isinstance(module, type(model.norm)):
            module.eps = getattr(hf_config, 'rms_norm_eps', 1e-6)
    
    model.eval()

    print(f'[3/4] 映射权重参数...')
    name_map = build_hf_to_our_name_mapping(num_layers)

    our_state = model.state_dict()
    loaded = set()
    skipped = []

    for hf_name, hf_tensor in hf_state.items():
        if hf_name in name_map:
            our_name = name_map[hf_name]
            if our_name in our_state:
                # 检查形状是否匹配
                if hf_tensor.shape == our_state[our_name].shape:
                    our_state[our_name].copy_(hf_tensor.to(
                        our_state[our_name].dtype))
                    loaded.add(our_name)
                else:
                    skipped.append(
                        f'{hf_name} -> {our_name}: shape不匹配 '
                        f'{list(hf_tensor.shape)} vs '
                        f'{list(our_state[our_name].shape)}'
                    )
            else:
                skipped.append(f'{hf_name} -> {our_name}: 目标参数不存在')
        else:
            # 跳过不需要的key（如旋转编码缓存、buffer等）
            pass

    # 输出统计
    total = len(our_state)
    loaded_count = len(loaded)
    print(f'  成功加载: {loaded_count}/{total}')
    if skipped:
        print(f'  跳过 {len(skipped)} 个不匹配参数（部分可忽略）:')
        for s in skipped[:5]:
            print(f'    - {s}')
        if len(skipped) > 5:
            print(f'    ... 还有 {len(skipped)-5} 个')

    # 移到目标设备
    if device == "cuda":
        model = model.cuda()
        if torch_dtype == torch.float16:
            model = model.half()

    print(f'[4/4] 加载完成!')
    return model

def chat_demo(model, tokenizer, prompt, max_new_tokens=100, device='cpu'):
    """
    用加载好的模型跑一段对话demo
    """
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    input_ids = inputs['input_ids']

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_k=50,
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='加载预训练LLaMA权重')
    parser.add_argument('--model_name', type=str,
                        default='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                        help='HuggingFace模型名称')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda'],
                        help='运行设备')
    parser.add_argument('--prompt', type=str,
                        default='请用中文介绍一下什么是深度学习:',
                        help='测试提示词')
    parser.add_argument('--max_new_tokens', type=int, default=50,
                        help='最大生成长度')
    parser.add_argument('--list_models', action='store_true',
                        help='列出支持的模型')

    args = parser.parse_args()

    if args.list_models:
        print('支持的预配置模型:')
        for name, cfg in MODEL_CONFIGS.items():
            params = (cfg['d_model'] * cfg['n_layers'] * 4) / 1e9
            print(f'  {name}')
            print(f'    dim={cfg["d_model"]}, layers={cfg["n_layers"]}, '
                  f'heads={cfg["n_heads"]}, kv_heads={cfg["n_kv_heads"]}')
            print(f'    max_seq_len={cfg["max_seq_len"]}, '
                  f'rope_theta={cfg["rope_theta"]}')
        sys.exit(0)

    # 检查transformers
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print('需要安装transformers: pip install transformers')
        sys.exit(1)

    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'自动选择设备: {device}')
    elif device == 'cuda' and not torch.cuda.is_available():
        print('CUDA不可用，回退到CPU')
        device = 'cpu'

# device already set in load_pretrained

    # 加载模型
    model = load_pretrained(args.model_name, device=device,
                            torch_dtype=dtype)
# device already set in load_pretrained


    # 加载tokenizer
    print(f'加载tokenizer: {args.model_name}')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 对话测试
    print(f'\n提示词: {args.prompt}')
    response = chat_demo(model, tokenizer, args.prompt,
                         args.max_new_tokens, device)
    print(f'回复: {response}')
    print('\n完成!')
