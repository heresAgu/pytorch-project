"""
加载 ViT 预训练权重脚本

从 timm 库加载 ImageNet 预训练权重，自动将 key 映射到本项目的模型结构。
如果 timm 下载失败（如 SSL 问题），自动降级为 urllib 下载。

使用方法：
    python model_vit/load_pretrained.py                         # 下载+保存
    python model_vit/load_pretrained.py --num-classes 10        # 自定义类别
    python model_vit/load_pretrained.py --local path/to/weights.pth  # 加载本地
"""
import torch
import torch.nn as nn
import sys
import os
import argparse
import ssl
import urllib.request
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_vit.vit_model import make_vit

# ViT-Base/16 预训练权重下载地址
HF_VIT_URL = (
    "https://huggingface.co/timm/"
    "vit_base_patch16_224.augreg2_in21k_ft_in1k/"
    "resolve/main/model.safetensors"
)


def _load_weights_directly():
    """使用 urllib 直接下载 safetensors 权重（降级方案，解决SSL证书问题）"""
    import timm
    from safetensors.torch import load_file

    print("使用 urllib 直接下载权重（降级方案）...")

    #创建 timm 参考模型 获取标准 key 结构
    ref_model = timm.create_model("vit_base_patch16_224", pretrained=False)
    ref_state = ref_model.state_dict()

    ctx = ssl._create_unverified_context()
    req = urllib.request.Request(
        HF_VIT_URL, headers={"User-Agent": "Mozilla/5.0"}
    )

    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
        print(f"正在下载 (约 330MB)...")
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                tmp.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    pct = downloaded / total_size * 100
                    print(f"\r  进度: {pct:.1f}% ({downloaded/1024/1024:.0f}MB)", end="")

    print(f"\n  下载完成! 正在解析权重...")
    raw_state = load_file(tmp_path)
    os.unlink(tmp_path)

    #只保留与 timm 结构匹配的 key
    filtered = {}
    for k in raw_state:
        if k in ref_state and raw_state[k].shape == ref_state[k].shape:
            filtered[k] = raw_state[k]

    print(f"  读取 {len(raw_state)} 个权重，匹配 {len(filtered)} 个")
    return filtered


def _split_qkv(pretrained_state, model_state, num_layers):
    """将 timm 中 fused 的 QKV 权重拆分为三个独立线性层"""
    count = 0
    for i in range(num_layers):
        for k in [f"blocks.{i}.attn.qkv.weight", f"blocks.{i}.attn.qkv.bias"]:
            if k not in pretrained_state:
                continue
            is_bias = "bias" in k
            t = pretrained_state[k]
            q_p, k_p, v_p = torch.chunk(t, 3, dim=0)
            for j, part in enumerate([q_p, k_p, v_p]):
                our_key = f"encoder.layers.{i}.self_attn.linears.{j}.{'bias' if is_bias else 'weight'}"
                model_state[our_key] = part
                count += 1
    return count


def _map_timm_to_ours(pretrained_state, model, num_classes):
    """将 timm 格式的 key 映射到我们模型的 key 格式"""
    model_state = model.state_dict()
    num_layers = len(model.encoder.layers)
    total = 0

    #直接映射的 key
    direct_map = {
        "patch_embed.proj.weight": "patch_embed.proj.weight",
        "patch_embed.proj.bias": "patch_embed.proj.bias",
        "cls_token": "cls_token",
        "pos_embed": "pos_embed",
        "norm.weight": "encoder.norm.a_2",
        "norm.bias": "encoder.norm.b_2",
        "head.weight": "head.1.weight",
        "head.bias": "head.1.bias",
    }

    for timm_k, our_k in direct_map.items():
        if timm_k in pretrained_state and our_k in model_state:
            if pretrained_state[timm_k].shape == model_state[our_k].shape:
                model_state[our_k] = pretrained_state[timm_k]
                total += 1

    #逐 block 映射
    for i in range(num_layers):
        block_items = {
            f"blocks.{i}.norm1.weight": f"encoder.layers.{i}.sublayer.0.norm.a_2",
            f"blocks.{i}.norm1.bias": f"encoder.layers.{i}.sublayer.0.norm.b_2",
            f"blocks.{i}.norm2.weight": f"encoder.layers.{i}.sublayer.1.norm.a_2",
            f"blocks.{i}.norm2.bias": f"encoder.layers.{i}.sublayer.1.norm.b_2",
            f"blocks.{i}.attn.proj.weight": f"encoder.layers.{i}.self_attn.linears.3.weight",
            f"blocks.{i}.attn.proj.bias": f"encoder.layers.{i}.self_attn.linears.3.bias",
            f"blocks.{i}.mlp.fc1.weight": f"encoder.layers.{i}.feed_forward.w_1.weight",
            f"blocks.{i}.mlp.fc1.bias": f"encoder.layers.{i}.feed_forward.w_1.bias",
            f"blocks.{i}.mlp.fc2.weight": f"encoder.layers.{i}.feed_forward.w_2.weight",
            f"blocks.{i}.mlp.fc2.bias": f"encoder.layers.{i}.feed_forward.w_2.bias",
        }
        for timm_k, our_k in block_items.items():
            if timm_k in pretrained_state and our_k in model_state:
                if pretrained_state[timm_k].shape == model_state[our_k].shape:
                    model_state[our_k] = pretrained_state[timm_k]
                    total += 1

    #QKV 拆分
    total += _split_qkv(pretrained_state, model_state, num_layers)

    return model_state, total


def load_pretrained_weights(model, num_classes=1000, model_name="vit_base_patch16_224"):
    """
    加载预训练权重到 ViT 模型中
    自动尝试: timm下载 -> urllib降级下载
    """
    try:
        import timm
    except ImportError:
        print("正在安装 timm...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "timm", "-q"])
        import timm

    pretrained_state = None
    source = ""

    #方案一：timm 自动下载
    try:
        print(f"尝试从 timm 下载: {model_name}")
        ref = timm.create_model(model_name, pretrained=True)
        pretrained_state = ref.state_dict()
        source = "timm"
    except Exception as e:
        print(f"  timm 失败: {type(e).__name__}")
        #方案二：urllib 降级
        try:
            pretrained_state = _load_weights_directly()
            source = "urllib"
        except Exception as e2:
            print(f"\n所有下载方式均失败!")
            print(f"请手动下载: {HF_VIT_URL}")
            print(f"然后运行: python {__file__} --local model.safetensors")
            return model

    #执行 key 映射
    model_state, loaded = _map_timm_to_ours(pretrained_state, model, num_classes)
    model.load_state_dict(model_state, strict=False)

    total_params = sum(p.numel() for p in model.parameters())
    head_msg = "已加载(ImageNet)" if num_classes == 1000 else "已跳过(自定义类别)"
    print(f"加载 {loaded}/{len(model_state)} 个权重 ({source})")
    print(f"分类头: {head_msg}")
    print(f"参数量: {total_params/1e6:.2f}M")
    return model


def load_from_local(model, pth_path, num_classes=1000):
    """从本地 .pth / .safetensors 文件加载（自动识别 timm/本项目 两种key格式）"""
    import timm

    print(f"从本地加载: {pth_path}")

    #读取权重文件
    if pth_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        raw_state = load_file(pth_path)
    else:
        raw_state = torch.load(pth_path, map_location="cpu", weights_only=True)

    #判断 key 格式：看第一个 key 是否含 "blocks." (timm格式)
    first_key = list(raw_state.keys())[0]
    is_timm_format = "blocks." in first_key

    if is_timm_format:
        #timm 格式 -> 映射到本项目
        ref_model = timm.create_model("vit_base_patch16_224", pretrained=False)
        ref_state = ref_model.state_dict()
        filtered = {}
        for k in raw_state:
            if k in ref_state and raw_state[k].shape == ref_state[k].shape:
                filtered[k] = raw_state[k]
        print(f"检测到 timm 格式, 匹配 {len(filtered)}/{len(raw_state)} 个")
        model_state, loaded = _map_timm_to_ours(filtered, model, num_classes)
    else:
        #本项目的 key 格式 -> 直接加载
        print(f"检测到本项目格式, 共 {len(raw_state)} 个权重")
        model_state = model.state_dict()
        loaded = 0
        for k in raw_state:
            if k in model_state and raw_state[k].shape == model_state[k].shape:
                model_state[k] = raw_state[k]
                loaded += 1

    model.load_state_dict(model_state, strict=False)
    print(f"成功加载 {loaded}/{len(model_state)} 个权重")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ViT 预训练权重加载器")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--local", type=str, default=None)
    parser.add_argument("--save", action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 40)
    print("ViT 预训练权重加载器")
    print("=" * 40)

    model = make_vit(num_classes=args.num_classes)
    print(f"ViT-Base/16 | 类别: {args.num_classes}")

    if args.local:
        model = load_from_local(model, args.local, num_classes=args.num_classes)
    else:
        model = load_pretrained_weights(model, num_classes=args.num_classes)

    #前向测试
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, 224, 224))
    print(f"前向测试: ({out.shape[0]}, {out.shape[1]})")

    #保存（以本项目 key 格式保存）
    if args.save:
        save_path = os.path.join(os.path.dirname(__file__), "vit_base_patch16_224_pretrained.pth")
        torch.save(model.state_dict(), save_path)
        print(f"已保存: {save_path}")
