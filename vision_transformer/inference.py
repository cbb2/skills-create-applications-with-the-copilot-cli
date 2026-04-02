"""
inference.py — ViT CIFAR-10 推理脚本
======================================
功能:
  1. 加载训练好的 checkpoint
  2. 对单张图片或整个文件夹的图片进行分类
  3. 显示 Top-K 预测结果和置信度

使用方式:
  # 对单张图片推理
  python inference.py --image path/to/image.jpg

  # 对文件夹内所有图片推理
  python inference.py --image-dir path/to/folder/

  # 指定 checkpoint 文件
  python inference.py --image img.jpg --checkpoint checkpoints/best_model.pth

  # 显示 Top-3 预测
  python inference.py --image img.jpg --top-k 3
"""

import argparse
import os
import sys
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CIFAR10_CLASSES, CIFAR10_CLASSES_ZH, CIFAR10_MEAN, CIFAR10_STD
from vit import VisionTransformer


# ============================================================
# 推理变换（与验证集相同，无随机增强）
# ============================================================

def get_inference_transform(image_size: int = 224) -> transforms.Compose:
    """
    推理时的图像预处理变换
    与训练验证集变换一致（保证分布相同）
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ============================================================
# 模型加载
# ============================================================

def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[VisionTransformer, dict]:
    """
    从 checkpoint 文件加载模型

    返回:
      (model, args_dict): 模型实例 + 训练时的参数字典
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到 checkpoint 文件: {checkpoint_path}\n"
            f"请先运行 train.py 训练模型，或检查路径是否正确。"
        )

    # 加载 checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})

    # 根据保存的超参数重建模型结构
    model = VisionTransformer(
        image_size=saved_args.get("image_size", 224),
        patch_size=saved_args.get("patch_size", 16),
        in_channels=3,
        num_classes=10,
        embed_dim=saved_args.get("embed_dim", 192),
        depth=saved_args.get("depth", 12),
        num_heads=saved_args.get("num_heads", 3),
        mlp_ratio=saved_args.get("mlp_ratio", 4.0),
        qkv_bias=True,
        pool="cls",
        mlp_head=False,
        dropout=0.0,   # 推理时 dropout 设为 0
    )

    # 加载模型权重
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()  # 切换到推理模式（关闭 Dropout）

    val_acc = checkpoint.get("val_acc", "未知")
    epoch   = checkpoint.get("epoch", "未知")
    print(f"✅ 加载 checkpoint: {checkpoint_path}")
    print(f"   训练 epoch: {epoch}  验证准确率: {val_acc:.4f}" if isinstance(val_acc, float) else "")

    return model, saved_args


# ============================================================
# 单张图片推理
# ============================================================

@torch.no_grad()
def predict_image(
    model: VisionTransformer,
    image_path: str,
    transform: transforms.Compose,
    device: torch.device,
    top_k: int = 3,
) -> List[Tuple[str, str, float]]:
    """
    对单张图片进行分类，返回 Top-K 预测结果

    参数:
      model      (VisionTransformer): 推理模型
      image_path (str): 图片文件路径
      transform  (transforms.Compose): 图像预处理变换
      device     (torch.device): 计算设备
      top_k      (int): 返回前 K 个最可能的类别

    返回:
      List[(class_en, class_zh, confidence)]: Top-K 预测结果
    """
    # 加载图片（兼容 PNG/JPG/BMP 等格式）
    try:
        image = Image.open(image_path).convert("RGB")  # 统一转为 RGB（去掉 alpha 通道）
    except Exception as e:
        raise ValueError(f"无法读取图片 {image_path}: {e}")

    # 预处理: PIL Image → (1, C, H, W) 张量
    tensor = transform(image).unsqueeze(0).to(device)  # unsqueeze(0) 加上 batch 维

    # 前向传播
    logits, _ = model(tensor)  # logits 形状: (1, 10)

    # 计算概率（softmax）
    probs = F.softmax(logits[0], dim=0)  # (10,) 概率分布

    # 取 Top-K
    top_probs, top_indices = probs.topk(min(top_k, len(CIFAR10_CLASSES)))

    results = []
    for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
        results.append((CIFAR10_CLASSES[idx], CIFAR10_CLASSES_ZH[idx], prob))

    return results


# ============================================================
# 主推理函数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViT CIFAR-10 推理脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None,
                        help="单张图片路径")
    parser.add_argument("--image-dir", type=str, default=None,
                        help="图片目录（对目录内所有图片推理）")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "checkpoints", "best_model.pth"),
                        help="checkpoint 文件路径")
    parser.add_argument("--top-k", type=int, default=3,
                        help="显示 Top-K 预测结果")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备: auto/cpu/cuda")
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    # 加载模型
    model, saved_args = load_model(args.checkpoint, device)
    image_size = saved_args.get("image_size", 224)
    transform = get_inference_transform(image_size)

    print(f"\n使用设备: {device}")
    print(f"图像尺寸: {image_size}×{image_size}")
    print()

    # 收集要推理的图片列表
    image_paths: List[str] = []

    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ 图片不存在: {args.image}")
            sys.exit(1)
        image_paths.append(args.image)

    if args.image_dir:
        if not os.path.isdir(args.image_dir):
            print(f"❌ 目录不存在: {args.image_dir}")
            sys.exit(1)
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for fname in sorted(os.listdir(args.image_dir)):
            if os.path.splitext(fname)[1].lower() in supported_exts:
                image_paths.append(os.path.join(args.image_dir, fname))

    if not image_paths:
        print("请指定 --image 或 --image-dir 参数。")
        print("示例: python inference.py --image test.jpg")
        sys.exit(0)

    # 逐张推理
    print(f"开始推理 {len(image_paths)} 张图片...\n")
    print("=" * 60)

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        try:
            results = predict_image(model, img_path, transform, device, args.top_k)
            print(f"图片: {fname}")
            print(f"  预测结果 (Top-{args.top_k}):")
            for rank, (cls_en, cls_zh, conf) in enumerate(results, 1):
                bar = "█" * int(conf * 20)  # 简单的进度条
                print(f"    {rank}. {cls_en:12s} ({cls_zh})  {conf:.4f}  {bar}")
        except Exception as e:
            print(f"  ❌ 推理失败: {e}")
        print()

    print("=" * 60)
    print("推理完成！✅")


if __name__ == "__main__":
    main()
