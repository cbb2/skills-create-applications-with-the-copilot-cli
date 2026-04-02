"""
dataset.py — CIFAR-10 数据集加载与预处理
=========================================
数据集介绍:
  CIFAR-10 是深度学习领域最经典的图像分类基准数据集之一。
  - 来源: 多伦多大学 Alex Krizhevsky (2009)
  - 下载: torchvision 内置，首次运行时自动从官网下载 (~170MB)
  - 规模: 60,000 张 32×32 彩色图像，10 个类别
  - 划分: 50,000 张训练集 + 10,000 张测试集，每类各 6,000 张

10 个类别:
  0: airplane (飞机)
  1: automobile (汽车)
  2: bird (鸟)
  3: cat (猫)
  4: deer (鹿)
  5: dog (狗)
  6: frog (青蛙)
  7: horse (马)
  8: ship (船)
  9: truck (卡车)

为什么选 CIFAR-10?
  - 无需注册，torchvision 一行代码自动下载
  - 图像小 (32×32)，训练快，适合教学和实验
  - 10 类均衡，评估指标直观
  - ViT 在此数据集上有大量公开 benchmark，便于对比

ViT 在 32×32 图像上的注意事项:
  - 原始 ViT 设计用于 224×224 图像，直接用 32×32 效果不佳
  - 本实现采用两种策略之一:
    策略A: 将图像上采样到 224×224 (与原始 ViT 完全兼容，但训练较慢)
    策略B: 使用 patch_size=4，在 32×32 上切出 8×8=64 个块 (更轻量)
  - 默认使用策略A (IMAGE_SIZE=224) 以保证与 vit.py 的完全兼容性
  - 通过 IMAGE_SIZE 和 PATCH_SIZE 常量可自由切换
"""

import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ============================================================
# 数据集与预处理配置
# ============================================================

# 训练图像目标尺寸 (ViT 原版为 224；若使用 patch_size=4 可改为 32 加快训练)
IMAGE_SIZE: int = 224

# 图像块大小（与 vit.py 的 patch_size 保持一致）
PATCH_SIZE: int = 16

# CIFAR-10 的 RGB 均值与标准差（在完整 CIFAR-10 训练集上统计）
# 这些值来自官方统计，用于将像素值标准化到近似标准正态分布
CIFAR10_MEAN: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
CIFAR10_STD:  Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

# 10 个类别名称（index → 中文/英文）
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CIFAR10_CLASSES_ZH = [
    "飞机", "汽车", "鸟", "猫", "鹿",
    "狗", "青蛙", "马", "船", "卡车",
]

# 数据存储目录（相对于本文件所在目录）
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data")


# ============================================================
# 数据增强变换
# ============================================================

def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    训练集数据增强变换
    ------------------
    采用常规的图像分类数据增强策略：
      1. 随机水平翻转：以 50% 概率水平镜像，增加样本多样性
      2. 随机裁剪：先填充 4px 再随机裁回原尺寸，模拟位移不变性
      3. ColorJitter：随机调整亮度/对比度/饱和度，增强颜色鲁棒性
      4. Resize → 上采样到 image_size（ViT 原版需要 224×224）
      5. ToTensor：PIL Image → [0,1] 浮点张量
      6. Normalize：减均值除标准差，加速收敛

    参数:
      image_size (int): 目标图像尺寸，与 VisionTransformer 的 image_size 一致
    """
    return transforms.Compose([
        # 随机水平翻转（p=0.5）: 水平镜像是图像分类最基础的增强
        transforms.RandomHorizontalFlip(p=0.5),

        # 随机裁剪: 先用 0 填充 4 像素边框，再随机裁回原始尺寸
        # 模拟物体在图像中位置的小幅偏移，提升平移不变性
        transforms.RandomCrop(32, padding=4),

        # 颜色抖动: 随机调整亮度(±20%)、对比度(±20%)、饱和度(±20%)
        # 让模型对光照变化更鲁棒
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

        # 上采样到目标尺寸（CIFAR-10 原始为 32×32，ViT 需要更大尺寸）
        # BICUBIC 插值质量最高，适合放大
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),

        # PIL Image → FloatTensor，像素值从 [0,255] 转为 [0,1]
        transforms.ToTensor(),

        # 标准化: (x - mean) / std，将数据分布调整为近似标准正态
        # 有利于梯度计算，加速收敛
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


def get_val_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    验证集/测试集变换（无随机增强）
    --------------------------------
    验证和测试时不做随机增强，确保评估结果的确定性和可复现性。
    只做必要的尺寸调整和标准化。
    """
    return transforms.Compose([
        # 仅做尺寸调整（确定性操作，无随机性）
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),

        # PIL Image → FloatTensor
        transforms.ToTensor(),

        # 用与训练集相同的均值/标准差做标准化（非常重要！不能用验证集统计值）
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
    ])


# ============================================================
# 数据集加载器
# ============================================================

def get_cifar10_loaders(
    data_dir: str = DATA_DIR,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 64,
    num_workers: int = 2,
    val_split: float = 0.1,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载 CIFAR-10 数据集，返回训练/验证/测试 DataLoader

    数据划分策略:
      - 官方训练集 (50,000) 中取 val_split 比例作为验证集
        例如 val_split=0.1 → 训练 45,000 + 验证 5,000
      - 官方测试集 (10,000) 不动，最终评估用

    参数:
      data_dir    (str): 数据存储目录，不存在时自动创建并下载
      image_size  (int): 目标图像尺寸（需与 VisionTransformer 配置一致）
      batch_size  (int): 每批次样本数，影响内存用量和梯度估计质量
      num_workers (int): 数据加载工作进程数，通常设为 CPU 核心数的一半
      val_split   (float): 从训练集中划分为验证集的比例 (0~1)
      pin_memory  (bool): 固定内存，加速 CPU→GPU 数据传输（只有 GPU 训练时有效）

    返回:
      (train_loader, val_loader, test_loader) 三元组
    """
    os.makedirs(data_dir, exist_ok=True)  # 确保数据目录存在

    train_transform = get_train_transform(image_size)
    val_transform   = get_val_transform(image_size)

    # ---- 官方训练集（50,000 张，含数据增强）----
    # download=True: 若数据不存在则自动从 CIFAR-10 官网下载 (~170MB)
    full_train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    # ---- 从训练集中划分验证集 ----
    # 固定随机种子保证划分可复现（每次运行得到相同的训练/验证分割）
    total = len(full_train_dataset)                          # 50,000
    val_size   = int(total * val_split)                      # 例如 5,000
    train_size = total - val_size                            # 例如 45,000

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # 固定种子，可复现
    )

    # 验证集不需要随机增强，但 random_split 后无法直接换 transform
    # 解决方案: 重新加载一份无增强版本的完整训练集，取相同的索引
    full_train_no_aug = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,            # 已下载，不再重复下载
        transform=val_transform,   # 使用验证集 transform（无随机增强）
    )
    # 用相同的随机分割取出验证集部分（transform 不同，但样本相同）
    _, val_dataset = random_split(
        full_train_no_aug,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # 必须相同的种子
    )

    # ---- 官方测试集 (10,000 张，无增强) ----
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,    # train=False 加载官方测试集
        download=True,
        transform=val_transform,
    )

    # ---- 构建 DataLoader ----
    # shuffle=True: 训练时每 epoch 打乱顺序，防止模型记住样本顺序
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,   # 丢弃最后不完整的 batch，保持 batch size 稳定
    )

    # shuffle=False: 验证/测试不需要打乱，确保结果可复现
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,   # 验证时不需要梯度，batch size 可以更大
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(f"数据集加载完成:")
    print(f"  训练集: {len(train_dataset):,} 张")
    print(f"  验证集: {len(val_dataset):,} 张")
    print(f"  测试集: {len(test_dataset):,} 张")
    print(f"  图像尺寸: {image_size}×{image_size}")
    print(f"  Batch size: {batch_size} (训练) / {batch_size*2} (验证/测试)")

    return train_loader, val_loader, test_loader


# ============================================================
# 数据集信息展示（调试用）
# ============================================================

def show_dataset_info(data_dir: str = DATA_DIR) -> None:
    """
    打印数据集基本统计信息（用于调试/验证数据加载是否正确）
    """
    print("=" * 50)
    print("CIFAR-10 数据集信息")
    print("=" * 50)
    print(f"数据目录: {data_dir}")
    print(f"类别数量: {len(CIFAR10_CLASSES)}")
    print()
    print("类别列表:")
    for i, (en, zh) in enumerate(zip(CIFAR10_CLASSES, CIFAR10_CLASSES_ZH)):
        print(f"  {i}: {en} ({zh})")
    print()
    print(f"图像预处理 (训练集):")
    print(f"  原始尺寸: 32×32")
    print(f"  目标尺寸: {IMAGE_SIZE}×{IMAGE_SIZE}")
    print(f"  归一化均值: {CIFAR10_MEAN}")
    print(f"  归一化标准差: {CIFAR10_STD}")
    print("=" * 50)


if __name__ == "__main__":
    # 运行此文件时执行数据集下载验证
    show_dataset_info()
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=32,
        num_workers=0,  # Windows 下需要设为 0
    )

    # 取一个批次验证形状
    images, labels = next(iter(train_loader))
    print(f"\n训练批次形状验证:")
    print(f"  images: {tuple(images.shape)}  dtype={images.dtype}")
    print(f"  labels: {tuple(labels.shape)}  dtype={labels.dtype}")
    print(f"  像素值范围: [{images.min():.3f}, {images.max():.3f}]")
    print(f"  类别示例: {[CIFAR10_CLASSES[l] for l in labels[:8].tolist()]}")
    print("\n✅ 数据集加载验证通过！")
