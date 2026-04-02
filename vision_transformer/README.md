# Vision Transformer (ViT) — 从零实现带详细注释

## 简介

本目录包含一个 **从零实现的基础 Vision Transformer (ViT)** 的完整 Python 代码，每一行代码都附有中文注释，帮助深度学习初学者系统完整地理解 ViT 架构。

ViT 论文：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)（Dosovitskiy et al., Google Brain, 2021）

---

## 架构概览

```
输入图像 (N, 3, 224, 224)
        ↓
  ① Patch Embedding       将图像切分为 16×16 的块，线性投影到 D 维
        ↓
  ② 拼接 CLS Token        在序列开头加入可学习的分类 Token
        ↓
  ③ 加 Position Embedding 给每个位置加入可学习的位置编码
        ↓
  ④ Dropout
        ↓
  ⑤ × L Transformer Encoder Block
     ├─ LayerNorm
     ├─ Multi-Head Self-Attention (MHSA)
     ├─ 残差连接
     ├─ LayerNorm
     ├─ Feed-Forward Network (FFN / MLP)
     └─ 残差连接
        ↓
  ⑥ 取 CLS Token 输出     形状 (N, D)
        ↓
  ⑦ LayerNorm + Linear     分类头
        ↓
分类预测 (N, num_classes)
```

---

## 文件结构

```
vision_transformer/
├── vit.py            # 完整 ViT 实现，每行带中文注释
├── test_vit.py       # 测试套件，验证各模块的输入输出形状
├── requirements.txt  # Python 依赖
└── README.md         # 本文件
```

---

## 快速开始

### 1. 安装依赖

```bash
# 仅 CPU 环境 (推荐先用这个)
pip install torch torchvision

# 或根据你的 CUDA 版本选择:
# CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. 运行演示

```bash
cd vision_transformer
python vit.py
```

预期输出:
```
============================================================
Vision Transformer (ViT) 演示
============================================================

使用设备: cpu

模型总参数量:   5,717,770
可训练参数量:   5,717,770

输入张量形状: torch.Size([4, 3, 224, 224])  (N=4, C=3, H=224, W=224)
输出张量形状: torch.Size([4, 10])  (N=4, num_classes=10)
预测类别:     [3, 7, 1, 5]

图像块数量: 196  (每行 14 个，共 14 行)
Transformer 序列长度 (含 CLS Token): 197

演示完成！✅
```

### 3. 运行测试

```bash
cd vision_transformer
python test_vit.py
```

---

## 模块说明

### `PatchEmbedding` — 图像块嵌入

将图像切分为不重叠的图像块，并投影到 D 维嵌入空间。

- 使用 `Conv2d(kernel_size=patch_size, stride=patch_size)` 同时完成切分和投影
- 输入: `(N, C, H, W)` → 输出: `(N, L, D)`，其中 `L = (H/P)²`

### `MultiHeadSelfAttention` — 多头自注意力

计算序列中各 token 之间的注意力关系。

- 公式: `Attention(Q, K, V) = softmax(QK^T / √d) × V`
- 多头机制：将 D 维分成 h 个子空间独立计算，再拼接

### `FeedForward` — 前馈神经网络

对每个 token 独立进行非线性变换。

- 结构: `Linear(D→4D) → GELU → Dropout → Linear(4D→D) → Dropout`

### `TransformerEncoderBlock` — Transformer 编码器块

将 MHSA 和 FFN 组合，使用 Pre-Norm 和残差连接。

- `x = x + MHSA(LayerNorm(x))`
- `x = x + FFN(LayerNorm(x))`

### `VisionTransformer` — 完整模型

组装所有组件，提供端到端的图像分类功能。

---

## 预置配置

| 配置       | `embed_dim` | `depth` | `num_heads` | 参数量  |
|------------|------------|---------|------------|--------|
| `vit_tiny` | 192        | 12      | 3          | ~5.7M  |
| `vit_small`| 384        | 12      | 6          | ~22M   |
| `vit_base` | 768        | 12      | 12         | ~86M   |
| `vit_large`| 1024       | 24      | 16         | ~307M  |

```python
from vit import vit_tiny, vit_small, vit_base
import torch

# 创建 ViT-Tiny 用于 10 分类
model = vit_tiny(num_classes=10)

# 输入一张 224×224 RGB 图像
x = torch.randn(1, 3, 224, 224)
logits = model(x)           # 形状: (1, 10)
pred = logits.argmax(dim=1) # 预测类别
```

---

## 关键概念速查

| 概念 | 解释 |
|------|------|
| **Patch** | 图像被切分成的小块，每块大小 `P×P` |
| **CLS Token** | 可学习的分类 Token，聚合整幅图像信息 |
| **Position Embedding** | 告知模型 token 位置的可学习编码 |
| **Self-Attention** | 序列中每个 token 关注所有 token 的机制 |
| **多头** | 在多个子空间独立计算注意力，捕获不同层面的关系 |
| **残差连接** | `y = x + f(x)`，缓解梯度消失，加速训练 |
| **LayerNorm** | 对特征维度归一化，稳定训练 |
| **GELU** | 平滑非线性激活函数，比 ReLU 在 Transformer 中表现更好 |
