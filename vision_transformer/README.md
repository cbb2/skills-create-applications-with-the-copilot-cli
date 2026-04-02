# Vision Transformer (ViT) — 从零实现带详细注释

## 简介

本目录包含一个 **从零实现的基础 Vision Transformer (ViT)** 的完整 Python 代码，每一行代码都附有中文注释，帮助深度学习初学者系统完整地理解 ViT 架构。

本实现参考了 GitHub 上最顶级的开源 ViT 实现，综合了它们各自的优点：

| 参考仓库 | ⭐ | 贡献点 |
|---------|------|--------|
| [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) | 25k | 合并 QKV 投影、Mean Pooling 选项 |
| [labmlai/annotated_deep_learning](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | 66k | 2 层 MLP 分类头（原论文设置）|
| [jeonsworld/ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) | 2k | 注意力权重可视化、`eps=1e-6`|
| [huggingface/pytorch-image-models (timm)](https://github.com/huggingface/pytorch-image-models) | 36k | `qkv_bias` 参数、权重初始化策略 |

论文：[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)（Dosovitskiy et al., Google Brain, 2021）

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
  ⑤ × depth 个 Transformer Encoder Block
     ├─ LayerNorm (eps=1e-6)
     ├─ Multi-Head Self-Attention (MHSA, 支持 qkv_bias, 可返回注意力权重)
     ├─ 残差连接
     ├─ LayerNorm (eps=1e-6)
     ├─ Feed-Forward Network (FFN / MLP, GELU 激活)
     └─ 残差连接
        ↓
  ⑥ 池化 (CLS Token 或 Mean Pooling 二选一)  → (N, D)
        ↓
  ⑦ LayerNorm
        ↓
  ⑧ 分类头 (单层 Linear 或 2层 MLP)
        ↓
分类预测 (N, num_classes)
```

---

## 文件结构

```
vision_transformer/
├── vit.py            # 完整 ViT 实现，每行带中文注释（8 个部分）
├── test_vit.py       # 测试套件（24 个测试），验证各模块正确性
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
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
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

模型总参数量:   5,526,346
可训练参数量:   5,526,346

输入张量形状: (4, 3, 224, 224)  (N=4, C=3, H=224, W=224)
输出张量形状: (4, 10)  (N=4, num_classes=10)
预测类别:     [8, 8, 8, 8]

图像块数量: 196  (每行 14 个，共 14 行)
Transformer 序列长度 (含 CLS Token): 197

---- 注意力权重可视化演示 ----
最后一层注意力权重形状: (3, 197, 197)
  第0头对 CLS Token 的注意力分布 (前5个位置):
  [...]

---- Mean Pooling 变体演示 ----
Mean Pooling 输出形状: (4, 10)

演示完成！✅
```

### 3. 运行测试 (24 个测试全部通过 ✅)

```bash
cd vision_transformer
python test_vit.py
```

---

## 模块说明 (vit.py 共 8 个部分)

### Part 1 · `PatchEmbedding` — 图像块嵌入

将图像切分为不重叠的图像块，并投影到 D 维嵌入空间。

- 使用 `Conv2d(kernel_size=patch_size, stride=patch_size)` 同时完成切分和投影（labmlai 有详细解释）
- 输入: `(N, C, H, W)` → 输出: `(N, L, D)`，其中 `L = (H/P)²`

### Part 2 · `MultiHeadSelfAttention` — 多头自注意力

计算序列中各 token 之间的注意力关系。

- 公式: `Attention(Q, K, V) = softmax(QK^T / √d) × V`
- **合并 QKV 投影** (lucidrains 风格)，比分开的 Q/K/V 更高效
- **`qkv_bias`**: 可选偏置参数（原始论文使用，timm 也支持）
- **`return_attn_weights`**: 可返回注意力权重矩阵用于可视化（jeonsworld 风格）
- 多头机制：将 D 维分成 h 个子空间独立计算，再拼接

### Part 3 · `FeedForward` — 前馈神经网络

对每个 token 独立进行非线性变换。

- 结构: `Linear(D→4D) → GELU → Dropout → Linear(4D→D) → Dropout`

### Part 4 · `TransformerEncoderBlock` — Transformer 编码器块

将 MHSA 和 FFN 组合，使用 Pre-Norm 和残差连接。

- **Pre-Norm** (先 LayerNorm 再子层): 所有主流实现的共同选择，训练更稳定
- **`eps=1e-6`**: 比默认 1e-5 更数值稳定（jeonsworld & timm 均如此）
- `x = x + MHSA(LayerNorm(x))` → `x = x + FFN(LayerNorm(x))`

### Part 5 · `MLPHead` — 2 层 MLP 分类头

原论文预训练时使用的分类头（labmlai 实现）。

- 结构: `Linear(D→D) → GELU → Linear(D→num_classes)`

### Part 6 · `VisionTransformer` — 完整模型

组装所有组件，提供端到端的图像分类功能，支持：

- **`pool='cls'`**: CLS Token 池化（默认，原论文）
- **`pool='mean'`**: Mean Pooling（lucidrains 支持，有时效果更好）
- **`mlp_head=True`**: 2 层 MLP 分类头（原论文预训练设置）
- **`mlp_head=False`**: 单层线性分类头（fine-tuning 设置，默认）
- **`get_attention_map()`**: 获取注意力权重图（可视化）

### Part 7 · 工厂函数

| 函数 | `embed_dim` | `depth` | `num_heads` | 参数量 |
|------|------------|---------|------------|--------|
| `vit_tiny()` | 192 | 12 | 3 | ~5.5M |
| `vit_small()` | 384 | 12 | 6 | ~22M |
| `vit_base()` | 768 | 12 | 12 | ~86M |
| `vit_large()` | 1024 | 24 | 16 | ~307M |

> 所有配置的每头维度均为 **64**，这是原始 ViT 论文的标准设置。

### Part 8 · 演示代码

---

## 使用示例

```python
from vit import vit_tiny, vit_small, vit_base, VisionTransformer
import torch

# ---- 1. 使用工厂函数快速创建 ----
model = vit_tiny(num_classes=10)     # CIFAR-10
model = vit_small(num_classes=1000)  # ImageNet

# ---- 2. 自定义配置 ----
model = VisionTransformer(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=10,
    embed_dim=192,
    depth=12,
    num_heads=3,
    mlp_ratio=4.0,
    qkv_bias=True,      # 原论文使用 bias
    pool="cls",          # 或 "mean"
    mlp_head=False,      # fine-tuning 时用单层 Linear
    dropout=0.1,
)

# ---- 3. 前向传播 ----
x = torch.randn(4, 3, 224, 224)
logits, _ = model(x)          # 形状: (4, 10)
pred = logits.argmax(dim=1)   # 预测类别

# ---- 4. 获取注意力权重 (可视化) ----
logits, attn_list = model(x, return_attn_weights=True)
# attn_list: 每层一个 (N, h, S, S) 张量

# 单张图片的注意力图
single_img = torch.randn(1, 3, 224, 224)
attn_map = model.get_attention_map(single_img, layer_idx=-1)
# attn_map 形状: (h, 197, 197)  h=注意力头数
```

---

## 关键概念速查

| 概念 | 解释 |
|------|------|
| **Patch** | 图像被切分成的小块，每块大小 `P×P` |
| **CLS Token** | 可学习的分类 Token，聚合整幅图像信息 |
| **Mean Pooling** | 对所有图像块 token 求均值，作为 CLS Token 的替代 |
| **Position Embedding** | 告知模型 token 位置的可学习编码（Transformer 本身是置换不变的）|
| **Self-Attention** | 序列中每个 token 关注所有 token 的机制 |
| **多头** | 在多个子空间独立计算注意力，捕获不同层面的关系 |
| **Pre-Norm** | 先 LayerNorm 再子层，训练比 Post-Norm 更稳定 |
| **残差连接** | `y = x + f(x)`，缓解梯度消失，加速训练 |
| **LayerNorm** | 对特征维度归一化，稳定训练，`eps=1e-6` 更数值稳定 |
| **GELU** | 平滑非线性激活函数，比 ReLU 在 Transformer 中表现更好 |
| **qkv_bias** | QKV 投影的偏置项，原始论文默认为 True |
| **MLP Head** | 2 层分类头（预训练）vs 单层 Linear（fine-tuning）|

---

## 与主流开源实现的对比

| 特性 | 本实现 | lucidrains | labmlai | jeonsworld | timm |
|------|--------|-----------|---------|------------|------|
| 合并 QKV 投影 | ✅ | ✅ | ❌ | ❌ | ✅ |
| `qkv_bias` 选项 | ✅ | ❌ | ✅ | ✅ | ✅ |
| CLS / Mean 池化 | ✅ | ✅ | ❌ | ❌ | ✅ |
| 注意力权重返回 | ✅ | ❌ | ❌ | ✅ | ✅ |
| 2 层 MLP 头 | ✅ | ❌ | ✅ | ❌ | ✅ |
| `eps=1e-6` | ✅ | ❌ | ❌ | ✅ | ✅ |
| 每行中文注释 | ✅ | ❌ | ❌ | ❌ | ❌ |
