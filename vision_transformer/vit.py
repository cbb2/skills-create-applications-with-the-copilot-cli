"""
Vision Transformer (ViT) - 视觉变换器 完整实现
========================================
论文参考: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
作者: Dosovitskiy et al. (Google Brain), 2021
arXiv: https://arxiv.org/abs/2010.11929

架构总览:
  1. 将输入图片切分成固定大小的图像块 (Patch)
  2. 将每个图像块线性投影为嵌入向量 (Patch Embedding)
  3. 在序列开头加入可学习的分类 Token (CLS Token)
  4. 为每个位置加入可学习的位置编码 (Position Embedding)
  5. 通过多层 Transformer Encoder 处理序列
  6. 取 CLS Token 对应位置的输出，经过分类头得到预测结果

数学符号说明:
  N  = 图片批次大小 (batch size)
  H  = 图片高度
  W  = 图片宽度
  C  = 图片通道数 (RGB=3)
  P  = 每个图像块的边长 (patch size)
  L  = 图像块数量 = (H/P) × (W/P)
  D  = 嵌入维度 (embedding dimension / hidden size)
  h  = 多头注意力的头数 (number of heads)
  d  = 每个注意力头的维度 = D / h
"""

import math          # 数学运算，用于计算注意力缩放因子
import torch         # PyTorch 深度学习框架核心库
import torch.nn as nn  # PyTorch 神经网络模块，提供各种网络层


# ============================================================
# 第一部分：图像块嵌入 (Patch Embedding)
# ============================================================

class PatchEmbedding(nn.Module):
    """
    图像块嵌入层 (Patch Embedding)
    ----------------------------------
    功能:
      将形状为 (N, C, H, W) 的图像张量转换为
      形状为 (N, L, D) 的图像块嵌入序列。

    实现方式:
      使用一个卷积层 (Conv2d) 完成两件事:
        1. 将图像切分为不重叠的图像块 (kernel_size=patch_size, stride=patch_size)
        2. 将每个图像块线性投影到 D 维嵌入空间 (out_channels=embed_dim)

    参数:
      image_size  (int): 输入图像的边长 (假设图像为正方形)，例如 224
      patch_size  (int): 每个图像块的边长，例如 16
      in_channels (int): 输入图像的通道数，彩色图像为 3
      embed_dim   (int): 嵌入向量的维度 D，例如 768
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()  # 调用父类 nn.Module 的初始化方法，必须调用

        # 断言图像尺寸必须能被图像块尺寸整除，否则切分不均匀
        assert image_size % patch_size == 0, (
            f"图像尺寸 {image_size} 必须能被图像块尺寸 {patch_size} 整除"
        )

        self.image_size = image_size  # 保存图像尺寸
        self.patch_size = patch_size  # 保存图像块尺寸

        # 计算图像块的数量: (H/P) × (W/P)
        # 例如: 图像 224×224, 块大小 16×16 → 14×14 = 196 个块
        self.num_patches = (image_size // patch_size) ** 2

        # 定义投影卷积层:
        #   in_channels  = 输入图像通道数 (如 RGB=3)
        #   embed_dim    = 输出通道数，即嵌入维度 D
        #   kernel_size  = patch_size：每个卷积核恰好覆盖一个图像块
        #   stride       = patch_size：步长等于块大小，确保不重叠切分
        # 效果: 将 (N, C, H, W) → (N, D, H/P, W/P)
        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入:  x 形状为 (N, C, H, W)
                N = batch_size (批次大小)
                C = in_channels (通道数)
                H = image_size (图像高度)
                W = image_size (图像宽度)

        输出: 形状为 (N, L, D)
                L = num_patches (图像块数量)
                D = embed_dim (嵌入维度)
        """
        # 步骤1: 卷积投影
        # x: (N, C, H, W) → (N, D, H/P, W/P)
        x = self.projection(x)

        # 步骤2: 将空间维度展平
        # (N, D, H/P, W/P) → (N, D, L)   其中 L = (H/P)*(W/P)
        # flatten(2) 表示从第2个维度开始展平
        x = x.flatten(2)

        # 步骤3: 调换维度，使序列长度在第1维
        # (N, D, L) → (N, L, D)
        # transpose(1, 2) 交换第1维和第2维
        x = x.transpose(1, 2)

        return x  # 返回形状 (N, L, D) 的图像块嵌入序列


# ============================================================
# 第二部分：多头自注意力 (Multi-Head Self-Attention)
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制 (Multi-Head Self-Attention, MHSA)
    --------------------------------------------------
    自注意力原理:
      每个位置的特征通过查询 (Query)、键 (Key)、值 (Value) 三个矩阵变换，
      计算序列中所有位置对当前位置的"注意力权重"，然后加权聚合值向量。

    注意力计算公式:
      Attention(Q, K, V) = softmax(Q × K^T / √d) × V
      其中 d = D/h 是每个注意力头的维度

    多头机制:
      将 D 维嵌入分成 h 个子空间，在每个子空间中独立计算注意力，
      再将所有头的输出拼接后通过线性变换得到最终输出。
      好处：让模型能从不同的"角度"关注不同的信息。

    参数:
      embed_dim   (int): 嵌入维度 D
      num_heads   (int): 注意力头数 h，需满足 D % h == 0
      dropout     (float): Dropout 比率，用于防止过拟合
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        # 确保嵌入维度可以被头数整除
        assert embed_dim % num_heads == 0, (
            f"嵌入维度 {embed_dim} 必须能被注意力头数 {num_heads} 整除"
        )

        self.embed_dim = embed_dim   # 总嵌入维度 D
        self.num_heads = num_heads   # 注意力头数 h

        # 每个注意力头的维度: d = D / h
        # 例如: D=768, h=12 → d=64
        self.head_dim = embed_dim // num_heads

        # 注意力缩放因子: 1 / √d
        # 缩放的目的：防止点积结果过大导致 softmax 进入梯度极小的饱和区
        self.scale = self.head_dim ** -0.5

        # QKV 投影矩阵 (一次性计算 Q、K、V 三个矩阵，效率更高)
        # 输入维度: D，输出维度: 3*D (Q、K、V 各占 D 维)
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # 输出投影矩阵: 将多头注意力的输出映射回 D 维
        self.proj = nn.Linear(embed_dim, embed_dim)

        # Dropout 层，在注意力权重上应用，防止过拟合
        self.attn_dropout = nn.Dropout(dropout)

        # Dropout 层，在输出投影后应用
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入:  x 形状为 (N, L+1, D)
                N   = batch_size
                L+1 = 序列长度 (图像块数量 + 1个CLS token)
                D   = embed_dim

        输出: 形状为 (N, L+1, D)，与输入形状相同
        """
        N, S, D = x.shape  # 获取 batch size N，序列长度 S，嵌入维度 D

        # 步骤1: 计算 Q、K、V
        # x: (N, S, D) → qkv: (N, S, 3*D)
        qkv = self.qkv(x)

        # 步骤2: 重塑维度，分离出 Q、K、V 并划分多头
        # (N, S, 3*D)
        # → reshape → (N, S, 3, h, d)  按头分割
        # → permute → (3, N, h, S, d)  调整维度顺序便于后续操作
        qkv = qkv.reshape(N, S, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # 从 qkv 中分离出 Q、K、V，每个形状均为 (N, h, S, d)
        q, k, v = qkv.unbind(0)  # unbind(0) 在第0维上分解张量

        # 步骤3: 计算注意力分数 (scaled dot-product attention)
        # Q: (N, h, S, d)  K^T: (N, h, d, S)
        # 矩阵乘法结果: (N, h, S, S)
        # 每个元素 attn[n,h,i,j] 表示第 i 个 token 对第 j 个 token 的注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # 乘以缩放因子

        # 步骤4: 归一化注意力权重
        # 对最后一维 (即所有 token) 做 softmax，使权重之和为 1
        # (N, h, S, S) → (N, h, S, S)  (形状不变，但值变为概率分布)
        attn = attn.softmax(dim=-1)

        # 步骤5: 对注意力权重应用 Dropout (训练时随机丢弃部分注意力连接)
        attn = self.attn_dropout(attn)

        # 步骤6: 用注意力权重对值 V 进行加权聚合
        # attn: (N, h, S, S)  v: (N, h, S, d)
        # 结果: (N, h, S, d)
        x = attn @ v

        # 步骤7: 重新合并所有注意力头
        # (N, h, S, d) → transpose → (N, S, h, d)
        x = x.transpose(1, 2)
        # (N, S, h, d) → reshape → (N, S, h*d) = (N, S, D)
        # contiguous() 确保内存连续，reshape 操作需要
        x = x.contiguous().reshape(N, S, D)

        # 步骤8: 输出投影，将合并后的多头输出映射回 D 维
        # (N, S, D) → (N, S, D)
        x = self.proj(x)

        # 步骤9: 在输出上应用 Dropout
        x = self.proj_dropout(x)

        return x  # 形状: (N, S, D)


# ============================================================
# 第三部分：前馈神经网络 (Feed-Forward Network, FFN)
# ============================================================

class FeedForward(nn.Module):
    """
    前馈神经网络 (Feed-Forward Network / MLP Block)
    -----------------------------------------------
    结构:
      Linear(D → hidden_dim) → GELU激活 → Dropout → Linear(hidden_dim → D) → Dropout

    在 ViT 中, hidden_dim 通常是 embed_dim 的 4 倍。
    GELU (Gaussian Error Linear Unit) 是一种平滑的非线性激活函数，
    在 Transformer 中比 ReLU 表现更好。

    参数:
      embed_dim   (int): 输入/输出维度 D
      hidden_dim  (int): 隐藏层维度，通常为 4*D
      dropout     (float): Dropout 比率
    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()

        # 定义 MLP 的各个层，Sequential 将它们串联起来
        self.net = nn.Sequential(
            # 第一个线性层：扩展维度 D → hidden_dim (通常为 4*D)
            nn.Linear(embed_dim, hidden_dim),

            # GELU 激活函数：引入非线性，使网络能学习复杂特征
            # GELU(x) = x * Φ(x)，其中 Φ 是正态分布的累积分布函数
            nn.GELU(),

            # Dropout：随机将部分神经元输出置零，防止过拟合
            nn.Dropout(dropout),

            # 第二个线性层：压缩维度 hidden_dim → D
            nn.Linear(hidden_dim, embed_dim),

            # 再次应用 Dropout
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入:  x 形状为 (N, S, D)
        输出: 形状为 (N, S, D)，形状不变
        """
        return self.net(x)  # 顺序通过 Sequential 中的所有层


# ============================================================
# 第四部分：Transformer 编码器块 (Transformer Encoder Block)
# ============================================================

class TransformerEncoderBlock(nn.Module):
    """
    Transformer 编码器块 (Transformer Encoder Block)
    -------------------------------------------------
    结构 (Pre-LayerNorm 变体，实践中更稳定):
      x = x + MHSA(LayerNorm(x))    ← 残差连接 + 多头自注意力
      x = x + FFN(LayerNorm(x))     ← 残差连接 + 前馈神经网络

    残差连接 (Residual Connection):
      将输入直接加到子层输出上: output = x + sublayer(x)
      好处: 解决深层网络的梯度消失问题，允许梯度直接流过网络

    层归一化 (Layer Normalization):
      对每个样本的特征维度做归一化，稳定训练过程
      与 BatchNorm 不同，LayerNorm 对每个 token 的 D 维向量独立归一化

    参数:
      embed_dim   (int): 嵌入维度 D
      num_heads   (int): 注意力头数
      mlp_ratio   (float): FFN 隐藏层维度与 embed_dim 的比值，通常为 4.0
      dropout     (float): Dropout 比率
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # 第一个层归一化，在自注意力前应用 (Pre-Norm)
        self.norm1 = nn.LayerNorm(embed_dim)

        # 多头自注意力层
        self.attention = MultiHeadSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # 第二个层归一化，在前馈网络前应用 (Pre-Norm)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 前馈神经网络
        # 隐藏维度 = embed_dim * mlp_ratio (例如 768 * 4 = 3072)
        self.feed_forward = FeedForward(
            embed_dim=embed_dim,
            hidden_dim=int(embed_dim * mlp_ratio),
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入:  x 形状为 (N, S, D)
        输出: 形状为 (N, S, D)，形状不变
        """
        # 分支1: 自注意力 + 残差连接
        # 先对 x 做 LayerNorm，再通过 MHSA，最后加上原始 x (残差)
        x = x + self.attention(self.norm1(x))

        # 分支2: 前馈网络 + 残差连接
        # 先对 x 做 LayerNorm，再通过 FFN，最后加上原始 x (残差)
        x = x + self.feed_forward(self.norm2(x))

        return x  # 形状: (N, S, D)


# ============================================================
# 第五部分：完整 Vision Transformer 模型
# ============================================================

class VisionTransformer(nn.Module):
    """
    完整的 Vision Transformer 模型 (ViT)
    -------------------------------------
    完整数据流:
      输入图像 (N, C, H, W)
        ↓ PatchEmbedding
      图像块序列 (N, L, D)
        ↓ 拼接 CLS Token → (N, L+1, D)
        ↓ 加 Position Embedding → (N, L+1, D)
        ↓ Dropout
        ↓ N × TransformerEncoderBlock
      编码后序列 (N, L+1, D)
        ↓ 取 CLS Token 位置 → (N, D)
        ↓ LayerNorm
        ↓ 分类头 (Linear)
      分类预测 (N, num_classes)

    参数:
      image_size    (int): 输入图像边长，默认 224
      patch_size    (int): 图像块边长，默认 16
      in_channels   (int): 输入通道数，默认 3 (RGB)
      num_classes   (int): 分类数量
      embed_dim     (int): 嵌入维度 D，默认 768
      depth         (int): Transformer Encoder 块的数量，默认 12
      num_heads     (int): 注意力头数，默认 12
      mlp_ratio     (float): FFN 隐藏层比例，默认 4.0
      dropout       (float): Dropout 比率，默认 0.0
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ------ 1. 图像块嵌入层 ------
        # 将图像切分并投影到 D 维嵌入空间
        self.patch_embedding = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # 记录图像块总数量，方便后续使用
        num_patches = self.patch_embedding.num_patches

        # ------ 2. 分类 Token (CLS Token) ------
        # 形状为 (1, 1, D) 的可学习参数
        # 在序列开头拼接，充当整幅图像的全局表示
        # nn.Parameter 将张量注册为可学习参数，会被优化器更新
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ------ 3. 位置嵌入 (Position Embedding) ------
        # 形状为 (1, num_patches + 1, D) 的可学习参数
        # +1 是因为加入了 CLS Token
        # 位置嵌入让模型知道每个 token 在序列中的位置信息
        # (Transformer 本身是置换不变的，不区分位置，需要位置编码)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim)
        )

        # ------ 4. 嵌入 Dropout ------
        # 在图像块嵌入 + 位置嵌入后应用 Dropout
        self.embed_dropout = nn.Dropout(dropout)

        # ------ 5. Transformer 编码器 ------
        # 由 depth 个 TransformerEncoderBlock 串联组成
        # nn.ModuleList 将多个模块注册为子模块列表
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(depth)  # 循环创建 depth 个编码器块
        ])

        # ------ 6. 最终层归一化 ------
        # 在取出 CLS Token 输出后应用，稳定最终特征表示
        self.norm = nn.LayerNorm(embed_dim)

        # ------ 7. 分类头 ------
        # 将 D 维 CLS Token 特征映射到 num_classes 个类别得分
        self.head = nn.Linear(embed_dim, num_classes)

        # ------ 8. 权重初始化 ------
        # 对关键参数进行适当初始化，有助于训练稳定性
        self._init_weights()

    def _init_weights(self):
        """初始化模型权重"""
        # 用截断正态分布初始化 CLS Token，标准差 0.02 是 ViT 的常用设置
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 用截断正态分布初始化位置嵌入
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        # 遍历所有子模块，对线性层和层归一化进行初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 线性层权重：截断正态分布初始化
                nn.init.trunc_normal_(module.weight, std=0.02)
                # 线性层偏置：初始化为 0
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                # LayerNorm 偏置：初始化为 0
                nn.init.zeros_(module.bias)
                # LayerNorm 缩放因子：初始化为 1
                nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        输入:  x 形状为 (N, C, H, W)
                N = batch_size, C = 通道数, H/W = 图像尺寸

        输出: 形状为 (N, num_classes) 的分类预测分数 (logits)
        """
        N = x.shape[0]  # 获取 batch size

        # ---- 步骤1: 图像块嵌入 ----
        # (N, C, H, W) → (N, L, D)   L = num_patches
        x = self.patch_embedding(x)

        # ---- 步骤2: 拼接 CLS Token ----
        # cls_token 形状: (1, 1, D)
        # 通过 expand 扩展到当前 batch 大小: (N, 1, D)
        # expand 不会复制数据 (节省内存)，只是扩展视图
        cls_tokens = self.cls_token.expand(N, -1, -1)  # -1 表示该维度不变

        # 沿序列维度 (dim=1) 拼接 CLS Token 和图像块嵌入
        # (N, 1, D) + (N, L, D) → (N, L+1, D)
        x = torch.cat([cls_tokens, x], dim=1)

        # ---- 步骤3: 添加位置嵌入 ----
        # position_embedding 形状: (1, L+1, D)，会自动广播到 (N, L+1, D)
        # 直接相加，每个位置的嵌入向量都加上对应位置的位置编码
        x = x + self.position_embedding

        # ---- 步骤4: 嵌入 Dropout ----
        # 在嵌入层输出后应用 Dropout (仅在训练阶段有效)
        x = self.embed_dropout(x)

        # ---- 步骤5: 逐层通过 Transformer Encoder ----
        # 循环通过每个 Transformer 编码器块
        # 每个块的输入和输出形状均为 (N, L+1, D)
        for block in self.transformer_blocks:
            x = block(x)  # x: (N, L+1, D) → (N, L+1, D)

        # ---- 步骤6: 最终层归一化 ----
        # 对整个序列做 LayerNorm，形状不变: (N, L+1, D)
        x = self.norm(x)

        # ---- 步骤7: 提取 CLS Token 对应的输出 ----
        # CLS Token 位于序列第 0 个位置
        # x[:, 0, :]: 取所有样本的第 0 个 token → (N, D)
        cls_output = x[:, 0, :]

        # ---- 步骤8: 分类头 ----
        # (N, D) → (N, num_classes)
        logits = self.head(cls_output)

        return logits  # 形状: (N, num_classes)


# ============================================================
# 第六部分：常用 ViT 模型配置工厂函数
# ============================================================

def vit_tiny(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """
    ViT-Tiny: 最小的 ViT 变体，参数量最少，适合快速实验
    参数量约 5.7M
    """
    return VisionTransformer(
        image_size=224, patch_size=16, in_channels=3,
        num_classes=num_classes,
        embed_dim=192,   # 嵌入维度 D = 192
        depth=12,        # Transformer 层数 = 12
        num_heads=3,     # 注意力头数 = 3 (每头 64 维)
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_small(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """
    ViT-Small: 小型 ViT，参数量和精度的较好平衡
    参数量约 22M
    """
    return VisionTransformer(
        image_size=224, patch_size=16, in_channels=3,
        num_classes=num_classes,
        embed_dim=384,   # 嵌入维度 D = 384
        depth=12,        # Transformer 层数 = 12
        num_heads=6,     # 注意力头数 = 6 (每头 64 维)
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_base(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """
    ViT-Base: 原始论文中的基础配置 (ViT-B/16)
    参数量约 86M
    """
    return VisionTransformer(
        image_size=224, patch_size=16, in_channels=3,
        num_classes=num_classes,
        embed_dim=768,   # 嵌入维度 D = 768
        depth=12,        # Transformer 层数 = 12
        num_heads=12,    # 注意力头数 = 12 (每头 64 维)
        mlp_ratio=4.0,
        **kwargs,
    )


def vit_large(num_classes: int = 1000, **kwargs) -> VisionTransformer:
    """
    ViT-Large: 大型 ViT (ViT-L/16)
    参数量约 307M
    """
    return VisionTransformer(
        image_size=224, patch_size=16, in_channels=3,
        num_classes=num_classes,
        embed_dim=1024,  # 嵌入维度 D = 1024
        depth=24,        # Transformer 层数 = 24
        num_heads=16,    # 注意力头数 = 16 (每头 64 维)
        mlp_ratio=4.0,
        **kwargs,
    )


# ============================================================
# 第七部分：简单演示
# ============================================================

if __name__ == "__main__":
    """
    运行此脚本时执行的演示代码
    使用方法: python vit.py
    """
    print("=" * 60)
    print("Vision Transformer (ViT) 演示")
    print("=" * 60)

    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)

    # 选择计算设备: 优先使用 GPU (CUDA)，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # ---- 构建 ViT-Tiny 模型 (参数量较小，适合演示) ----
    model = vit_tiny(num_classes=10)  # 假设 10 分类任务 (如 CIFAR-10)
    model = model.to(device)          # 将模型移到对应设备

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量:   {total_params:,}")
    print(f"可训练参数量:   {trainable_params:,}")

    # ---- 创建随机输入张量模拟一批图像 ----
    batch_size = 4      # 批次大小
    channels = 3        # RGB 图像
    height = 224        # 图像高度
    width = 224         # 图像宽度
    x = torch.randn(batch_size, channels, height, width).to(device)
    print(f"\n输入张量形状: {x.shape}  (N={batch_size}, C={channels}, H={height}, W={width})")

    # ---- 前向传播 ----
    model.eval()  # 切换到评估模式 (Dropout 不生效)
    with torch.no_grad():  # 不计算梯度 (节省内存)
        logits = model(x)

    print(f"输出张量形状: {logits.shape}  (N={batch_size}, num_classes=10)")

    # ---- 计算预测类别 ----
    # argmax 找到概率最大的类别索引
    predictions = logits.argmax(dim=-1)
    print(f"预测类别:     {predictions.tolist()}")

    # ---- 打印图像块信息 ----
    num_patches = model.patch_embedding.num_patches
    print(f"\n图像块数量: {num_patches}  (每行 {int(num_patches**0.5)} 个，共 {int(num_patches**0.5)} 行)")
    print(f"Transformer 序列长度 (含 CLS Token): {num_patches + 1}")

    print("\n演示完成！✅")
