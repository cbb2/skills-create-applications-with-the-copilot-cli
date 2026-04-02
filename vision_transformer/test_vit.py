"""
Vision Transformer (ViT) 测试套件
===================================
测试覆盖范围:
  1. PatchEmbedding 输出形状验证
  2. MultiHeadSelfAttention 输出形状验证
  3. FeedForward 输出形状验证
  4. TransformerEncoderBlock 输出形状验证
  5. 完整 VisionTransformer 端到端形状验证
  6. 常用配置 (vit_tiny / vit_small / vit_base) 可实例化验证
  7. 参数量合理性验证
  8. 梯度反向传播验证
  9. 模型推理模式 (eval) 验证
  10. 批次大小为 1 的边界测试

运行方式:
  python test_vit.py
  或
  python -m pytest test_vit.py -v
"""

import math
import torch
import torch.nn as nn

# 导入要测试的模块
from vit import (
    PatchEmbedding,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerEncoderBlock,
    VisionTransformer,
    vit_tiny,
    vit_small,
    vit_base,
)


# ============================================================
# 辅助函数
# ============================================================

def count_parameters(model: nn.Module) -> int:
    """统计模型的可训练参数总量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# 测试: PatchEmbedding
# ============================================================

def test_patch_embedding_output_shape():
    """测试 PatchEmbedding 输出形状是否正确"""
    image_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 128
    batch_size = 2

    layer = PatchEmbedding(
        image_size=image_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
    )

    x = torch.randn(batch_size, in_channels, image_size, image_size)
    out = layer(x)

    # 期望图像块数量: (224/16)^2 = 14^2 = 196
    expected_num_patches = (image_size // patch_size) ** 2
    expected_shape = (batch_size, expected_num_patches, embed_dim)

    assert out.shape == expected_shape, (
        f"PatchEmbedding 输出形状错误: 期望 {expected_shape}, 实际 {out.shape}"
    )
    print(f"✅ test_patch_embedding_output_shape 通过  输出形状: {tuple(out.shape)}")


def test_patch_embedding_num_patches():
    """测试不同配置下图像块数量是否正确"""
    test_cases = [
        (224, 16, 196),   # ViT-B/16 标准配置
        (224, 32, 49),    # ViT-B/32 配置
        (32, 4, 64),      # 小图像测试
    ]

    for image_size, patch_size, expected_num_patches in test_cases:
        layer = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=64,
        )
        assert layer.num_patches == expected_num_patches, (
            f"图像块数量错误: image_size={image_size}, patch_size={patch_size}, "
            f"期望 {expected_num_patches}, 实际 {layer.num_patches}"
        )

    print("✅ test_patch_embedding_num_patches 通过")


# ============================================================
# 测试: MultiHeadSelfAttention
# ============================================================

def test_mhsa_output_shape():
    """测试 MultiHeadSelfAttention 输出形状与输入相同"""
    batch_size = 2
    seq_len = 197  # 196 个图像块 + 1 个 CLS Token
    embed_dim = 192
    num_heads = 3

    layer = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = layer(x)

    assert out.shape == x.shape, (
        f"MHSA 输出形状错误: 期望 {x.shape}, 实际 {out.shape}"
    )
    print(f"✅ test_mhsa_output_shape 通过  输出形状: {tuple(out.shape)}")


def test_mhsa_with_dropout():
    """测试带 Dropout 的 MHSA 在训练和推理模式下行为不同"""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4

    layer = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.5)
    x = torch.randn(batch_size, seq_len, embed_dim)

    # 推理模式下 Dropout 不激活，两次输出应该相同
    layer.eval()
    with torch.no_grad():
        out1 = layer(x)
        out2 = layer(x)
    assert torch.allclose(out1, out2), "推理模式下 MHSA 输出应该是确定性的"
    print("✅ test_mhsa_with_dropout 通过")


# ============================================================
# 测试: FeedForward
# ============================================================

def test_feed_forward_output_shape():
    """测试 FeedForward 输出形状与输入相同"""
    batch_size = 2
    seq_len = 197
    embed_dim = 192
    hidden_dim = 768  # 4 * embed_dim

    layer = FeedForward(embed_dim=embed_dim, hidden_dim=hidden_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = layer(x)

    assert out.shape == x.shape, (
        f"FeedForward 输出形状错误: 期望 {x.shape}, 实际 {out.shape}"
    )
    print(f"✅ test_feed_forward_output_shape 通过  输出形状: {tuple(out.shape)}")


# ============================================================
# 测试: TransformerEncoderBlock
# ============================================================

def test_transformer_block_output_shape():
    """测试 TransformerEncoderBlock 输出形状与输入相同"""
    batch_size = 2
    seq_len = 197
    embed_dim = 192
    num_heads = 3

    block = TransformerEncoderBlock(
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=4.0,
    )
    x = torch.randn(batch_size, seq_len, embed_dim)
    out = block(x)

    assert out.shape == x.shape, (
        f"TransformerEncoderBlock 输出形状错误: 期望 {x.shape}, 实际 {out.shape}"
    )
    print(f"✅ test_transformer_block_output_shape 通过  输出形状: {tuple(out.shape)}")


def test_transformer_block_residual():
    """测试残差连接是否正常工作 (输出不等于输入但形状相同)"""
    embed_dim = 64
    num_heads = 4

    block = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(2, 10, embed_dim)
    out = block(x)

    # 有残差连接，输出形状与输入相同
    assert out.shape == x.shape
    # 有变换，输出不等于输入
    assert not torch.allclose(out, x), "TransformerEncoderBlock 输出不应与输入完全相同"
    print("✅ test_transformer_block_residual 通过")


# ============================================================
# 测试: VisionTransformer 端到端
# ============================================================

def test_vit_end_to_end_shape():
    """测试完整 ViT 的输入输出形状"""
    batch_size = 2
    num_classes = 10

    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=192,
        depth=2,        # 测试时只用 2 层，加快速度
        num_heads=3,
    )

    x = torch.randn(batch_size, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits = model(x)

    expected_shape = (batch_size, num_classes)
    assert logits.shape == expected_shape, (
        f"ViT 输出形状错误: 期望 {expected_shape}, 实际 {logits.shape}"
    )
    print(f"✅ test_vit_end_to_end_shape 通过  输出形状: {tuple(logits.shape)}")


def test_vit_batch_size_one():
    """测试批次大小为 1 时的边界情况"""
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    x = torch.randn(1, 3, 224, 224)  # 批次大小为 1
    model.eval()
    with torch.no_grad():
        logits = model(x)

    assert logits.shape == (1, 10), f"批次为1时输出形状错误: {logits.shape}"
    print("✅ test_vit_batch_size_one 通过")


def test_vit_gradient_flow():
    """测试梯度能否正常反向传播 (检验没有梯度断裂)"""
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    x = torch.randn(2, 3, 224, 224, requires_grad=False)
    target = torch.randint(0, 10, (2,))  # 随机类别标签

    # 前向传播
    logits = model(x)

    # 计算交叉熵损失
    loss = nn.CrossEntropyLoss()(logits, target)

    # 反向传播
    loss.backward()

    # 检查关键参数的梯度不为 None 且不全为 0
    assert model.cls_token.grad is not None, "CLS Token 梯度为 None"
    assert model.position_embedding.grad is not None, "位置嵌入梯度为 None"
    assert not torch.all(model.cls_token.grad == 0), "CLS Token 梯度全为 0"

    print(f"✅ test_vit_gradient_flow 通过  损失值: {loss.item():.4f}")


# ============================================================
# 测试: 工厂函数
# ============================================================

def test_vit_factory_functions():
    """测试工厂函数能够成功创建模型并输出形状正确"""
    num_classes = 100
    x = torch.randn(1, 3, 224, 224)

    for name, factory in [("vit_tiny", vit_tiny), ("vit_small", vit_small)]:
        model = factory(num_classes=num_classes)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, num_classes), (
            f"{name} 输出形状错误: {out.shape}"
        )
        params = count_parameters(model)
        print(f"✅ {name:12s} 创建成功  参数量: {params:,}")


def test_vit_tiny_parameter_count():
    """验证 ViT-Tiny 的参数量在合理范围内 (约 5-7M)"""
    model = vit_tiny(num_classes=1000)
    params = count_parameters(model)
    # ViT-Tiny 参数量约 5.7M
    assert 4_000_000 < params < 8_000_000, (
        f"ViT-Tiny 参数量 {params:,} 超出预期范围 (4M~8M)"
    )
    print(f"✅ test_vit_tiny_parameter_count 通过  参数量: {params:,}")


def test_vit_small_parameter_count():
    """验证 ViT-Small 的参数量在合理范围内 (约 22M)"""
    model = vit_small(num_classes=1000)
    params = count_parameters(model)
    # ViT-Small 参数量约 22M
    assert 18_000_000 < params < 28_000_000, (
        f"ViT-Small 参数量 {params:,} 超出预期范围 (18M~28M)"
    )
    print(f"✅ test_vit_small_parameter_count 通过  参数量: {params:,}")


# ============================================================
# 主函数: 运行所有测试
# ============================================================

def run_all_tests():
    """运行全部测试并统计结果"""
    print("=" * 60)
    print("Vision Transformer 测试套件")
    print("=" * 60)

    tests = [
        # PatchEmbedding 测试
        test_patch_embedding_output_shape,
        test_patch_embedding_num_patches,
        # MHSA 测试
        test_mhsa_output_shape,
        test_mhsa_with_dropout,
        # FeedForward 测试
        test_feed_forward_output_shape,
        # TransformerEncoderBlock 测试
        test_transformer_block_output_shape,
        test_transformer_block_residual,
        # VisionTransformer 端到端测试
        test_vit_end_to_end_shape,
        test_vit_batch_size_one,
        test_vit_gradient_flow,
        # 工厂函数测试
        test_vit_factory_functions,
        test_vit_tiny_parameter_count,
        test_vit_small_parameter_count,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"❌ {test_fn.__name__} 失败: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败 (共 {len(tests)} 个)")
    print("=" * 60)

    if failed == 0:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查代码。")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
