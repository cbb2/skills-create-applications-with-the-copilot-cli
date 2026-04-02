"""
Vision Transformer (ViT) 测试套件
===================================
测试覆盖范围:
  1.  PatchEmbedding 输出形状验证
  2.  MultiHeadSelfAttention 输出形状 + 注意力权重返回
  3.  FeedForward 输出形状验证
  4.  TransformerEncoderBlock 输出形状 + 注意力权重传播
  5.  完整 VisionTransformer 端到端形状验证
  6.  CLS Token 池化 vs Mean Pooling 两种模式
  7.  MLP 分类头 vs 单层线性分类头
  8.  注意力权重可视化接口 (get_attention_map)
  9.  常用配置 (vit_tiny / vit_small) 可实例化验证
  10. 参数量合理性验证
  11. 梯度反向传播验证
  12. 批次大小为 1 的边界测试
  13. qkv_bias 参数是否生效验证

运行方式:
  python test_vit.py
  或
  python -m pytest test_vit.py -v
"""

import torch
import torch.nn as nn

# 导入要测试的模块
from vit import (
    PatchEmbedding,
    MultiHeadSelfAttention,
    FeedForward,
    TransformerEncoderBlock,
    VisionTransformer,
    MLPHead,
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


def make_small_vit(**kwargs) -> VisionTransformer:
    """创建一个用于快速测试的小型 ViT (2层，节省时间)"""
    defaults = dict(
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=10,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )
    defaults.update(kwargs)
    return VisionTransformer(**defaults)


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
    out, attn = layer(x, return_attn_weights=False)

    assert out.shape == x.shape, (
        f"MHSA 输出形状错误: 期望 {x.shape}, 实际 {out.shape}"
    )
    assert attn is None, "return_attn_weights=False 时 attn 应为 None"
    print(f"✅ test_mhsa_output_shape 通过  输出形状: {tuple(out.shape)}")


def test_mhsa_returns_attention_weights():
    """测试 MHSA 在 return_attn_weights=True 时返回正确形状的注意力权重"""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4

    layer = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    layer.eval()
    with torch.no_grad():
        out, attn = layer(x, return_attn_weights=True)

    # 注意力权重形状应为 (N, h, S, S)
    expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn is not None, "return_attn_weights=True 时 attn 不应为 None"
    assert attn.shape == expected_attn_shape, (
        f"注意力权重形状错误: 期望 {expected_attn_shape}, 实际 {attn.shape}"
    )

    # 注意力权重应为概率分布 (每行之和约为 1)
    row_sums = attn.sum(dim=-1)  # (N, h, S)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
        "注意力权重行和不等于 1，softmax 可能未正确应用"
    )
    print(f"✅ test_mhsa_returns_attention_weights 通过  注意力权重形状: {tuple(attn.shape)}")


def test_mhsa_qkv_bias():
    """测试 qkv_bias 参数是否正确控制 QKV 投影层的偏置"""
    embed_dim = 64
    num_heads = 4

    # 有偏置的版本
    layer_with_bias = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=True)
    assert layer_with_bias.qkv.bias is not None, "qkv_bias=True 时应有偏置参数"

    # 无偏置的版本
    layer_no_bias = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, qkv_bias=False)
    assert layer_no_bias.qkv.bias is None, "qkv_bias=False 时不应有偏置参数"

    print("✅ test_mhsa_qkv_bias 通过")


def test_mhsa_eval_deterministic():
    """测试推理模式下 MHSA 输出是确定性的 (Dropout 不激活)"""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4

    layer = MultiHeadSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.5)
    x = torch.randn(batch_size, seq_len, embed_dim)

    layer.eval()
    with torch.no_grad():
        out1, _ = layer(x)
        out2, _ = layer(x)
    assert torch.allclose(out1, out2), "推理模式下 MHSA 输出应该是确定性的"
    print("✅ test_mhsa_eval_deterministic 通过")


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
# 测试: MLPHead
# ============================================================

def test_mlp_head_output_shape():
    """测试 2 层 MLP 分类头的输出形状"""
    batch_size = 4
    embed_dim = 192
    num_classes = 100

    head = MLPHead(embed_dim=embed_dim, num_classes=num_classes)
    x = torch.randn(batch_size, embed_dim)
    out = head(x)

    expected_shape = (batch_size, num_classes)
    assert out.shape == expected_shape, (
        f"MLPHead 输出形状错误: 期望 {expected_shape}, 实际 {out.shape}"
    )
    print(f"✅ test_mlp_head_output_shape 通过  输出形状: {tuple(out.shape)}")


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
    out, attn = block(x)

    assert out.shape == x.shape, (
        f"TransformerEncoderBlock 输出形状错误: 期望 {x.shape}, 实际 {out.shape}"
    )
    assert attn is None, "默认不返回注意力权重时 attn 应为 None"
    print(f"✅ test_transformer_block_output_shape 通过  输出形状: {tuple(out.shape)}")


def test_transformer_block_attn_weights():
    """测试 TransformerEncoderBlock 能正确传播注意力权重"""
    batch_size = 2
    seq_len = 10
    embed_dim = 64
    num_heads = 4

    block = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(batch_size, seq_len, embed_dim)

    block.eval()
    with torch.no_grad():
        out, attn = block(x, return_attn_weights=True)

    assert attn is not None, "return_attn_weights=True 时 attn 不应为 None"
    expected_attn_shape = (batch_size, num_heads, seq_len, seq_len)
    assert attn.shape == expected_attn_shape, (
        f"TransformerEncoderBlock 注意力权重形状错误: 期望 {expected_attn_shape}, 实际 {attn.shape}"
    )
    print(f"✅ test_transformer_block_attn_weights 通过  注意力权重形状: {tuple(attn.shape)}")


def test_transformer_block_residual():
    """测试残差连接是否正常工作 (输出不等于输入但形状相同)"""
    embed_dim = 64
    num_heads = 4

    block = TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads)
    x = torch.randn(2, 10, embed_dim)
    out, _ = block(x)

    assert out.shape == x.shape
    assert not torch.allclose(out, x), "TransformerEncoderBlock 输出不应与输入完全相同"
    print("✅ test_transformer_block_residual 通过")


# ============================================================
# 测试: VisionTransformer 端到端
# ============================================================

def test_vit_end_to_end_shape():
    """测试完整 ViT 的输入输出形状"""
    batch_size = 2
    num_classes = 10

    model = make_small_vit(num_classes=num_classes)
    x = torch.randn(batch_size, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits, attn_list = model(x)

    expected_shape = (batch_size, num_classes)
    assert logits.shape == expected_shape, (
        f"ViT 输出形状错误: 期望 {expected_shape}, 实际 {logits.shape}"
    )
    assert attn_list is None, "默认不返回注意力权重时 attn_list 应为 None"
    print(f"✅ test_vit_end_to_end_shape 通过  输出形状: {tuple(logits.shape)}")


def test_vit_cls_pooling():
    """测试 CLS Token 池化模式 (默认)"""
    model = make_small_vit(pool="cls")
    assert model.pool == "cls"

    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)

    assert logits.shape == (2, 10)
    print("✅ test_vit_cls_pooling 通过")


def test_vit_mean_pooling():
    """测试 Mean Pooling 模式 (lucidrains 风格的替代方案)"""
    model = make_small_vit(pool="mean")
    assert model.pool == "mean"

    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)

    assert logits.shape == (2, 10)
    print("✅ test_vit_mean_pooling 通过")


def test_vit_mlp_head():
    """测试 MLP 分类头模式 (原论文预训练设置)"""
    model = make_small_vit(mlp_head=True)

    # 分类头应是 MLPHead 实例，而非单层 Linear
    assert isinstance(model.head, MLPHead), (
        f"mlp_head=True 时头部应为 MLPHead，实际为 {type(model.head)}"
    )

    x = torch.randn(2, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)

    assert logits.shape == (2, 10)
    print("✅ test_vit_mlp_head 通过")


def test_vit_linear_head():
    """测试单层线性分类头模式 (fine-tuning 设置)"""
    model = make_small_vit(mlp_head=False)

    # 分类头应是 Linear 实例
    assert isinstance(model.head, nn.Linear), (
        f"mlp_head=False 时头部应为 nn.Linear，实际为 {type(model.head)}"
    )
    print("✅ test_vit_linear_head 通过")


def test_vit_attention_weights():
    """测试 return_attn_weights=True 时返回正确数量和形状的注意力权重"""
    depth = 2
    num_heads = 4
    embed_dim = 64
    seq_len = 197  # 196 patches + 1 CLS

    model = make_small_vit(depth=depth, num_heads=num_heads, embed_dim=embed_dim)
    x = torch.randn(1, 3, 224, 224)

    model.eval()
    with torch.no_grad():
        logits, attn_list = model(x, return_attn_weights=True)

    # 应返回 depth 个注意力权重矩阵
    assert len(attn_list) == depth, (
        f"注意力权重列表长度错误: 期望 {depth}, 实际 {len(attn_list)}"
    )

    # 每个注意力权重形状应为 (N, h, S, S)
    expected_attn_shape = (1, num_heads, seq_len, seq_len)
    for i, attn in enumerate(attn_list):
        assert attn.shape == expected_attn_shape, (
            f"第 {i} 层注意力权重形状错误: 期望 {expected_attn_shape}, 实际 {attn.shape}"
        )

    print(f"✅ test_vit_attention_weights 通过  {depth} 层，每层形状: {tuple(expected_attn_shape)}")


def test_vit_get_attention_map():
    """测试 get_attention_map 辅助方法"""
    num_heads = 4
    seq_len = 197  # 196 patches + 1 CLS

    model = make_small_vit(num_heads=num_heads)
    single_img = torch.randn(1, 3, 224, 224)

    # 获取最后一层的注意力权重图
    attn_map = model.get_attention_map(single_img, layer_idx=-1)

    # 形状应为 (h, S, S)
    expected_shape = (num_heads, seq_len, seq_len)
    assert attn_map.shape == expected_shape, (
        f"get_attention_map 输出形状错误: 期望 {expected_shape}, 实际 {attn_map.shape}"
    )
    print(f"✅ test_vit_get_attention_map 通过  形状: {tuple(attn_map.shape)}")


def test_vit_batch_size_one():
    """测试批次大小为 1 时的边界情况"""
    model = make_small_vit()
    x = torch.randn(1, 3, 224, 224)
    model.eval()
    with torch.no_grad():
        logits, _ = model(x)
    assert logits.shape == (1, 10), f"批次为1时输出形状错误: {logits.shape}"
    print("✅ test_vit_batch_size_one 通过")


def test_vit_gradient_flow():
    """测试梯度能否正常反向传播"""
    model = make_small_vit()
    x = torch.randn(2, 3, 224, 224)
    target = torch.randint(0, 10, (2,))

    logits, _ = model(x)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

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
            out, _ = model(x)
        assert out.shape == (1, num_classes), (
            f"{name} 输出形状错误: {out.shape}"
        )
        params = count_parameters(model)
        print(f"✅ {name:12s} 创建成功  参数量: {params:,}")


def test_vit_tiny_parameter_count():
    """验证 ViT-Tiny 的参数量在合理范围内 (约 5-7M)"""
    model = vit_tiny(num_classes=1000)
    params = count_parameters(model)
    assert 4_000_000 < params < 9_000_000, (
        f"ViT-Tiny 参数量 {params:,} 超出预期范围 (4M~9M)"
    )
    print(f"✅ test_vit_tiny_parameter_count 通过  参数量: {params:,}")


def test_vit_small_parameter_count():
    """验证 ViT-Small 的参数量在合理范围内 (约 22M)"""
    model = vit_small(num_classes=1000)
    params = count_parameters(model)
    assert 18_000_000 < params < 30_000_000, (
        f"ViT-Small 参数量 {params:,} 超出预期范围 (18M~30M)"
    )
    print(f"✅ test_vit_small_parameter_count 通过  参数量: {params:,}")


def test_vit_invalid_pool():
    """测试非法 pool 参数能触发断言错误"""
    raised = False
    try:
        VisionTransformer(
            image_size=224, patch_size=16, in_channels=3,
            num_classes=10, embed_dim=64, depth=1, num_heads=4,
            pool="invalid",
        )
    except AssertionError:
        raised = True
    assert raised, "非法 pool 参数应触发 AssertionError"
    print("✅ test_vit_invalid_pool 通过")


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
        test_mhsa_returns_attention_weights,
        test_mhsa_qkv_bias,
        test_mhsa_eval_deterministic,
        # FeedForward 测试
        test_feed_forward_output_shape,
        # MLPHead 测试
        test_mlp_head_output_shape,
        # TransformerEncoderBlock 测试
        test_transformer_block_output_shape,
        test_transformer_block_attn_weights,
        test_transformer_block_residual,
        # VisionTransformer 端到端测试
        test_vit_end_to_end_shape,
        test_vit_cls_pooling,
        test_vit_mean_pooling,
        test_vit_mlp_head,
        test_vit_linear_head,
        test_vit_attention_weights,
        test_vit_get_attention_map,
        test_vit_batch_size_one,
        test_vit_gradient_flow,
        test_vit_invalid_pool,
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
