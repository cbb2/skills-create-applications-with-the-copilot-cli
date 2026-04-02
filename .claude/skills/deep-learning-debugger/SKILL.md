---
name: deep-learning-debugger
description: 深度学习调试专家技能。当用户遇到 PyTorch 模型训练问题时自动激活，包括：loss 不收敛、梯度异常（爆炸/消失）、维度不匹配报错、CUDA OOM、精度异常等。使用结构化的逐步诊断流程定位根因并给出修复方案，所有解释使用中文，适合深度学习初学者。
---

# 深度学习调试专家

## 核心原则

**永远先理解再修改。** 调试深度学习问题不能靠猜测，要系统地缩小问题范围。
使用 sequential-thinking MCP 分解复杂调试任务，使用 filesystem MCP 读取相关源码。

---

## 调试决策树

```
收到错误/异常行为
        ↓
① 分类问题类型（见下表）
        ↓
② 按对应诊断流程操作
        ↓
③ 定位根因 → 给出修复代码 → 解释原理
```

### 问题类型速查表

| 症状 | 类型 | 跳转 |
|------|------|------|
| `RuntimeError: shape mismatch` / 维度报错 | 张量形状错误 | [→ 形状调试](#形状调试) |
| `loss = nan` 或 `loss` 不下降 | 训练不收敛 | [→ 收敛调试](#收敛调试) |
| `CUDA out of memory` | 显存溢出 | [→ OOM 调试](#oom-调试) |
| 梯度为 None 或全为 0 | 梯度问题 | [→ 梯度调试](#梯度调试) |
| 模型在训练集好但验证集差 | 过拟合 | [→ 过拟合调试](#过拟合调试) |
| 训练慢、GPU 利用率低 | 性能问题 | [→ 性能调试](#性能调试) |

---

## 形状调试

**触发条件**: `RuntimeError`, `size mismatch`, `shape`, `expected ... got ...`

### 诊断步骤

1. **定位出错行**，找到形状不匹配发生在哪一层

2. **打印前后形状**，在出错行前后插入：
   ```python
   print(f"[调试] 输入形状: {x.shape}")
   # ... 出错的操作 ...
   print(f"[调试] 输出形状: {output.shape}")
   ```

3. **追踪 ViT 数据流**（本项目标准形状）：
   ```
   输入图像:        (N, 3, 224, 224)
   PatchEmbedding:  (N, 196, D)      ← L = (224/16)² = 196
   +CLS Token:      (N, 197, D)      ← 197 = 196 + 1
   +Pos Embedding:  (N, 197, D)
   每层 Block 输入: (N, 197, D)
   每层 Block 输出: (N, 197, D)      ← 形状不变！
   池化后:          (N, D)
   分类头输出:      (N, num_classes)
   ```

4. **常见根因**：
   - `image_size % patch_size != 0` → 切分不整除
   - `num_heads` 不整除 `embed_dim` → MHSA 维度错误
   - 忘记 `transpose(1, 2)` 调换序列维度
   - `flatten` 参数错误（`flatten(2)` vs `flatten(1)`）

5. **给出修复**: 提供修正后的代码片段，并解释为什么这样改正确

---

## 收敛调试

**触发条件**: loss 不下降、loss = nan、loss 剧烈震荡

### 诊断清单（按优先级）

**第一步：检查数值稳定性**
```python
# 在每个 epoch 开始时检查
assert not torch.isnan(loss), f"Loss 为 NaN，epoch={epoch}"
assert not torch.isinf(loss), f"Loss 为 Inf，epoch={epoch}"

# 检查模型输出
print(f"logits 均值: {logits.mean():.4f}, 标准差: {logits.std():.4f}")
print(f"logits 最大值: {logits.max():.4f}, 最小值: {logits.min():.4f}")
```

**第二步：检查学习率**
```python
# ViT 推荐学习率范围
# - AdamW optimizer: lr = 1e-3 ~ 1e-4
# - 小数据集 (如 CIFAR-10): lr = 1e-4
# - 配合 warmup scheduler 效果更好
print(f"当前学习率: {optimizer.param_groups[0]['lr']}")
```

**第三步：检查 loss 函数**
```python
# 分类任务确认使用正确的 loss
criterion = nn.CrossEntropyLoss()
# ✅ 正确: logits 不需要 softmax，CrossEntropyLoss 内部已包含
loss = criterion(logits, labels)

# ❌ 错误示范: 对 logits 先 softmax 再 CrossEntropyLoss
```

**第四步：梯度裁剪**（ViT 训练标准做法）
```python
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
optimizer.step()
```

**第五步：检查数据标准化**
```python
# ImageNet 标准化（迁移学习推荐）
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
```

### NaN 根因速查
- **权重初始化不当** → 确认使用 `trunc_normal_(std=0.02)`
- **学习率过大** → 尝试除以 10
- **LayerNorm eps 太小** → 使用 `eps=1e-6`（本项目已正确设置）
- **输入数据未归一化** → 检查数据预处理管道

---

## OOM 调试

**触发条件**: `CUDA out of memory`

### 快速解决方案（按效果排序）

```python
# 1. 减小 batch_size（最直接）
batch_size = 16  # 从 64 减到 16

# 2. 使用梯度累积模拟大 batch
accumulation_steps = 4  # 等效 batch_size * 4
for i, (images, labels) in enumerate(dataloader):
    loss = criterion(model(images), labels) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 3. 推理时使用 torch.no_grad()
with torch.no_grad():
    logits, _ = model(images)

# 4. 启用混合精度训练（节省约 50% 显存）
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    logits, _ = model(images)
    loss = criterion(logits, labels)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()

# 5. 清理显存缓存（调试用）
torch.cuda.empty_cache()

# 6. 查看显存使用情况
print(f"已用显存: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"缓存显存: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
```

### ViT 各配置的显存参考（batch_size=32, 224×224）
| 模型 | 参数量 | 训练显存 (约) |
|------|--------|--------------|
| ViT-Tiny | 5.7M | ~2 GB |
| ViT-Small | 22M | ~6 GB |
| ViT-Base | 86M | ~20 GB |

---

## 梯度调试

**触发条件**: 梯度为 None、梯度全为 0、训练不更新

### 诊断步骤

```python
# 1. 检查所有参数的梯度
for name, param in model.named_parameters():
    if param.grad is None:
        print(f"⚠️  {name}: 梯度为 None（参数未参与计算图）")
    elif param.grad.abs().max() < 1e-7:
        print(f"⚠️  {name}: 梯度接近 0（可能梯度消失）")
    elif param.grad.abs().max() > 100:
        print(f"⚠️  {name}: 梯度过大={param.grad.abs().max():.2f}（可能梯度爆炸）")
    else:
        print(f"✅ {name}: 梯度正常，最大值={param.grad.abs().max():.4f}")

# 2. 注册梯度钩子（实时监控）
def grad_hook(name):
    def hook(grad):
        print(f"[梯度钩子] {name}: max={grad.abs().max():.4f}, mean={grad.abs().mean():.6f}")
    return hook

for name, param in model.named_parameters():
    param.register_hook(grad_hook(name))
```

### 常见根因
- **`torch.no_grad()` 包裹了前向传播** → 梯度不会被计算
- **模型处于 `eval()` 模式时调用了 backward** → 切换到 `model.train()`
- **参数 `requires_grad=False`** → 检查是否意外冻结了层

---

## 过拟合调试

**触发条件**: 训练 acc 高 (>90%) 但验证 acc 低 (<60%)

### 推荐解决方案

```python
# 1. 增大 Dropout（本项目 VisionTransformer 接受 dropout 参数）
model = VisionTransformer(
    ...
    dropout=0.1,  # 从 0.0 增加到 0.1
)

# 2. 数据增强（推荐 torchvision transforms）
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),  # 对 CIFAR-10
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# 3. 减少模型规模（使用 vit_tiny 而非 vit_base）
from vision_transformer.vit import vit_tiny
model = vit_tiny(num_classes=10)

# 4. 使用预训练权重（最有效）
# ViT 在小数据集上容易过拟合，预训练权重至关重要
# 可使用 timm 加载预训练 ViT-Tiny
# pip install timm
# import timm; model = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=10)
```

---

## 性能调试

**触发条件**: GPU 利用率低、训练速度慢

```python
# 1. 启用混合精度（最推荐）
# 参考上方 OOM 调试中的 autocast 用法

# 2. 使用 DataLoader 多进程加载
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)

# 3. 使用 torch.compile（PyTorch 2.0+ 加速推理/训练）
model = torch.compile(model)

# 4. Profile 找到瓶颈
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    logits, _ = model(images)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## 完整调试报告模板

当诊断完成后，按以下格式输出结果：

```
## 调试报告

**问题类型**: [形状错误 / 训练不收敛 / OOM / 梯度问题 / 过拟合 / 性能]

**根本原因**:
[简洁说明根因，1-2句话]

**修复代码**:
```python
[提供可直接运行的修复代码]
```

**原理解释**:
[用初学者能理解的中文解释为什么这样修复是正确的，结合 ViT 架构知识]

**预防建议**:
[下次如何避免同类问题]
```

---

## 参考资源

- ViT 原论文: https://arxiv.org/abs/2010.11929
- PyTorch 调试文档: https://pytorch.org/docs/stable/notes/faq.html
- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- timm 预训练模型: https://github.com/huggingface/pytorch-image-models
