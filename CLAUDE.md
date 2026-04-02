# Vision Transformer 学习项目 — Claude Code 项目说明

## 项目概述

本仓库是一个面向深度学习初学者的 **Vision Transformer (ViT)** 教学项目。
核心代码位于 `vision_transformer/` 目录，每一行都有中文注释，帮助初学者系统理解 ViT 架构。

论文参考: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (Dosovitskiy et al., 2021)

---

## 项目结构

```
vision_transformer/
├── vit.py            # 完整 ViT 实现（8 个模块，每行带中文注释）
├── test_vit.py       # 测试套件（24 个测试）
├── requirements.txt  # Python 依赖
└── README.md         # 使用指南

.claude/
├── skills/           # Claude Code Skills（3 个领域专项技能）
│   ├── deep-learning-debugger/
│   ├── pytorch-code-reviewer/
│   └── vit-explainer/
└── .mcp.json         # MCP 服务器配置（已迁移到项目根目录）

.mcp.json             # MCP 服务器配置（Claude Code 根目录发现）
CLAUDE.md             # 本文件
```

---

## 快速命令

```bash
# 安装依赖
pip install torch torchvision

# 运行演示
cd vision_transformer && python vit.py

# 运行测试（24 个测试）
cd vision_transformer && python test_vit.py

# 检查特定模块
cd vision_transformer && python -c "from vit import vit_tiny; m = vit_tiny(); print(sum(p.numel() for p in m.parameters()))"
```

---

## 架构速查

| 模块 | 类名 | 输入 → 输出 |
|------|------|------------|
| 图像块嵌入 | `PatchEmbedding` | `(N,C,H,W)` → `(N,L,D)` |
| 多头自注意力 | `MultiHeadSelfAttention` | `(N,S,D)` → `(N,S,D)` |
| 前馈网络 | `FeedForward` | `(N,S,D)` → `(N,S,D)` |
| 编码器块 | `TransformerEncoderBlock` | `(N,S,D)` → `(N,S,D)` |
| MLP 分类头 | `MLPHead` | `(N,D)` → `(N,C)` |
| 完整模型 | `VisionTransformer` | `(N,C,H,W)` → `(N,num_classes)` |

符号: N=batch, C=通道, H/W=图像尺寸, L=图像块数量, D=嵌入维度, S=序列长度, C=类别数

---

## 代码规范

- **语言**: Python 3.8+，所有注释使用中文（面向中文初学者）
- **框架**: PyTorch (torch ≥ 2.0)，不使用外部深度学习库
- **测试**: 使用标准 `assert` + 手写测试函数，不依赖 pytest（可用 pytest 运行）
- **类型注解**: 所有函数签名使用 `typing` 模块的类型注解
- **LayerNorm**: 始终使用 `eps=1e-6`（比默认 1e-5 更数值稳定）
- **命名规范**: 类名 PascalCase，函数/变量 snake_case

---

## 可用的 Claude Code Skills

本项目配置了 3 个专项技能，Claude Code 会自动从 `.claude/skills/` 加载：

### 1. `deep-learning-debugger` — 深度学习调试专家
当遇到以下问题时自动激活：
- 训练 loss 不下降、梯度爆炸/消失
- 维度不匹配报错 (`RuntimeError`, `shape mismatch`)
- CUDA 内存溢出 (OOM)
- 模型精度异常

### 2. `pytorch-code-reviewer` — PyTorch 代码审查
当需要以下操作时激活：
- 审查新增的 PyTorch 代码
- 检查是否符合 ViT 架构规范
- 发现潜在的性能或正确性问题

### 3. `vit-explainer` — ViT 架构解释器
当用户询问以下内容时激活：
- 解释某个模块的原理
- 对比不同实现方案
- 生成教学示例代码

---

## MCP 配置说明

项目根目录的 `.mcp.json` 配置了以下 MCP 服务器（Claude Code 自动发现）：

| 服务器 | 功能 |
|--------|------|
| `filesystem` | 安全地读写项目文件 |
| `sequential-thinking` | 分步推理复杂的调试和分析任务 |
| `fetch` | 获取 arXiv 论文、PyTorch 文档等网络资源 |

> **注意**: `filesystem` MCP 的访问范围已限制为本项目目录，确保安全。
