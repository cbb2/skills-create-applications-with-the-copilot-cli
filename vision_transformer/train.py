"""
train.py — ViT 在 CIFAR-10 上的完整训练脚本
============================================
功能:
  1. 自动下载 CIFAR-10 数据集（首次运行）
  2. 构建 ViT-Tiny 模型（轻量，适合 CPU/单 GPU）
  3. 训练循环：含 CosineAnnealing 学习率调度 + 梯度裁剪
  4. 每 epoch 在验证集上评估，保存最优 checkpoint
  5. 记录 TensorBoard 日志（可用 tensorboard --logdir runs/ 可视化）

快速开始:
  python train.py                      # 使用默认配置（GPU 优先）
  python train.py --epochs 5           # 快速验证（5 个 epoch）
  python train.py --image-size 32 --patch-size 4  # 原始分辨率（更快）

注意:
  - 在 CPU 上 IMAGE_SIZE=224 训练很慢，建议 --image-size 32 --patch-size 4
  - GPU 训练使用默认配置（224×224），精度更高
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 确保可以 import 同目录的模块
sys.path.insert(0, os.path.dirname(__file__))
from dataset import CIFAR10_CLASSES, get_cifar10_loaders
from vit import VisionTransformer


# ============================================================
# 训练配置（通过命令行参数覆盖）
# ============================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数，所有参数都有合理默认值"""
    parser = argparse.ArgumentParser(
        description="ViT CIFAR-10 训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 模型架构 ----
    parser.add_argument("--image-size", type=int, default=224,
                        help="训练图像尺寸（ViT 输入）")
    parser.add_argument("--patch-size", type=int, default=16,
                        help="图像块尺寸")
    parser.add_argument("--embed-dim", type=int, default=192,
                        help="嵌入维度（192=ViT-Tiny, 384=ViT-Small）")
    parser.add_argument("--depth", type=int, default=12,
                        help="Transformer 编码器层数")
    parser.add_argument("--num-heads", type=int, default=3,
                        help="注意力头数")
    parser.add_argument("--mlp-ratio", type=float, default=4.0,
                        help="FFN 隐藏层维度比例")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout 比率")

    # ---- 训练超参数 ----
    parser.add_argument("--epochs", type=int, default=30,
                        help="训练总轮数")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="训练批次大小")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=0.05,
                        help="AdamW 权重衰减")
    parser.add_argument("--warmup-epochs", type=int, default=5,
                        help="学习率预热 epoch 数")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="梯度裁剪最大范数（0=不裁剪）")

    # ---- 数据 ----
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data"),
                        help="CIFAR-10 数据存储目录")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="数据加载工作进程数（Windows 下设为 0）")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="从训练集中划分为验证集的比例")

    # ---- 输出 ----
    parser.add_argument("--checkpoint-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "checkpoints"),
                        help="模型 checkpoint 保存目录")
    parser.add_argument("--log-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "runs"),
                        help="TensorBoard 日志保存目录")
    parser.add_argument("--save-every", type=int, default=5,
                        help="每隔多少 epoch 保存一次 checkpoint（0=仅保存最优）")

    # ---- 其他 ----
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备: auto/cpu/cuda/mps")

    return parser.parse_args()


# ============================================================
# 辅助函数
# ============================================================

def set_seed(seed: int) -> None:
    """设置全局随机种子，保证实验可复现"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """根据参数和硬件自动选择最优计算设备"""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")   # Apple Silicon
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def build_model(args: argparse.Namespace) -> VisionTransformer:
    """根据参数构建 VisionTransformer 模型"""
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=3,
        num_classes=10,         # CIFAR-10 共 10 类
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        qkv_bias=True,
        pool="cls",             # 使用 CLS Token 池化
        mlp_head=False,         # 使用单层线性分类头（fine-tuning 设置）
        dropout=args.dropout,
    )
    return model


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    构建带 Warmup 的余弦退火学习率调度器

    策略:
      - 前 warmup_epochs 个 epoch: 学习率从 0 线性增加到 args.lr
        防止训练初期损失震荡（ViT 对初始学习率敏感）
      - 之后: 按余弦曲线从 args.lr 衰减到 args.lr × 1e-2
        余弦退火让学习率平滑下降，比阶梯式衰减效果更好
    """
    def lr_lambda(current_epoch: int) -> float:
        if current_epoch < args.warmup_epochs:
            # 线性预热: 从 0 增加到 1.0 (乘以 base_lr)
            return float(current_epoch + 1) / float(args.warmup_epochs)
        # 余弦退火: 从 1.0 降到 min_lr / base_lr
        progress = (current_epoch - args.warmup_epochs) / max(
            1, args.epochs - args.warmup_epochs
        )
        return 0.01 + 0.5 * (1.0 - 0.01) * (1 + torch.cos(torch.tensor(progress * 3.14159265)).item())

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================
# 单 epoch 训练
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    writer: SummaryWriter,
) -> Dict[str, float]:
    """
    训练一个 epoch，返回平均 loss 和 accuracy

    关键步骤:
      1. 前向传播: 计算预测 logits
      2. 计算损失: CrossEntropyLoss（含 softmax）
      3. 反向传播: 计算梯度
      4. 梯度裁剪: 防止梯度爆炸
      5. 参数更新: AdamW 步进
    """
    model.train()  # 切换到训练模式（开启 Dropout、BatchNorm 等）

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = len(loader)

    start_time = time.time()

    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)  # 非阻塞传输，提升 GPU 利用率
        labels = labels.to(device, non_blocking=True)

        # ---- 前向传播 ----
        optimizer.zero_grad(set_to_none=True)  # set_to_none 比 zero_grad() 更省内存
        logits, _ = model(images)              # logits 形状: (N, num_classes)
        loss = criterion(logits, labels)       # 标量 loss

        # ---- 反向传播 ----
        loss.backward()  # 计算所有参数的梯度

        # ---- 梯度裁剪（防止梯度爆炸）----
        # ViT 训练初期梯度可能很大，裁剪确保训练稳定
        if grad_clip > 0.0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        # ---- 参数更新 ----
        optimizer.step()

        # ---- 统计指标 ----
        batch_size = images.size(0)
        total_loss    += loss.item() * batch_size
        preds          = logits.argmax(dim=1)                  # 预测类别
        total_correct += (preds == labels).sum().item()        # 正确数量
        total_samples += batch_size

        # 每 50 个 batch 打印进度
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(
                f"  Epoch [{epoch}] Batch [{batch_idx+1}/{num_batches}] "
                f"Loss: {loss.item():.4f}  "
                f"Acc: {total_correct/total_samples:.4f}  "
                f"Time: {elapsed:.1f}s"
            )

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples

    # 记录到 TensorBoard
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Accuracy/train", avg_acc, epoch)

    return {"loss": avg_loss, "accuracy": avg_acc}


# ============================================================
# 验证评估
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    split: str = "val",
) -> Dict[str, float]:
    """
    在验证集或测试集上评估模型性能

    @torch.no_grad(): 关闭梯度计算，节省内存和时间
    model.eval(): 关闭 Dropout，使用 BatchNorm 的统计值（固定模式）
    """
    model.eval()  # 切换到评估模式

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = model(images)
        loss = criterion(logits, labels)

        batch_size     = images.size(0)
        total_loss    += loss.item() * batch_size
        preds          = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc  = total_correct / total_samples

    # 记录到 TensorBoard
    writer.add_scalar(f"Loss/{split}", avg_loss, epoch)
    writer.add_scalar(f"Accuracy/{split}", avg_acc, epoch)

    return {"loss": avg_loss, "accuracy": avg_acc}


# ============================================================
# 主训练流程
# ============================================================

def main() -> None:
    args = parse_args()

    # ---- 初始化 ----
    set_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 60)
    print("ViT CIFAR-10 训练")
    print("=" * 60)
    print(f"设备:       {device}")
    print(f"图像尺寸:   {args.image_size}×{args.image_size}")
    print(f"块大小:     {args.patch_size}×{args.patch_size}")
    print(f"嵌入维度:   {args.embed_dim}")
    print(f"层数/头数:  {args.depth} / {args.num_heads}")
    print(f"训练 epoch: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"学习率:     {args.lr}")
    print()

    # ---- 数据 ----
    train_loader, val_loader, _ = get_cifar10_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
    )

    # ---- 模型 ----
    model = build_model(args).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量:   {total_params:,}")
    print(f"可训练参数量:   {trainable_params:,}")

    # ---- 损失函数 ----
    # CrossEntropyLoss = LogSoftmax + NLLLoss，不需要提前对 logits 做 softmax
    # label_smoothing=0.1: 防止模型过于自信（DeiT 的标准配置）
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ---- 优化器: AdamW ----
    # AdamW 是 Adam + 正确的 weight decay（权重衰减不作用于 bias 和 LayerNorm）
    # ViT 原论文和所有主流实现均使用 AdamW
    # 对 bias 和 LayerNorm 参数不应用 weight decay（它们不会过拟合）
    no_decay = {"bias", "norm"}
    params_with_decay    = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    params_without_decay = [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)]
    optimizer = torch.optim.AdamW([
        {"params": params_with_decay,    "weight_decay": args.weight_decay},
        {"params": params_without_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.999))

    # ---- 学习率调度器 ----
    scheduler = get_lr_scheduler(optimizer, args)

    # ---- TensorBoard 记录器 ----
    writer = SummaryWriter(log_dir=args.log_dir)

    # ---- 训练循环 ----
    best_val_acc = 0.0
    best_epoch   = 0
    history = []

    print("\n开始训练...\n")
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch}/{args.epochs}]  LR={current_lr:.6f}")

        # 训练一个 epoch
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            args.grad_clip, epoch, writer,
        )

        # 在验证集上评估
        val_metrics = evaluate(
            model, val_loader, criterion, device, epoch, writer, split="val",
        )

        # 更新学习率
        scheduler.step()

        # 记录学习率到 TensorBoard
        writer.add_scalar("LR", current_lr, epoch)

        # 打印本 epoch 统计
        epoch_time = time.time() - epoch_start
        print(
            f"  Train Loss={train_metrics['loss']:.4f}  Train Acc={train_metrics['accuracy']:.4f}"
        )
        print(
            f"  Val   Loss={val_metrics['loss']:.4f}  Val   Acc={val_metrics['accuracy']:.4f}"
            f"  [{epoch_time:.1f}s]"
        )

        # 记录历史
        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc":  train_metrics["accuracy"],
            "val_loss":   val_metrics["loss"],
            "val_acc":    val_metrics["accuracy"],
            "lr":         current_lr,
        })

        # ---- 保存最优模型 ----
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_epoch   = epoch
            save_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_acc":     best_val_acc,
                "args": vars(args),
            }, save_path)
            print(f"  💾 保存最优模型  Val Acc={best_val_acc:.4f}  → {save_path}")

        # ---- 定期保存 checkpoint ----
        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pth")
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "val_acc":     val_metrics["accuracy"],
                "args": vars(args),
            }, ckpt_path)

        print()

    # ---- 训练完成 ----
    total_time = time.time() - train_start
    writer.close()

    print("=" * 60)
    print(f"训练完成！")
    print(f"总用时:     {total_time/60:.1f} 分钟")
    print(f"最优 epoch: {best_epoch}")
    print(f"最优验证准确率: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"最优模型路径: {os.path.join(args.checkpoint_dir, 'best_model.pth')}")
    print(f"\n查看训练曲线: tensorboard --logdir {args.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
