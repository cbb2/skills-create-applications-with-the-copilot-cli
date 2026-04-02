"""
quick_start.py — 端到端快速验证脚本
======================================
此脚本演示完整的 ViT CIFAR-10 流程：
  1. 数据预处理（自动下载 CIFAR-10）
  2. 模型训练（轻量配置，3 个 epoch 快速验证）
  3. 推理（随机从测试集取若干张图）
  4. 输出标准测试报告

用途:
  - 验证代码是否能正常运行（端到端冒烟测试）
  - 在 CPU 环境下快速体验完整流程
  - 作为真正训练前的"沙箱"实验

注意:
  仅 3 epoch 训练精度不高（约 20~35%），这是正常的！
  完整训练（30~100 epoch + GPU）可达 70~80%。

运行:
  python quick_start.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from dataset import (
    CIFAR10_CLASSES,
    CIFAR10_CLASSES_ZH,
    get_cifar10_loaders,
    show_dataset_info,
)
from vit import VisionTransformer


# ============================================================
# 快速验证配置（使用更小的模型和更低分辨率）
# ============================================================

CFG = {
    # 图像配置: 使用原始 32×32 + patch_size=4 (更快，无需 224×224 上采样)
    "image_size": 32,
    "patch_size": 4,

    # 模型: 极简 ViT (快速实验)
    "embed_dim": 64,
    "depth": 4,
    "num_heads": 4,
    "mlp_ratio": 2.0,
    "dropout": 0.1,

    # 训练
    "epochs": 3,
    "batch_size": 128,
    "lr": 1e-3,
    "weight_decay": 0.05,
    "grad_clip": 1.0,

    # 数据
    "num_workers": 0,  # 0 = 主进程加载，兼容所有操作系统
    "val_split": 0.1,

    # 输出
    "checkpoint_dir": os.path.join(os.path.dirname(__file__), "checkpoints"),
    "results_dir": os.path.join(os.path.dirname(__file__), "results"),
    "data_dir": os.path.join(os.path.dirname(__file__), "data"),
}


# ============================================================
# 工具函数
# ============================================================

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算批次准确率"""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ============================================================
# 训练函数
# ============================================================

def train(model, loader, optimizer, criterion, device, grad_clip):
    """训练一个 epoch，返回 (avg_loss, avg_acc)"""
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += compute_accuracy(logits, labels) * bs
        n += bs
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估，返回 (avg_loss, avg_acc)"""
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)
        loss = criterion(logits, labels)
        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc  += compute_accuracy(logits, labels) * bs
        n += bs
    return total_loss / n, total_acc / n


# ============================================================
# 测试集完整评估（混淆矩阵 + 分类报告）
# ============================================================

@torch.no_grad()
def full_test_evaluation(model, test_loader, device):
    """
    在测试集上运行完整评估，计算 Top-1/Top-5 准确率、
    每类别 Precision/Recall/F1 以及混淆矩阵
    """
    model.eval()
    num_classes = 10

    all_preds  = []
    all_labels = []
    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    start = time.time()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)

        # Top-1
        preds = logits.argmax(dim=1)
        top1_correct += (preds == labels).sum().item()

        # Top-5
        _, top5 = logits.topk(5, dim=1)
        correct5 = top5.eq(labels.unsqueeze(1).expand_as(top5))
        top5_correct += correct5.any(dim=1).sum().item()

        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_samples += images.size(0)

    elapsed = time.time() - start

    # ---- 混淆矩阵 ----
    cm = [[0] * num_classes for _ in range(num_classes)]
    for label, pred in zip(all_labels, all_preds):
        cm[label][pred] += 1

    # ---- 每类别指标 ----
    tp = [cm[i][i] for i in range(num_classes)]
    fp = [sum(cm[j][i] for j in range(num_classes)) - cm[i][i] for i in range(num_classes)]
    fn = [sum(cm[i][j] for j in range(num_classes)) - cm[i][i] for i in range(num_classes)]

    precision, recall, f1, support = [], [], [], []
    for i in range(num_classes):
        p  = tp[i] / (tp[i] + fp[i]) if tp[i] + fp[i] > 0 else 0.0
        r  = tp[i] / (tp[i] + fn[i]) if tp[i] + fn[i] > 0 else 0.0
        f  = 2 * p * r / (p + r) if p + r > 0 else 0.0
        s  = tp[i] + fn[i]
        precision.append(p); recall.append(r); f1.append(f); support.append(s)

    return {
        "top1_acc":    top1_correct / total_samples,
        "top5_acc":    top5_correct / total_samples,
        "precision":   precision,
        "recall":      recall,
        "f1":          f1,
        "support":     support,
        "conf_matrix": cm,
        "total":       total_samples,
        "elapsed":     elapsed,
    }


def print_test_report(results, save_path=None):
    """打印标准测试报告（同时可选保存到文件）"""
    num_classes = 10
    lines = []

    lines.append("=" * 70)
    lines.append("ViT CIFAR-10  标准测试报告")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"  测试样本数:     {results['total']:,}")
    lines.append(f"  推理耗时:       {results['elapsed']:.2f}s  "
                 f"({results['elapsed']/results['total']*1000:.2f}ms / 张)")
    lines.append(f"  Top-1 准确率:   {results['top1_acc']:.4f}  "
                 f"({results['top1_acc']*100:.2f}%)")
    lines.append(f"  Top-5 准确率:   {results['top5_acc']:.4f}  "
                 f"({results['top5_acc']*100:.2f}%)")
    lines.append("")

    # 宏平均
    mp  = sum(results["precision"]) / num_classes
    mr  = sum(results["recall"])    / num_classes
    mf1 = sum(results["f1"])        / num_classes
    lines.append(f"  宏平均 Precision: {mp:.4f}")
    lines.append(f"  宏平均 Recall:    {mr:.4f}")
    lines.append(f"  宏平均 F1:        {mf1:.4f}")
    lines.append("")

    # 每类别
    lines.append("【每类别指标】")
    lines.append(f"  {'类别':<17} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    lines.append("  " + "-" * 57)
    for i in range(num_classes):
        name = f"{CIFAR10_CLASSES[i]}({CIFAR10_CLASSES_ZH[i]})"
        lines.append(
            f"  {name:<17} "
            f"{results['precision'][i]:>10.4f} "
            f"{results['recall'][i]:>10.4f} "
            f"{results['f1'][i]:>10.4f} "
            f"{results['support'][i]:>8d}"
        )
    lines.append("  " + "-" * 57)
    lines.append(f"  {'macro avg':<17} {mp:>10.4f} {mr:>10.4f} {mf1:>10.4f} {results['total']:>8d}")
    lines.append("")

    # 混淆矩阵
    lines.append("【混淆矩阵】（行=真实，列=预测）")
    col_h = "               " + "".join(f"{CIFAR10_CLASSES[j][:5]:>7}" for j in range(num_classes))
    lines.append(col_h)
    lines.append("  " + "-" * (13 + 7 * num_classes))
    cm = results["conf_matrix"]
    for i in range(num_classes):
        row = f"  {CIFAR10_CLASSES[i]:<13}" + "".join(f"{cm[i][j]:>7}" for j in range(num_classes))
        lines.append(row)
    lines.append("")

    # 最易混淆对
    lines.append("【最易混淆的 5 个类别对】")
    errors = sorted(
        [(cm[i][j], i, j) for i in range(num_classes) for j in range(num_classes) if i != j],
        reverse=True
    )
    for cnt, ti, pi in errors[:5]:
        t = f"{CIFAR10_CLASSES[ti]}({CIFAR10_CLASSES_ZH[ti]})"
        p = f"{CIFAR10_CLASSES[pi]}({CIFAR10_CLASSES_ZH[pi]})"
        lines.append(f"  真实={t:<18} → 预测={p:<18} {cnt} 次")
    lines.append("")
    lines.append("=" * 70)

    report = "\n".join(lines)
    print(report)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\n📄 测试报告已保存到: {save_path}")

    return report


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("ViT CIFAR-10 端到端快速验证")
    print("=" * 60)
    print()

    device = get_device()
    print(f"计算设备: {device}")
    print(f"图像尺寸: {CFG['image_size']}×{CFG['image_size']}")
    print(f"块大小:   {CFG['patch_size']}×{CFG['patch_size']}")
    print(f"嵌入维度: {CFG['embed_dim']}")
    print(f"训练轮数: {CFG['epochs']}（快速验证，完整训练建议 30~100 epoch）")
    print()

    # ── 步骤 1: 数据集 ─────────────────────────────────────
    print("步骤 1/4: 加载数据集")
    print("-" * 40)
    show_dataset_info(CFG["data_dir"])
    print()

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=CFG["data_dir"],
        image_size=CFG["image_size"],
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_workers"],
        val_split=CFG["val_split"],
        pin_memory=(device.type == "cuda"),
    )
    print()

    # ── 步骤 2: 构建模型 ───────────────────────────────────
    print("步骤 2/4: 构建模型")
    print("-" * 40)
    model = VisionTransformer(
        image_size=CFG["image_size"],
        patch_size=CFG["patch_size"],
        in_channels=3,
        num_classes=10,
        embed_dim=CFG["embed_dim"],
        depth=CFG["depth"],
        num_heads=CFG["num_heads"],
        mlp_ratio=CFG["mlp_ratio"],
        dropout=CFG["dropout"],
        pool="cls",
        mlp_head=False,
        qkv_bias=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    patch_count  = model.patch_embedding.num_patches
    seq_len      = patch_count + 1

    print(f"总参数量:       {total_params:,}")
    print(f"可训练参数量:   {trainable:,}")
    print(f"图像块数量:     {patch_count}  ({CFG['image_size']//CFG['patch_size']}×{CFG['image_size']//CFG['patch_size']})")
    print(f"Transformer 序列长度: {seq_len}  (含 CLS Token)")
    print()

    # ── 步骤 3: 训练 ───────────────────────────────────────
    print("步骤 3/4: 训练")
    print("-" * 40)

    # 损失函数: CrossEntropyLoss + Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器: AdamW（不对 bias/norm 参数做 weight decay）
    no_decay = {"bias", "norm"}
    wd_params    = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_wd_params = [p for n, p in model.named_parameters() if     any(nd in n for nd in no_decay)]
    optimizer = torch.optim.AdamW([
        {"params": wd_params,    "weight_decay": CFG["weight_decay"]},
        {"params": no_wd_params, "weight_decay": 0.0},
    ], lr=CFG["lr"])

    # 学习率调度: CosineAnnealingLR（简单版）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG["epochs"], eta_min=CFG["lr"] * 0.01
    )

    best_val_acc = 0.0
    best_state   = None

    print(f"{'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Time':>7}")
    print("  " + "-" * 57)

    for epoch in range(1, CFG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train(
            model, train_loader, optimizer, criterion, device, CFG["grad_clip"]
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        marker = " ◀ best" if val_acc > best_val_acc else ""
        print(
            f"  {epoch:>4}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
            f"{val_loss:>9.4f}  {val_acc:>8.4f}  {elapsed:>5.1f}s{marker}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # 保存最优模型权重
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 保存 checkpoint
    os.makedirs(CFG["checkpoint_dir"], exist_ok=True)
    ckpt_path = os.path.join(CFG["checkpoint_dir"], "best_model.pth")
    torch.save({
        "epoch":       CFG["epochs"],
        "model_state": best_state,
        "val_acc":     best_val_acc,
        "args": {
            "image_size": CFG["image_size"],
            "patch_size": CFG["patch_size"],
            "embed_dim":  CFG["embed_dim"],
            "depth":      CFG["depth"],
            "num_heads":  CFG["num_heads"],
            "mlp_ratio":  CFG["mlp_ratio"],
        },
    }, ckpt_path)
    print(f"\n💾 最优模型已保存: {ckpt_path}  (验证准确率: {best_val_acc:.4f})")
    print()

    # ── 步骤 4: 测试评估 ───────────────────────────────────
    print("步骤 4/4: 测试集评估")
    print("-" * 40)

    # 加载最优权重
    model.load_state_dict(best_state)

    results = full_test_evaluation(model, test_loader, device)

    report_path = os.path.join(CFG["results_dir"], "test_results.txt")
    print_test_report(results, save_path=report_path)

    # ── 快速推理演示 ───────────────────────────────────────
    print("\n【单张图片推理演示（从测试集随机取 5 张）】")
    model.eval()
    sample_images, sample_labels = next(iter(test_loader))
    sample_images = sample_images[:5]
    sample_labels = sample_labels[:5]

    with torch.no_grad():
        logits, _ = model(sample_images.to(device))
        probs = F.softmax(logits, dim=1).cpu()

    for i in range(5):
        true_cls = CIFAR10_CLASSES[sample_labels[i].item()]
        true_zh  = CIFAR10_CLASSES_ZH[sample_labels[i].item()]
        pred_idx = probs[i].argmax().item()
        pred_cls = CIFAR10_CLASSES[pred_idx]
        pred_zh  = CIFAR10_CLASSES_ZH[pred_idx]
        conf     = probs[i][pred_idx].item()
        correct  = "✅" if pred_idx == sample_labels[i].item() else "❌"
        print(
            f"  {correct} 真实: {true_cls}({true_zh})<18  "
            f"预测: {pred_cls}({pred_zh})  置信度: {conf:.4f}"
        )

    print()
    print("=" * 60)
    print("端到端验证完成！✅")
    print()
    print("下一步:")
    print(f"  完整训练:  python train.py --epochs 30")
    print(f"  测试评估:  python evaluate.py")
    print(f"  单图推理:  python inference.py --image <path/to/image.jpg>")
    print(f"  训练曲线:  tensorboard --logdir runs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
