"""
evaluate.py — ViT CIFAR-10 标准测试评估脚本
============================================
功能:
  1. 在 CIFAR-10 官方测试集 (10,000 张) 上评估训练好的模型
  2. 输出标准分类报告（precision/recall/F1，每类别 + 宏平均）
  3. 生成混淆矩阵（文本版，无需 matplotlib）
  4. 输出 Top-1 / Top-5 准确率
  5. 将结果保存到 results/test_results.txt

使用方式:
  python evaluate.py                          # 使用默认 checkpoint
  python evaluate.py --checkpoint path/to/ckpt.pth
  python evaluate.py --batch-size 128        # 调整测试批次大小
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from dataset import CIFAR10_CLASSES, CIFAR10_CLASSES_ZH, get_cifar10_loaders
from vit import VisionTransformer


# ============================================================
# 命令行参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ViT CIFAR-10 测试集评估脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "checkpoints", "best_model.pth"),
                        help="要评估的 checkpoint 路径")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "data"),
                        help="CIFAR-10 数据目录")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="测试批次大小（不影响结果，只影响速度）")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="数据加载工作进程数")
    parser.add_argument("--device", type=str, default="auto",
                        help="计算设备: auto/cpu/cuda")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"),
                        help="结果文件保存目录")
    return parser.parse_args()


# ============================================================
# 辅助函数
# ============================================================

def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[VisionTransformer, dict]:
    """加载 checkpoint 中的模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到 checkpoint: {checkpoint_path}\n"
            f"请先运行 python train.py 训练模型。"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    saved_args = checkpoint.get("args", {})

    model = VisionTransformer(
        image_size=saved_args.get("image_size", 224),
        patch_size=saved_args.get("patch_size", 16),
        in_channels=3,
        num_classes=10,
        embed_dim=saved_args.get("embed_dim", 192),
        depth=saved_args.get("depth", 12),
        num_heads=saved_args.get("num_heads", 3),
        mlp_ratio=saved_args.get("mlp_ratio", 4.0),
        qkv_bias=True,
        pool="cls",
        mlp_head=False,
        dropout=0.0,  # 推理时关闭 dropout
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()
    return model, saved_args


# ============================================================
# 核心评估逻辑
# ============================================================

@torch.no_grad()
def run_evaluation(
    model: VisionTransformer,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> Dict:
    """
    在测试集上运行前向传播，收集所有预测结果

    返回包含以下字段的字典:
      - all_preds:   所有样本的预测类别 (N,)
      - all_labels:  所有样本的真实类别 (N,)
      - all_probs:   所有样本的预测概率 (N, num_classes)
      - top1_acc:    Top-1 准确率
      - top5_acc:    Top-5 准确率
      - total_samples: 测试样本总数
      - elapsed_time:  推理耗时（秒）
    """
    all_preds:  List[int] = []
    all_labels: List[int] = []
    all_probs:  List[torch.Tensor] = []

    top1_correct = 0
    top5_correct = 0
    total_samples = 0

    start_time = time.time()

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, _ = model(images)              # (N, 10)
        probs = F.softmax(logits, dim=1)       # (N, 10) 概率

        # Top-1 准确率
        top1_preds = logits.argmax(dim=1)      # (N,) 最大 logit 的类别
        top1_correct += (top1_preds == labels).sum().item()

        # Top-5 准确率
        # topk 返回最大的 k 个值的索引
        _, top5_preds = logits.topk(5, dim=1)  # (N, 5)
        # 对每个样本检查真实标签是否在 Top-5 预测中
        correct_top5 = top5_preds.eq(labels.unsqueeze(1).expand_as(top5_preds))
        top5_correct += correct_top5.any(dim=1).sum().item()

        # 收集预测结果
        all_preds.extend(top1_preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_probs.append(probs.cpu())

        total_samples += images.size(0)

    elapsed = time.time() - start_time

    return {
        "all_preds":     all_preds,
        "all_labels":    all_labels,
        "all_probs":     torch.cat(all_probs, dim=0),  # (N, 10)
        "top1_acc":      top1_correct / total_samples,
        "top5_acc":      top5_correct / total_samples,
        "total_samples": total_samples,
        "elapsed_time":  elapsed,
    }


# ============================================================
# 指标计算（不依赖 scikit-learn）
# ============================================================

def compute_per_class_metrics(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int = 10,
) -> Dict[str, List[float]]:
    """
    手动计算每个类别的 Precision、Recall、F1、Support

    指标定义:
      Precision = TP / (TP + FP)  该类的预测中有多少是对的
      Recall    = TP / (TP + FN)  该类的真实样本有多少被正确找出
      F1        = 2 × P × R / (P + R)  精确率和召回率的调和平均

    其中:
      TP (True Positive):  预测为该类，且真实也是该类
      FP (False Positive): 预测为该类，但真实不是该类
      FN (False Negative): 真实是该类，但预测不是该类
    """
    # 初始化每类的计数器
    tp = [0] * num_classes  # 真正例
    fp = [0] * num_classes  # 假正例
    fn = [0] * num_classes  # 假负例

    for pred, label in zip(all_preds, all_labels):
        if pred == label:
            tp[label] += 1       # 预测正确: 该类的 TP +1
        else:
            fp[pred]  += 1       # 预测错误: 预测类的 FP +1
            fn[label] += 1       # 预测错误: 真实类的 FN +1

    # 计算每类指标
    precision_list = []
    recall_list    = []
    f1_list        = []
    support_list   = []

    for c in range(num_classes):
        support = tp[c] + fn[c]  # 该类的真实样本总数

        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        precision_list.append(p)
        recall_list.append(r)
        f1_list.append(f1)
        support_list.append(support)

    return {
        "precision": precision_list,
        "recall":    recall_list,
        "f1":        f1_list,
        "support":   support_list,
    }


def compute_confusion_matrix(
    all_preds: List[int],
    all_labels: List[int],
    num_classes: int = 10,
) -> List[List[int]]:
    """
    计算混淆矩阵 (num_classes × num_classes)

    混淆矩阵定义:
      matrix[i][j] = 真实类别为 i 且预测类别为 j 的样本数
      对角线: 预测正确
      非对角线: 预测错误（具体错成了哪类）
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for label, pred in zip(all_labels, all_preds):
        matrix[label][pred] += 1
    return matrix


# ============================================================
# 结果格式化与输出
# ============================================================

def format_results(
    eval_results: Dict,
    per_class: Dict,
    conf_matrix: List[List[int]],
    checkpoint_path: str,
    saved_args: dict,
) -> str:
    """
    将评估结果格式化为标准测试报告字符串
    """
    num_classes = 10
    lines = []

    # ---- 标题 ----
    lines.append("=" * 70)
    lines.append("ViT CIFAR-10 测试集评估报告")
    lines.append("=" * 70)
    lines.append("")

    # ---- 模型信息 ----
    lines.append("【模型配置】")
    lines.append(f"  Checkpoint:   {checkpoint_path}")
    lines.append(f"  图像尺寸:     {saved_args.get('image_size', 224)}×{saved_args.get('image_size', 224)}")
    lines.append(f"  块大小:       {saved_args.get('patch_size', 16)}×{saved_args.get('patch_size', 16)}")
    lines.append(f"  嵌入维度:     {saved_args.get('embed_dim', 192)}")
    lines.append(f"  层数 / 头数:  {saved_args.get('depth', 12)} / {saved_args.get('num_heads', 3)}")
    lines.append("")

    # ---- 总体指标 ----
    lines.append("【总体指标】")
    lines.append(f"  测试样本数:       {eval_results['total_samples']:,}")
    lines.append(f"  推理耗时:         {eval_results['elapsed_time']:.2f} 秒")
    lines.append(f"  每样本推理时间:   {eval_results['elapsed_time']/eval_results['total_samples']*1000:.2f} ms")
    lines.append(f"  Top-1 准确率:     {eval_results['top1_acc']:.4f}  ({eval_results['top1_acc']*100:.2f}%)")
    lines.append(f"  Top-5 准确率:     {eval_results['top5_acc']:.4f}  ({eval_results['top5_acc']*100:.2f}%)")
    lines.append("")

    # ---- 宏平均指标 ----
    macro_p  = sum(per_class["precision"]) / num_classes
    macro_r  = sum(per_class["recall"])    / num_classes
    macro_f1 = sum(per_class["f1"])        / num_classes
    lines.append(f"  宏平均 Precision: {macro_p:.4f}")
    lines.append(f"  宏平均 Recall:    {macro_r:.4f}")
    lines.append(f"  宏平均 F1-Score:  {macro_f1:.4f}")
    lines.append("")

    # ---- 每类别详细指标 ----
    lines.append("【每类别指标】")
    header = f"  {'类别':<14} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}"
    lines.append(header)
    lines.append("  " + "-" * 56)

    for i in range(num_classes):
        cls_label = f"{CIFAR10_CLASSES[i]} ({CIFAR10_CLASSES_ZH[i]})"
        lines.append(
            f"  {cls_label:<14} "
            f"{per_class['precision'][i]:>10.4f} "
            f"{per_class['recall'][i]:>10.4f} "
            f"{per_class['f1'][i]:>10.4f} "
            f"{per_class['support'][i]:>10d}"
        )

    lines.append("  " + "-" * 56)
    lines.append(
        f"  {'macro avg':<14} "
        f"{macro_p:>10.4f} "
        f"{macro_r:>10.4f} "
        f"{macro_f1:>10.4f} "
        f"{eval_results['total_samples']:>10d}"
    )
    lines.append("")

    # ---- 混淆矩阵 ----
    lines.append("【混淆矩阵】（行=真实类别，列=预测类别）")
    lines.append("")

    # 列标题
    col_header = "              " + "".join(f"{CIFAR10_CLASSES[j][:6]:>8}" for j in range(num_classes))
    lines.append(col_header)
    lines.append("  " + "-" * (12 + 8 * num_classes))

    for i in range(num_classes):
        row_label = f"  {CIFAR10_CLASSES[i][:10]:<12}"
        row_values = "".join(f"{conf_matrix[i][j]:>8}" for j in range(num_classes))
        lines.append(row_label + row_values)

    lines.append("")

    # ---- 最容易混淆的类别对 ----
    lines.append("【最容易混淆的 5 个类别对】")
    errors = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and conf_matrix[i][j] > 0:
                errors.append((conf_matrix[i][j], i, j))
    errors.sort(reverse=True)

    for count, true_cls, pred_cls in errors[:5]:
        true_name = f"{CIFAR10_CLASSES[true_cls]}({CIFAR10_CLASSES_ZH[true_cls]})"
        pred_name = f"{CIFAR10_CLASSES[pred_cls]}({CIFAR10_CLASSES_ZH[pred_cls]})"
        lines.append(f"  真实={true_name:<16} 预测={pred_name:<16} 次数={count}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


# ============================================================
# 主函数
# ============================================================

def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    print("=" * 60)
    print("ViT CIFAR-10 测试集评估")
    print("=" * 60)
    print(f"设备:       {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print()

    # ---- 加载模型 ----
    model, saved_args = load_model(args.checkpoint, device)
    image_size = saved_args.get("image_size", 224)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    print()

    # ---- 加载测试集 ----
    print("加载测试数据集...")
    _, _, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print()

    # ---- 运行评估 ----
    print("正在评估（这可能需要几分钟）...")
    eval_results = run_evaluation(model, test_loader, device)
    print(f"评估完成，耗时 {eval_results['elapsed_time']:.1f}s")
    print()

    # ---- 计算指标 ----
    per_class  = compute_per_class_metrics(eval_results["all_preds"], eval_results["all_labels"])
    conf_matrix = compute_confusion_matrix(eval_results["all_preds"], eval_results["all_labels"])

    # ---- 格式化报告 ----
    report = format_results(eval_results, per_class, conf_matrix, args.checkpoint, saved_args)

    # ---- 打印到控制台 ----
    print(report)

    # ---- 保存到文件 ----
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "test_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 评估报告已保存到: {output_path}")


if __name__ == "__main__":
    main()
