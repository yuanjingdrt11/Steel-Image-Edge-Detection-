#!/usr/bin/env python3
"""在验证集上按指定指标选择逐类置信度阈值。

不同应用场景对精确率与召回率的侧重不同（如工业质检通常召回优先），因此除 F2 外还需
支持 F1、F0.5 等工作点。本脚本对每个类别扫描完整阈值曲线，取指定指标的逐类最优点，
并报告在该阈值组合下的微平均 Precision / Recall / F1 / F2 与 mAP50。

用法:
  python v1/src/select_thresholds.py \
    --predictions val_predictions.json \
    --annotations train \
    --output thresholds_f1.json \
    --metric f1
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_mixed as ev  # noqa: E402


def curve(predictions, ground_truth, name):
    """按分数分组返回 (threshold, tp, fp, fn, P, R, F1, F2, Fbeta) 曲线。"""
    targets = {img: [o["bbox"] for o in objs if o["category_name"] == name]
               for img, objs in ground_truth.items()}
    total = sum(len(v) for v in targets.values())
    ranked = sorted((p for p in predictions if p["category_name"] == name),
                    key=lambda p: -p["score"])
    used = defaultdict(set)
    rows, tp = [], 0
    for index, pred in enumerate(ranked):
        img = pred["image_id"]
        best = max(((ev.iou(pred["bbox"], box), j)
                    for j, box in enumerate(targets[img]) if j not in used[img]),
                   default=(0, -1))
        if best[0] >= 0.5:
            used[img].add(best[1])
            tp += 1
        if index + 1 < len(ranked) and ranked[index + 1]["score"] == pred["score"]:
            continue  # 同分必须整组纳入，避免阈值切在并分中间
        fp = (index + 1) - tp
        fn = total - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / total if total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f2 = 5 * tp / (5 * tp + 4 * fn + fp) if total else 0.0
        rows.append({"threshold": pred["score"], "tp": tp, "fp": fp, "fn": fn,
                     "precision": precision, "recall": recall, "f1": f1, "f2": f2})
    return rows, total


def score(row, metric, beta):
    if metric == "f1":
        return row["f1"]
    if metric == "f2":
        return row["f2"]
    if metric == "fbeta":
        p, r = row["precision"], row["recall"]
        b2 = beta * beta
        return (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0
    raise ValueError(f"unknown metric: {metric}")


def micro(predictions, ground_truth, thresholds):
    tp = fp = fn = 0
    for name in ev.CLASSES:
        targets = {img: [o["bbox"] for o in objs if o["category_name"] == name]
                   for img, objs in ground_truth.items()}
        total = sum(len(v) for v in targets.values())
        selected = sorted((p for p in predictions
                           if p["category_name"] == name
                           and p["score"] >= thresholds.get(name, 0.01)),
                          key=lambda p: -p["score"])
        used = defaultdict(set)
        hit = 0
        for pred in selected:
            img = pred["image_id"]
            best = max(((ev.iou(pred["bbox"], box), j)
                        for j, box in enumerate(targets[img]) if j not in used[img]),
                       default=(0, -1))
            if best[0] >= 0.5:
                used[img].add(best[1])
                hit += 1
        tp += hit
        fp += len(selected) - hit
        fn += total - hit
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * tp / (5 * tp + 4 * fn + fp) if tp + fn + fp else 0.0
    return {"boxes": tp + fp, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1, "f2": f2}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--annotations", default="train")
    p.add_argument("--output", required=True)
    p.add_argument("--metric", default="f1", choices=["f1", "f2", "fbeta"])
    p.add_argument("--beta", type=float, default=1.0, help="--metric fbeta 时的 beta")
    p.add_argument("--fallback-floor", type=float, default=0.10,
                   help="验证样本少于 --min-gt 的类别使用的保守阈值下限")
    p.add_argument("--min-gt", type=int, default=10,
                   help="低于该 GT 数量的类别视为小样本，阈值不可靠")
    p.add_argument("--report", default=None)
    args = p.parse_args()

    cov = Path(args.predictions).with_suffix(Path(args.predictions).suffix + ".coverage.json")
    names = json.loads(cov.read_text(encoding="utf-8"))["images"]
    images = [Path(args.annotations) / n for n in names]
    missing = [x for x in images if not x.exists()]
    if missing:
        raise SystemExit(f"缺少 {len(missing)} 张验证图片，例如 {missing[0]}")
    gt = ev.load_ground_truth(images, args.annotations)

    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    for item in predictions:
        item["bbox"] = [float(v) for v in item["bbox"]]
        item["score"] = float(item["score"])

    thresholds, per_class = {}, {}
    for name in ev.CLASSES:
        rows, total = curve(predictions, gt, name)
        if not rows or not total:
            thresholds[name] = args.fallback_floor
            per_class[name] = {"ground_truth": total, "threshold": args.fallback_floor,
                               "note": "无 GT 或无候选，使用保守下限"}
            continue
        best = max(rows, key=lambda r: (score(r, args.metric, args.beta), r["threshold"]))
        threshold = best["threshold"]
        if total < args.min_gt:
            threshold = min(threshold, args.fallback_floor)
            note = f"GT={total} 少于 {args.min_gt}，阈值回退到 {args.fallback_floor}"
        else:
            note = ""
        thresholds[name] = threshold
        per_class[name] = {"ground_truth": total, "threshold": threshold,
                           "precision": best["precision"], "recall": best["recall"],
                           "f1": best["f1"], "f2": best["f2"], "note": note}

    summary = micro(predictions, gt, thresholds)
    aps = []
    for name in ev.CLASSES:
        m, _ = ev.class_metrics(name, predictions, gt)
        if m["ap50"] is not None:
            aps.append(m["ap50"])
    summary["map50_present_classes"] = sum(aps) / len(aps) if aps else None
    summary["metric"] = args.metric
    summary["images"] = len(images)

    Path(args.output).write_text(json.dumps(thresholds, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    if args.report:
        Path(args.report).write_text(
            json.dumps({"per_class": per_class, "summary": summary},
                       ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"指标 {args.metric}  验证集 {len(images)} 张")
    print(f"{'类别':<16}{'GT':>6}{'阈值':>8}{'P':>8}{'R':>8}{'F1':>8}  备注")
    print("-" * 72)
    for name in ev.CLASSES:
        row = per_class[name]
        if "precision" in row:
            print(f"{name:<16}{row['ground_truth']:>6}{row['threshold']:>8.3f}"
                  f"{row['precision']:>8.3f}{row['recall']:>8.3f}{row['f1']:>8.3f}  {row['note']}")
        else:
            print(f"{name:<16}{row['ground_truth']:>6}{row['threshold']:>8.3f}"
                  f"{'-':>8}{'-':>8}{'-':>8}  {row['note']}")
    print("-" * 72)
    print(f"微平均: 框数 {summary['boxes']}  P {summary['precision']:.3f}  "
          f"R {summary['recall']:.3f}  F1 {summary['f1']:.3f}  F2 {summary['f2']:.3f}")
    print(f"mAP50(有GT类别): {summary['map50_present_classes']:.4f}")


if __name__ == "__main__":
    main()
