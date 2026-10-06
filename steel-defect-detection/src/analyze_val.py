#!/usr/bin/env python3
"""本地验证集分析：对比不同推理配置或提交策略的指标。

用法:
  python v1/src/analyze_val.py --predictions <预测JSON> [--predictions <另一个> ...]

预测 JSON 需要同名 `.coverage.json`（推理脚本自动生成）以确定验证集图片清单。
Ground Truth 从 --annotations（默认 train/）按文件名主干匹配。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_mixed as ev  # noqa: E402


def load_gt(images, annotations):
    return ev.load_ground_truth(images, annotations)


def micro_pr(preds, gt, classes):
    """把给定预测全部当作正例，逐类贪心一对一匹配后汇总。"""
    tp = fp = fn = 0
    for name in classes:
        targets = {img: [o["bbox"] for o in objs if o["category_name"] == name]
                   for img, objs in gt.items()}
        n_gt = sum(len(v) for v in targets.values())
        ranked = sorted((p for p in preds if p["category_name"] == name),
                        key=lambda p: -p["score"])
        used = defaultdict(set)
        hit = 0
        for p in ranked:
            img = p["image_id"]
            best = max(((ev.iou(p["bbox"], b), j)
                        for j, b in enumerate(targets[img]) if j not in used[img]),
                       default=(0, -1))
            if best[0] >= 0.5:
                used[img].add(best[1])
                hit += 1
        tp += hit
        fp += len(ranked) - hit
        fn += n_gt - hit
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def class_ap(preds, gt, classes):
    aps = {}
    for name in classes:
        m, _ = ev.class_metrics(name, preds, gt)
        aps[name] = m["ap50"]
    present = [v for v in aps.values() if v is not None]
    return aps, (sum(present) / len(present) if present else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", action="append", required=True)
    p.add_argument("--labels", action="append", default=None)
    p.add_argument("--annotations", default="train")
    a = p.parse_args()
    labels = a.labels or [Path(x).stem for x in a.predictions]

    cov_path = Path(a.predictions[0]).with_suffix(Path(a.predictions[0]).suffix + ".coverage.json")
    names = json.loads(cov_path.read_text(encoding="utf-8"))["images"]
    images = [Path(a.annotations) / n for n in names]
    missing = [x for x in images if not x.exists()]
    if missing:
        raise SystemExit(f"缺少 {len(missing)} 张验证图片，例如 {missing[0]}")
    gt = load_gt(images, a.annotations)

    print(f"验证集 {len(images)} 张\n")
    header = f"{'配置':<22}{'框数':>8}{'mAP50':>9}{'微P':>8}{'微R':>8}{'微F1':>8}"
    print(header)
    print("-" * len(header))
    for path, label in zip(a.predictions, labels):
        preds = json.loads(Path(path).read_text(encoding="utf-8"))
        for x in preds:
            x["bbox"] = [float(v) for v in x["bbox"]]
            x["score"] = float(x["score"])
        _, map50 = class_ap(preds, gt, ev.CLASSES)
        _, _, _, precision, recall, f1 = micro_pr(preds, gt, ev.CLASSES)
        print(f"{label:<22}{len(preds):>8}{map50:>9.4f}{precision:>8.3f}{recall:>8.3f}{f1:>8.3f}")


if __name__ == "__main__":
    main()
