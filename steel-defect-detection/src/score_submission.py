#!/usr/bin/env python3
"""按官方评分公式评估预测：总分 = 0.2*P + 0.6*R + 0.2*mAP50

P、R 为全部提交框在 IoU=0.5 下的微平均（逐类一对一贪心匹配），
mAP50 为各类 AP50 的均值（仅含有 GT 的类别）。
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate_mixed as ev


def prf(preds, gt):
    tp = fp = fn = 0
    for name in ev.CLASSES:
        tg = {i: [o["bbox"] for o in v if o["category_name"] == name] for i, v in gt.items()}
        n = sum(len(x) for x in tg.values())
        sel = sorted((p for p in preds if p["category_name"] == name), key=lambda p: -p["score"])
        used = defaultdict(set); hit = 0
        for p in sel:
            b = max(((ev.iou(p["bbox"], bb), j) for j, bb in enumerate(tg[p["image_id"]])
                     if j not in used[p["image_id"]]), default=(0, -1))
            if b[0] >= 0.5:
                used[p["image_id"]].add(b[1]); hit += 1
        tp += hit; fp += len(sel) - hit; fn += n - hit
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return P, R, tp, fp, fn


def map50(preds, gt):
    aps = []
    for name in ev.CLASSES:
        m, _ = ev.class_metrics(name, preds, gt)
        if m["ap50"] is not None:
            aps.append(m["ap50"])
    return sum(aps) / len(aps) if aps else 0.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", action="append", required=True)
    p.add_argument("--labels", action="append", default=None)
    p.add_argument("--annotations", default="train")
    p.add_argument("--exclude-classes", default="qilie")
    a = p.parse_args()
    labels = a.labels or [Path(x).stem for x in a.predictions]
    excluded = {c.strip() for c in a.exclude_classes.split(",") if c.strip()}

    cov = Path(a.predictions[0]).with_suffix(Path(a.predictions[0]).suffix + ".coverage.json")
    names = json.loads(cov.read_text(encoding="utf-8"))["images"]
    images = [Path(a.annotations) / n for n in names]
    missing = [x for x in images if not x.exists()]
    if missing:
        raise SystemExit(f"缺少 {len(missing)} 张验证图片，例如 {missing[0]}")
    gt = ev.load_ground_truth(images, a.annotations)

    print(f"验证集 {len(images)} 张 | 剔除类别: {sorted(excluded) or '无'}")
    print(f"{'模型':<18}{'框数':>8}{'P':>8}{'R':>8}{'mAP50':>8}{'总分':>9}{'×100':>8}")
    print("-" * 68)
    for path, label in zip(a.predictions, labels):
        preds = json.loads(Path(path).read_text(encoding="utf-8"))
        preds = [x for x in preds if x["category_name"] not in excluded]
        for x in preds:
            x["bbox"] = [float(v) for v in x["bbox"]]
            x["score"] = float(x["score"])
        P, R, tp, fp, fn = prf(preds, gt)
        M = map50(preds, gt)
        S = 0.2 * P + 0.6 * R + 0.2 * M
        print(f"{label:<18}{len(preds):>8}{P:>8.3f}{R:>8.3f}{M:>8.3f}{S:>9.4f}{S*100:>8.2f}")


if __name__ == "__main__":
    main()
