#!/usr/bin/env python3
"""把多尺度/多视角的预测合并为一个集合（同类同图内做贪心 NMS）。

用于测试时增强（TTA）：同一模型在不同尺度或翻转下推理，合并后召回通常上升。
"""
import argparse, json
from collections import defaultdict
from pathlib import Path


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def _wbf(items, iou_thr, conf_thr):
    """加权框融合：把重叠框按置信度加权平均，输出单一框。"""
    clusters = []
    for item in items:
        box = item["bbox"]
        best, best_iou = None, 0.0
        for cluster in clusters:
            overlap = iou(box, cluster["box"])
            if overlap > best_iou:
                best, best_iou = cluster, overlap
        if best is not None and best_iou > iou_thr:
            n = best["n"]
            best["box"] = [(box[i] * item["score"] + best["box"][i] * best["score_sum"]) /
                           (item["score"] + best["score_sum"]) for i in range(4)]
            best["score_sum"] += item["score"]
            best["n"] = n + 1
        else:
            clusters.append({"box": list(box), "score_sum": item["score"], "n": 1,
                             "image_id": item["image_id"], "category_name": item["category_name"]})
    out = []
    for cluster in clusters:
        score = cluster["score_sum"] / cluster["n"]
        if score < conf_thr:
            continue
        out.append({"image_id": cluster["image_id"], "category_name": cluster["category_name"],
                    "bbox": cluster["box"], "score": score})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", action="append", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--iou", type=float, default=0.55)
    p.add_argument("--mode", choices=["nms", "wbf"], default="nms",
                   help="nms=贪心去重；wbf=加权框融合（通常 mAP 更高）")
    p.add_argument("--wbf-iou", type=float, default=0.55)
    p.add_argument("--wbf-conf", type=float, default=0.001)
    a = p.parse_args()

    pool = []
    for path in a.predictions:
        for item in json.loads(Path(path).read_text(encoding="utf-8")):
            item["bbox"] = [float(v) for v in item["bbox"]]
            item["score"] = float(item["score"])
            pool.append(item)

    groups = defaultdict(list)
    for item in pool:
        groups[(item["image_id"], item["category_name"])].append(item)

    kept = []
    for _, items in groups.items():
        items.sort(key=lambda x: -x["score"])
        if a.mode == "wbf":
            kept.extend(_wbf(items, a.wbf_iou, a.wbf_conf))
        else:
            chosen = []
            for item in items:
                if all(iou(item["bbox"], c["bbox"]) <= a.iou for c in chosen):
                    chosen.append(item)
            kept.extend(chosen)

    kept.sort(key=lambda x: (x["image_id"], -x["score"], x["category_name"]))
    Path(a.output).write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"inputs": len(a.predictions), "pool": len(pool), "merged": len(kept)}))


if __name__ == "__main__":
    main()
