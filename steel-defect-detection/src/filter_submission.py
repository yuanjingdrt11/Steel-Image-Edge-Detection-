#!/usr/bin/env python3
"""把保留全部候选的预测结果，按验证集选出的逐类阈值过滤成最终输出文件。

用法:
  python src/filter_submission.py \
    --predictions all_predictions.json \
    --thresholds thresholds.json \
    --output result.json \
    --images path/to/images

设计说明:
- 推理阶段使用极低置信度下限保留全部候选，避免为不同阈值重复推理。
- 阈值过滤在本地完成，最终输出文件由本地生成。
- 阈值来自独立验证集的逐类 F2 最优点；无 GT 的类别保持默认下限不拟合。
"""
import argparse
import json
import math
from pathlib import Path

CLASSES = ["mamianmakeng", "jieba", "yiwuyaru", "yanghuatiepi", "zonglie",
           "gunyin", "jiaza", "huashang", "qilie"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def load_thresholds(path, default=0.01):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "thresholds" in data:
        data = data["thresholds"]
    if not isinstance(data, dict):
        raise ValueError("thresholds file must be a JSON object")
    unknown = set(data) - set(CLASSES)
    if unknown:
        raise ValueError(f"unknown classes in thresholds: {sorted(unknown)}")
    result = {name: float(default) for name in CLASSES}
    for name, value in data.items():
        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"invalid threshold for {name}: {value}")
        result[name] = value
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", required=True)
    p.add_argument("--thresholds", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--images", default=None,
                   help="可选：测试图片目录，用于核对覆盖与文件名合法性")
    p.add_argument("--report", default=None, help="可选：输出过滤统计 JSON")
    p.add_argument("--float-bbox", action="store_true",
                   help="保留小数坐标；默认输出整数坐标以匹配官方示例格式")
    p.add_argument("--exclude-classes", default="",
                   help="逗号分隔的类别名，这些类别的预测整体丢弃。"
                        "用于平台校验器不接受某些类名的情况（例如 qilie）")
    a = p.parse_args()

    excluded = {c.strip() for c in a.exclude_classes.split(",") if c.strip()}
    unknown_excluded = excluded - set(CLASSES)
    if unknown_excluded:
        raise ValueError(f"unknown classes in --exclude-classes: {sorted(unknown_excluded)}")

    thresholds = load_thresholds(a.thresholds)
    items = json.loads(Path(a.predictions).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("predictions must be a JSON list")

    image_ids = None
    if a.images:
        root = Path(a.images)
        image_ids = {x.name for x in root.rglob("*")
                     if x.is_file() and x.suffix.lower() in EXTS}
        if not image_ids:
            raise FileNotFoundError(f"no images found under {root}")

    kept, dropped, dropped_excluded = [], 0, 0
    per_class = {name: 0 for name in CLASSES}
    seen = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"item {index} is not an object")
        if set(item) != {"image_id", "category_name", "bbox", "score"}:
            raise ValueError(f"item {index}: unexpected fields {sorted(item)}")
        name = item["category_name"]
        if name not in CLASSES:
            raise ValueError(f"item {index}: unknown class {name}")
        box = item["bbox"]
        if not isinstance(box, list) or len(box) != 4 or \
                not all(isinstance(v, (int, float)) for v in box):
            raise ValueError(f"item {index}: invalid bbox {box}")
        x1, y1, x2, y2 = box
        if not (x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2):
            raise ValueError(f"item {index}: invalid bbox {box}")
        score = float(item["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"item {index}: invalid score {score}")
        if image_ids is not None and item["image_id"] not in image_ids:
            raise ValueError(f"item {index}: unknown image_id {item['image_id']}")
        seen.add(item["image_id"])
        if name in excluded:
            dropped_excluded += 1
            continue
        if score < thresholds[name]:
            dropped += 1
            continue
        if a.float_bbox:
            coords = [round(float(v), 2) for v in box]
        else:
            coords = [int(round(float(v))) for v in box]
            if coords[2] <= coords[0]:
                coords[2] = coords[0] + 1
            if coords[3] <= coords[1]:
                coords[3] = coords[1] + 1
        kept.append({"image_id": item["image_id"], "category_name": name,
                     "bbox": coords, "score": round(score, 6)})
        per_class[name] += 1

    kept.sort(key=lambda x: (x["image_id"], -x["score"], x["category_name"]))
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = {
        "input_candidates": len(items),
        "kept": len(kept),
        "dropped_by_threshold": dropped,
        "dropped_excluded_classes": dropped_excluded,
        "excluded_classes": sorted(excluded),
        "images_with_predictions": len({k["image_id"] for k in kept}),
        "images_total": len(image_ids) if image_ids else None,
        "detections_per_class": per_class,
        "thresholds_used": thresholds,
        "output": str(out.resolve()),
    }
    if a.report:
        Path(a.report).write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
