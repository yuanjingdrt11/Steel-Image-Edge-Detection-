#!/usr/bin/env python3
"""Evaluate official xyxy predictions on original XML, including empty images.

AP50 is the area under the all-points interpolated precision/recall envelope.
Thresholds maximize F2 over complete tied-score groups; ties prefer the higher
threshold. Classes without GT have null recall/AP/F2 and retain the candidate
confidence floor (0.01); no threshold is fitted to these unsupported classes.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

CLASSES = ["mamianmakeng", "jieba", "yiwuyaru", "yanghuatiepi", "zonglie", "gunyin", "jiaza", "huashang", "qilie"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_paths(source):
    source = Path(source)
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if not paths:
        raise FileNotFoundError(f"no images found: {source}")
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("duplicate image basenames")
    if len({p.stem for p in paths}) != len(paths):
        raise ValueError("duplicate image stems cannot uniquely identify XML annotations")
    return paths


def valid_box(values):
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        raise ValueError("bbox must be four xyxy coordinates")
    box = [float(v) for v in values]
    if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"invalid bbox: {values}")
    return box


def load_ground_truth(images, annotations):
    """Match by XML stem: official XML filename elements can end in .xml."""
    wanted = {p.stem: p.name for p in images}
    xmls = {}
    for path in sorted(Path(annotations).rglob("*")):
        if path.suffix.lower() != ".xml" or path.stem not in wanted:
            continue
        if path.stem in xmls:
            raise ValueError(f"duplicate annotation stem: {path.stem}")
        xmls[path.stem] = path
    missing = sorted(set(wanted) - set(xmls))
    if missing:
        raise ValueError(f"missing XML annotations: {missing}")
    gt = {p.name: [] for p in images}
    for stem, path in xmls.items():
        root = ET.parse(path).getroot()
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            if name not in CLASSES:
                raise ValueError(f"unknown XML class {name!r}: {path}")
            bnd = obj.find("bndbox")
            if bnd is None:
                raise ValueError(f"missing bndbox: {path}")
            box = valid_box([bnd.findtext(k) for k in ("xmin", "ymin", "xmax", "ymax")])
            # All original objects count, including difficult/truncated defects.
            gt[wanted[stem]].append({"category_name": name, "bbox": box})
    return gt


def load_predictions(path, image_ids):
    path = Path(path)
    predictions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("predictions must be an official-format JSON array")
    coverage_path = path.with_suffix(path.suffix + ".coverage.json")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    for key in ("images", "processed"):
        values = coverage.get(key)
        if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
            raise ValueError(f"coverage {key} must be an image-id list")
        if len(values) != len(set(values)) or set(values) != set(image_ids):
            raise ValueError(f"coverage {key} must cover exactly the evaluation images")
    if coverage.get("missing") != []:
        raise ValueError("coverage reports missing images")
    if coverage.get("detections") != len(predictions):
        raise ValueError("coverage detection count mismatch")
    for pred in predictions:
        if not isinstance(pred, dict) or pred.get("image_id") not in image_ids:
            raise ValueError(f"unknown prediction image: {pred}")
        if pred.get("category_name") not in CLASSES:
            raise ValueError(f"unknown prediction class: {pred}")
        pred["bbox"] = valid_box(pred.get("bbox"))
        score = float(pred["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"invalid prediction score: {score}")
        pred["score"] = score
    return predictions


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union


def class_metrics(name, predictions, ground_truth):
    targets = {image: [o["bbox"] for o in objs if o["category_name"] == name] for image, objs in ground_truth.items()}
    count = sum(map(len, targets.values()))
    ranked = sorted((p for p in predictions if p["category_name"] == name), key=lambda p: (-p["score"], p["image_id"], tuple(p["bbox"])))
    matched = defaultdict(set)
    curve = []
    tp = fp = 0
    for index, pred in enumerate(ranked):
        image = pred["image_id"]
        choices = [(iou(pred["bbox"], box), j) for j, box in enumerate(targets[image]) if j not in matched[image]]
        overlap, j = max(choices, default=(0, -1))
        if overlap >= .5:
            matched[image].add(j)
            tp += 1
        else:
            fp += 1
        # Scores tied at a threshold must be admitted together.
        if index + 1 == len(ranked) or ranked[index+1]["score"] != pred["score"]:
            precision = tp / (tp + fp)
            recall = tp / count if count else None
            f2 = 5 * tp / (5*tp + 4*(count-tp) + fp) if count else None
            curve.append({"threshold": pred["score"], "tp": tp, "fp": fp, "precision": precision, "recall": recall, "f2": f2})
    baseline = {"threshold": .01, "tp": 0, "fp": 0, "precision": 0., "recall": 0. if count else None, "f2": 0. if count else None}
    if count:
        best = max(curve, key=lambda row: (row["f2"], row["threshold"])) if curve else baseline
        envelope = [row["precision"] for row in curve]
        for j in range(len(envelope)-2, -1, -1):
            envelope[j] = max(envelope[j], envelope[j+1])
        ap = 0.
        last_recall = 0.
        for row, precision in zip(curve, envelope):
            ap += (row["recall"] - last_recall) * precision
            last_recall = row["recall"]
        status = "ok"
    else:
        selected = [p for p in ranked if p["score"] >= .01]
        best = {**baseline, "fp": len(selected)}
        ap = None
        status = "no_ground_truth"
    return {"status": status, "ground_truth": count, "predictions": len(ranked), "ap50": ap, **best, "fn": count-best["tp"], "threshold_calibrated": bool(count and curve)}, best["threshold"]


def evaluate(predictions, ground_truth):
    per_class = {}
    thresholds = {}
    for name in CLASSES:
        per_class[name], thresholds[name] = class_metrics(name, predictions, ground_truth)
    empty = {image for image, objs in ground_truth.items() if not objs}
    raw = [p for p in predictions if p["image_id"] in empty]
    selected = [p for p in raw if p["score"] >= thresholds[p["category_name"]]]
    aps = [v["ap50"] for v in per_class.values() if v["ap50"] is not None]
    report = {"iou_threshold": .5, "ap_method": "all_points_interpolated_score_groups", "class_order": CLASSES,
              "images": len(ground_truth), "coverage_verified": True, "per_class": per_class,
              "map50_present_classes": sum(aps)/len(aps) if aps else None,
              "empty_images": {"count": len(empty), "raw_false_positives": len(raw),
                               "raw_images_with_false_positives": len({p["image_id"] for p in raw}),
                               "selected_false_positives": len(selected),
                               "selected_images_with_false_positives": len({p["image_id"] for p in selected})},
              "notes": "Metrics at per-class best F2 thresholds; AP uses all candidates. No-GT classes have null recall/AP/F2 and an uncalibrated 0.01 threshold. Thresholds are selected and reported on the same validation set."}
    return report, thresholds


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("predictions", "images", "annotations", "output", "thresholds-output"):
        p.add_argument("--"+flag, required=True)
    args = p.parse_args()
    paths = image_paths(args.images)
    gt = load_ground_truth(paths, args.annotations)
    predictions = load_predictions(args.predictions, set(gt))
    report, thresholds = evaluate(predictions, gt)
    for filename, data in ((args.output, report), (args.thresholds_output, thresholds)):
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"images": len(gt), "map50_present_classes": report["map50_present_classes"], "report": args.output}))


if __name__ == "__main__":
    main()
