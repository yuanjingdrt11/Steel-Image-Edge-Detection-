#!/usr/bin/env python3
"""Validate competition submission JSON against an image directory."""
import argparse
import json
from pathlib import Path

CLASSES = {
    "mamianmakeng", "jieba", "yiwuyaru", "yanghuatiepi", "zonglie",
    "gunyin", "jiaza", "huashang", "qilie",
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission")
    p.add_argument("images")
    a = p.parse_args()
    items = json.loads(Path(a.submission).read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("top-level JSON value must be a list")
    image_root = Path(a.images)
    image_ids = {x.name for x in image_root.rglob("*") if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    seen = set()
    for i, item in enumerate(items):
        if set(item) != {"image_id", "category_name", "bbox", "score"}:
            raise ValueError(f"item {i}: unexpected fields {sorted(item)}")
        if item["image_id"] not in image_ids:
            raise ValueError(f"item {i}: unknown image_id {item['image_id']}")
        if item["category_name"] not in CLASSES:
            raise ValueError(f"item {i}: unknown category {item['category_name']}")
        box = item["bbox"]
        if not isinstance(box, list) or len(box) != 4 or not all(isinstance(v, (int, float)) for v in box):
            raise ValueError(f"item {i}: bbox must contain four numbers")
        x1, y1, x2, y2 = box
        if not (x1 >= 0 and y1 >= 0 and x1 < x2 and y1 < y2):
            raise ValueError(f"item {i}: invalid bbox {box}")
        score = item["score"]
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"item {i}: invalid score {score}")
        seen.add(item["image_id"])
    print(json.dumps({"images": len(image_ids), "images_with_predictions": len(seen), "detections": len(items)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
