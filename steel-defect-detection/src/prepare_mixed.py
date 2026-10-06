#!/usr/bin/env python3
"""Prepare grouped full-image/tile supervision without modifying source data."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import xml.etree.ElementTree as ET

from PIL import Image
import yaml

CLASSES = ['mamianmakeng', 'jieba', 'yiwuyaru', 'yanghuatiepi', 'zonglie',
           'gunyin', 'jiaza', 'huashang', 'qilie']
GIB = 1024 ** 3


def group_key(stem):
    """Keep all views/frames of one product in the same split."""
    return re.split(r'[_-]Raw|[_-]V(?=\d)', stem, maxsplit=1, flags=re.I)[0]


def positions(length, tile):
    return sorted(set(range(0, max(1, length - tile + 1), tile)) | {max(0, length - tile)})


def select_tiles(boxes, width, height, tile, limit, rng):
    w, h = min(tile, width), min(tile, height)
    centers = [((x1+x2)/2, (y1+y2)/2) for _, x1, y1, x2, y2 in boxes]
    if not centers:
        centers = [(width/2, height/2)]
    candidates = sorted({(max(0, min(width-w, int(cx-w/2))),
                          max(0, min(height-h, int(cy-h/2))), w, h)
                         for cx, cy in centers})
    if len(candidates) > limit:
        candidates = sorted(rng.sample(candidates, limit))
    return candidates


def read_boxes(xml, width, height):
    root = ET.parse(xml).getroot()
    size = root.find('size')
    if size is not None and (int(size.findtext('width', str(width))),
                             int(size.findtext('height', str(height)))) != (width, height):
        raise ValueError(f'XML/image dimensions differ: {xml}')
    boxes = []
    for obj in root.findall('object'):
        name = (obj.findtext('name') or '').strip()
        if name not in CLASSES:
            raise ValueError(f'Unknown class {name!r}: {xml}')
        b = obj.find('bndbox')
        if b is None:
            raise ValueError(f'Missing bounding box: {xml}')
        coords = [float(b.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
        if not all(math.isfinite(x) for x in coords):
            raise ValueError(f'Non-finite bounding box: {xml}')
        x1, y1, x2, y2 = coords
        x1, x2 = max(0, min(width, x1)), max(0, min(width, x2))
        y1, y2 = max(0, min(height, y1)), max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            boxes.append((CLASSES.index(name), x1, y1, x2, y2))
    return boxes


def clipped_labels(boxes, x, y, width, height):
    labels = []
    for cls, x1, y1, x2, y2 in boxes:
        left, top = max(x1, x), max(y1, y)
        right, bottom = min(x2, x + width), min(y2, y + height)
        if right > left and bottom > top:
            labels.append((cls, (left + right - 2*x)/(2*width),
                           (top + bottom - 2*y)/(2*height),
                           (right-left)/width, (bottom-top)/height))
    return labels


def write_labels(path, labels):
    with path.open('x') as stream:
        for cls, *coords in labels:
            stream.write(f'{cls} ' + ' '.join(f'{v:.8f}' for v in coords) + '\n')


def link_image(src, dst):
    """Never open linked destination images for writing."""
    try:
        os.link(src, dst)
        return 'hardlink'
    except OSError:
        if dst.exists() or dst.is_symlink():
            raise
    try:
        dst.symlink_to(src.resolve())
        return 'symlink'
    except OSError:
        if dst.exists() or dst.is_symlink():
            raise
    if shutil.disk_usage(dst.parent).free < src.stat().st_size + 1024*1024:
        raise OSError(f'Insufficient disk space for full-image copy fallback: {dst}')
    with src.open('rb') as source, dst.open('xb') as target:
        shutil.copyfileobj(source, target)
    return 'copy'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', default='train')
    p.add_argument('--out', default='v1/data/mixed')
    p.add_argument('--tile', type=int, default=1280)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--val', type=float, default=0.2)
    p.add_argument('--min-free-gb', type=float, default=35.0,
                   help='minimum free GiB; actual requirement also covers estimated output')
    p.add_argument('--max-tiles-per-image', type=int, default=4,
                   help='maximum target-centered crops per training image')
    p.add_argument('--allow-unpaired', action='store_true')
    p.add_argument('--dry-run', action='store_true', help='scan and print plan; do not write files')
    return p


def prepare(a):
    src, out = Path(a.src).resolve(), Path(a.out).resolve()
    if a.tile <= 0 or a.max_tiles_per_image <= 0 or not 0 <= a.val < 1 or not math.isfinite(a.min_free_gb) or a.min_free_gb < 0:
        raise ValueError('Require tile > 0, 0 <= val < 1, and finite min-free-gb >= 0')
    if not src.is_dir():
        raise ValueError(f'Source directory does not exist: {src}')
    if src == out or src in out.parents or out in src.parents:
        raise ValueError('Source and output must be separate, non-nested directories')
    if out.exists() and (not out.is_dir() or any(p.name != 'manifest.json' for p in out.iterdir())):
        raise ValueError(f'Output is not empty; use a new output directory: {out}')
    if (out / 'manifest.json').is_symlink():
        raise ValueError('Refusing symlinked output manifest')
    images = {p.stem: p for p in src.iterdir() if p.is_file() and p.suffix.lower() == '.jpg'}
    xmls = {p.stem: p for p in src.iterdir() if p.is_file() and p.suffix.lower() == '.xml'}
    missing_xml = sorted(str(images[s]) for s in images.keys() - xmls.keys())
    missing_images = sorted(str(xmls[s]) for s in xmls.keys() - images.keys())
    manifest = {'source': str(src), 'seed': a.seed, 'tile': a.tile, 'val_fraction': a.val,
                'unpaired_images': missing_xml, 'unpaired_annotations': missing_images}
    if (missing_xml or missing_images) and not a.allow_unpaired:
        if not a.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(json.dumps(manifest, indent=2))
        raise ValueError(f'Unpaired data: {len(missing_xml)} JPG without XML, '
                         f'{len(missing_images)} XML without JPG. See manifest; '
                         'use --allow-unpaired to explicitly use only matched pairs.')
    stems = sorted(images.keys() & xmls.keys())
    if not stems:
        raise ValueError('No paired JPG/XML files found')
    groups = sorted({group_key(s) for s in stems})
    if a.val > 0 and len(groups) < 2:
        raise ValueError('At least two product groups are required for train/validation splitting')
    rng = random.Random(a.seed)
    rng.shuffle(groups)
    nval = max(1, min(len(groups)-1, round(len(groups)*a.val))) if a.val else 0
    val_groups = set(groups[:nval])
    records, estimated, tile_count = [], 0, 0
    counts = {split: dict.fromkeys(CLASSES, 0) for split in ('train', 'val')}
    for stem in stems:
        with Image.open(images[stem]) as image:
            width, height = image.size
        boxes = read_boxes(xmls[stem], width, height)
        split = 'val' if group_key(stem) in val_groups else 'train'
        tiles = select_tiles(boxes, width, height, a.tile, a.max_tiles_per_image, rng) if split == 'train' else []
        # Budget conservative RGB-sized JPEGs; full images use links (copy checked separately).
        estimated += 4096
        estimated += sum(w*h*3 + 65536 for _, _, w, h in tiles)
        tile_count += len(tiles)
        for cls, *_ in boxes:
            counts[split][CLASSES[cls]] += 1
        records.append({'stem': stem, 'group': group_key(stem), 'split': split,
                        'width': width, 'height': height, 'boxes': boxes, 'tiles': tiles})
    probe = out
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    required = max(a.min_free_gb*GIB, math.ceil(estimated*1.2))
    manifest.update({'images': len(records), 'tiles': tile_count, 'classes': counts,
                     'train_groups': sorted(set(groups)-val_groups), 'val_groups': sorted(val_groups),
                     'estimated_bytes': estimated, 'required_free_bytes': required,
                     'free_bytes': free, 'space_ok': free >= required,
                     'records': [{k: v for k, v in r.items() if k not in ('boxes', 'tiles')} for r in records]})
    manifest['group_counts'] = {'train': len(groups)-nval, 'val': nval}
    manifest['image_counts'] = {split: sum(r['split'] == split for r in records) for split in ('train', 'val')}
    manifest['max_tiles_per_image'] = a.max_tiles_per_image
    manifest['warnings'] = [f'Class {name}: validation fraction {counts["val"][name]/total:.1%} '
                            f'differs from requested group fraction {a.val:.1%}; product groups remain intact'
                            for name in CLASSES
                            if (total := counts['train'][name] + counts['val'][name]) > 0
                            and abs(counts['val'][name]/total - a.val) > 0.25]
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in ('records', 'train_groups', 'val_groups')}, indent=2))
    if a.dry_run:
        return manifest
    if free < required:
        raise ValueError(f'Insufficient disk space at {probe}: free {free/GIB:.2f} GiB, '
                         f'required {required/GIB:.2f} GiB (minimum {a.min_free_gb:g} GiB; '
                         'output estimate + 20% reserve). Free space or set --min-free-gb explicitly.')
    for split in ('train', 'val'):
        for kind in ('images', 'labels'):
            (out / kind / split).mkdir(parents=True, exist_ok=True)
    strategies = dict.fromkeys(('hardlink', 'symlink', 'copy'), 0)
    for index, r in enumerate(records, 1):
        if index % 100 == 0:
            print(f'Preparing {index}/{len(records)} images', flush=True)
        stem, split = r['stem'], r['split']
        image_dir, label_dir = out/'images'/split, out/'labels'/split
        strategies[link_image(images[stem], image_dir/f'{stem}.jpg')] += 1
        write_labels(label_dir/f'{stem}.txt', clipped_labels(r['boxes'], 0, 0, r['width'], r['height']))
        if r['tiles']:
            with Image.open(images[stem]) as original:
                rgb = original.convert('RGB')
                try:
                    for x, y, w, h in r['tiles']:
                        name = f'{stem}__tile_{x}_{y}'
                        with (image_dir/f'{name}.jpg').open('xb') as stream:
                            rgb.crop((x, y, x+w, y+h)).save(stream, 'JPEG', quality=95)
                        write_labels(label_dir/f'{name}.txt', clipped_labels(r['boxes'], x, y, w, h))
                finally:
                    rgb.close()
    manifest['image_storage'] = strategies
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    with (out/'data.yaml').open('x') as stream:
        yaml.safe_dump({'path': str(out), 'train': 'images/train', 'val': 'images/val',
                        'names': dict(enumerate(CLASSES))}, stream, sort_keys=False)
    return manifest


def main(argv=None):
    p = parser()
    try:
        return prepare(p.parse_args(argv))
    except (ValueError, OSError, ET.ParseError) as exc:
        p.exit(2, f'prepare_mixed: error: {exc}\n')


if __name__ == '__main__':
    main()
