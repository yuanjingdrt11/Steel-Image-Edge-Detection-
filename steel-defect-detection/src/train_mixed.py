#!/usr/bin/env python3
"""Train YOLO11s mixed-scale supervision, or resume an explicit last.pt."""
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='v1/data/mixed/data.yaml')
    p.add_argument('--project', default='v1/runs')
    p.add_argument('--name', default='mixed_scale')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device', default='0')
    p.add_argument('--patience', type=int, default=40,
                   help='验证指标连续多少轮无提升后早停')
    p.add_argument('--mosaic', type=float, default=0.15)
    p.add_argument('--close-mosaic', type=int, default=15)
    p.add_argument('--mixup', type=float, default=0.0)
    p.add_argument('--scale', type=float, default=0.2)
    p.add_argument('--translate', type=float, default=0.03)
    p.add_argument('--hsv-v', type=float, default=0.0)
    p.add_argument('--hsv-s', type=float, default=0.0)
    p.add_argument('--degrees', type=float, default=0.0)
    p.add_argument('--erasing', type=float, default=0.0)
    p.add_argument('--resume', metavar='LAST_PT', help='resume only an existing last.pt checkpoint')
    return p


def main(argv=None):
    p = parser()
    a = p.parse_args(argv)
    if min(a.epochs, a.imgsz, a.batch) <= 0 or a.workers < 0:
        p.error('epochs, imgsz and batch must be positive; workers must be nonnegative')
    if a.resume:
        checkpoint = Path(a.resume).expanduser()
        if checkpoint.name != 'last.pt' or not checkpoint.is_file() or checkpoint.resolve().name != 'last.pt':
            p.error('--resume requires an existing last.pt, not best.pt or another model')
        from ultralytics import YOLO
        print(f'Resuming explicit last.pt: {checkpoint.resolve()}', flush=True)
        # Let Ultralytics restore the saved epoch target, optimizer and augmentation settings.
        return YOLO(str(checkpoint.resolve())).train(resume=True, device=a.device,
                                                     workers=a.workers)
    if not Path(a.data).is_file():
        p.error(f'Dataset YAML does not exist: {a.data}')
    run_dir = Path(a.project) / a.name
    if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
        p.error(f'Run directory is not empty: {run_dir}; choose a new --name or use --resume last.pt')
    weights = PROJECT_ROOT / 'yolo11s.pt'
    initialization = str(weights) if weights.is_file() else 'yolo11s.pt'
    print(f'New training: YOLO11s COCO initialization from {initialization}; '
          'no old steel checkpoint is loaded.', flush=True)
    from ultralytics import YOLO
    model = YOLO(initialization)
    if len(model.names) != 80 or model.names[0] != 'person':
        p.error('Initialization must be the 80-class COCO yolo11s.pt, not old steel weights')
    return model.train(
        data=a.data, project=a.project, name=a.name, epochs=a.epochs,
        imgsz=a.imgsz, batch=a.batch, workers=a.workers, device=a.device,
        pretrained=True, resume=False, exist_ok=True, patience=a.patience, amp=True, cache=False,
        optimizer='AdamW', lr0=0.0015, lrf=0.01, cos_lr=True,
        weight_decay=0.0005, box=8.5, cls=1.5, dfl=1.5,
        mosaic=a.mosaic, close_mosaic=a.close_mosaic, mixup=a.mixup,
        hsv_h=0.0, hsv_s=a.hsv_s, hsv_v=a.hsv_v,
        fliplr=0.5, flipud=0.0, scale=a.scale, translate=a.translate,
        degrees=a.degrees, erasing=a.erasing,
        plots=True, save_period=-1,
    )


if __name__ == '__main__':
    main()
