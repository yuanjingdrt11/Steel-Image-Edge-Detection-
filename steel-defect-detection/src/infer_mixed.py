#!/usr/bin/env python3
"""Full, tiled, or fused inference for steel-sheet defect images."""
import argparse, json, math
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

CLASSES = ["mamianmakeng", "jieba", "yiwuyaru", "yanghuatiepi", "zonglie", "gunyin", "jiaza", "huashang", "qilie"]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def positions(length, tile, overlap=.25):
    if tile >= length: return [0]
    step = max(1, int(tile * (1-overlap)))
    out = list(range(0, length-tile+1, step))
    if not out or out[-1] + tile < length: out.append(length-tile)
    return out

def image_paths(source):
    source = Path(source)
    paths = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    if not paths: raise FileNotFoundError(f"no image files found under {source}")
    names = [p.name for p in paths]
    if len(names) != len(set(names)): raise ValueError("duplicate image basenames are not allowed")
    return paths

def nms(dets, iou=.5):
    keep=[]
    for cls in sorted({int(d[5]) for d in dets}):
        ix=[i for i,d in enumerate(dets) if int(d[5])==cls]
        ix.sort(key=lambda i:dets[i][4], reverse=True)
        while ix:
            i=ix.pop(0); keep.append(i)
            a=np.array(dets[i][:4], float); rest=[]
            for j in ix:
                b=np.array(dets[j][:4], float)
                inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                union=max(0,a[2]-a[0])*max(0,a[3]-a[1])+max(0,b[2]-b[0])*max(0,b[3]-b[1])-inter
                if union == 0 or inter/union <= iou: rest.append(j)
            ix=rest
    return keep

def model_names(model):
    vals = model.names
    vals = vals.items() if isinstance(vals, dict) else enumerate(vals)
    names={int(k):str(v) for k,v in vals}
    bad=set(names.values())-set(CLASSES)
    if bad: raise ValueError(f"weights contain unknown classes: {sorted(bad)}")
    if set(names) != set(range(9)) or [names[i] for i in range(9)] != CLASSES:
        raise ValueError("weights must contain the exact nine classes in official order")
    return names

def predict_image(model, image, mode, imgsz, tile, batch, device, conf, iou=.5, overlap=.25):
    if mode not in ("full", "tiled", "mixed"): raise ValueError("invalid mode")
    names = model_names(model)
    h,w=image.shape[:2]; origins=[]
    if mode in ("full","mixed"): origins.append((0,0,image))
    if mode in ("tiled","mixed"):
        for y in positions(h,tile,overlap):
            for x in positions(w,tile,overlap):
                crop=np.zeros((tile,tile,3),np.uint8); part=image[y:min(y+tile,h),x:min(x+tile,w)]
                crop[:part.shape[0],:part.shape[1]]=part; origins.append((x,y,crop))
    detections=[]
    for start in range(0,len(origins),batch):
        group=origins[start:start+batch]
        results=list(model.predict([g[2] for g in group], conf=conf, iou=iou, device=device, imgsz=imgsz, verbose=False, batch=len(group)))
        if len(results) != len(group): raise RuntimeError("model result count differs from input count")
        for result,(x0,y0,crop) in zip(results,group):
            if result.boxes is None: continue
            for box in result.boxes:
                xy=box.xyxy[0].tolist(); raw_cls=float(box.cls[0]); score=float(box.conf[0])
                if not math.isfinite(raw_cls) or not raw_cls.is_integer() or int(raw_cls) not in names:
                    raise ValueError(f"invalid class id: {raw_cls}")
                if not all(math.isfinite(v) for v in xy) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("model emitted nonfinite coordinates or invalid score")
                cls=int(raw_cls)
                cw,ch=min(crop.shape[1],w-x0),min(crop.shape[0],h-y0)
                x1,x2=[max(0,min(cw,v)) + x0 for v in (xy[0],xy[2])]
                y1,y2=[max(0,min(ch,v)) + y0 for v in (xy[1],xy[3])]
                if x2 > x1 and y2 > y1: detections.append([x1,y1,x2,y2,score,cls])
    return detections

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source", required=True); p.add_argument("--weights", required=True); p.add_argument("--output", required=True)
    p.add_argument("--mode", choices=["mixed","full","tiled"], default="mixed"); p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--tile", type=int, default=1280); p.add_argument("--batch", type=int, default=2)
    p.add_argument("--device", default="0"); p.add_argument("--thresholds", default=None); p.add_argument("--conf", type=float, default=.01)
    p.add_argument("--iou", type=float, default=.5); p.add_argument("--output-dir", default=None)
    p.add_argument("--overlap", type=float, default=.25, help="切片重叠率，越大切片越密")
    p.add_argument("--hflip", action="store_true", help="水平翻转 TTA：翻转后推理再把框翻回来")
    a=p.parse_args()
    if a.imgsz<=0 or a.tile<=0 or a.batch<=0 or not 0<=a.conf<=1 or not 0<=a.iou<=1: raise ValueError("invalid imgsz/tile/batch/conf/iou")
    thresholds={c:a.conf for c in CLASSES}
    if a.thresholds:
        with open(a.thresholds,encoding="utf-8") as f: data=json.load(f)
        if not isinstance(data,dict): raise ValueError("thresholds must be an object")
        thresholds.update({k:float(v) for k,v in data.get("thresholds",data).items()})
    if set(thresholds)-set(CLASSES): raise ValueError("unknown threshold class")
    if any(not math.isfinite(v) or not 0<=v<=1 for v in thresholds.values()): raise ValueError("thresholds must be finite values in [0,1]")
    inference_conf=min(a.conf,*thresholds.values())
    paths=image_paths(a.source); model=YOLO(a.weights); names=model_names(model); all_preds=[]; rendered=set()
    outdir=Path(a.output_dir) if a.output_dir else None
    if outdir: outdir.mkdir(parents=True,exist_ok=True)
    for image_index,path in enumerate(paths,1):
        image=cv2.imread(str(path),cv2.IMREAD_COLOR)
        if image is None: raise RuntimeError(f"cannot read image: {path}")
        if a.hflip: image=cv2.flip(image,1)
        dets=predict_image(model,image,a.mode,a.imgsz,a.tile,a.batch,a.device,inference_conf,a.iou,a.overlap)
        if a.hflip:
            # 翻转推理后把框翻回原坐标系，再复原图像供后续裁剪/绘制使用
            width_flipped=image.shape[1]
            dets=[[width_flipped-d[2], d[1], width_flipped-d[0], d[3], d[4], d[5]] for d in dets]
            image=cv2.flip(image,1)
        kept=nms(dets,a.iou)
        for i in kept:
            x1,y1,x2,y2,score,cls=dets[i]; name=names.get(int(cls))
            if name is None: raise ValueError(f"invalid class id {cls}")
            if score < thresholds.get(name,a.conf): continue
            x1=max(0,min(image.shape[1],x1)); x2=max(0,min(image.shape[1],x2)); y1=max(0,min(image.shape[0],y1)); y2=max(0,min(image.shape[0],y2))
            if x2<=x1 or y2<=y1: continue
            coords=[round(v,3) for v in (x1,y1,x2,y2)]
            if coords[2]<=coords[0] or coords[3]<=coords[1]: continue
            all_preds.append({"image_id":path.name,"category_name":name,"bbox":coords,"score":score})
            if outdir: cv2.rectangle(image,(round(x1),round(y1)),(round(x2),round(y2)),(0,0,255),3)
        if outdir: cv2.imwrite(str(outdir/path.name),image)
        rendered.add(path.name)
        if image_index % 25 == 0 or image_index == len(paths): print(f"processed {image_index}/{len(paths)}", flush=True)
    all_preds.sort(key=lambda x:(x["image_id"],-x["score"],x["category_name"]))
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(all_preds,ensure_ascii=False,indent=2),encoding="utf-8")
    coverage={"images":[p.name for p in paths],"processed":sorted(rendered),"missing":sorted(set(p.name for p in paths)-rendered),"detections":len(all_preds)}
    output.with_suffix(output.suffix+".coverage.json").write_text(json.dumps(coverage,ensure_ascii=False,indent=2),encoding="utf-8")
    if coverage["missing"]: raise RuntimeError(f"inference coverage missing: {coverage['missing']}")
    print(json.dumps({"images":len(paths),"detections":len(all_preds),"output":str(output.resolve())},ensure_ascii=False))
if __name__=="__main__": main()
