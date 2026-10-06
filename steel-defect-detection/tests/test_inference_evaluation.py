"""Synthetic contract tests: no dataset weights or GPU are needed."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import cv2
import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
def load(name):
    spec = importlib.util.spec_from_file_location(name, SRC / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
infer = load("infer_mixed")
evalmod = load("evaluate_mixed")

class FakeModel:
    names = dict(enumerate(infer.CLASSES))
    def __init__(self, outputs=None):
        self.outputs = outputs or []
        self.calls = []
        self.index = 0
    def predict(self, images, **kwargs):
        self.calls.append((images, kwargs))
        results = []
        for image in images:
            rows = self.outputs[self.index] if self.index < len(self.outputs) else []
            self.index += 1
            boxes = [SimpleNamespace(xyxy=np.array([r[:4]]), conf=np.array([r[4]]), cls=np.array([r[5]])) for r in rows]
            results.append(SimpleNamespace(boxes=boxes))
        return results

def pred(image, score, box=(0,0,10,10), name="mamianmakeng"):
    return {"image_id": image, "score": score, "bbox": list(box), "category_name": name}

class InferenceTests(unittest.TestCase):
    def test_positions_cover_edges_and_overlap(self):
        self.assertEqual(infer.positions(3000,1280),[0,960,1720])
        self.assertEqual(infer.positions(10,1280),[0])

    def test_fused_nms_padding_coordinates_and_channels(self):
        model = FakeModel([
            [[1,1,9,7,.95,0], [1,1,9,7,.9,1]],
            [[1,1,9,7,.8,0], [9,0,13,9,.7,2]],
            [[0,0,4,8,.8,3]],
        ])
        image = np.zeros((8,14,3),np.uint8)
        dets = infer.predict_image(model,image,"mixed",1024,10,2,"cpu",.01)
        kept=[dets[i] for i in infer.nms(dets)]
        self.assertEqual(len(kept),4)
        self.assertEqual(next(d for d in kept if d[5]==2)[:4],[9,0,10,8])
        self.assertEqual(next(d for d in kept if d[5]==3)[:4],[4,0,8,8])
        self.assertTrue(all(im.shape[2]==3 for ims,_ in model.calls for im in ims))
        self.assertTrue(all(kw["imgsz"]==1024 for _,kw in model.calls))
        self.assertEqual([len(ims) for ims,_ in model.calls],[2,1])

    def test_short_results_and_invalid_classes_fail(self):
        model=FakeModel()
        model.predict=lambda *args,**kw: []
        with self.assertRaisesRegex(RuntimeError,"count"):
            infer.predict_image(model,np.zeros((8,8,3),np.uint8),"full",32,8,2,"cpu",.01)
        model=FakeModel([[[0,0,4,4,.5,99]]])
        with self.assertRaisesRegex(ValueError,"class"):
            infer.predict_image(model,np.zeros((8,8,3),np.uint8),"full",32,8,2,"cpu",.01)
        model.names={0:"mamianmakeng"}
        with self.assertRaisesRegex(ValueError,"nine"):
            infer.model_names(model)

    def test_cli_empty_coverage_grayscale_small_box_and_threshold(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); imgs=root/"images"; imgs.mkdir()
            cv2.imwrite(str(imgs/"a.png"),np.zeros((8,8),np.uint8))
            cv2.imwrite(str(imgs/"empty.png"),np.zeros((8,8),np.uint8))
            output=root/"p.json"; thresholds=root/"t.json"
            thresholds.write_text(json.dumps({"mamianmakeng":.001}))
            model=FakeModel([[[.1,.1,.4,.4,.005,0]]])
            argv=["infer", "--source",str(imgs),"--weights","fake", "--output",str(output),"--mode","full","--thresholds",str(thresholds)]
            with patch.object(sys,"argv",argv),patch.object(infer,"YOLO",return_value=model): infer.main()
            data=json.loads(output.read_text()); side=json.loads(output.with_suffix(".json.coverage.json").read_text())
            self.assertEqual(len(data),1); self.assertEqual(data[0]["bbox"],[.1,.1,.4,.4])
            self.assertEqual(side["processed"],["a.png","empty.png"])
            self.assertEqual(model.calls[0][1]["conf"],.001)
            self.assertEqual(model.calls[0][0][0].shape,(8,8,3))
            (imgs/"nested").mkdir(); cv2.imwrite(str(imgs/"nested"/"a.png"),np.zeros((8,8),np.uint8))
            with self.assertRaisesRegex(ValueError,"duplicate"): infer.image_paths(imgs)

    def test_invalid_thresholds_and_unreadable_images_fail(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); image=root/"broken.png"; image.write_bytes(b"broken")
            output=root/"p.json"; threshold=root/"t.json"
            base=["infer","--source",str(image),"--weights","fake","--output",str(output)]
            for value in [float("nan"),1.1,-.1]:
                threshold.write_text(json.dumps({"jieba":value}))
                with patch.object(sys,"argv",base+["--thresholds",str(threshold)]),self.assertRaises(ValueError): infer.main()
            with patch.object(sys,"argv",base),patch.object(infer,"YOLO",return_value=FakeModel()),self.assertRaisesRegex(RuntimeError,"cannot read"):
                infer.main()
            self.assertFalse(output.exists())

class EvaluationTests(unittest.TestCase):
    def test_one_to_one_ap_f2_empty_images_and_no_gt(self):
        gt={"a.png":[{"category_name":"mamianmakeng","bbox":[0,0,10,10]}, {"category_name":"mamianmakeng","bbox":[20,0,30,10]}],"empty.png":[]}
        predictions=[pred("a.png",.9),pred("a.png",.8),pred("a.png",.7,(20,0,30,10)),pred("empty.png",.6),pred("empty.png",.5,name="jieba")]
        report,thresholds=evalmod.evaluate(predictions,gt)
        metrics=report["per_class"]["mamianmakeng"]
        self.assertAlmostEqual(metrics["ap50"],5/6)
        self.assertAlmostEqual(metrics["precision"],2/3)
        self.assertEqual(metrics["recall"],1)
        self.assertAlmostEqual(metrics["f2"],10/11)
        self.assertEqual(thresholds["mamianmakeng"],.7)
        self.assertEqual(metrics["tp"],2); self.assertEqual(metrics["fp"],1)
        self.assertIsNone(report["per_class"]["jieba"]["ap50"])
        self.assertEqual(report["per_class"]["jieba"]["status"],"no_ground_truth")
        self.assertEqual(report["empty_images"]["raw_false_positives"],2)
        self.assertEqual(report["empty_images"]["selected_false_positives"],1)

    def test_iou_boundary_and_score_ties(self):
        gt={"a.png":[{"category_name":"mamianmakeng","bbox":[0,0,10,10]}],"b.png":[]}
        report,_=evalmod.evaluate([pred("a.png",.5,(0,0,20,10)),pred("b.png",.5)],gt)
        cls=report["per_class"]["mamianmakeng"]
        self.assertEqual(cls["tp"],1); self.assertEqual(cls["fp"],1)
        self.assertEqual(cls["ap50"],.5)

    def test_original_xml_cli_coverage_and_rejections(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); images=root/"images"; images.mkdir(); anns=root/"ann"; anns.mkdir()
            for name in ["a","empty"]: cv2.imwrite(str(images/(name+".png")),np.zeros((12,12,3),np.uint8))
            (anns/"a.xml").write_text('<annotation><filename>a.xml</filename><object><name>mamianmakeng</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>10</xmax><ymax>10</ymax></bndbox></object></annotation>')
            (anns/"empty.xml").write_text('<annotation/>')
            path=root/"pred.json"; path.write_text(json.dumps([pred("a.png",.9)]))
            side=path.with_suffix(".json.coverage.json")
            coverage={"images":["a.png","empty.png"],"processed":["a.png","empty.png"],"missing":[],"detections":1}
            side.write_text(json.dumps(coverage))
            result=subprocess.run([sys.executable,str(SRC/"evaluate_mixed.py"),"--predictions",str(path),"--images",str(images),"--annotations",str(anns),"--output",str(root/"report.json"),"--thresholds-output",str(root/"thresholds.json")],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/"report.json").read_text()); self.assertEqual(report["per_class"]["mamianmakeng"]["ap50"],1)
            coverage["processed"]=["a.png"]; side.write_text(json.dumps(coverage))
            with self.assertRaisesRegex(ValueError,"coverage"): evalmod.load_predictions(path,{"a.png","empty.png"})
            coverage["processed"]=["a.png","empty.png"]; side.write_text(json.dumps(coverage))
            path.write_text(json.dumps([pred("a.png",.9,name="unknown")]))
            with self.assertRaisesRegex(ValueError,"class"): evalmod.load_predictions(path,{"a.png","empty.png"})
            with self.assertRaisesRegex(ValueError,"missing XML"): evalmod.load_ground_truth([images/"absent.png"],anns)

if __name__ == "__main__": unittest.main()
