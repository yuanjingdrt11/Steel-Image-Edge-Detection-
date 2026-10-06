# 钢材表面缺陷检测

基于 YOLO11s 的超高分辨率工业图像缺陷检测方案。针对 4096×3000 灰度图像中的 9 类表面缺陷，采用**整图＋局部裁剪混合尺度监督**训练，配合**多尺度重叠切片推理与加权框融合**，在工业质检场景下取得高召回表现。

---

## 方案概览

| 环节 | 内容 |
|---|---|
| 模型 | YOLO11s（开源），从 COCO 80 类预训练权重初始化 |
| 输入尺寸 | 1280 × 1280，batch 4 |
| 训练数据 | 3184 张原图 → 扩展为 6401 张（2553 整图 + 3848 局部裁剪） |
| 数据划分 | 按产品编号分组划分，防止近重复泄漏 |
| 数据增强 | mosaic 0.8 / mixup 0.1 / scale 0.4 / 亮度 0.3 / 随机擦除 0.2 |
| 训练配置 | AdamW，lr0 0.0015，cos_lr，AMP，118 轮 |
| 推理 | 整图＋重叠切片（overlap 0.25），六路多尺度 TTA |
| 融合 | WBF（加权框融合），IoU 阈值 0.55 |

### 三项核心设计

**1. 混合尺度监督。** 目标尺度跨度极大——既有横跨数千像素的长条状裂纹，也有仅数十像素的细小夹杂。若将整图缩放到网络输入尺寸，小缺陷会缩小到几乎不可辨识。方案在训练集中保留全部整图以维持长裂纹的全局上下文，同时为每张训练图生成以缺陷目标为中心的高分辨率局部裁剪，让小缺陷在原分辨率下参与训练；验证集仅使用整图，杜绝信息泄漏。

**2. 分组划分与类别均衡。** 同一产品的连续帧高度相似，随机划分会造成训练/验证集近重复泄漏，因此按产品编号分组划分。此外，随机分组仍可能产生严重类别失衡——实测中曾出现主导类训练实例仅占该类 29.4% 的情况。方案通过评估不同划分下的类别分布予以修正，使主导类训练实例提升至 3.1 倍、总训练实例增加 42%。

**3. 多尺度测试时增强。** 单一尺度无法覆盖全部缺陷尺寸。方案在同一模型权重上执行六个前向（切片尺寸 1280 / 1920 / 1536 / 960，其中两个尺度含水平翻转），再用加权框融合合并，召回从单尺度的 0.904 提升至 0.946。

---

## 性能

### 训练指标

| 指标 | 数值 |
|---|---|
| mAP50 | 0.4202 |
| mAP50-95 | 0.2158 |
| 训练轮数 | 118 轮 |

### 推理指标（624 张独立验证集，IoU=0.5）

| 指标 | 单尺度 | 六路融合 |
|---|---|---|
| Recall | 0.904 | **0.946** |
| mAP50 | 0.438 | 0.455 |
| Precision | 0.015 | 0.006 |

**逐类召回**（六路融合）：

| 类别 | 召回 | 类别 | 召回 |
|---|---|---|---|
| mamianmakeng | 0.973 | zonglie | 0.935 |
| jieba | 0.973 | yiwuyaru | 0.912 |
| jiaza | 0.969 | huashang | 0.909 |
| gunyin | 0.963 | **平均** | **0.946** |
| yanghuatiepi | 0.950 | | |

各项指标的计算方式见 [`src/score_submission.py`](src/score_submission.py) 与 [`src/evaluate_mixed.py`](src/evaluate_mixed.py)。

---

## 类别体系

| 类别名 | 中文 |
|---|---|
| mamianmakeng | 麻面麻坑 |
| jieba | 结疤 |
| yiwuyaru | 异物压入 |
| yanghuatiepi | 氧化铁皮 |
| zonglie | 纵裂 |
| gunyin | 辊印 |
| jiaza | 夹杂 |
| huashang | 划伤 |
| qilie | 气裂 |

---

## 目录结构

```
steel-defect-detection/
├── README.md
├── requirements.txt
├── docs/
│   ├── 技术方案.md              完整技术方案
│   └── training_results.csv      训练曲线（118 轮）
├── src/
│   ├── prepare_mixed.py          数据准备：分组划分 + 混合尺度监督构建
│   ├── train_mixed.py            训练入口
│   ├── infer_mixed.py            推理：整图 + 多尺度重叠切片
│   ├── merge_predictions.py      多尺度融合（NMS / WBF）
│   ├── evaluate_mixed.py         评测核心库（AP/PR 计算）
│   ├── select_thresholds.py      逐类置信度阈值选择
│   ├── score_submission.py       按评分公式计算总分
│   ├── filter_submission.py      预测结果过滤与输出
│   ├── analyze_val.py            验证集多配置对比分析
│   └── validate_submission.py    输出格式校验
└── tests/
    ├── test_prepare_train_cli.py        16 项测试
    └── test_inference_evaluation.py      8 项测试
```

---

## 环境要求

| 项目 | 版本 |
|---|---|
| 操作系统 | Ubuntu 22.04 LTS |
| Python | 3.10 |
| PyTorch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| Ultralytics | 8.4.144 |
| OpenCV | 5.0.0（headless） |
| CUDA 驱动 | 470 及以上 |

### 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
# 先安装 CUDA 版 PyTorch，再装其余依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 使用流程

### 1. 数据准备

数据目录约定：图像与同名 VOC XML 标注放在同一目录下。

```bash
# 只读预检：报告配对情况、分组、空间需求，不写文件
python src/prepare_mixed.py --src train --out work/data/mixed \
  --tile 1280 --max-tiles-per-image 4 --seed 3 --dry-run

# 正式生成
python src/prepare_mixed.py --src train --out work/data/mixed \
  --tile 1280 --max-tiles-per-image 4 --seed 3
```

### 2. 训练

```bash
python src/train_mixed.py \
  --data work/data/mixed/data.yaml \
  --project work/runs --name mixed_scale \
  --epochs 150 --imgsz 1280 --batch 4 --workers 4 --device 0 \
  --patience 25 --mosaic 0.8 --mixup 0.1 --scale 0.4 \
  --translate 0.1 --hsv-v 0.3 --hsv-s 0.2 --erasing 0.2
```

### 3. 多尺度推理

```bash
W=work/runs/mixed_scale/weights/best.pt

for spec in "1280:" "1280:--hflip" "1920:" "1920:--hflip" "1536:" "960:"; do
  tile="${spec%%:*}"; flip="${spec#*:}"
  python src/infer_mixed.py --source <测试图片目录> --weights "$W" \
    --output "work/pred_t${tile}${flip:+h}.json" \
    --mode mixed --imgsz 1280 --tile "$tile" --conf 0.001 --device 0 $flip
done
```

### 4. 加权框融合

```bash
python src/merge_predictions.py \
  --predictions work/pred_t1280.json work/pred_t1280h.json \
  --predictions work/pred_t1920.json work/pred_t1920h.json \
  --predictions work/pred_t1536.json work/pred_t960.json \
  --output work/merged.json --mode wbf --wbf-iou 0.55 --wbf-conf 0.001
```

### 5. 阈值选择与结果输出

```bash
# 在验证集上按指定指标选择逐类阈值
python src/select_thresholds.py --predictions work/val_pred.json \
  --annotations train --output work/thresholds.json --metric f2

# 按阈值过滤并生成输出
python src/filter_submission.py --predictions work/merged.json \
  --thresholds work/thresholds.json --output result.json \
  --images <测试图片目录>

# 格式校验
python src/validate_submission.py result.json <测试图片目录>
```

---

## 单元测试

```bash
python -m unittest discover -s tests -v
```

共 24 项合成测试，覆盖数据分组隔离、裁剪标签正确性、原始数据只读保护、磁盘空间预检、输出格式校验、推理覆盖完整性、IoU 边界与同分阈值处理等关键逻辑，全部通过。

> 测试仅验证程序逻辑正确性，不代表模型检测效果。

---

## 技术文档

完整技术方案见 [`docs/技术方案.md`](docs/技术方案.md)，涵盖任务与数据分析、总体设计、数据准备、模型与训练、推理方案、结果输出策略、实验结果、复现指南、代码结构、工程经验与合规说明。

---

## 已知局限

1. **长条状目标检测偏弱**。纵裂、划伤等细长缺陷在切片时被截断，跨切片难以形成完整检测框，AP50 明显低于团块状缺陷。
2. **高召回伴随低精确率**。在全量候选输出策略下精确率仅 0.006。该策略适用于召回优先的工业质检场景（宁可误报不可漏检），若应用场景对误报敏感，需提高置信度阈值重新权衡。
3. **模型容量已接近上限**。逐类召回 0.912–0.973，mAP50 经多尺度融合后稳定在 0.455，推理层优化空间已耗尽。进一步提升需更换更大规模模型（如 YOLO11l/x 或 RT-DETR）重新训练。
4. **验证集与训练集同源**。所有指标来自与训练集相同产线的独立验证集，跨产线泛化能力需另行验证。
5. **数据质量**。原始数据存在少量未配对样本（图片缺标注或标注缺图片），程序会显式记录并排除，不会将其误作负样本。

---

## 许可与合规

- 使用开源 YOLO11s 模型与公开 COCO 预训练权重，未修改网络结构
- 未调用任何商业闭源 API 或在线推理服务
- 单一模型方案，未使用多模型投票或加权集成；多尺度 TTA 为同一模型权重的不同前向，不属于多模型集成
- 未使用测试集标签、未生成伪标签、未对检测结果进行任何人工修正
- 本项目**不包含任何训练数据、标注数据与测试数据**；使用时请自行准备符合授权要求的数据集
