# CSDD 基线

本仓库提供 CSDD 数据集的训练 / 评测 / 可视化代码：
- 缺陷目标检测：COCO mAP + 各类别 AP
- 缺陷分割：IoU / mIoU + 各类别 IoU

类别含义：
- 检测：Scratches（划痕）、Spots（斑点）、Rusts（锈蚀）
- 分割：background、Scratches（划痕）、Spots（斑点）、Rusts（锈蚀）

## 数据集目录

- 检测（Detection）
  - `data/CSDD_det/images/{train2017,val2017,test2017}/*.jpg`
  - `data/CSDD_det/labels/{train2017,val2017,test2017}/*.txt`（YOLO：`cls xc yc w h`）
- 分割（Segmentation）
  - `data/CSDD_seg/img/{train,val,test}/*.jpg`
  - `data/CSDD_seg/ground_truth/{train,val,test}/*_mask.png`

## 环境

```bash
conda create -n csdd python=3.10 -y
conda activate csdd
pip install -r requirements.txt
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
python -m mim install "mmengine==0.10.7" "mmcv==2.1.0" "mmdet==3.3.0" "mmsegmentation==1.2.2"
```

## 快速开始

### 检测（mAP）

训练（首次运行会在缺失时自动把 YOLO 标签转换成 COCO JSON，输出到 `data/CSDD_det/annotations/`）：
```bash
python -m csdd.tools.train_det --gin configs/det.gin
```

评测：
```bash
python -m csdd.tools.test_det --gin configs/det.gin --split val2017  --checkpoint work_dirs/det/best_coco_bbox_mAP.pth
python -m csdd.tools.test_det --gin configs/det.gin --split test2017 --checkpoint work_dirs/det/best_coco_bbox_mAP.pth
```

可视化：
```bash
python -m csdd.tools.vis_det --gin configs/det.gin --split test2017 --checkpoint work_dirs/det/best_coco_bbox_mAP.pth --out-dir work_dirs/det/vis_test
```

只进行 COCO 标注转换（可选）：
```bash
python -m csdd.tools.prepare_det --data-root data/CSDD_det
```
强制重新生成 COCO JSON（例如你修改过类别名称后）：
```bash
python -m csdd.tools.prepare_det --data-root data/CSDD_det --force
```

### 分割（IoU/mIoU）

训练：
```bash
python -m csdd.tools.train_seg --gin configs/seg.gin
```

评测：
```bash
python -m csdd.tools.test_seg --gin configs/seg.gin --split val  --checkpoint work_dirs/seg/best_mIoU.pth
python -m csdd.tools.test_seg --gin configs/seg.gin --split test --checkpoint work_dirs/seg/best_mIoU.pth
```

可视化：
```bash
python -m csdd.tools.vis_seg --gin configs/seg.gin --split test --checkpoint work_dirs/seg/best_mIoU.pth --out-dir work_dirs/seg/vis_test
```

## 配置（gin）

- 检测：`configs/det.gin`
- 分割：`configs/seg.gin`

命令行覆盖参数示例：
```bash
python -m csdd.tools.train_det --gin configs/det.gin --bind csdd.recipes.det_fasterrcnn.build_det_cfg.max_epochs=12
```
