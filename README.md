# CSDD Baselines

This repo provides end-to-end training / evaluation / visualization for the CSDD dataset:
- Detection: COCO mAP + per-class AP
- Segmentation: IoU / mIoU + per-class IoU

Classes:
- Detection classes: `Scratches`, `Spots`, `Rusts`
- Segmentation classes: `background`, `Scratches`, `Spots`, `Rusts`

## Dataset Layout

- Detection
  - `data/CSDD_det/images/{train2017,val2017,test2017}/*.jpg`
  - `data/CSDD_det/labels/{train2017,val2017,test2017}/*.txt` (YOLO: `cls xc yc w h`)
- Segmentation
  - `data/CSDD_seg/img/{train,val,test}/*.jpg`
  - `data/CSDD_seg/ground_truth/{train,val,test}/*_mask.png`

## Environment

```bash
conda create -n csdd python=3.10 -y
conda activate csdd
pip install -r requirements.txt
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2+cu121 --index-url https://download.pytorch.org/whl/cu121
python -m mim install "mmengine==0.10.7" "mmcv==2.1.0" "mmdet==3.3.0" "mmsegmentation==1.2.2"
```

## Quickstart

### Detection (mAP)

Training (will auto-convert YOLO labels to COCO JSON under `data/CSDD_det/annotations/` if missing):
```bash
python -m csdd.tools.train_det --gin configs/det.gin
```

Eval:
```bash
python -m csdd.tools.test_det --gin configs/det.gin --split val2017  --checkpoint work_dirs/det/best_coco_bbox_mAP.pth
python -m csdd.tools.test_det --gin configs/det.gin --split test2017 --checkpoint work_dirs/det/best_coco_bbox_mAP.pth
```

Visualization:
```bash
python -m csdd.tools.vis_det --gin configs/det.gin --split test2017 --checkpoint work_dirs/det/best_coco_bbox_mAP.pth --out-dir work_dirs/det/vis_test
```

COCO annotation conversion only:
```bash
python -m csdd.tools.prepare_det --data-root data/CSDD_det
```
Force-regenerate COCO JSON (e.g. after changing class names):
```bash
python -m csdd.tools.prepare_det --data-root data/CSDD_det --force
```

### Segmentation (IoU/mIoU)

Training:
```bash
python -m csdd.tools.train_seg --gin configs/seg.gin
```

Eval:
```bash
python -m csdd.tools.test_seg --gin configs/seg.gin --split val  --checkpoint work_dirs/seg/best_mIoU.pth
python -m csdd.tools.test_seg --gin configs/seg.gin --split test --checkpoint work_dirs/seg/best_mIoU.pth
```

Visualization:
```bash
python -m csdd.tools.vis_seg --gin configs/seg.gin --split test --checkpoint work_dirs/seg/best_mIoU.pth --out-dir work_dirs/seg/vis_test
```

## Config (gin)

- Detection: `configs/det.gin`
- Segmentation: `configs/seg.gin`

Override an option via `--bind`, e.g.:
```bash
python -m csdd.tools.train_det --gin configs/det.gin --bind csdd.recipes.det_fasterrcnn.build_det_cfg.max_epochs=12
```
