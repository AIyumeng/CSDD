from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.runner import Runner

from csdd.data.det_yolo_to_coco import ensure_csdd_det_coco_annotations
from csdd.openmmlab import register_all
from csdd.tools._common import add_gin_args, load_cfg_from_gin


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CSDD defect detection (MMDetection).")
    add_gin_args(ap)
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    ap.add_argument("--split", choices=["val2017", "test2017"], default="test2017")
    ap.add_argument("--no-prepare", action="store_true", help="Skip YOLO->COCO conversion step.")
    args = ap.parse_args()

    register_all("det")

    from csdd.recipes.det_fasterrcnn import build_det_cfg

    cfg = load_cfg_from_gin(gin_files=args.gin, bindings=args.bind, build_fn=build_det_cfg)
    cfg.load_from = args.checkpoint

    data_root = Path(cfg.test_dataloader.dataset.data_root)
    ann_dir = data_root / "annotations"

    if not args.no_prepare:
        classes = list(cfg.train_dataloader.dataset.metainfo["classes"])
        ensure_csdd_det_coco_annotations(data_root, class_names=classes)

    if args.split == "val2017":
        cfg.test_dataloader.dataset.ann_file = str((ann_dir / "instances_val2017.json").resolve())
        cfg.test_dataloader.dataset.data_prefix = {"img": "images/val2017/"}
        cfg.test_evaluator.ann_file = str((ann_dir / "instances_val2017.json").resolve())
    else:
        cfg.test_dataloader.dataset.ann_file = str((ann_dir / "instances_test2017.json").resolve())
        cfg.test_dataloader.dataset.data_prefix = {"img": "images/test2017/"}
        cfg.test_evaluator.ann_file = str((ann_dir / "instances_test2017.json").resolve())

    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    print(metrics)


if __name__ == "__main__":
    main()
