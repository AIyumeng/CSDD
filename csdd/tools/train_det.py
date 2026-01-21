from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.runner import Runner

from csdd.data.det_yolo_to_coco import ensure_csdd_det_coco_annotations
from csdd.openmmlab import register_all
from csdd.tools._common import add_gin_args, load_cfg_from_gin
from csdd.utils import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CSDD defect detection (MMDetection).")
    add_gin_args(ap)
    ap.add_argument("--no-prepare", action="store_true", help="Skip YOLO->COCO conversion step.")
    ap.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if present.")
    args = ap.parse_args()

    register_all("det")

    from csdd.recipes.det_fasterrcnn import build_det_cfg

    cfg = load_cfg_from_gin(gin_files=args.gin, bindings=args.bind, build_fn=build_det_cfg)

    # Create COCO jsons (train/val/test) if missing.
    if not args.no_prepare:
        classes = list(cfg.train_dataloader.dataset.metainfo["classes"])
        ensure_csdd_det_coco_annotations(Path(cfg.train_dataloader.dataset.data_root), class_names=classes)

    seed_everything(int(cfg.randomness.seed))
    cfg.resume = bool(args.resume)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()

