from __future__ import annotations

import argparse

from mmengine.runner import Runner

from csdd.openmmlab import register_all
from csdd.tools._common import add_gin_args, load_cfg_from_gin
from csdd.utils import seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CSDD defect segmentation (MMSegmentation).")
    add_gin_args(ap)
    ap.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint if present.")
    args = ap.parse_args()

    register_all("seg")
    # Required for mmsegmentation==1.2.2: ensure the custom dataset class is registered.
    import csdd.data.seg_dataset  # noqa: F401
    # Register custom metrics.
    import csdd.metrics.fg_iou_metric  # noqa: F401

    from csdd.recipes.seg_unet import build_seg_cfg

    cfg = load_cfg_from_gin(gin_files=args.gin, bindings=args.bind, build_fn=build_seg_cfg)
    seed_everything(int(cfg.randomness.seed))
    cfg.resume = bool(args.resume)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == "__main__":
    main()
