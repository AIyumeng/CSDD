from __future__ import annotations

import argparse

from mmengine.runner import Runner

from csdd.openmmlab import register_all
from csdd.tools._common import add_gin_args, load_cfg_from_gin


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate CSDD defect segmentation (MMSegmentation).")
    add_gin_args(ap)
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    args = ap.parse_args()

    register_all("seg")
    # Required for mmsegmentation==1.2.2: ensure the custom dataset class is registered.
    import csdd.data.seg_dataset  # noqa: F401

    from csdd.recipes.seg_unet import build_seg_cfg

    cfg = load_cfg_from_gin(gin_files=args.gin, bindings=args.bind, build_fn=build_seg_cfg)
    cfg.load_from = args.checkpoint

    cfg.test_dataloader.dataset.split = args.split
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    # Convenience: also report mIoU excluding background if IoU array is available.
    ious = metrics.get("IoU", None)
    if ious is not None:
        try:
            ious_list = list(ious)
            if len(ious_list) > 1:
                miou_no_bg = sum(ious_list[1:]) / max(1, (len(ious_list) - 1))
                metrics["mIoU_no_bg"] = float(miou_no_bg)
        except TypeError:
            pass
    print(metrics)


if __name__ == "__main__":
    main()
