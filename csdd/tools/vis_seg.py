from __future__ import annotations

import argparse
from pathlib import Path

import mmcv

from csdd.openmmlab import register_all
from csdd.tools._common import add_gin_args, ensure_dir, load_cfg_from_gin


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize CSDD segmentation predictions (mask overlay).")
    add_gin_args(ap)
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out-dir", required=True, help="Output directory for visualizations.")
    ap.add_argument("--opacity", type=float, default=0.6)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    register_all("seg")
    # Required for mmsegmentation==1.2.2: ensure the custom dataset class is registered.
    import csdd.data.seg_dataset  # noqa: F401

    from mmseg.apis import inference_model, init_model
    from mmseg.visualization import SegLocalVisualizer

    from csdd.recipes.seg_unet import build_seg_cfg

    cfg = load_cfg_from_gin(gin_files=args.gin, bindings=args.bind, build_fn=build_seg_cfg)
    model = init_model(cfg, args.checkpoint, device=args.device)

    data_root = Path(cfg.test_dataloader.dataset.data_root)
    img_dir = data_root / "img" / args.split
    out_dir = ensure_dir(args.out_dir)

    classes = tuple(cfg.test_dataloader.dataset.metainfo["classes"])
    palette = cfg.test_dataloader.dataset.metainfo.get("palette", None)
    visualizer = SegLocalVisualizer(alpha=args.opacity)
    visualizer.dataset_meta = {"classes": classes, "palette": palette}

    img_paths = sorted(img_dir.glob("*.jpg"))[: args.limit]
    for p in img_paths:
        img = mmcv.imread(str(p), channel_order="rgb")
        pred = inference_model(model, str(p))
        out_file = out_dir / p.name
        visualizer.add_datasample(
            name=p.stem,
            image=img,
            data_sample=pred,
            draw_gt=False,
            out_file=str(out_file),
        )
        print(f"[OK] {out_file}")


if __name__ == "__main__":
    main()
