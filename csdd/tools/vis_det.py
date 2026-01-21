from __future__ import annotations

import argparse
from pathlib import Path

import mmcv

from csdd.openmmlab import register_all
from csdd.tools._common import add_gin_args, load_cfg_from_gin, ensure_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize CSDD detection predictions (bbox).")
    add_gin_args(ap)
    ap.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint.")
    ap.add_argument("--split", choices=["val2017", "test2017"], default="test2017")
    ap.add_argument("--out-dir", required=True, help="Output directory for visualizations.")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    register_all("det")

    from mmdet.apis import inference_detector, init_detector
    from mmdet.visualization import DetLocalVisualizer

    from csdd.recipes.det_fasterrcnn import build_det_cfg

    cfg = load_cfg_from_gin(gin_files=args.gin, bindings=args.bind, build_fn=build_det_cfg)
    model = init_detector(cfg, args.checkpoint, device=args.device)

    data_root = Path(cfg.test_dataloader.dataset.data_root)
    img_dir = data_root / "images" / args.split
    out_dir = ensure_dir(args.out_dir)

    classes = tuple(cfg.train_dataloader.dataset.metainfo["classes"])
    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = {"classes": classes}

    img_paths = sorted(img_dir.glob("*.jpg"))[: args.limit]
    for p in img_paths:
        img = mmcv.imread(str(p), channel_order="rgb")
        pred = inference_detector(model, str(p))
        out_file = out_dir / p.name
        visualizer.add_datasample(
            name=p.stem,
            image=img,
            data_sample=pred,
            draw_gt=False,
            pred_score_thr=args.score_thr,
            out_file=str(out_file),
        )
        print(f"[OK] {out_file}")


if __name__ == "__main__":
    main()
