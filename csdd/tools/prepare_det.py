from __future__ import annotations

import argparse
from pathlib import Path

from csdd.data.det_yolo_to_coco import ensure_csdd_det_coco_annotations


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare CSDD detection COCO annotations from YOLO txt labels.")
    ap.add_argument("--data-root", default="data/CSDD_det", help="Dataset root containing images/ and labels/.")
    ap.add_argument(
        "--classes",
        nargs="+",
        default=["Scratches", "Spots", "Rusts"],
        help="Class names for YOLO label ids 0..N-1.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Regenerate COCO annotations even if the JSON files already exist.",
    )
    args = ap.parse_args()

    out = ensure_csdd_det_coco_annotations(Path(args.data_root), class_names=list(args.classes), force=args.force)
    for split, p in out.items():
        print(f"[OK] {split}: {p}")


if __name__ == "__main__":
    main()
