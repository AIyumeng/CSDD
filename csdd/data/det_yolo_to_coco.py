from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class CocoCategory:
    id: int
    name: str
    supercategory: str = "defect"


def _yolo_to_xywh_abs(
    xc: float, yc: float, w: float, h: float, img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    bw = w * img_w
    bh = h * img_h
    x = (xc * img_w) - bw / 2.0
    y = (yc * img_h) - bh / 2.0
    return x, y, bw, bh


def convert_split_yolo_to_coco(
    *,
    images_dir: Path,
    labels_dir: Path,
    out_json: Path,
    class_names: list[str],
) -> Path:
    """Convert a YOLO-format split into COCO instances JSON.

    YOLO format: one txt per image, lines: `<cls> <xc> <yc> <w> <h>` in [0,1].
    COCO category ids are 1..N (COCO convention).
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_json = Path(out_json)

    out_json.parent.mkdir(parents=True, exist_ok=True)

    categories = [CocoCategory(id=i + 1, name=name).__dict__ for i, name in enumerate(class_names)]

    images = []
    annotations = []
    ann_id = 1

    img_paths = sorted(images_dir.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No .jpg images found in: {images_dir}")

    for img_id, img_path in enumerate(tqdm(img_paths, desc=f"YOLO->COCO {images_dir.name}"), start=1):
        with Image.open(img_path) as im:
            w, h = im.size

        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
            }
        )

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            # Allow images with no instances.
            continue

        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls_s, xc_s, yc_s, bw_s, bh_s = line.split()
                cls = int(cls_s)
                if cls < 0 or cls >= len(class_names):
                    raise ValueError(
                        f"Invalid class id {cls} in {label_path} (expected 0..{len(class_names) - 1})."
                    )
                xc, yc, bw, bh = float(xc_s), float(yc_s), float(bw_s), float(bh_s)

                x, y, bw_abs, bh_abs = _yolo_to_xywh_abs(xc, yc, bw, bh, w, h)
                # Clamp to image bounds (annotation noise / rounding).
                x = max(0.0, min(x, w - 1.0))
                y = max(0.0, min(y, h - 1.0))
                bw_abs = max(0.0, min(bw_abs, w - x))
                bh_abs = max(0.0, min(bh_abs, h - y))

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls + 1,  # map 0..N-1 -> 1..N
                        "bbox": [x, y, bw_abs, bh_abs],
                        "area": float(bw_abs * bh_abs),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}
    out_json.write_text(json.dumps(coco, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


def _coco_categories_match(out_json: Path, class_names: list[str]) -> bool:
    """Return True if `out_json` uses exactly `class_names` as COCO categories (id=1..N)."""
    coco = json.loads(Path(out_json).read_text(encoding="utf-8"))
    cats = coco.get("categories", [])
    cats_by_id = {int(c["id"]): str(c["name"]) for c in cats}
    if len(cats_by_id) != len(class_names):
        return False
    names_by_id = [cats_by_id.get(i + 1) for i in range(len(class_names))]
    if any(n is None for n in names_by_id):
        return False
    return names_by_id == list(class_names)


def ensure_csdd_det_coco_annotations(
    data_root: Path, class_names: list[str], *, force: bool = False
) -> dict[str, Path]:
    """Ensure COCO json exists for CSDD detection splits; return paths.

    If JSON exists but its categories don't match `class_names`, it will be regenerated.
    """
    data_root = Path(data_root)
    ann_dir = data_root / "annotations"

    mapping = {
        "train2017": ("images/train2017", "labels/train2017", ann_dir / "instances_train2017.json"),
        "val2017": ("images/val2017", "labels/val2017", ann_dir / "instances_val2017.json"),
        "test2017": ("images/test2017", "labels/test2017", ann_dir / "instances_test2017.json"),
    }

    out: dict[str, Path] = {}
    for split, (img_rel, lbl_rel, out_json) in mapping.items():
        needs_regen = force or (not out_json.exists())
        if (not needs_regen) and (not _coco_categories_match(out_json, class_names)):
            needs_regen = True

        if needs_regen:
            convert_split_yolo_to_coco(
                images_dir=data_root / img_rel,
                labels_dir=data_root / lbl_rel,
                out_json=out_json,
                class_names=class_names,
            )
        out[split] = out_json
    return out
