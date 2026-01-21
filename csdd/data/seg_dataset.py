from __future__ import annotations

from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class CSDDSegDataset(BaseSegDataset):
    """CSDD semantic segmentation dataset.

    Expected structure (relative to data_root):
      img/{train,val,test}/*.jpg
      ground_truth/{train,val,test}/*_mask.png
    """

    METAINFO = {
        # Defaults; override via cfg.dataset.metainfo for your own names/palette.
        "classes": ("background", "Scratches", "Spots", "Rusts"),
        "palette": [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)],
    }

    def __init__(
        self,
        data_root: str,
        split: str,
        **kwargs,
    ):
        split = split.lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"split must be train/val/test, got {split!r}")

        # mmsegmentation==1.2.2 BaseSegDataset uses `data_prefix` rather than `img_dir/ann_dir`.
        # With `ann_file=''`, it will scan `data_prefix.img_path` and map to seg_map by suffix.
        super().__init__(
            data_root=data_root,
            ann_file="",
            data_prefix=dict(img_path=f"img/{split}", seg_map_path=f"ground_truth/{split}"),
            img_suffix=".jpg",
            seg_map_suffix="_mask.png",
            reduce_zero_label=False,
            **kwargs,
        )
