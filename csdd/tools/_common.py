from __future__ import annotations

import argparse
from pathlib import Path

from mmengine.config import Config

from csdd.gin_utils import parse_gin


def add_gin_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gin", nargs="+", required=True, help="Path(s) to gin config file(s).")
    p.add_argument(
        "--bind",
        nargs="*",
        default=None,
        help='Extra gin bindings, e.g. csdd.recipes.det_fasterrcnn.build_det_cfg.lr=0.02',
    )


def load_cfg_from_gin(*, gin_files: list[str], bindings: list[str] | None, build_fn) -> Config:
    parse_gin(gin_files, bindings)
    cfg_dict = build_fn()
    return Config(cfg_dict)


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

