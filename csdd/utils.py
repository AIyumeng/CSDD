from __future__ import annotations

from mmengine.runner import set_random_seed


def seed_everything(seed: int) -> None:
    set_random_seed(seed, deterministic=False)
