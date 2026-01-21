from __future__ import annotations


def register_all(task: str) -> None:
    """Register OpenMMLab modules and set default scope."""
    task = task.lower().strip()
    if task in {"det", "detect", "detection"}:
        from mmdet.utils import register_all_modules

        register_all_modules(init_default_scope=True)
        return
    if task in {"seg", "segment", "segmentation"}:
        from mmseg.utils import register_all_modules

        register_all_modules(init_default_scope=True)
        return
    raise ValueError(f"Unknown task: {task!r} (expected det/seg)")
