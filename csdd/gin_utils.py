from __future__ import annotations

from collections.abc import Iterable

import gin


def parse_gin(files: Iterable[str], bindings: Iterable[str] | None = None) -> None:
    """Parse gin files/bindings (and clear any previous gin state)."""
    gin.clear_config()
    for f in files:
        gin.parse_config_file(f)
    if bindings:
        for b in bindings:
            gin.parse_config(b)
    gin.finalize()

