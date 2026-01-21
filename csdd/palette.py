from __future__ import annotations


def default_palette(n: int) -> list[tuple[int, int, int]]:
    """A small, deterministic palette for visualizations."""
    base = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
    ]
    if n <= len(base):
        return base[:n]
    out = list(base)
    # Deterministic extension: cycle with a simple offset.
    i = 0
    while len(out) < n:
        r, g, b = base[i % len(base)]
        out.append(((r + 64) % 256, (g + 128) % 256, (b + 192) % 256))
        i += 1
    return out[:n]

