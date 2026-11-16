"""Shared evaluation helpers."""

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion. Better than p ± 1.96·se for small n or p near 0 or 1,
    which matters for the 88 bubble games and the 5 neutral-site ones."""
    if n <= 0:
        raise ValueError("need at least one trial")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
