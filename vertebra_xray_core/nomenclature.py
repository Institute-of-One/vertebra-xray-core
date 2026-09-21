"""Vertebral level names and the arithmetic of stepping between them.

A single ordered vocabulary runs from C1 down to S1. Every label in this
package is one of these strings, and every "two levels below T10" style
question goes through :func:`step`, so an off-by-one can only ever be wrong
in one place.
"""

from __future__ import annotations

__all__ = [
    "CERVICAL",
    "THORACIC",
    "LUMBAR",
    "SACRAL",
    "ALL_LEVELS",
    "index_of",
    "level_at",
    "step",
    "span",
    "region_of",
    "THORACOLUMBAR_JUNCTION",
    "CERVICOTHORACIC_JUNCTION",
    "LUMBOSACRAL_JUNCTION",
]

CERVICAL = tuple(f"C{i}" for i in range(1, 8))
THORACIC = tuple(f"T{i}" for i in range(1, 13))
LUMBAR = tuple(f"L{i}" for i in range(1, 6))
SACRAL = ("S1",)

#: Cranial to caudal, the only ordering used anywhere in this package.
ALL_LEVELS: tuple[str, ...] = CERVICAL + THORACIC + LUMBAR + SACRAL

_INDEX = {name: i for i, name in enumerate(ALL_LEVELS)}

#: The level immediately *cranial* to each anatomical transition. The
#: thoracolumbar junction is the T12/L1 disc, so the anchor level is T12.
THORACOLUMBAR_JUNCTION = "T12"
CERVICOTHORACIC_JUNCTION = "C7"
LUMBOSACRAL_JUNCTION = "L5"


def index_of(level: str) -> int:
    """Position of ``level`` in :data:`ALL_LEVELS` (0 = C1)."""
    try:
        return _INDEX[level]
    except KeyError:
        raise ValueError(f"unknown vertebral level {level!r}") from None


def level_at(index: int) -> str:
    """Level name at ``index``, counting caudally from C1."""
    if not 0 <= index < len(ALL_LEVELS):
        raise ValueError(f"vertebral index {index} is outside C1..S1")
    return ALL_LEVELS[index]


def step(level: str, n: int) -> str:
    """The level ``n`` positions caudal to ``level`` (negative ``n`` = cranial)."""
    return level_at(index_of(level) + n)


def span(cranial: str, caudal: str) -> tuple[str, ...]:
    """Inclusive run of levels from ``cranial`` down to ``caudal``."""
    a, b = index_of(cranial), index_of(caudal)
    if a > b:
        raise ValueError(f"{cranial!r} is not cranial to {caudal!r}")
    return ALL_LEVELS[a : b + 1]


def region_of(level: str) -> str:
    """Return ``cervical``, ``thoracic``, ``lumbar`` or ``sacral``."""
    index_of(level)  # validates
    return {"C": "cervical", "T": "thoracic", "L": "lumbar", "S": "sacral"}[level[0]]
