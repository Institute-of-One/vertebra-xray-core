"""Adapters for public datasets.

Each adapter turns a published dataset into this package's own types and
nothing else: no measurement logic lives here, so a dataset's quirks cannot
leak into the numbers. Adapters are optional imports -- reading VerSe needs
nibabel, which the core does not.
"""

from __future__ import annotations

__all__ = ["verse"]
