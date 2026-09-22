"""The manuscript's typed tables must agree with the package they describe.

Most numbers in the paper are printed by a script and copied once. The
normative pedicle table is different: it is a constant in the package and also
a table in section 3.4, typed out. The two drifted -- four values, after the
constants were synced to what ``tools/normative_pedicles.py`` measures -- and
nothing caught it until the .docx was read by eye on submission day.

So it is checked here instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vertebra_xray_core.pedicles import NORMATIVE_PEDICLES_BY_LEVEL

MANUSCRIPT = Path(__file__).resolve().parents[1] / "paper" / "qims_manuscript.md"


def _row(table: str, name: str) -> list[str]:
    for line in table.splitlines():
        if line.startswith(f"| {name} "):
            return [cell.strip() for cell in line.strip("|").split("|")][1:]
    raise AssertionError(f"no {name!r} row in the pedicle table")


@pytest.mark.skipif(not MANUSCRIPT.exists(), reason="manuscript not in this checkout")
def test_section_3_4_pedicle_table_matches_the_package() -> None:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^\| level \|.*?(?=\n\n)", text, re.MULTILINE | re.DOTALL)
    assert match, "the pedicle table in section 3.4 has moved or changed shape"
    table = match.group(0)

    levels = _row(table, "level")
    separations = _row(table, "half-separation (mm)")
    offsets = _row(table, "posterior offset (mm)")
    assert len(levels) == len(separations) == len(offsets)

    for level, separation, offset in zip(levels, separations, offsets, strict=True):
        shipped = NORMATIVE_PEDICLES_BY_LEVEL[level]
        assert separation == f"{shipped.half_separation:.1f}", (
            f"{level} half-separation: manuscript says {separation}, "
            f"the package says {shipped.half_separation:.1f}"
        )
        assert offset == f"{shipped.posterior_offset:.1f}", (
            f"{level} posterior offset: manuscript says {offset}, "
            f"the package says {shipped.posterior_offset:.1f}"
        )


@pytest.mark.skipif(not MANUSCRIPT.exists(), reason="manuscript not in this checkout")
def test_every_reference_is_cited_and_numbered_consecutively() -> None:
    """Vancouver numbering is in order of first appearance, with no gaps.

    A reference that is listed but never cited, or a citation to a number that
    does not exist, is the kind of thing a copy-editor finds and an author
    does not.
    """
    text = MANUSCRIPT.read_text(encoding="utf-8")
    body, _, tail = text.partition("## References")
    assert tail, "no reference list"
    # The list runs to the next top-level heading; Figures and Tables follow it
    # and are numbered lists of their own.
    references = re.split(r"^## ", tail, maxsplit=1, flags=re.MULTILINE)[0]

    listed = [int(n) for n in re.findall(r"^(\d+)\. ", references, re.MULTILINE)]
    assert listed == list(range(1, len(listed) + 1)), f"reference list is not consecutive: {listed}"

    cited: set[int] = set()
    for group in re.findall(r"\((\d+(?:,\d+)*)\)", body):
        cited.update(int(n) for n in group.split(","))

    assert not cited - set(listed), f"cited but not listed: {sorted(cited - set(listed))}"
    assert not set(listed) - cited, f"listed but never cited: {sorted(set(listed) - cited)}"
