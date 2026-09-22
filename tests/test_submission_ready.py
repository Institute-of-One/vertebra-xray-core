"""Checks that would have caught what nearly went out in the QIMS submission.

Each one is here because something got as far as the upload screen, or past
it, before being noticed by eye:

- a working note at the top of the manuscript, addressed to nobody but the
  author, about the journal's word limit;
- bracketed placeholders still marking what had not been decided;
- the repository URL in the data-availability statement, whose organisation
  name identifies the author, in a manuscript for double-anonymised review;
- a title page missing one of the nine items the journal lists.

None of these is about the science. All of them cost a round trip if an editor
finds them, and all of them are mechanical enough to check.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
MANUSCRIPT = PAPER / "qims_manuscript.md"
MASKED = PAPER / "qims_manuscript_masked.md"
TITLE_PAGE = PAPER / "title_page.md"
COVER_LETTER = PAPER / "cover_letter.md"

#: Anything that names the author, the institution, or where the code lives.
IDENTIFYING = re.compile(
    r"Yamamoto|Institute of One|LISIT|TexelCraft|ORCID|0000-0001|github\.com|zenodo|IORN",
    re.IGNORECASE,
)

SUBMITTED = [MANUSCRIPT, MASKED, TITLE_PAGE, COVER_LETTER]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", SUBMITTED, ids=lambda p: p.name)
def test_no_placeholders_survive_into_a_submitted_file(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"{path.name} not in this checkout")
    offenders = [
        f"line {i}: {line.strip()[:70]}"
        for i, line in enumerate(_read(path).splitlines(), 1)
        if "TBC" in line or "TODO" in line
    ]
    assert not offenders, f"{path.name} still carries placeholders:\n" + "\n".join(offenders)


@pytest.mark.parametrize("path", SUBMITTED, ids=lambda p: p.name)
def test_no_working_notes_survive_into_a_submitted_file(path: Path) -> None:
    """Notes to self read as notes to self, and they were nearly submitted."""
    if not path.exists():
        pytest.skip(f"{path.name} not in this checkout")
    text = _read(path)
    for pattern in (r"Prepared for [A-Z]", r"Word limit \d", r"\bText headings:"):
        assert not re.search(pattern, text), (
            f"{path.name} contains a working note matching {pattern!r}"
        )


def test_the_masked_manuscript_names_nobody() -> None:
    """QIMS reviews double-anonymised, so the review copy must not identify."""
    if not MASKED.exists():
        pytest.skip("masked manuscript not in this checkout")
    leaks = [
        f"line {i}: {line.strip()[:70]}"
        for i, line in enumerate(_read(MASKED).splitlines(), 1)
        if IDENTIFYING.search(line)
    ]
    assert not leaks, "the anonymised copy identifies the author:\n" + "\n".join(leaks)


def test_the_masked_manuscript_still_claims_the_code_is_public() -> None:
    """Anonymising must not quietly drop the reproducibility claim."""
    if not MASKED.exists():
        pytest.skip("masked manuscript not in this checkout")
    text = _read(MASKED)
    assert "public repository" in text
    assert "archived under a DOI" in text


def test_the_title_page_carries_every_item_the_journal_lists() -> None:
    """The nine items QIMS enumerates. Degree information was the one missing."""
    if not TITLE_PAGE.exists():
        pytest.skip("title page not in this checkout")
    text = _read(TITLE_PAGE)
    required = {
        "title": "**Title:**",
        "running title": "**Running title:**",
        "affiliation": "Institute of One",
        "degree": "Doctor of",
        "telephone": "Telephone:",
        "e-mail": "@",
        "ORCID": "0000-0001-9211-1071",
        "word count": "Main text approximately",
        "figure and table counts": "figures;",
        "author contributions": "Author Contributions",
    }
    missing = [name for name, probe in required.items() if probe not in text]
    assert not missing, f"the title page is missing: {', '.join(missing)}"


def test_the_fields_the_form_limits_are_within_their_limits() -> None:
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not in this checkout")
    text = _read(MANUSCRIPT)

    head = re.search(r"\*\*Running title\.\*\* (.+)", text).group(1).strip()
    assert len(head) <= 60, f"running head is {len(head)} characters, limit 60"

    keywords = re.search(r"\*\*Keywords\.\*\* (.+?)\n\n", text, re.S).group(1)
    count = len([k for k in " ".join(keywords.split()).split(";") if k.strip()])
    assert 3 <= count <= 5, f"{count} keywords, the journal allows 3 to 5"

    abstract = text[text.index("**Background.**") : text.index("## 1. Introduction")]
    words = len(re.sub(r"\*\*|---", "", abstract).split())
    assert words <= 450, f"abstract is {words} words, limit 450"


def test_subscripts_are_subscripts_and_filenames_are_not() -> None:
    """R_y came out of Word as three literal characters and read as code."""
    if not MANUSCRIPT.exists():
        pytest.skip("manuscript not in this checkout")
    text = _read(MANUSCRIPT)
    for matrix in ("R_y", "R_x", "R_z", "e_z"):
        assert matrix not in text, f"{matrix} should use subscript markup, not an underscore"
    # The other direction: a filename's underscore is the character itself.
    assert "normative_pedicles.py" in text


def test_the_built_word_file_carries_no_author_metadata() -> None:
    """Word keeps an author in docProps even when the text does not."""
    built = PAPER / "qims_manuscript_masked.docx"
    if not built.exists():
        pytest.skip("the masked .docx has not been built in this checkout")
    zipfile = pytest.importorskip("zipfile")
    with zipfile.ZipFile(built) as archive:
        for name in archive.namelist():
            if name.startswith("docProps/"):
                blob = archive.read(name).decode("utf-8", errors="replace")
                assert not IDENTIFYING.search(blob), f"{name} identifies the author"
