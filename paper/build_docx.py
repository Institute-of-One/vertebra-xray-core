"""Convert the manuscript, title page and cover letter to .docx for submission.

AME's portal takes Word, not Markdown. Nothing is retyped: the .docx files are
built from the Markdown that is under version control, so the submitted text
and the text the repository shows are the same text.

    python paper/build_docx.py

Pandoc is reached through pypandoc. It does not emit line or page numbers, and
it does not number figures in Word output; the figure legends here are a
numbered list written by hand, so that does not bite. Figures are uploaded as
separate files and are not embedded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pypandoc

HERE = Path(__file__).resolve().parent
TARGETS = ("qims_manuscript", "title_page", "cover_letter")


def build(stem: str) -> Path:
    source = HERE / f"{stem}.md"
    if not source.exists():
        raise SystemExit(f"missing {source}")
    out = HERE / f"{stem}.docx"
    pypandoc.convert_file(
        str(source),
        "docx",
        format="markdown+pipe_tables+yaml_metadata_block",
        outputfile=str(out),
        extra_args=["--standalone"],
    )
    return out


def main() -> int:
    print(f"pandoc {pypandoc.get_pandoc_version()}")
    for stem in TARGETS:
        out = build(stem)
        print(f"  wrote {out.name}  ({out.stat().st_size:,} bytes)")

    remaining = []
    for stem in TARGETS:
        text = (HERE / f"{stem}.md").read_text(encoding="utf-8")
        remaining += [
            f"{stem}.md:{i}" for i, line in enumerate(text.splitlines(), 1) if "TBC" in line
        ]
    if remaining:
        print("\nPlaceholders still in the source, and therefore in the .docx:")
        for where in remaining:
            print(f"  {where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
