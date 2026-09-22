"""Emit the submission form's fields as plain text, ready to paste.

The portal's step 1 takes the abstract and the cover letter into textareas,
not as files, so they have to go in as text rather than Markdown. Asterisks
pasted into a web form stay asterisks.

Everything is read from the manuscript and the cover letter under version
control, so what goes into the form is what the repository holds.

    python paper/form_fields.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNING_HEAD_LIMIT = 60
ABSTRACT_LIMIT = 450


def plain(text: str) -> str:
    """Markdown emphasis and links out, readable text in."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = text.replace("<", "").replace(">", "")
    return text


def unwrap(text: str) -> str:
    """One paragraph to one line, which is how a textarea wants it."""
    out = []
    for block in re.split(r"\n\s*\n", text.strip()):
        out.append(" ".join(block.split()))
    return "\n\n".join(out)


def field(name: str, value: str, note: str = "") -> None:
    print(f"\n{'=' * 70}\n{name}{'  — ' + note if note else ''}\n{'=' * 70}")
    print(value)


def main() -> int:
    manuscript = (HERE / "qims_manuscript.md").read_text(encoding="utf-8")
    letter = (HERE / "cover_letter.md").read_text(encoding="utf-8")

    title = re.search(r"\*\*Title\.\*\* (.+?)\n\n", manuscript, re.DOTALL).group(1)
    title = " ".join(title.split())

    head = re.search(r"\*\*Running title\.\*\* (.+)", manuscript).group(1).strip()

    keywords = re.search(r"\*\*Keywords\.\*\* (.+?)\n\n", manuscript, re.DOTALL).group(1)
    keywords = [k.strip() for k in " ".join(keywords.split()).split(";") if k.strip()]

    abstract = manuscript[
        manuscript.index("**Background.**") : manuscript.index("## 1. Introduction")
    ]
    abstract = unwrap(plain(abstract.replace("---", "").strip()))

    funding = re.search(r"\*Funding:\* (.+?)\n\n", manuscript, re.DOTALL).group(1)
    funding = unwrap(plain(funding))

    body = letter[letter.index("Dear Editor,") :]
    body = body[: body.index("Yours sincerely,")]
    body = unwrap(plain(body))

    field("ARTICLE TYPE", "Original Article", "already selected")
    field("TITLE", title, f"{len(title)} characters")
    field("RUNNING HEAD", head, f"{len(head)} of {RUNNING_HEAD_LIMIT} characters including spaces")
    field("KEYWORDS", ", ".join(keywords), "the form asks for commas, not semicolons")
    field("ABSTRACT", abstract, f"{len(abstract.split())} words, limit {ABSTRACT_LIMIT}")
    field("FUNDING", funding)
    field("COVER LETTER", body, "paste as text; the .docx is for the file upload step")
    field("INVITED ARTICLE", "No", "already selected")

    problems = []
    if len(head) > RUNNING_HEAD_LIMIT:
        problems.append(
            f"running head is {len(head)} characters, over the {RUNNING_HEAD_LIMIT} limit"
        )
    if len(abstract.split()) > ABSTRACT_LIMIT:
        problems.append(
            f"abstract is {len(abstract.split())} words, over the {ABSTRACT_LIMIT} limit"
        )
    if not 3 <= len(keywords) <= 5:
        problems.append(f"{len(keywords)} keywords, the guidelines allow 3 to 5")
    for name, value in (("abstract", abstract), ("cover letter", body), ("title", title)):
        if "TBC" in value:
            problems.append(f"the {name} still contains a TBC placeholder")

    # Non-ASCII survives most forms and is mangled by a few. Know what is in
    # there before pasting, so that checking afterwards is possible.
    exotic: dict[str, int] = {}
    for chunk in (title, head, abstract, funding, body):
        for ch in chunk:
            if ord(ch) > 127:
                exotic[ch] = exotic.get(ch, 0) + 1
    if exotic:
        print(f"\n{'=' * 70}\nNON-ASCII TO CHECK AFTER PASTING\n{'=' * 70}")
        for ch, n in sorted(exotic.items(), key=lambda kv: -kv[1]):
            print(f"  {ch!r}  U+{ord(ch):04X}  {n} time(s)")

    print(f"\n{'=' * 70}")
    if problems:
        for p in problems:
            print(f"PROBLEM: {p}")
        return 1
    print("All fields within their limits.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
