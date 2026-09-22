"""Build the anonymised main document QIMS's double-blind review requires.

The journal reviews double-anonymised, which is why it asks for the title page
as a separate file. Everything that names the author, the company, the
repository or the archive comes out of the main document and stays on the
title page, where the editorial office reads it and the reviewers do not.

The repository is the awkward one. The paper's claim is that every number is
reproducible from it, and a reviewer cannot check that against a withheld URL.
The masked version says the repository exists, is public and is archived under
a DOI, and that both are on the title page -- so the claim is on the record and
the editor can release it if a reviewer asks.

    python paper/build_masked.py
"""

import re
from pathlib import Path

HERE = Path(r"D:\DevGit\vertebra-xray-core\paper")
source = HERE / "qims_manuscript.md"
target = HERE / "qims_manuscript_masked.md"

s = source.read_text(encoding="utf-8")

WITHHELD = "[repository URL withheld for double-anonymised review; given on the title page]"
ARCHIVE = "[archive DOI withheld for double-anonymised review; given on the title page]"


def cut_section(text: str, heading: str) -> str:
    start = text.index(heading)
    nxt = text.find("\n## ", start + 1)
    end = len(text) if nxt == -1 else nxt + 1
    return text[:start] + text[end:]


# -- title page: keep the title and the bibliographic counts, drop the person
old_block = re.search(r"\*\*Author\.\*\*.*?\*\*Correspondence\.\*\*[^\n]*\n", s, re.S).group(0)
s = s.replace(
    old_block,
    "*Author details are given in the separate title page, as the journal's\n"
    "double-anonymised review requires.*\n",
    1,
)

# -- repository and archive, in both places they appear
s = s.replace(
    "https://github.com/Institute-of-One/vertebra-xray-core,\narchived at https://doi.org/10.5281/zenodo.22893477,",
    f"{WITHHELD},\n{ARCHIVE},",
)
s = s.replace(
    "The software is available at https://github.com/Institute-of-One/vertebra-xray-core\n"
    "under the MIT licence, and the version used in this study, v0.1.0, is\n"
    "archived at https://doi.org/10.5281/zenodo.22893478.",
    "The software is available in a public repository under the MIT licence, and\n"
    "the version used in this study is archived under a DOI. Both are given on\n"
    "the title page and are withheld here for double-anonymised review.",
)

# -- sections that exist to name people and money
s = cut_section(s, "## Author Contributions")
s = cut_section(s, "## Acknowledgment")

s = re.sub(
    r"\*Funding:\*.*?(?=\n\*Conflicts of Interest:\*)",
    "*Funding:* Stated on the title page, withheld here for double-anonymised\nreview.\n\n",
    s,
    flags=re.S,
)
s = re.sub(
    r"\*Conflicts of Interest:\*.*?(?=\n\*Ethical Statement:\*)",
    "*Conflicts of Interest:* Declared in full on the title page and on the\n"
    "ICMJE forms accompanying this submission, and withheld here for\n"
    "double-anonymised review. The author has a commercial interest in\n"
    "downstream products that may use the software described; that software is\n"
    "released under the MIT licence and no patient or customer data were used.\n\n",
    s,
    flags=re.S,
)

s = s.replace(
    "*Every number reported here is produced",
    "*This is the anonymised copy for review; author, affiliation, funding and\n"
    "conflicts are on the separate title page. Every number reported here is\n"
    "produced",
    1,
)

target.write_text(s, encoding="utf-8")

leaks = [
    (i, line)
    for i, line in enumerate(s.splitlines(), 1)
    if re.search(
        r"Yamamoto|Institute of One|LISIT|TexelCraft|ORCID|0000-0001|github\.com|zenodo|IORN",
        line,
        re.I,
    )
]
print(f"wrote {target.name}  ({len(s.split())} words)")
if leaks:
    print("STILL IDENTIFYING:")
    for i, line in leaks:
        print(f"  {i}: {line.strip()[:90]}")
else:
    print("no identifying strings remain")
