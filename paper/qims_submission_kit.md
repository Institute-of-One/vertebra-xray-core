# QIMS submission kit — IORN-014

Fields in the order the form asks for them.

> **Verify before submitting.** The requirements below were read from the QIMS
> guidelines for authors on 2026-09-22. AME updates these pages; re-check the
> live guidelines and the APC page immediately before submitting. Anything
> marked **[verify]** could not be settled from a primary source and needs the
> editorial office.

---

## 0. The one thing to decide before opening the form

The submission checklist asks you to confirm that **institutional review board
approval and informed consent were acquired**. For this study they were not,
because they were not required: computational phantoms and a public
de-identified collection, no recruitment, no new human data.

Do not tick past it silently. The position is set out in the cover letter,
which says what the study did and invites the editorial office to direct a
different route. Attach the cover letter, and if the form blocks you, ask the
editorial office before going further.

The submission runbook §8.3 is the reasoning: the question is whether human
*participation data* were used, not whether participants were recruited, and
stopping at "it was not required" reads as research nobody approved. The
approval exists — VerSe holds it, under Proposal 27/19 S-SR.

---

## 0b. The other thing the checklist asks that needs a word

Item 1 asks you to confirm the submission has not been previously published,
and to explain in the cover letter otherwise. It has not been published or
submitted anywhere. But the manuscript source is versioned alongside the code,
so it is visible in the public repository and a copy sits inside the archived
release. The guidelines state no preprint or repository policy.

The cover letter discloses this and offers to remove the manuscript from the
live repository if the journal prefers. Tick item 1 and let the letter carry
the explanation, which is what the item itself directs.

---

## 1. Files to upload

| Purpose | File |
|---|---|
| Manuscript | `paper/qims_manuscript.md` (convert to .docx before upload) |
| Cover letter | `paper/cover_letter.md` |
| Figures 1–7 | `paper/figures/fig{1..7}_*.png`, 400 dpi, separate files |
| ICMJE disclosure form | completed by the author |

Figures are already above the 300 dpi floor. They are original: six are
rendered from computational phantoms and Figure 3 plots measurements, so **no
copyright permission is needed from anyone**, and no VerSe image appears
anywhere.

---

## 2. Form fields

**Article type:** Original Article.

**Title:**
> Two more landmarks: identifiability of the three-dimensional Cobb angle
> measured from biplanar radiographs

**Running title** (60 characters, at the limit — do not lengthen it):
> Identifiability of the biplanar three-dimensional Cobb angle

**Author:** Shuji Yamamoto — sole author, and also the corresponding author.

**Affiliation:** if the form has separate fields,
Department = `Institute of One`, Institution = `LISIT Co., Ltd.`,
City = `Tokyo`, Country = `Japan`.
If it has one field: `Institute of One, LISIT Co., Ltd., Tokyo 150-0044, Japan`.

**ORCID:** 0000-0001-9211-1071 — connect the account rather than typing it.

**E-mail:** yamamoto@lisit.jp (matches the company domain, which is what makes
the affiliation verifiable).

**Keywords** (5; the guidelines allow 3–5):
scoliosis; Cobb angle; biplanar radiography; vertebral rotation; measurement
uncertainty

**Abstract:** copy from the manuscript. Structured, 435 words against a 450
ceiling, so it has almost no headroom — if the form reports a different count,
something was truncated or doubled on paste.

**Counts:** main text about 3,100 words; 7 figures; 3 tables; 23 references.

---

## 3. Declarations, which must match the manuscript word for word

The ICMJE form and the manuscript are read side by side. Disagreement between
them is a revision request.

**Conflicts of interest.** The author is the representative of LISIT Co., Ltd.
(Tokyo, Japan) and Chief Executive Officer of TexelCraft OÜ. Institute of One
is the open-research initiative of LISIT Co., Ltd., which provides
institutional oversight and accountability for this work. The measurement core
is released under the MIT licence; the author has a commercial interest in
downstream products that may use it, including a graphical application under
development at LISIT Co., Ltd. That product is not required to reproduce or to
use any result in the paper. No patient or customer data were used.

Declare the same on the ICMJE form. **Never describe Institute of One as an
"independent research organization"** — the omission of the oversight
relationship is what got IORN-002 rejected by medRxiv.

**Funding.** None. The APC is an internal LISIT cost, not an external grant.

**Ethics.** See §0. The wording is in the manuscript footnote and section 2.9.

**Use of artificial intelligence.** Declared in the manuscript footnote and the
cover letter. If the form asks separately, the same wording applies.

**Author contributions.** Sole author, all seven roles, with
provision-of-materials marked not applicable because no patients were involved.

---

## 4. Data and code

- Repository: <https://github.com/Institute-of-One/vertebra-xray-core> (MIT)
- Concept DOI (all versions): <https://doi.org/10.5281/zenodo.22893477>
- Version DOI (v0.1.0, the version this paper used):
  <https://doi.org/10.5281/zenodo.22893478>

VerSe is not redistributed. Its three required citations are references 15–17.

---

## 5. Portal behaviour worth knowing

Carried over from other AME and ScholarOne submissions in the programme; none
is confirmed for this particular portal. **[verify on the day]**

- **Enter keywords one at a time.** Pasted together, the newlines are eaten and
  they become a single keyword.
- **Do not use any "auto-fill from PDF" button.** It has restored stale author
  details on other portals.
- **Do not paste the abstract into the title field.** It breaks the article
  page, Crossref and Scholar, and unpicking it costs a round trip.
- **Check long pasted text end to end.** Paste into the ethics and disclosure
  boxes has been silently truncated before. If the form shows a word count,
  compare it with the manuscript.
- **Author information cannot be changed after submission**, per the checklist.

---

## 6. Still open

| | |
|---|---|
| Reporting checklist | No EQUATOR guideline appears to cover a measurement-layer study. The cover letter asks. **[verify]** |
| Ethics route | See §0. Raise with the editorial office if the form will not accept the true position. |
| Open Access statement | Completed on acceptance. |
| ICMJE form | Author to complete; must match §3. |

**APC: 1,900 USD** on acceptance, per the checklist screen. Copyright transfers
to QIMS on acceptance — which is why no ShareAlike-licensed image appears
anywhere in the manuscript.
