# Two more landmarks: identifiability of the three-dimensional Cobb angle measured from biplanar radiographs

*Prepared for Quantitative Imaging in Medicine and Surgery, Original Article.
Word limit 5,000 including abstract, excluding references, tables and figures.
Structured abstract, 450 words, headings Background / Methods / Results /
Conclusions. Text headings: Introduction, Materials and Methods, Results,
Discussion, Acknowledgment, Disclosure, References.*

*Every number below is produced by the scripts named in section 5 and is
reproducible from the repository and the public VerSe release. Placeholders
marked **[TBC]** need the author's input.*

---

## Title page

**Title.** Two more landmarks: identifiability of the three-dimensional Cobb
angle measured from biplanar radiographs

**Running title.** Identifiability of the biplanar three-dimensional Cobb angle

**Author.** Shuji Yamamoto

**Affiliation.** Institute of One, LISIT Co., Ltd., Tokyo 150-0044, Japan

**ORCID.** 0000-0001-9211-1071

**Correspondence.** Shuji Yamamoto, Institute of One, LISIT Co., Ltd., Tokyo 150-0044, Japan. E-mail: yamamoto@lisit.jp

**Counts.** Main text approximately 3,100 words (excluding the title page,
abstract, references, figure legends and tables); abstract 435 words; 7
figures; 3 tables; 23 references.

**Keywords.** scoliosis; Cobb angle; biplanar radiography; vertebral rotation;
measurement uncertainty

---

## Abstract

**Background.** The Cobb angle is defined on a single coronal projection,
although spinal deformity is three-dimensional. Methods now exist that compute
a three-dimensional scoliosis angle from the endplate landmarks of a frontal
and a lateral radiograph, and automatic landmark detectors make that
computation attractive to run at scale. We asked whether the quantity such a
computation returns is identifiable from the landmarks it uses, what the
shortfall costs, and what a detector would have to deliver to remove it.

**Methods.** A frontal radiograph yields the tilt of the line joining an
endplate's left and right edges; a lateral radiograph yields the tilt of the
line joining its anterior and posterior edges. Two measurements constrain
three rotations, so axial rotation is unconstrained. We added the two pedicle
centroids, visible on the frontal view and the basis of the Nash-Moe and
Perdriolle gradings, as a third constraint: axial rotation displaces their
projected midpoint by *b* sin(ψ), where *b* is their posterior offset from the
vertebral body centre. All quantities were validated against phantoms with
closed-form angles and against 28 spines (394 vertebrae) from the VerSe CT
benchmark, which supplies real vertebral orientations including real axial
rotation. Detector error was injected as isotropic Gaussian landmark noise.

**Results.** Using four corners per view alone, the recovered endplate normal
was in error by a median of 5.60° (90th percentile 12.82°, maximum 22.84°)
across the anatomical range of orientations, increasing linearly to 12.8° at
30° of axial rotation. With the pedicles the solution was exact. Under
injected localisation error the residual was a median of 0.43° at 1 mm, the
90th percentile crossing 1° at about 0.85 mm; at the 2–4 mm detectors report,
the median bias still fell four- to six-fold. A normative pedicle table in
error by 2 mm left a median residual of 0.27°, so patient-specific geometry is
unnecessary, and that table is reported. Across 70 curves in 26 VerSe spines
the three-dimensional angle exceeded the coronal angle in every one, medians
19.3° against 9.5°, three definitions agreeing within 1.0°, and the plane in
which the deformity is largest lay a median of 53.6° from coronal. The largest
coronal curve in that cohort is 26.2°, so the direction and mechanism of the
shortfall are established on real anatomy but its magnitude in scoliosis is
not.

**Conclusions.** The three-dimensional Cobb angle computed from biplanar
endplate landmarks is not identifiable as usually formulated. Two additional
landmarks, which a detector already processing the frontal view can produce,
make it identifiable; placing them to about a millimetre makes the residual
negligible, and placing them no better than the corners already are removes
most of the bias.

---

## 1. Introduction

The Cobb angle is the measurement on which scoliosis is diagnosed, monitored, braced and operated,
and it is defined on one coronal projection (1). Spinal deformity is
not confined to that plane.

That the coronal projection under-reads the deformity has been known since Péloux,
Fauchet, Faucon and Stagnara proposed the *plan d'élection*, an oblique view
taken perpendicular to the plane through the end vertebrae and the apex
(2), and it is formalised in the Scoliosis Research Society's
three-dimensional terminology (3). Recent work measures the plane of
maximum curvature from computed tomography (4) and estimates it
computationally (5). **This paper claims none of that as new.**

A recent method computes a three-dimensional scoliosis angle from four endplate angles
measured on a standing frontal and lateral pair, validated in 41 patients
against computed tomography, and reports a mean Cobb angle of 54° against a
mean three-dimensional angle of 60° (6,7). It is an appealing
construction because automatic landmark detectors produce exactly those inputs
(8). The construction infers a quantity that depends on three
rotations from two measurements. We ask whether it is identifiable, and if
not, what closes it.

Full biplanar reconstruction does not have this problem. EOS and the deformable-model
pipelines fit a vertebra shape model to both silhouettes and recover axial
rotation to about 1.4–1.9° RMS (9,10). The gap is specific
to measurement from a small set of endplate landmarks, which is the route a
fully automatic three-dimensional Cobb angle would take.

This paper contributes the following.

1. The biplanar endplate-landmark measurement is shown to be unidentifiable,
   and the resulting bias is quantified on phantoms and on real vertebral
   orientations.
2. Two pedicle landmarks are shown to close the system exactly, and the
   localisation accuracy required is stated.
3. A normative table of pedicle geometry from 394 vertebrae is reported,
   together with the demonstration that patient-specific geometry is not
   needed.
4. Secondary: the exact coupling between the measured lateral tilt and the
   true sagittal tilt; the invariance of the coronal Cobb angle to cone-beam
   divergence at zero axial rotation; and the agreement of three definitions
   of the three-dimensional angle.

---

## 2. Materials and Methods

### 2.1 Coordinate conventions and vertebral orientation

Right-handed patient frame: *X* towards the patient's left, *Y* anterior, *Z*
cranial. Vertebral orientation is *R* = *R_y*(θ)·*R_x*(φ)·*R_z*(ψ), being
coronal tilt, sagittal tilt and axial rotation.

The order is not arbitrary. Because ψ acts first, about the vertebra's own
cranio-caudal axis, *R*·**e**_z = *R_y*·*R_x*·**e**_z: **the endplate normal,
and therefore every three-dimensional angle reported here, is independent of
axial rotation.** The endplate's lateral and antero-posterior directions,
which are what a radiograph shows, are not. That asymmetry is the source of
the identifiability problem.

### 2.2 The biplanar measurement and what it constrains

A frontal radiograph gives α, the tilt of the line joining an endplate's left
and right edges. A lateral radiograph gives β, the tilt of the line joining
its anterior and posterior edges. For a vertebra at (θ, φ, ψ),

  tan β = −cos θ · tan φ    (ψ = 0)

so the measured lateral tilt is not the sagittal tilt; the discrepancy grows
with the coronal deformity. Two measurements, three unknowns: with ψ supplied
from outside, (θ, φ) follows by a two-parameter least-squares solve that was
exact over a grid of 2,601 orientations spanning ±40° in each angle (maximum
endplate-normal error < 10⁻⁶ °). With ψ unknown, it is not determined.

### 2.3 Pedicles as the third constraint

**Figure 1** shows what this amounts to on a film: the four corners a detector
already places, and the two pedicle points it is being asked to add.

The pedicles lie lateral to the midline and, decisively, posterior to the
vertebral body. Writing their centroids in the vertebra's frame as (±*a*,
−*b*, 0), the projected midpoint of the pair leaves the body centre by
*b* sin(ψ) while their separation narrows as 2*a* cos(ψ). The midpoint is the
better-conditioned signal: its derivative at ψ = 0 is *b*, whereas the
separation's is zero. Both were used, the separation down-weighted by 0.3.

This is the geometry Nash and Moe graded visually (11) and Perdriolle
measured with a torsionmeter (12). Its use here is as the missing
constraint in a joint solve for all three rotations.

### 2.4 Which plane the three-dimensional angle is measured in

"The largest Cobb angle" requires the family of admissible planes to be
stated. **The maximum over all planes is degenerate**: for any two
non-parallel endplates there is a viewing direction making their traces
perpendicular, so the unrestricted maximum is 90° for every curve and carries
no information. Restricting to planes containing the cranio-caudal axis makes
the quantity well defined, and the restriction is physical: a radiograph of a
standing patient rotating about their own long axis realises exactly that
family.

Three quantities were computed and compared: the angle in the coronal plane;
the maximum over planes containing the cranio-caudal axis; and the dihedral
angle between the two endplates. The classical *plan d'élection* angle, in the
plane through the two end vertebrae's centroids and the apex, was computed as
a fourth.

### 2.5 Phantoms

Spines with prescribed Cobb angles, kyphosis, lordosis and axial rotation in
closed form, so that measured values can be compared with the values put in
rather than with another measurement. End-vertebra tilts are chained rather
than superposed, and tilt ramps are linear so curvature is piecewise constant, after the piecewise-circular description of normal sagittal alignment (13).

### 2.6 A rendered phantom radiograph

The phantom of section 2.5 is given vertebral bodies with cortical endplates,
pedicles, laminae, spinous and transverse processes, ribs, a shoulder girdle
and a pelvis, and placed in a torso containing aerated lungs, the mediastinum
and the abdomen; the volume is line-integrated along the beam to give the
films in Figure 1.

Anthropomorphic digital phantoms and digitally reconstructed radiographs are
long established (14), and nothing here is new as imaging physics:
scatter, beam hardening and trabecular texture are not modelled, and the
attenuation values were chosen to order correctly rather than to match a
measured spectrum. **No result in this paper is measured from it.** It is
used because the renderer and the measurement code share one geometry object,
so landmarks projected by the forward model of section 2.2 fall on the
anatomy that was drawn, and because it carries no patient data and no
third-party licence.

### 2.7 Real vertebral orientations from CT

VerSe (15,16,17) supplies hand-corrected
vertebral labels and segmentation masks. Extracting orientation from a mask is
not trivial and required three decisions, each invisible on a box phantom and
decisive on real vertebrae:

1. **The body is separated from the posterior elements by thickness, not
   direction.** A morphological opening by a 6 mm ball removes pedicles,
   laminae and processes and restores the body. Cutting along the
   antero-posterior axis at the pedicle narrowing instead leaves the
   transverse processes attached, giving body widths of 71–90 mm against a
   true 30–50.
2. **The in-plane orientation cannot come from the body's principal axes.** A
   thoracic body is as deep as it is wide (width/depth 0.8–1.3 in this
   cohort), so those moments are degenerate and the recovered axial rotation
   flips by 90° between neighbours: a median of 27° of apparent rotation on
   supine CT. The posterior elements resolve it.
3. **Nor can the endplate normal.** The two smaller moments of a lumbar body
   differ by under 20%. Planes were fitted to the opened body's superior and
   inferior surfaces instead.

Measured body dimensions agreed with published morphometry (18): heights within
1–2 mm at every level, widths within 2 mm below T4. An automated check on
dimensions and segmental angles flagged 2 of 30 scans, both at the cranial
edge of the field of view, and those two were the only scans where the
reconstruction was not exact.

### 2.8 Scope

Landmark detection from radiographic pixels was **not** evaluated. Landmarks
were projected from the CT-derived model and detector error injected as a
stated perturbation. This isolates the measurement layer, which is the subject
of the paper.

### 2.9 Ethics

This study used the publicly available VerSe collection of de-identified spine
computed tomography, together with synthetic phantoms. No participants were
recruited by the author and no identifiable data were accessed, so no further
review board approval was sought. The approval for the underlying imaging is
held by the source collection: its retrospective evaluation was approved by
the local institutional review board with written informed consent waived
under Proposal 27/19 S-SR, and the scans were anonymised and defaced before
release (16). The work conformed to the Declaration of Helsinki as revised in
2013. No image from the collection is reproduced here; the figures other than
Figure 3 are rendered from phantoms, and Figure 3 plots measurements rather
than images.

---

## 3. Results

### 3.1 Cohort

Of 80 scans in the VerSe 2019 training release, 48 were excluded because the
field of view did not span the thoracolumbar junction with sufficient
vertebrae on both sides and 2 failed to process, leaving 30; the automated
quality check passed 28. Those 28 hold 394 vertebrae with measurable
pedicles, and 26 of them carry at least one structural curve, 70 in all. Axial rotation
present: median 3.8°, 90th percentile 10.5°, maximum 26.4°.

### 3.2 Without the pedicles, the measurement is biased

Over orientations drawn uniformly across the anatomical range (coronal ±35°,
sagittal ±30°, axial ±30°), the endplate normal recovered from four corners
per view was in error by a median of 5.60°, 90th percentile 12.82°, maximum
22.84°, growing linearly with the rotation present to 12.8° at 30°.

**Table 1** and **Figure 2a**.

### 3.3 Two pedicle landmarks make it identifiable

With exact pedicles the solution was exact (maximum error 2.5 × 10⁻¹² ° over
500 random orientations). Under injected localisation error:

| pedicle localisation error | median | 90th pct | maximum |
|---|---|---|---|
| none (four corners only) | 5.60 | 12.82 | 22.84 |
| 3 mm | 1.25 | 3.37 | 7.23 |
| 2 mm | 0.82 | 2.56 | 5.57 |
| 1 mm | 0.43 | 1.18 | 1.97 |
| 0.5 mm | 0.21 | 0.57 | 1.50 |
| 0.25 mm | 0.10 | 0.27 | 0.68 |
| exact | 0.00 | 0.00 | 0.00 |

Degrees of endplate-normal error. **Figure 2b.**

The 90th-percentile residual is 0.57° at 0.5 mm and 1.18° at 1 mm, so it
crosses 1° at about 0.85 mm. Published whole-spine
landmark detectors report median localisation errors of 1.5–2.4 mm cervical,
2.1–3.0 mm lumbosacral and 2.4–4.3 mm thoracic (19,20,21), so **a
millimetre is a target rather than a current capability**. At 3 mm — the thoracic spine as
reported today, and where the deformity usually is — the median bias still
falls from 5.60° to 1.25°.

### 3.4 A normative pedicle table is sufficient

Taking the half-separation and posterior offset from a table rather than
measuring them cost a median of 0.13° at 1 mm of table error and 0.27° at
2 mm. The interquartile spread observed across the 394 vertebrae was 12.3–15.4 mm
in separation and 25.0–29.4 mm in posterior offset, inside the regime where it
does not matter. **Figure 2c**, **Table 2**.

| level | T1 | T4 | T7 | T10 | T12 | L1 | L3 | L5 |
|---|---|---|---|---|---|---|---|---|
| half-separation (mm) | 16.7 | 11.9 | 13.3 | 14.1 | 11.8 | 13.1 | 16.2 | 19.6 |
| posterior offset (mm) | 18.6 | 24.0 | 27.2 | 27.5 | 27.6 | 29.1 | 29.7 | 26.8 |

The posterior offset carries the rotation signal and runs from 19 mm at T1 to
about 29 mm in the lumbar spine, so one degree of rotation displaces the
projected midpoint by 0.33–0.51 mm. The table is regenerated by
`tools/normative_pedicles.py`, which reports the spread behind each median. That, not the endplate geometry, sets the
millimetre requirement.

### 3.5 Real orientations

Pooling the 394 VerSe vertebrae with measurable pedicles, the median endplate
normal error was 0.55° without the pedicles and 0.16° with them at 1 mm of
localisation error and the normative table. (Section 3.7 aggregates the same
quantity differently, taking the worst vertebra in each spine and then the
median across spines, which is the larger number a reader should hold a
pipeline to.) Rotations in this cohort are
modest (median 3.8°), so the phantom carries the scoliotic range.

### 3.6 The coronal projection always reports less, and the measurement plane is nowhere near coronal

Across all 70 curves, from 26 spines, the three-dimensional angle exceeded the
coronal angle; in none was the coronal angle the larger.

| | median | 90th pct | maximum |
|---|---|---|---|
| coronal | 9.53 | 16.18 | 26.19 |
| maximum over vertical planes | 19.30 | 33.62 | 45.64 |
| *plan d'élection* (centroid plane) | 19.75 | 33.65 | 45.55 |
| dihedral angle between endplates | 18.78 | 33.55 | 45.53 |

Degrees, per curve. The three three-dimensional definitions agree to within
1.0° at the median, so the conclusion does not depend on which is chosen. The
centroid plane is tilted a median of 5.2° (maximum 28.9°) from vertical, which
places it outside the family a rotating radiograph can realise and explains
the seven curves in which its angle exceeded the vertical-plane maximum.

**Figure 3.**

**The ratio is not the finding.** The plane of maximum curvature lies a median
of 53.6° from coronal (90th percentile 76.1°), and over the same level pairs
the sagittal angle has a median of 15.60° against the coronal 9.53°. In a
spine that is nearly straight coronally and normally kyphotic sagittally, the
maximising plane rotates towards the sagittal and reports the kyphosis, so a
three-dimensional angle around twice the coronal one is what the geometry
requires and is not evidence about scoliosis. Stratified by the size of the
coronal curve, the ratio falls rather than holds:

| coronal band | n | coronal | three-dimensional | ratio | plane from coronal |
|---|---|---|---|---|---|
| 0–10° | 38 | 6.62 | 12.49 | 1.89 | 57.5° |
| 10–20° | 29 | 12.06 | 21.66 | 1.80 | 51.5° |
| 20–30° | 3 | 24.12 | 29.66 | 1.23 | 36.6° |
| ≥ 30° | 0 | — | — | — | — |

Medians, degrees. The three curves at or above 20° give ratios of 1.13, 1.25
and 1.28 individually.

**Cohort caveat.** VerSe is a general and fracture CT collection, not a
scoliosis cohort. Of 80 scans, 30 span the thoracolumbar junction contiguously,
28 pass the dimension check of section 2.7, and 26 carry at least one
structural curve; the largest coronal curve among them is 26.2° and none
reaches 30°. What this establishes on real anatomy is the direction of the
shortfall, and its mechanism — the plane in which the deformity is largest is
not the plane the measurement is made in, by a wide margin. The magnitude of
the shortfall in scoliosis is not established here and cannot be from this
dataset.

### 3.7 Robustness to landmark error

Medians across the 28 spines of the **worst vertebra in each spine**, with
isotropic Gaussian error on every corner before any measurement. Per-vertebra
medians over the pooled 394 vertebrae are several times smaller (section 3.5);
the per-spine worst case is reported here because a measurement is only as
good as the level a reader happens to need.

| corner error | 0 mm | 0.5 mm | 1.0 mm | 2.0 mm |
|---|---|---|---|---|
| endplate normal error, ψ supplied | 0.00 | 2.12 | 4.22 | 8.34 |
| endplate normal error, ψ assumed zero | 3.26 | 3.62 | 5.25 | 8.73 |
| three-dimensional minus coronal | 8.52 | 6.20 | 5.65 | 6.90 |

Degrees. The geometric basis of measurement uncertainty in radiographic spinal angles has been set out recently (22); here the shortfall does not wash out. At about 1 mm the zero-rotation
assumption and the detector contribute equally; below that the assumption
dominates, so for a well-localised study it is the assumption, not the
detector, that limits the result.

### 3.8 Secondary findings

**Cone-beam divergence.** With no axial rotation the chord a reader marks on a
frontal film lies in a plane of constant depth, so both ends are magnified
identically and the coronal tilt is exactly invariant (difference < 10⁻¹³ °).
At 20° of axial rotation the same comparison gives 4.5° (**Figure 5**). Treating a long film
as a parallel projection is an identity, not an approximation, until the spine
rotates.

**Lateral films under-read kyphosis in scoliosis.** Coronal tilt swings the
body's antero-posterior axis out of the sagittal plane, so the
anterior-to-posterior chord is not the sagittal trace of the endplate. On
phantoms the T1–T12 sagittal angle was under-read by 0.0°, 1.7°, 3.5° and 8.0°
at main thoracic curves of 0°, 25°, 50° and 75°. **Figure 6.**

**Uncertainty decomposition.** Under corner error the choice of end vertebrae
contributed none of the variance below 1.5 mm, 6% at 2 mm, a third at 3 mm and
half at 4 mm. This is a lower bound on the discrete term: independent corner
jitter is a poor model of how two readers come to disagree about which
vertebra ends a curve (23). **Figure 7.**

---

## 4. Discussion

**What the result means.** The three-dimensional Cobb angle is attractive
precisely because it can be computed from landmarks an automatic detector
already produces. That computation, as formulated, asks the data for
something the data do not contain. The remedy is cheap: two more points, on a
view already being processed, whose anatomical basis has been used clinically
since 1969.

**Why the specification matters more than the bias.** Reporting that a method
is biased is of limited use unless the fix is actionable. The contribution
here is the accuracy target — under a millimetre for a 90th-percentile
residual below a degree — and the finding that pedicle geometry need not be measured per
patient, which removes what would otherwise be an obstacle to deployment.

**Detectors are not there yet, and it still helps.** At the 2.4–4.3 mm
reported for thoracic landmarks, the pedicles reduce the median bias by a
factor of four. The thoracic spine is simultaneously where detectors are
weakest and where the deformity usually is, so that is the row that matters.

**On the choice of measurement plane.** A three-dimensional Cobb angle needs
its plane family stated, because the unrestricted maximum is 90° for every
spine. The vertical-plane family is the physically realisable one. That three
definitions agree at the median is reassuring for anyone comparing across the
literature, and the cases where they diverge are those in which the *plan
d'élection* tilts furthest out of vertical.

**Limitations.**

1. Landmark detection from pixels is outside the scope by design, and no
   result here is measured from a radiograph, real or rendered: the
   measurement layer is characterised given landmarks, with the required
   landmark accuracy stated so that a detector can be held to it. An
   end-to-end evaluation on standing films with detected landmarks remains
   to be done.
2. VerSe is supine axial CT, not standing radiography, and is a general and
   fracture collection rather than a scoliosis cohort. Real axial rotations in
   it are modest.
3. The projection model treats the vertebral body as a box for corner
   generation.
4. The quality check on segmentation is a heuristic, though it flagged exactly
   the scans where the reconstruction failed.
A training-free method for assigning vertebral levels from the sagittal
curvature reversal was developed alongside this work and is **deliberately
held for a separate report**. It is correct in every phantom case tested with
standing geometry, but correct in only 4 of 30 supine VerSe scans and within
one level in 21 of 30, because supine positioning flattens lumbar lordosis and
moves the sagittal inflection from T12 to L1. That posture dependence is worth
reporting on its own; included here it would dilute the argument and could not
be evaluated fairly, since no public dataset provides standing biplanar series
with hand-checked levels.

**What would settle the open questions.** Standing biplanar series with
hand-checked vertebral levels and measured axial rotation would allow the
magnitude of the shortfall to be measured in scoliosis, and would permit a
fair evaluation of the labelling component.

---

## 5. Reproducibility

All results are produced by `tools/pedicle_requirement.py`,
`tools/validate_verse.py`, `tools/coronal_shortfall.py`,
`tools/normative_pedicles.py`,
`tools/make_figures.py` and `tools/make_verse_figure.py` in the accompanying
repository
https://github.com/Institute-of-One/vertebra-xray-core,
archived at https://doi.org/10.5281/zenodo.22893477, over the public VerSe release. Every figure
except Figure 3 is generated from phantoms; Figure 3 plots measurements
computed from the public collection rather than any image from it.

Each table states the population it is aggregated over, because they are not
the same: section 3.6 pools structural curves, section 3.5 pools vertebrae,
and section 3.7 takes the worst vertebra in each spine. No number in this
paper was transcribed between aggregations.

---

## Author Contributions

Shuji Yamamoto is the sole author and performed all of the following:

(I) Conception and design;
(II) Administrative support;
(III) Provision of study materials or patients — not applicable, no patients
were involved;
(IV) Collection and assembly of data;
(V) Data analysis and interpretation;
(VI) Manuscript writing;
(VII) Final approval of manuscript.

---

## Acknowledgment

*Funding:* None. The Article Processing Charge, if the article is accepted,
is met by LISIT Co., Ltd. as an internal cost; no external grant supported
this work.

None.

---

## Footnote

*Reporting Checklist:* **[TBC: none of the EQUATOR checklists covers a
measurement-theory study of this kind. Confirm with the editorial office
whether one is expected; if not, state that no reporting guideline applies.]**

*Data Sharing Statement:* All data underlying this article are public. The
vertebral orientations are measured from the VerSe 2019 training release,
which its authors distribute under a Creative Commons Attribution-ShareAlike
licence and which requires the three citations given here (15-17); no image
or mask from it is redistributed, and none is reproduced in any figure. Every
other quantity is generated by the phantoms in the accompanying repository.
The software is available at https://github.com/Institute-of-One/vertebra-xray-core
under the MIT licence, and the version used in this study, v0.1.0, is
archived at https://doi.org/10.5281/zenodo.22893478. Five
scripts in that repository produce every number and every figure reported
here.

*Conflicts of Interest:* The author is the representative of LISIT Co., Ltd.
(Tokyo, Japan) and Chief Executive Officer of TexelCraft OÜ. Institute of One
is the open-research initiative of LISIT Co., Ltd., which provides
institutional oversight and accountability for this work. The measurement core
described here is released under the MIT licence; the author has a commercial
interest in downstream products that may use it, including a graphical
application under development at LISIT Co., Ltd. That product is not required
to reproduce or to use any result in this paper, and no patient or customer
data were used at any point. Completed ICMJE uniform disclosure forms
accompany this submission.

*Ethical Statement:* The authors are accountable for all aspects of the work in ensuring that questions related to the accuracy or integrity of any part of the work are appropriately investigated and resolved. This study analysed computational phantoms and
the publicly available VerSe collection of de-identified spine computed
tomography. No participants were recruited by the author, no identifiable data
were accessed, and no new human or animal data were generated, so no further
review board approval was sought and no consent could be obtained by the
author.

The approval for the underlying imaging exists and is held by the source
collection rather than by this study: VerSe's retrospective evaluation of
imaging data was approved by its local institutional review board and written
informed consent was waived under Proposal 27/19 S-SR, and the scans were
anonymised by conversion to NIfTI and defaced before release (16). The work
conformed to the Declaration of Helsinki as revised in 2013. No image from the
collection is reproduced in this article.

*Use of Artificial Intelligence:* A generative AI assistant (Claude,
Anthropic) supported software development and drafted portions of this
manuscript. All results were produced by executable code in the released
repository and were re-run and verified by the author, who takes full
responsibility for this work. The AI system is not an author and does not meet
authorship criteria.

*Open Access Statement:* **[TBC: completed on acceptance.]**

---

## References

1. Cobb JR. Outline for the study of scoliosis. Instr Course Lect 1948;5:261-75.
2. Péloux J, Fauchet R, Faucon B, Stagnara P. Le plan d'élection pour l'examen radiologique des cypho-scolioses. Rev Chir Orthop Reparatrice Appar Mot 1965;51:517-24.
3. Stokes IAF; Scoliosis Research Society Working Group on 3-D Terminology of Spinal Deformity. Three-dimensional terminology of spinal deformity. Spine (Phila Pa 1976) 1994;19:236-48.
4. Wu HD, Chu WC, He CQ, Wong MS. Assessment of the plane of maximum curvature for patients with adolescent idiopathic scoliosis via computed tomography. Prosthet Orthot Int 2020;44:298-304.
5. Wu HD, He C, Chu WC, Wong MS. Estimation of plane of maximum curvature for the patients with adolescent idiopathic scoliosis via a purpose-design computational method. Eur Spine J 2021;30:668-75.
6. Główka P, Politarczyk W, Janusz P, Woźniak Ł, Kotwicki T. The method for measurement of the three-dimensional scoliosis angle from standard radiographs. BMC Musculoskelet Disord 2020;21:475.
7. Lechner R, Putzer D, Dammerer D, Liebensteiner M, Bach C, Thaler M. Comparison of two- and three-dimensional measurement of the Cobb angle in scoliosis. Int Orthop 2017;41:957-62.
8. Wang L, Xie C, Lin Y, Zhou HY, Chen K, Cheng D, et al. Evaluation and comparison of accurate automated spinal curvature estimation algorithms with spinal anterior-posterior X-ray images: the AASCE2019 challenge. Med Image Anal 2021;72:102115.
9. Dumas R, Le Bras A, Champain N, Savidan M, Mitton D, Kalifa G, Steib JP, de Guise JA, Skalli W. Validation of the relative 3D orientation of vertebrae reconstructed by bi-planar radiography. Med Eng Phys 2004;26:415-22.
10. Glaser DA, Doan J, Newton PO. Comparison of 3-dimensional spinal reconstruction accuracy: biplanar radiographs with EOS versus computed tomography. Spine (Phila Pa 1976) 2012;37:1391-7.
11. Nash CL Jr, Moe JH. A study of vertebral rotation. J Bone Joint Surg Am 1969;51:223-9.
12. Perdriolle R, Vidal J. Morphology of scoliosis: three-dimensional evolution. Orthopedics 1987;10:909-15.
13. Roussouly P, Gollogly S, Berthonnaud E, Dimnet J. Classification of the normal variation in the sagittal alignment of the human lumbar spine and pelvis in the standing position. Spine (Phila Pa 1976) 2005;30:346-53.
14. Segars WP, Sturgeon G, Mendonca S, Grimes J, Tsui BM. 4D XCAT phantom for multimodality imaging research. Med Phys 2010;37:4902-15.
15. Löffler MT, Sekuboyina A, Jacob A, Grau AL, Scharr A, El Husseini M, Kallweit M, Zimmer C, Baum T, Kirschke JS. A vertebral segmentation dataset with fracture grading. Radiol Artif Intell 2020;2:e190138.
16. Liebl H, Schinz D, Sekuboyina A, Malagutti L, Löffler MT, Bayat A, El Husseini M, Tetteh G, Grau K, Niederreiter E, Baum T, Wiestler B, Menze B, Braren R, Zimmer C, Kirschke JS. A computed tomography vertebral segmentation dataset with anatomical variations and multi-vendor scanner data. Sci Data 2021;8:284.
17. Sekuboyina A, Husseini ME, Bayat A, Löffler M, Liebl H, Li H, et al. VerSe: a vertebrae labelling and segmentation benchmark for multi-detector CT images. Med Image Anal 2021;73:102166.
18. Busscher I, Ploegmakers JJ, Verkerke GJ, Veldhuizen AG. Comparative anatomical dimensions of the complete human and porcine spine. Eur Spine J 2010;19:1104-14.
19. Noh SH, Lee G, Bae HJ, Han JY, Son SJ, Kim D, Park JY, Choi SK, Cho PG, Kim SH, Yuh WT, Lee SH, Park B, Kim KR, Kim KT, Ha Y. Deep learning method for precise landmark identification and structural assessment of whole-spine radiographs. Bioengineering (Basel) 2024;11:481.
20. Yeh YC, Weng CH, Huang YJ, Fu CJ, Tsai TT, Yeh CY. Deep learning approach for automatic landmark detection and alignment analysis in whole-spine lateral radiographs. Sci Rep 2021;11:7618.
21. Cina A, Bassani T, Panico M, Luca A, Masharawi Y, Brayda-Bruno M, Galbusera F. 2-step deep learning model for landmarks localization in spine radiographs. Sci Rep 2021;11:9482.
22. Chayer M, Phan P, Arnoux PJ, Aubin CÉ. Geometric foundations of measurement uncertainty and clinical relevance in radiographic spinal angle assessment. Spine Deform 2026. doi: 10.1007/s43390-026-01500-0.
23. Carman DL, Browne RH, Birch JG. Measurement of scoliosis and kyphosis radiographs. Intraobserver and interobserver variation. J Bone Joint Surg Am 1990;72:328-33.

---

## Figures

1. **`fig1_landmarks.png` — What a detector is being asked to place.** (a) a
   simulated frontal radiograph of the phantom; (b) the same with the
   landmarks overlaid, four corners per body in blue, the two pedicles in gold
   and the Cobb construction on the main curve; (c) the same spine in three
   dimensions with its measurement plane. All three panels are rendered from
   one phantom, so the pedicle shadows in the film and the pedicle landmarks
   drawn on it are the same points.
2. **`fig2_pedicles.png` — Two pedicle landmarks close the system.** (a)
   endplate-normal error against the axial rotation present, four corners
   alone versus plus pedicles at three localisation accuracies; (b) residual
   against pedicle localisation error, median and 90th percentile, against the
   no-pedicle baseline; (c) residual against error in the assumed pedicle
   geometry, with the observed anatomical spread shaded.
3. **`fig3_verse.png` — The coronal projection on 28 real spines.** (a)
   coronal against three-dimensional angle for all 70 curves, coloured by the
   orientation of the measurement plane, with the identity line; (b)
   distribution of the shortfall; (c) endplate-normal error against the axial
   rotation actually present, with the phantom prediction as an upper bound.
4. **`fig4_planes.png` — The angle depends on its plane.** (a) Cobb angle as a
   continuous function of the measurement plane's orientation for three
   curves, with the coronal reading marked as a circle and the maximum as a
   star; (b) the centreline seen from above with each curve's plane.
5. **`fig5_axial_rotation.png` — Two ways unmeasured rotation corrupts the
   measurement.** (a) the orientation solve is biased; (b) cone-beam
   divergence, exactly harmless at zero rotation, is let in.
6. **`fig6_kyphosis.png` — A lateral film under-reads kyphosis**, against the
   size of the coronal curve.
7. **`fig7_uncertainty.png` — Where the spread in a Cobb angle comes from.**
   (a) interval width against corner localisation error; (b) decomposition
   into the endplate-line and end-vertebra terms, with end-vertebra stability
   on the right axis.

Only Figure 3 derives from patient data, and it plots measurements rather than
images. Every other figure is rendered from the phantom of sections 2.5 and
2.6 and carries no patient data and no third-party licence, so a reader can
regenerate them from the repository alone.

---

## Tables

1. Endplate-normal error with and without pedicles, over the anatomical range.
2. Normative pedicle geometry per level, from 394 vertebrae.
3. The four angle definitions compared on 70 real curves.

