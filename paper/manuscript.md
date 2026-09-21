# Two more landmarks: making the three-dimensional Cobb angle identifiable from biplanar radiographs

**Draft, IORN-014.** Every number is produced by `tools/pedicle_requirement.py`,
`tools/validate_verse.py` and `tools/make_figures.py`, and is reproducible from
the repository and the public VerSe release.

---

## Abstract

**Purpose.** A three-dimensional scoliosis angle can be computed from the
endplate landmarks of a frontal and a lateral radiograph, and a published
method does so. We show that this measurement is **not identifiable**: four
corner landmarks per vertebral body per view give two constraints for three
unknown rotations, so the result carries whatever bias the unmeasured axial
rotation imposes. We then show that two further landmarks make it
identifiable, and state how accurately a detector would have to place them.

**Methods.** The frontal view gives the tilt of the line joining an endplate's
left and right edges, the lateral view the tilt of the line joining its
anterior and posterior edges. Adding the two pedicle centroids, visible on the
frontal view a landmark detector is already processing and the basis of the
Nash-Moe and Perdriolle rotation gradings, supplies the missing constraint:
axial rotation slides the projected pedicle midpoint by `b sin(psi)`, where
`b` is their posterior offset from the body centre. Validated against
closed-form phantoms and against 28 spines from the VerSe CT benchmark, which
supplies real vertebral orientations including real axial rotation.

**Results.** Without the pedicles the recovered endplate normal is in error by
a median of 5.50 deg (p90 12.99, max 22.84) over the anatomical range of
orientations, growing linearly to 12.8 deg at 30 deg of axial rotation. With
them the solution is exact. Under realistic detector error the residual is a
median of 0.43 deg at 1 mm of pedicle localisation error, and the 90th
percentile falls below 1 deg at about 0.9 mm. Patient-specific pedicle
geometry is unnecessary: taking it from a normative table wrong by 2 mm leaves
a median residual of 0.25 deg. That table, pedicle half-separation and
posterior offset per level over 400 vertebrae, is reported.

**Conclusion.** The three-dimensional angle from a biplanar landmark pair is
unidentifiable as usually computed, and the fix is two landmarks a detector
can already produce, placed to about a millimetre. Secondary findings: the
exact coupling between the measured lateral tilt and the true sagittal tilt,
the fact that cone-beam divergence cannot perturb a coronal Cobb angle at zero
axial rotation, and a per-patient interval on the Cobb angle.

**Stated limitations.** Landmark detection from radiographic pixels is outside
the scope of this work by design: landmarks are projected from the CT-derived
model, and detector error is injected as a stated perturbation. The VerSe
cohort is a general and fracture CT collection, not a scoliosis one, with a
median coronal Cobb of 9.5 deg and one curve of 70 above 25 deg, so real
rotations in it are modest (median 3.8 deg) and the phantom carries the
scoliotic range.

---

## 1. Introduction

*Points to make, in order:*

1. The Cobb angle is what scoliosis is diagnosed, monitored, braced and
   operated on, and it is defined on one coronal projection.
2. That scoliosis is three-dimensional, and that the coronal projection
   under-reads it, is long established: Stagnara's plane-of-maximum-curvature
   projection, the SRS three-dimensional terminology (Stokes 1994), and recent
   measurements of the plane of maximum curvature from CT and from
   radiographs. **This paper does not claim that as a finding.**
3. What is new is a question about *identifiability*. A recent method computes
   a three-dimensional scoliosis angle from four measured endplate angles on a
   standing frontal and lateral pair (n = 41; mean Cobb 54 deg against a mean
   three-dimensional angle of 60 deg). That construction takes two
   measurements and infers a quantity depending on three rotations. It is
   silent on axial rotation, and silent on whether the measured lateral angle
   is the sagittal tilt. We show it is neither, quantify the consequence, and
   close the gap.
4. The full biplanar reconstruction literature, EOS and the deformable-model
   pipelines, does not have this problem: it fits a vertebra shape model to
   both silhouettes and recovers axial rotation to about 1.4 to 1.9 deg RMS.
   **The problem is specific to the landmark-only route**, which is exactly
   what automatic Cobb detectors produce, and therefore exactly the route a
   fully automatic three-dimensional measurement would take.
5. The fix is not a new imaging modality but two more landmark points. The
   contribution is the specification: how accurately they must be placed, and
   whether their geometry has to be measured per patient.

---

## 2. Methods

### 2.1 Coordinate conventions

Two frames, converted once at the library boundary, stated because sign errors
here are the commonest defect in Cobb-angle code.

* **image**: `x` right, `y` down, as landmarks arrive.
* **math**: `x` right, `y` up. Every computation is in this frame.
* **patient**: `X` towards the patient's left, `Y` anterior, `Z` cranial.

Vertebral orientation is `R = R_y(θ) R_x(φ) R_z(ψ)`: coronal tilt, sagittal
tilt, axial rotation. The order is not arbitrary. Because `ψ` is applied first,
about the vertebra's own axis, `R e_z = R_y R_x e_z`: **the endplate normal,
and therefore every three-dimensional angle in this paper, is independent of
axial rotation.** The endplate *lateral* and *antero-posterior* directions,
which are what a radiograph shows, are not. That asymmetry is the whole
difficulty of the inverse problem.

### 2.2 Coronal Cobb angles

The signed endplate-tilt profile is decomposed into monotone runs; each run is
one structural curve, its turning points are the end vertebrae, and the angle
is taken between the superior endplate of the upper and the inferior endplate
of the lower (the SRS definition). Curves are named PT / MT / TL / L from the
anatomical level of their apex.

The AASCE benchmark convention — the maximum angle over all pairs of
mid-body axes — is also implemented, because published SMAPE figures are
computed against it. *Reporting both, and their difference, is deliberate: a
measurable part of the spread between published automatic systems is
definitional rather than algorithmic.*

### 2.3 The biplanar inverse problem

A frontal radiograph gives `α`, the tilt of the line joining the left and
right edges of an endplate; a lateral one gives `β`, the tilt of the line
joining its anterior and posterior edges. Two measurements, three unknowns.

`tan β = −cos θ · tan φ` — the measured sagittal tilt is coupled to the
coronal one, so reading the lateral tilt off as the sagittal tilt is wrong by
a factor that grows with the deformity.

Given `ψ` from any external source, `(θ, φ)` follows by a two-parameter
solve. Over a grid of 2601 orientations spanning ±40° in each angle the
solution is exact (maximum endplate-normal error < 10⁻⁶ °).

### 2.3b Pedicles, and why they close the system

Four corners per view give two constraints. The unknowns are three rotations,
so the system is short by one and axial rotation is what is missing.

The pedicles supply it. They sit lateral to the midline and, decisively,
*behind* the vertebral body, so rotation about the body's own axis swings them
across its projected width. Writing their positions in the vertebra's frame as
`(+-a, -b, 0)`, the projected midpoint of the pair leaves the body centre by
`b sin(psi)` while their separation narrows as `2a cos(psi)`. The midpoint is
the better-conditioned of the two signals: its derivative at zero rotation is
`b`, while the separation's is zero, so near the neutral position the
separation carries almost no information. Both are used, the separation
down-weighted.

This is the same geometry Nash and Moe graded by eye and Perdriolle measured
with a torsionmeter. The contribution here is not the observation but its use
as the missing constraint in a joint solve for all three rotations, and the
resulting specification on landmark accuracy.

### 2.4 The plane of maximum curvature

Rotating the measurement plane about the cranio-caudal axis changes the Cobb
angle continuously. The maximum over all such planes is the SRS
three-dimensional Cobb angle. Every curve is reported in the coronal plane, in
the sagittal plane, and in its plane of maximum curvature, with the
orientation of that plane.

### 2.5 Uncertainty

Landmarks are resampled under a stated isotropic Gaussian corner error and the
measurement repeated, separating two mechanisms: the endplate lines moving
with the end vertebrae held fixed, and the whole analysis rerun so the end
vertebrae are free to move.

### 2.6 Phantoms

Spines whose Cobb angles, kyphosis, lordosis and axial rotation are prescribed
in closed form. Validating only against human annotation cannot separate an
algorithm's error from the reference's; a phantom has no such floor.
*Construction notes worth stating: end-vertebra tilts are chained, not
superposed — superposing two curves adds their tilts at the shared end
vertebra and inflates both angles — and the tilt ramps are linear, so
curvature is piecewise constant and the curvature reversal is a point rather
than a flat region.*

### 2.7 Vertebral geometry from CT

VerSe supplies hand-corrected vertebral labels and, through its segmentation
masks, the orientations. Extracting them is not trivial and three distinct
degeneracies had to be resolved; each is invisible on a box phantom and
decisive on real vertebrae. **This subsection is a methods contribution in its
own right for anyone deriving vertebral orientation from a segmentation.**

1. **The body must be separated from the posterior elements by thickness, not
   direction.** The body is the only part of a vertebra thick in every
   direction. A morphological opening by a 6 mm ball removes the pedicles,
   laminae and processes and restores the body exactly. Cutting along the
   antero-posterior axis at the pedicle narrowing instead — the anatomically
   obvious approach — fails: it removes the spinous process but leaves the
   transverse processes, giving widths of 71–90 mm against a true 30–50.
2. **The in-plane orientation cannot come from the body's principal axes.** A
   thoracic body is as deep as it is wide (width/depth 0.8–1.3 on VerSe), so
   those moments are degenerate and the recovered axial rotation flips 90°
   between neighbours: a median of 27° of apparent rotation on supine CT,
   where the truth is near zero. The posterior elements settle it — the
   spinous process is always behind the body.
3. **The endplate normal cannot come from them either.** The two smaller
   moments of a lumbar body differ by under 20%, and their eigenvectors came
   out 40° and 51° from the cranio-caudal axis on a vertebra whose endplate is
   within 10° of horizontal. Fitting planes to the body's superior and
   inferior surfaces measures the endplate the way a Cobb angle is defined on
   it.

Measured body dimensions agree with published morphometry: heights within
1–2 mm at every level, widths within 2 mm below T4.

An automated quality check flags vertebrae whose dimensions are impossible for
their level or whose segmental angle disagrees with both neighbours. It
flagged 2 of 30 scans, both at the cranial end of the field of view.

### 2.8 What is and is not under test

Landmarks in the VerSe experiments are projections of the CT-derived model, so
**landmark detection from radiographic pixels is not evaluated**. What is
evaluated is everything that depends on real spinal anatomy rather than on a
phantom: whether the inverse problem recovers real orientations including real
axial rotation, and whether three-dimensional angles computed from two
projections match those computed directly from the CT. Robustness to detector
error is assessed by injecting a stated landmark error (section 3.5).

---

## 3. Results

### 3.1 The dataset

80 scans in the VerSe 2019 training release; 48 skipped because the field of
view does not span the thoracolumbar junction with enough bodies on both
sides, 2 failed, 30 usable, 28 passing the quality check.

### 3.2 The biplanar inverse problem is exact when axial rotation is known

Per scan, taking the worst vertebra in each (28 scans):

| | median | p90 | max |
|---|---|---|---|
| endplate normal error, ψ supplied | **0.00°** | **0.00°** | **0.00°** |
| endplate normal error, ψ assumed zero | 3.26° | 5.65° | 9.30° |
| PMC angle from two views vs CT truth | 0.00° | 0.00° | 0.00° |

Per vertebra, with ψ assumed zero: median 0.55°, p90 2.79°, max 9.30°. The
error tracks the rotation actually present and stays below the bound the
phantom predicts (Figure 5c).

Axial rotation present in the data: median 10.0°, p90 22.8°, max 41.4° per
scan.

The two scans the quality check flagged are the two where the reconstruction
is not exact, and both fail at the cranial end of the field of view where the
segmentation had merged a vertebral body with a neighbouring structure. The
flag is computed from the segmentation alone, without reference to the
reconstruction, so it is a usable pre-filter rather than a post-hoc exclusion.

### 3.2b Two pedicle landmarks make the measurement identifiable

Over orientations drawn uniformly across the anatomical range (coronal tilt
+-35 deg, sagittal +-30, axial rotation +-30) the endplate normal recovered
from four corners alone is in error by:

| | median | p90 | max |
|---|---|---|---|
| four corners per view only | **5.50** | **12.99** | **22.84** |
| plus pedicles, exact | 0.00 | 0.00 | 0.00 |
| plus pedicles, 0.5 mm error | 0.21 | 0.60 | 1.36 |
| plus pedicles, 1.0 mm error | 0.43 | 1.15 | 3.03 |
| plus pedicles, 2.0 mm error | 0.86 | 2.44 | 4.02 |

All values in degrees. The bias without pedicles grows linearly with the
rotation actually present, from zero to 12.8 deg at 30 deg. With them it is
flat: the solution does not degrade as the deformity worsens, which is the
property that matters, because axial rotation and coronal deformity grow
together.

**The requirement is about one millimetre.** The 90th-percentile residual
crosses 1 deg at 0.9 mm of pedicle localisation error; the median crosses it
at about 3 mm.

**Patient-specific pedicle geometry is unnecessary.** Taking the
half-separation and posterior offset from a normative table rather than
measuring them costs a median of 0.13 deg when the table is wrong by 1 mm and
0.25 deg at 2 mm. The interquartile spread actually observed across 400
vertebrae is 12.3 to 15.4 mm in separation and 25.0 to 29.4 mm in posterior
offset, so a table sits comfortably inside the regime where it does not
matter.

That table is a secondary deliverable. Medians over 400 vertebrae from 28
spines, in millimetres:

| level | T1 | T4 | T7 | T10 | T12 | L1 | L3 | L5 |
|---|---|---|---|---|---|---|---|---|
| half-separation | 16.7 | 11.9 | 13.3 | 14.1 | 11.8 | 13.1 | 16.2 | 19.6 |
| posterior offset | 18.6 | 24.0 | 27.2 | 27.5 | 27.6 | 29.1 | 29.7 | 26.8 |

The posterior offset carries the rotation signal, and it runs from 19 mm at T1
to about 29 mm in the lumbar spine, so one degree of rotation moves the
projected pedicle midpoint by 0.33 to 0.51 mm. That, and not the endplate
geometry, is what sets the millimetre requirement.

On VerSe, with real orientations and pedicle geometry measured from the
segmentations (394 vertebrae; axial rotation median 3.8 deg, max 26.4), the
same ordering holds: a median residual of 0.55 deg without pedicles against
0.16 with them at 1 mm of localisation error and the normative table.

*Figure 6.*

### 3.3 The coronal projection understates the deformity, in every curve measured

70 curves from 28 spines.

| plane of maximum curvature minus coronal | median | p90 | max |
|---|---|---|---|
| all 70 curves | **7.2°** | **22.8°** | **37.0°** |
| the 32 curves reaching 10° | 8.1° | 21.0° | — |

**In none of the 70 curves is the coronal angle the larger one.** The median
ratio of the three-dimensional angle to the coronal one is 1.79.

**This cohort is not a scoliosis cohort, and the absolute magnitudes must be
read with that in mind.** VerSe is a general and fracture CT collection: the
coronal Cobb angles in it have a median of 9.5°, a 90th percentile of 16.2°
and a maximum of 26.2°, and only one curve of 70 reaches 25°. The categorical
result — that the coronal projection never overstates and systematically
under-reads — is established here on real anatomy. The magnitude *in
scoliosis* is not, and cannot be from this dataset.

What the phantoms add is the mechanism, and it says where to expect the gap to
be large. The understatement is driven by the curve's **sagittal** component,
not by its coronal size: on a prescribed 50° main thoracic curve the gap is
only 2.2°, while on the 32° lumbar counter-curve beneath it, where lordosis
makes the sagittal component steep, it is 25.1°. Thoracolumbar and lumbar
curves are therefore the ones a coronal radiograph most under-reads — which is
consistent with the VerSe result, where normal lordosis supplies the sagittal
component.

*Figure 5a-b: per-curve scatter of coronal against PMC angle coloured by the
orientation of the PMC, and the distribution of the gap. Figure 1d: Cobb as a
continuous function of measurement-plane orientation on a phantom.*

### 3.4 Three independent costs of assuming no axial rotation

| effect | at ψ = 0 | at ψ = 20° |
|---|---|---|
| error in the recovered endplate normal | 0° | 8.3° |
| coronal tilt error from cone-beam divergence | **exactly 0°** | 4.5° |
| thoracolumbar labelling | correct | wrong, and flagged |

The middle row deserves its own sentence. With no axial rotation the chord a
reader marks on a frontal film lies in a plane of constant depth, so both ends
are magnified identically and the tilt is exactly invariant: treating a long
film as a parallel projection is not an approximation but an identity — until
the spine rotates.

### 3.5 Robustness to landmark error

Isotropic Gaussian error added to every annotated corner before any
measurement, 28 unflagged scans, medians across scans:

| corner error | 0 mm | 0.5 mm | 1.0 mm | 2.0 mm |
|---|---|---|---|---|
| endplate normal error, ψ supplied | 0.00° | 2.12° | 4.22° | 8.34° |
| endplate normal error, ψ assumed zero | 3.26° | 3.62° | 5.25° | 8.73° |
| PMC angle from two views vs CT truth | 0.00° | 0.76° | 1.49° | 5.28° |
| **PMC minus coronal** | **8.5°** | **6.2°** | **5.7°** | **6.9°** |

The headline effect does not wash out: the coronal projection still under-reads
by 6 to 7 degrees at the median with 2 mm of corner error, which is worse
localisation than a current detector achieves.

Two things are visible in the first two rows. Landmark error and the
zero-rotation assumption are comparable in size at about 1 mm and the
assumption dominates below that, so for a well-localised study it is the
assumption, not the detector, that limits the reconstruction.

### 3.6 A lateral radiograph under-reads kyphosis in scoliosis

Coronal tilt swings the body's antero-posterior axis out of the sagittal
plane, so the anterior-to-posterior chord is not the sagittal trace of the
endplate.

| main thoracic curve | 0° | 25° | 50° | 75° |
|---|---|---|---|---|
| T1–T12 sagittal truth | 35.0° | 25.9° | 26.9° | 29.5° |
| as read on a lateral film | 35.0° | 24.2° | 23.4° | 21.5° |
| under-read by | 0.0° | 1.7° | 3.5° | **8.0°** |

### 3.7 Where the spread in a Cobb angle comes from

| corner error | endplate lines (SD) | total (SD) | end-vertebra share of variance | end vertebrae unchanged |
|---|---|---|---|---|
| 0.5 mm | 1.54° | 1.54° | 0% | 100% |
| 1.0 mm | 3.09° | 3.09° | 0% | 98% |
| 2.0 mm | 6.19° | 6.37° | 6% | 74% |
| 3.0 mm | 9.31° | 11.35° | 33% | 47% |
| 4.0 mm | 12.48° | 17.20° | **47%** | 24% |

There is a threshold. Below about 1.5 mm a better detector buys a better angle
outright; above 3 mm a third or more of the error is a discrete choice between
vertebrae that no amount of line-fitting removes. This is a *lower bound* on
the discrete term: independent corner jitter is a poor model of how two
readers come to disagree about which vertebra ends a curve.

### 3.8 Vertebral levels without a trained model

A secondary component, reported with its limitation rather than around it.

The sagittal curvature reverses where thoracic kyphosis gives way to lumbar
lordosis. That reversal is anatomically anchored, so levels can be counted
outward from it with no training data, and two further checks — the fixed
twelve-body count between the C7/T1 and T12/L1 reversals, and the requirement
that the whole labelling fit between C1 and S1 — let the method contradict
itself rather than fail silently.

On standing-geometry phantoms this is correct in every case tested and holds
its level in 86–98% of resamples under 1 mm of corner noise.

On VerSe it is correct in 4 of 30 and within one level in 21 of 30, and the
error is systematic rather than random: **the anchor lands one level caudal to
T12 in 14 of 26 scans.** VerSe is supine, and supine positioning flattens
lumbar lordosis and moves the inflection caudally. The anchor level is
therefore posture-dependent — T12 standing, L1 supine — which is a finding in
itself and a prerequisite for anyone deploying a curvature anchor.

Two honest negatives: recalibrating the anchor to L1 raises exact agreement
only to 14 of 30, and the anchor-strength confidence signal does not predict
accuracy on supine data (56% / 62% / 50% above 0° / 20° / 25°). A fair
evaluation needs standing biplanar data with hand-checked levels, and no
public dataset provides it: SpineWeb, which hosted the AASCE landmarks, no
longer resolves, and the AASCE images are frontal only.

---

## 4. Discussion

*Points to make:*

* What the numbers mean clinically: a curve reported as 45° coronal may be
  52° in the plane it actually occupies, and the discrepancy is largest in the
  lumbar spine where it exceeded 30°.
* Axial rotation is the one quantity two films cannot supply and the one that
  most corrupts them. It should be measured, from pedicles or from a
  segmentation, not assumed.
* A Cobb angle reported without an interval is reported without its
  resolution. The interval is computable per patient.
* Limitations, stated: landmark detection is out of scope; VerSe is supine and
  axial CT, standing radiographs are neither; the phantom's corner model
  assumes a box body; the quality flag is a heuristic.
* What would settle the labelling question: standing biplanar series with
  hand-checked levels.

## 5. Reproducibility

Every number above is produced by two scripts over a public dataset and a
public repository. `tools/make_figures.py` builds every figure from phantoms,
so no figure carries patient data or a dataset licence.

VerSe is CC BY-SA (2.0 in the bundled licence file, 4.0 per the project
README) and its terms require three citations: Löffler 2020, Liebl 2021,
Sekuboyina 2021.

---

## 6. References to secure

Verified during the literature check; full citations still to be assembled.

**Establishes what this paper does not claim as new**

1. Stokes IAF, for the SRS Working Group on 3-D Terminology of Spinal
   Deformity. Three-dimensional terminology of spinal deformity. *Spine*
   1994;19:236-248. PMID 8153835. Defines the vertebral body line and the
   per-vertebra angulations this package's conventions follow.
2. Stagnara P. The plane-of-maximum-curvature projection: rotating the plate
   about the spine's long axis to view the curve where it is largest. Cited in
   the PMC literature below; primary reference to be traced.
3. Plane of maximum curvature from CT in AIS. PubMed 32693677 (2020).
4. Estimation of the plane of maximum curvature by a computational method.
   *Eur Spine J* 2020. PubMed 32767126.
5. Comparison of two- and three-dimensional measurement of the Cobb angle.
   *Int Orthop* 2016.

**The method this paper analyses**

6. The measurement of the three-dimensional scoliosis angle from standard
   radiographs. *BMC Musculoskelet Disord* 2020;21. PMC7372870. n = 41
   patients, 62 curves; mean Cobb 54 +- 17 deg against a mean 3-D angle of
   60 +- 15. Computes the angle between endplate normals built from four
   measured radiographic angles. **Does not discuss axial rotation, and does
   not examine whether the measured lateral angle is the sagittal tilt.**

**Why the problem is specific to the landmark-only route**

7. Validation of the relative 3D orientation of vertebrae reconstructed by
   biplanar radiography. PubMed 15147749. Orientation accuracy 1.5 deg except
   axial rotation, 3.3 deg moderate and 4.4 deg severe.
8. Comparison of 3-D spinal reconstruction accuracy: biplanar radiographs with
   EOS versus CT. PubMed 22415001. RMS axial rotation 1.9 deg, maximum 5.8.
9. Accuracy of vertebral rotation assessment using biplanar imaging in AIS.
   *Spine Deformity*, recent.

**Axial rotation grading this work builds on**

10. Nash CL, Moe JH. A study of vertebral rotation. *J Bone Joint Surg Am*
    1969;51:223-229.
11. Perdriolle R, Vidal J. Morphology of scoliosis: three-dimensional
    evolution. *Orthopedics* 1987.

**Uncertainty, which this paper reports rather than discovers**

12. Carman DL, Browne RH, Birch JG. Measurement of the Cobb angle on
    radiographs of patients who have scoliosis: evaluation of intrinsic error.
    *J Bone Joint Surg Am* 1990;72:328-333. PMID 2312527. Interobserver 95%
    limit 7.2 deg with self-selected end vertebrae against 6.3 with them
    pre-selected: the classical decomposition this package reproduces
    computationally and per patient.
13. Geometric foundations of measurement uncertainty and clinical relevance in
    radiographic spinal angle assessment. PubMed 42443638. Establishes
    CI95 ~ 111.4 R/L in landmark error R and vertebral size L.

**Datasets**

14. Loffler M, et al. A vertebral segmentation dataset with fracture grading.
    *Radiol Artif Intell* 2020. doi:10.1148/ryai.2020190138
15. Liebl H, et al. A CT vertebral segmentation dataset with anatomical
    variations and multi-vendor scanner data. *Sci Data* 2021;8:284.
16. Sekuboyina A, et al. VerSe: a vertebrae labelling and segmentation
    benchmark for multi-detector CT images. *Med Image Anal* 2021;73:102166.
    (14-16 are required by the VerSe terms of use.)
17. AASCE 2019 challenge: accurate automated spinal curvature estimation.
    *Med Image Anal* 2021. **The SpineWeb host of the training landmarks no
    longer resolves; the images are frontal only.**

**Context for the demoted labelling component**

18. Roussouly P, et al. Classification of the normal variation in the sagittal
    alignment of the human lumbar spine and pelvis. *Spine* 2005. Source of
    the piecewise-circular-arc idealisation the phantoms use.
19. Supine versus standing radiographic measurement in scoliosis: the sagittal
    flattening that moves the thoracolumbar inflection. To be traced.
