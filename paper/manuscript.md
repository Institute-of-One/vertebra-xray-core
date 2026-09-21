# A Cobb angle is a property of a spine and a projection: three-dimensional deformity measurement from biplanar radiographs, with the cost of ignoring axial rotation

**Draft — IORN-014.** Numbers in this draft are produced by
`tools/validate_verse.py` and `tools/make_figures.py`; every one of them is
reproducible from the repository and the public VerSe release.

---

## Abstract

*(to be written last; structure below)*

**Purpose.** A Cobb angle is measured on a coronal radiograph because that
projection is easy to acquire, not because it is where the deformity lies.
We quantify what the coronal projection omits, what a biplanar pair can and
cannot recover, and what it costs to assume — as published biplanar pipelines
do implicitly — that vertebrae are not axially rotated.

**Methods.** A measurement layer that takes four corner landmarks per
vertebral body from any source and reports coronal Cobb angles, the angle in
the plane of maximum curvature, and an interval on each. Validated against
closed-form phantoms and against 30 spines from the VerSe CT benchmark, which
supplies the vertebral orientations — including axial rotation — that no pair
of radiographs can.

**Results.** On 28 spines passing an automated quality check, the biplanar
inverse problem recovers every endplate normal exactly (median, 90th
percentile and maximum error 0.00°) when axial rotation is supplied, and with
a median error of 3.26° (p90 5.65°, max 9.30°) when it is assumed to be zero.
Across 70 curves the angle in the plane of maximum curvature exceeded the
coronal angle in **every one**, by a median of 7.2° (p90 22.8°, max 37.0°);
the effect persists under 2 mm of injected landmark error. Cone-beam
divergence cannot change a coronal Cobb angle at all when axial rotation is
zero, and changes it by 4.5° at 20° of rotation. A lateral radiograph
under-reads T1–T12 kyphosis by 8.0° on a 75° curve. Landmark error propagates
at roughly 3° of Cobb angle per millimetre, and the choice of end vertebrae
contributes none of the variance below 1.5 mm and half of it at 4 mm.

**Conclusion.** The coronal projection systematically under-reads spinal
deformity, and the shortfall is set by the curve's sagittal component rather
than its coronal size, so it is largest in the thoracolumbar and lumbar spine.
Axial rotation is the one quantity two radiographs cannot supply and the one
that most corrupts what they do supply; it should be measured rather than
assumed. A Cobb angle reported without an interval is reported without its
resolution, and the interval is computable per patient.

**Stated limitation.** The cohort is a general and fracture CT collection, not
a scoliosis one — median coronal Cobb 9.5°, one curve of 70 above 25° — so the
*direction* and *mechanism* of the shortfall are established here on real
anatomy but its *magnitude in scoliosis* is not. Landmark detection from
radiographic pixels is outside the scope of this work by design.

---

## 1. Introduction

*Points to make, in order:*

1. The Cobb angle is the measurement scoliosis is diagnosed, monitored,
   braced and operated on. It is defined on a single coronal projection.
2. Scoliosis is not a coronal deformity. It is a three-dimensional one, and
   the plane in which it is largest is not the coronal plane. This is not new
   (Stokes 1994; the SRS three-dimensional terminology), but the measurement
   that clinical practice runs on has not moved.
3. Biplanar systems (EOS and others) make a three-dimensional measurement
   possible, and a substantial literature reconstructs spines from them. What
   that literature does not state is what the reconstruction assumes and what
   the assumption costs.
4. The gap this paper fills: not a better detector, but a stated, auditable
   measurement layer, and quantitative answers to three questions that are
   normally left implicit — how much the coronal plane omits, what axial
   rotation does to a biplanar reconstruction, and how much of a reported Cobb
   angle is noise.
5. Explicit non-goal: landmark detection from radiographic pixels. That field
   is crowded and competitive. Keeping it outside makes a human annotation, a
   published detector and a CT segmentation directly comparable, because they
   are measured by identical code.

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
