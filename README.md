# vertebra-xray-core

Training-free vertebral labelling and three-dimensional Cobb angles from
biplanar spine radiographs, with a stated uncertainty on every number.

This is the measurement core. It does **not** detect vertebrae in an image,
and that is deliberate: detection is where the field already competes, and
keeping it outside means a human annotation, a published detector and a CT
segmentation are all measured by identical code and are therefore directly
comparable.

```python
from vertebra_xray_core import cobb, cobb3d, labeling, phantom
from vertebra_xray_core.spine3d import Projection, reconstruct_from_biplanar

model   = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=45.0, lumbar_deg=30.0)
frontal = model.project(Projection(view="pa")).with_labels(None)
lateral = model.project(Projection(view="lateral")).with_labels(None)

levels  = labeling.label_biplanar(frontal, lateral)        # no training data involved
spine   = reconstruct_from_biplanar(levels.apply(frontal), levels.apply(lateral))

for curve in cobb3d.cobb3d_for_curves(spine, cobb.cobb_angles(levels.apply(frontal)).curves):
    print(curve.describe())
# PT T1-T5:  coronal 20.0 deg, 3-D 22.1 deg in a plane -25.2 deg from coronal
# MT T5-T12: coronal 45.0 deg, 3-D 47.5 deg in a plane +19.8 deg from coronal
# L  T12-L5: coronal 30.0 deg, 3-D 55.4 deg in a plane +56.7 deg from coronal
```

## What is here

### Vertebral levels without a trained model

Which body is T12 decides every name in a spine report. The usual answer is a
network trained to recognise each level's appearance, which needs annotated
data, inherits that data's population, and fails silently.

There is a geometric fact that needs none of that. Walking down the spine, the
sagittal curvature reverses sign where thoracic kyphosis gives way to lumbar
lordosis, and that reversal is at the T12/L1 disc in essentially every human
spine. Curvature is the derivative of the endplate tilt profile, so the
reversal can be read straight off the landmarks. A second reversal at C7/T1
gives an independent anchor, the count between them is fixed at twelve, and
the whole labelling has to fit between C1 and S1 — so the method checks itself
three ways and **reports a disagreement instead of absorbing it**.

A lateral film alone is not quite enough, because coronal tilt and axial
rotation both distort the chord it shows. Reconstructing the spine first
removes both distortions. On a grid of main curves from 45 to 90 degrees
crossed with axial rotations from 0 to 30 degrees, a single lateral film gets
5 of 16 right and the two-pass `label_biplanar` route gets 16 of 16.

This reworks an idea from a 2007 MATLAB prototype that read levels off the
second derivative of a polynomial fitted to the *frontal* curve. The mechanism
was right and the plane was wrong: coronal inflections sit wherever the
scoliosis puts them, so they identify nothing.

### Cobb angles in three dimensions

A Cobb angle is a property of a spine *and a projection*. The coronal
radiograph is one particular projection, chosen because it is easy to acquire.
Rotating the measurement plane about the cranio-caudal axis changes the number
continuously, and the plane where it peaks is the one that describes the
deformity. Every curve is reported in the coronal plane, in the sagittal
plane, and in its plane of maximum curvature.

Recovering the three-dimensional endplate normals from two films is a closed
inverse problem that this package solves exactly, and it makes explicit what
the two films can and cannot determine — see *Axial rotation* below.

### An interval, not just a number

Landmark error propagates into a Cobb angle at roughly three degrees per
millimetre of corner error, which is the same order as the inter-observer
spread reported in reader studies. `uncertainty.bootstrap_cobb` resamples
under a stated localisation error and separates the two mechanisms behind
that spread: the endplate lines moving, and the **end vertebrae being chosen
differently**.

The split has a threshold in it. Under independent corner noise on a 50
degree main thoracic curve, the discrete choice contributes nothing up to
about 1.5 mm, 6% of the variance at 2 mm, a third at 3 mm and half at 4 mm.
Below the threshold, a better detector buys a better angle outright; above it,
half the error is a choice between vertebrae that no amount of line-fitting
removes.

That is a lower bound on the discrete term rather than an estimate of it:
independent corner jitter is a poor model of how two readers disagree about
which vertebra is the end vertebra, and the real effect is likely larger.

### Phantoms with exact ground truth

Validating against human annotation cannot separate an algorithm's error from
the reference's. `vertebra_xray_core.phantom` builds spines whose Cobb angles,
kyphosis, lordosis and axial rotation are prescribed in closed form, so an
implementation error of a tenth of a degree is visible. They are synthetic,
contain no patient data, and are reproducible from their specification alone.

## Axial rotation: what two films can and cannot tell you

The endplate normal is independent of axial rotation, because the rotation is
about the vertebra's own cranio-caudal axis. Every three-dimensional angle in
this package therefore depends only on coronal and sagittal tilt.

The *projections* are a different matter. What a film shows is the chord
joining two body margins, and rotation swings that chord out of the
measurement plane. Two films give two measurements for three unknowns, so
rotation must come from somewhere else — pedicle landmarks, or a
segmentation. Assuming it away, which published biplanar pipelines do
implicitly, has three consequences this package quantifies rather than hides:

| Effect | At 0 deg | At 20 deg |
|---|---|---|
| Error in the recovered endplate normal | 0 | 8.3 deg |
| Coronal tilt error from beam divergence | exactly 0 | 4.5 deg |
| Levels from a single lateral film | correct | wrong, and flagged |

The middle row is worth its own sentence: with no axial rotation, cone-beam
divergence cannot change a frontal Cobb angle at all, because the chord lies
in a plane of constant depth and both ends are magnified identically. Treating
a long film as a parallel projection is exact, not approximate — until the
spine rotates.

Supply the rotation and `reconstruct_from_biplanar` is exact to machine
precision, and `labeling.label_biplanar` anchors perfectly at every rotation
tested.

## Also quantified here

* A lateral radiograph **under-reads kyphosis in scoliosis**. Coronal tilt
  swings the body's antero-posterior axis out of the sagittal plane, so the
  anterior-to-posterior chord is not the sagittal trace of the endplate. On a
  50 degree curve the gap is several degrees. See `cobb3d.as_seen_on_film`.
* The measured sagittal tilt is coupled to the coronal one by
  `tan(beta) = -cos(theta) tan(phi)`. Reading the lateral tilt straight off as
  the sagittal tilt is wrong by that factor, and the error grows with the
  deformity.
* The SRS endplate definition and the AASCE mid-body definition of a Cobb
  angle are both implemented. A measurable part of the spread between
  published automatic systems is definitional rather than algorithmic.

## Install

```bash
pip install -e ".[dev]"
```

Python 3.10 or newer; NumPy and SciPy are the only runtime dependencies.

```bash
python -m pytest        # 133 tests, about seven seconds
python examples/reproduce_findings.py
```

## Validation

Two things are validated separately, because they fail for different reasons.

**The geometry** is checked against phantoms whose angles are known in closed
form, so an implementation error of a tenth of a degree is visible and there
is no annotator's spread underneath it.

**The anatomy** is checked against [VerSe](https://github.com/anjany/verse), a
CT benchmark with hand-corrected vertebral level identification. CT is the
only realistic source of ground-truth axial rotation -- no pair of radiographs
can supply it -- and VerSe is deliberately enriched with transitional
vertebrae and enumeration anomalies, which are exactly the spines a level
count cannot get right. `vertebra_xray_core.ct` measures each vertebra from a
segmentation mask and `tools/validate_verse.py` runs the pipeline over a
release.

Landmark detection from radiographic pixels is **not** under test anywhere:
this package is the measurement layer, and detector error is handled
analytically by `uncertainty`. Using VerSe requires the three citations its
terms specify, listed in `datasets/verse.py`.

## Figures

```bash
python tools/make_figures.py --out paper/figures
```

Every figure comes from a phantom, so none of them carries patient data or a
dataset licence, and they are identical on any machine. Real images are needed
for at most one qualitative panel.

## Status

Alpha. The measurement core, the labelling, the three-dimensional
reconstruction, the CT path, the phantoms, the uncertainty model, the figures
and the VerSe adapter are implemented and tested. Still to come: DICOM ingest
for long-film studies, corner extraction from TotalSegmentator output, and the
manuscript itself.

## Licence

MIT. See `LICENSE`.
