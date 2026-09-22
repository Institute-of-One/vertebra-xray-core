# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — unreleased

First public release, accompanying Institute of One research note IORN-014.

### Added

- Biplanar reconstruction of vertebral orientation from endplate landmarks,
  with `solve_orientation` and `reconstruct_from_biplanar`.
- Pedicle landmarks as the third constraint, closing the system that four
  endplate corners per view leaves underdetermined, with a normative pedicle
  table measured from 394 VerSe vertebrae.
- Three-dimensional Cobb angles: the maximum over vertical planes, the
  centroid plane, and the dihedral angle between endplates.
- Structural curve detection and Cobb measurement from a single projection.
- Per-case uncertainty by bootstrap, separating the endplate-line and
  end-vertebra contributions.
- Vertebral orientation measured from CT segmentation masks.
- Training-free vertebral level assignment from the sagittal curvature
  reversal. Implemented and tested, but **not reported in IORN-014**: it is
  correct on standing phantom geometry and correct in only 4 of 30 supine
  VerSe scans, because supine positioning moves the sagittal inflection from
  T12 to L1. That posture dependence deserves its own report.
- Phantoms with closed-form angles, and a renderer that projects them as
  radiographs for figures.
- Five scripts under `tools/` that produce every number and figure in the
  paper.
