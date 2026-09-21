"""Open, auditable measurement of spinal deformity from biplanar radiographs.

The package separates three things that are usually tangled together:

* **landmarks** -- four corners per vertebral body, from any source
* **labelling** -- which body is which, from sagittal curvature alone
* **measurement** -- Cobb angles in two and three dimensions, with intervals

Nothing here detects vertebrae in an image. That is deliberate: detection is
where the field already competes, and keeping it outside means a human
annotation, a published detector and a CT segmentation are all measured by
identical code and are therefore directly comparable.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import cobb, cobb3d, geometry, labeling, nomenclature, phantom, spine3d, uncertainty
from .cobb import CobbResult, Curve, cobb_angles
from .cobb3d import Cobb3D, measure_between_levels, plane_of_maximum_curvature
from .labeling import (
    LabelingResult,
    label_biplanar,
    label_by_sagittal_inflection,
    label_from_model,
)
from .landmarks import SpineLandmarks
from .spine3d import Projection, SpineModel3D, reconstruct_from_biplanar

__all__ = [
    "__version__",
    "Cobb3D",
    "CobbResult",
    "Curve",
    "LabelingResult",
    "Projection",
    "SpineLandmarks",
    "SpineModel3D",
    "cobb",
    "cobb3d",
    "cobb_angles",
    "geometry",
    "label_biplanar",
    "label_by_sagittal_inflection",
    "label_from_model",
    "labeling",
    "measure_between_levels",
    "nomenclature",
    "phantom",
    "plane_of_maximum_curvature",
    "reconstruct_from_biplanar",
    "spine3d",
    "uncertainty",
]
