"""Reading the VerSe vertebral segmentation benchmark.

VerSe is CT with per-vertebra segmentation masks and, crucially, *correct
level identification* done by hand. That makes it the reference this package
validates labelling against, and the only realistic source of ground-truth
axial rotation, which no pair of radiographs can supply.

It is also, deliberately, full of the cases that break level counting: the
2020 release is enriched with transitional vertebrae and enumeration
anomalies. Those are not a nuisance here. A method that claims to assign
levels without training has to say what it does when the spine has thirteen
thoracic vertebrae, and VerSe is where that gets measured.

Licence and citation
--------------------
The bundled ``license.txt`` states Creative Commons Attribution-ShareAlike
2.0, while the project's README says 4.0; either way the ShareAlike term
applies, so derived images need care before they go into a copyrighted
article. The dataset's own readme requires three citations in any work using
it, listed in :data:`REQUIRED_CITATIONS`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import nomenclature as nom
from ..ct import VertebraGeometry, model_from_segmentation
from ..spine3d import SpineModel3D

__all__ = [
    "VERSE_LEVELS",
    "ANOMALOUS_LABELS",
    "REQUIRED_CITATIONS",
    "VerseSample",
    "find_samples",
    "load_model",
]

#: VerSe's integer labels for the levels this package can name. 1-7 are the
#: cervical spine, 8-19 thoracic, 20-24 lumbar, 25 the sacrum.
VERSE_LEVELS: dict[int, str] = {
    **{i: f"C{i}" for i in range(1, 8)},
    **{7 + i: f"T{i}" for i in range(1, 13)},
    **{19 + i: f"L{i}" for i in range(1, 6)},
    25: "S1",
}

#: Labels for vertebrae this package's C1-to-S1 vocabulary cannot name: a
#: thirteenth thoracic vertebra and a sixth lumbar one. Their presence is the
#: marker of an enumeration anomaly, and a spine containing one is a case
#: where counting levels from an anatomical anchor is expected to fail. They
#: are reported, never silently renamed.
ANOMALOUS_LABELS: dict[int, str] = {26: "coccyx", 27: "T13", 28: "L6"}

#: Required by the dataset's own terms; any work using VerSe must cite all three.
REQUIRED_CITATIONS = (
    "Loffler M, et al. A Vertebral Segmentation Dataset with Fracture Grading. "
    "Radiology: Artificial Intelligence, 2020. doi:10.1148/ryai.2020190138",
    "Liebl H, et al. A computed tomography vertebral segmentation dataset with "
    "anatomical variations and multi-vendor scanner data. Sci Data 8:284, 2021. "
    "doi:10.1038/s41597-021-01060-0",
    "Sekuboyina A, et al. VerSe: A Vertebrae labelling and segmentation benchmark "
    "for multi-detector CT images. Med Image Anal 73:102166, 2021. "
    "doi:10.1016/j.media.2021.102166",
)


@dataclass(frozen=True)
class VerseSample:
    """One VerSe scan: the paths, not the pixels."""

    subject: str
    mask_path: Path
    centroid_path: Path | None
    ct_path: Path | None

    def centroids(self) -> list[dict]:
        """The dataset's own centroid annotation, or an empty list.

        The first entry of a VerSe centroid file is a direction header rather
        than a vertebra, and is dropped here.
        """
        if self.centroid_path is None:
            return []
        with open(self.centroid_path, encoding="utf-8") as handle:
            data = json.load(handle)
        return [entry for entry in data if isinstance(entry, dict) and "label" in entry]

    def annotated_labels(self) -> tuple[int, ...]:
        """Integer labels VerSe says are present, from the centroid file."""
        return tuple(int(entry["label"]) for entry in self.centroids())

    def has_enumeration_anomaly(self) -> bool:
        """Whether the scan contains a T13 or an L6.

        These are the spines where counting outward from a curvature anchor
        cannot give the right answer, because the spine does not have the
        number of vertebrae the count assumes.
        """
        return any(label in (27, 28) for label in self.annotated_labels())


def find_samples(root: str | Path) -> list[VerseSample]:
    """Discover VerSe scans under ``root``, however the release is laid out.

    Matching is by the segmentation mask, because that is the one file every
    usable scan must have; the CT and the centroid JSON are attached when a
    sibling with the same subject prefix exists. Releases differ in directory
    structure and in whether files sit in ``rawdata``/``derivatives``, so the
    search is a recursive glob rather than a fixed path.
    """
    root = Path(root)
    samples: list[VerseSample] = []
    for mask in sorted(root.rglob("*_seg-vert_msk.nii.gz")):
        subject = mask.name.split("_seg-vert_msk")[0]
        folder = mask.parent
        centroid = next(iter(sorted(folder.glob(f"{subject}*_ctd.json"))), None)
        ct_candidates = sorted(root.rglob(f"{subject}*_ct.nii.gz"))
        samples.append(
            VerseSample(
                subject=subject,
                mask_path=mask,
                centroid_path=centroid,
                ct_path=ct_candidates[0] if ct_candidates else None,
            )
        )
    return samples


def load_model(
    sample: VerseSample,
    *,
    include_sacrum: bool = False,
    min_voxels: int = 200,
    drop_boundary: bool = True,
) -> tuple[SpineModel3D, dict[int, VertebraGeometry]]:
    """Measure one VerSe scan into a :class:`SpineModel3D`.

    The sacrum is excluded by default: it is not a vertebral body, its mask
    has none of the proportions the body isolation expects, and no Cobb angle
    is measured to it.

    Raises ``ImportError`` if nibabel is missing, and ``ValueError`` if the
    scan has too few usable vertebrae -- a cervical-only or lumbar-only field
    of view, which VerSe has plenty of and which this package cannot anchor.
    """
    try:
        import nibabel
    except ImportError:  # pragma: no cover - environment dependent
        raise ImportError(
            "reading VerSe needs nibabel; install it with pip install 'vertebra-xray-core[ct]'"
        ) from None

    image = nibabel.load(str(sample.mask_path))
    canonical = nibabel.as_closest_canonical(image)  # force RAS+
    mask = np.asarray(canonical.dataobj).astype(np.int16)

    names = dict(VERSE_LEVELS)
    if not include_sacrum:
        names.pop(25, None)
    return model_from_segmentation(
        mask,
        canonical.affine,
        names,
        min_voxels=min_voxels,
        drop_boundary=drop_boundary,
    )


def coverage(model: SpineModel3D) -> dict[str, object]:
    """What a scan actually covers, which decides whether it can be labelled.

    Anchoring levels on the thoracolumbar curvature reversal needs the spine
    on both sides of it. A scan reaching neither T12 nor L1, or covering only
    a handful of bodies, has to be excluded up front rather than producing a
    labelling that happens to be wrong.
    """
    levels = set(model.labels)
    thoracic = [lab for lab in model.labels if nom.region_of(lab) == "thoracic"]
    lumbar = [lab for lab in model.labels if nom.region_of(lab) == "lumbar"]
    return {
        "n_vertebrae": len(model),
        "cranial": model.labels[0],
        "caudal": model.labels[-1],
        "n_thoracic": len(thoracic),
        "n_lumbar": len(lumbar),
        "spans_thoracolumbar": bool({"T12", "L1"} & levels) and len(thoracic) >= 3
        and len(lumbar) >= 2,
        "contiguous": tuple(model.labels) == nom.span(model.labels[0], model.labels[-1]),
    }
