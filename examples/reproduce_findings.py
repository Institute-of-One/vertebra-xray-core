"""Reproduce every quantitative claim in the README, from phantoms alone.

Run with ``python examples/reproduce_findings.py``. Nothing here reads patient
data: the spines are built from their specification, so the output is
identical on any machine and is the starting point for the manuscript figures.
"""

from __future__ import annotations

import numpy as np

from vertebra_xray_core import cobb, cobb3d, labeling, phantom, uncertainty
from vertebra_xray_core.spine3d import (
    Projection,
    axial_rotation_bias,
    reconstruct_from_biplanar,
)

PA = Projection(view="pa")
LAT = Projection(view="lateral")
CONE_PA = Projection(view="pa", sod_mm=1000.0, sdd_mm=1200.0)


def rule(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def coronal_versus_three_dimensional() -> None:
    rule("1. The coronal film understates the deformity")
    model = phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=45.0, lumbar_deg=30.0)
    frontal = model.project(PA)
    for result in cobb3d.cobb3d_for_curves(model, cobb.cobb_angles(frontal).curves):
        print(f"  {result.describe()}")
        print(f"      understated by {result.coronal_underestimate_deg:5.1f} deg")


def labelling_without_training() -> None:
    rule("2. Levels from sagittal curvature alone")
    cases = {
        "normal adult": phantom.normal_adult_spine(),
        "adolescent idiopathic": phantom.adolescent_idiopathic_scoliosis(),
        "severe, 75 deg": phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=75.0),
    }
    for name, model in cases.items():
        lateral = model.project(LAT).with_labels(None)
        result = labeling.label_by_sagittal_inflection(lateral)
        ok = "correct" if result.labels == model.labels else "WRONG"
        stability = uncertainty.labelling_stability(lateral, 1.0, resamples=300)
        print(
            f"  {name:24s} {ok:8s} confidence {result.confidence:8s} "
            f"stable in {stability['exact']:.0%} of resamples at 1 mm noise"
        )


def single_film_versus_biplanar() -> None:
    rule("2b. One lateral film against the two-pass biplanar route")
    print("  main curve   axial rotation   single film   biplanar")
    single_ok = both_ok = total = 0
    for main in (45.0, 60.0, 75.0, 90.0):
        for psi in (0.0, 10.0, 20.0, 30.0):
            model = phantom.adolescent_idiopathic_scoliosis(
                main_thoracic_deg=main, axial_rotation_deg=psi
            )
            frontal = model.project(PA).with_labels(None)
            lateral = model.project(LAT).with_labels(None)
            try:
                one = labeling.label_by_sagittal_inflection(lateral).labels == model.labels
            except ValueError:
                one = False
            two = (
                labeling.label_biplanar(
                    frontal, lateral, axial_rotation_deg=np.full(len(model), psi)
                ).labels
                == model.labels
            )
            single_ok += one
            both_ok += two
            total += 1
            print(
                f"  {main:7.0f} deg {psi:12.0f} deg   {'ok' if one else 'WRONG':>11s}"
                f"   {'ok' if two else 'WRONG':>8s}"
            )
    print(f"\n  correct: single film {single_ok}/{total}, biplanar {both_ok}/{total}")


def axial_rotation_costs() -> None:
    rule("3. What assuming no axial rotation costs")
    print("  psi   normal error   coronal tilt error     labelling anchor")
    print("  deg      deg            from divergence         strength")
    for psi in (0.0, 5.0, 10.0, 20.0, 30.0):
        bias = axial_rotation_bias(20.0, -15.0, psi)["normal_error_deg"]
        model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=psi)
        divergence = np.abs(model.project(PA).body_tilt() - model.project(CONE_PA).body_tilt()).max()
        lateral = model.project(LAT).with_labels(None)
        try:
            anchors = labeling.label_by_sagittal_inflection(lateral).anchors
            strength = f"{max(a.reversal_deg for a in anchors):.1f} deg"
        except ValueError:
            strength = "none found"
        print(f"  {psi:4.0f}   {bias:7.2f}        {divergence:10.2f}          {strength:>12s}")

    print("\n  with the rotation supplied, the reconstruction is exact:")
    for psi in (0.0, 20.0, 40.0):
        model = phantom.adolescent_idiopathic_scoliosis(axial_rotation_deg=psi)
        rebuilt = reconstruct_from_biplanar(
            model.project(PA), model.project(LAT), axial_rotation_deg=np.full(len(model), psi)
        )
        error = np.abs(rebuilt.endplate_normals - model.endplate_normals).max()
        levels = labeling.label_from_model(rebuilt).labels == model.labels
        print(f"    psi={psi:4.0f}  normal error {error:.1e}   levels {'correct' if levels else 'WRONG'}")


def kyphosis_under_read() -> None:
    rule("4. A lateral film under-reads kyphosis in scoliosis")
    for main in (0.0, 25.0, 50.0, 75.0):
        model = (
            phantom.normal_adult_spine()
            if main == 0.0
            else phantom.adolescent_idiopathic_scoliosis(main_thoracic_deg=main)
        )
        true_angle = cobb3d.measure_between_levels(model, "T1", "T12").sagittal_deg
        on_film = cobb3d.as_seen_on_film(model, "T1", "T12", LAT)
        print(
            f"  coronal curve {main:4.0f} deg: sagittal truth {true_angle:5.1f} deg, "
            f"film reads {on_film:5.1f} deg, short by {true_angle - on_film:4.1f}"
        )


def intervals_not_numbers() -> None:
    rule("5. Where the uncertainty in a Cobb angle comes from")
    frontal = phantom.adolescent_idiopathic_scoliosis(
        main_thoracic_deg=45.0, lumbar_deg=30.0
    ).project(PA)
    for sigma in (0.5, 1.0, 2.0, 4.0):
        result = uncertainty.bootstrap_cobb(frontal, sigma, resamples=400)[0]
        print(f"  corner error {sigma:.1f} mm -> {result.describe()}")


if __name__ == "__main__":
    coronal_versus_three_dimensional()
    labelling_without_training()
    single_film_versus_biplanar()
    axial_rotation_costs()
    kyphosis_under_read()
    intervals_not_numbers()
    print()
