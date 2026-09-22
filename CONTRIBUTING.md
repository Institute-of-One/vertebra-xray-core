# Contributing

Issues and pull requests are welcome.

## Running the checks

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy vertebra_xray_core
pytest
```

All four must pass. CI runs the same four on Python 3.10, 3.11 and 3.12.

## What this package is, and is not

It is the measurement layer: given landmarks, it computes angles and states
their uncertainty. Landmark detection from pixels is out of scope by design,
and so is anything that would require patient data.

**No patient data, in any form, ever enters this repository** — not DICOM, not
NIfTI, not a derived mask, not a figure containing one. Validation uses
synthetic phantoms and public collections, and public collections are read
where they live rather than redistributed here.

## Two conventions worth knowing before you change anything

**Numbers in the documentation are generated, not typed.** Every figure in
`paper/figures/` and every quantity in the manuscript comes from a script in
`tools/`. If you change something that moves a number, re-run the script that
produces it and commit the new output. Do not edit the number by hand.

**Tests encode geometry that was got wrong once.** Several look arbitrary and
are not: corner canonicalisation by the local spine axis rather than global y,
the Euler order that leaves the endplate normal independent of axial rotation,
chained rather than superposed phantom curves. Each one replaced something
that produced plausible wrong answers. If a test like that fails, the change
is probably wrong, not the test.
