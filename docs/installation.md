# Installation and Reproducibility

## Runtime

Use Python 3.11 or newer. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[flow]'
```

The core requires grid-data-models 2.4.0 or newer. Flow integration requires
the released gdm-flow 0.5.0 or newer. Imports use `gdm.systems.distribution`.

## Book and Notebook

```sh
python -m pip install '.[docs,dev,opendss]'
python -m ipykernel install --user --name gdm-reduce-docs --display-name 'GDM Reduce Docs'
cd docs
jupyter-book build --html --strict
```

This is a Jupyter Book 2 / MyST project configured by `myst.yml`. HTML builds
render committed notebook outputs without network model downloads. To reproduce
the computation before building:

```sh
cd ..
jupyter nbconvert --to notebook --execute --inplace docs/comparison.ipynb \
  --ExecutePreprocessor.kernel_name=gdm-reduce-docs \
  --ExecutePreprocessor.timeout=9000
cd docs
jupyter-book build --html --strict
```

For numerical runs, set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1` to avoid nested thread oversubscription. Build artifacts
and notebook-generated model exports are under ignored `docs/_build/`.
Use `jupyter-book start` from `docs` to serve a local preview.

## Model Preparation

Ordinary fixture tests and the notebook use copied JSON and companion
time-series stores. They do not require runtime DiTTo or sibling repositories.
Refreshing copies requires a working DiTTo installation:

```sh
python -m pip install '.[conversion]'
python examples/prepare_circuit_models.py \
  /path/to/ditto/tests/data/opendss_circuit_models
```

Preparation for this book used local DiTTo source and its own environment,
which passed its circuit conversion tests. To reproduce that setup:

```sh
PYTHONPATH=/path/to/ditto/src /path/to/ditto/.venv/bin/python \
  examples/prepare_circuit_models.py /path/to/ditto/tests/data/opendss_circuit_models
```

The separate `opendss_transformer_models` corpus is not the collection used here.

## Verification

```sh
pytest tests/test_circuit_documentation.py -q
pytest --cov=gdm_reduce --cov-report=term-missing
```

Focused tests round-trip every copied primary circuit and reproduce the IEEE-13
export and held-out comparison. The full suite retains the 90% coverage gate.
Serialization alone does not establish nonlinear convergence or export support.

## Automation and Releases

CI runs tests and package builds on Python 3.11, 3.12, and 3.13, executes the
comparison notebook, builds the MyST book strictly, and runs pre-commit. Install
the existing Ruff lint and formatting hooks locally:

```sh
python -m pip install '.[dev]'
pre-commit install
pre-commit run --all-files
```

In GitHub repository Settings > Pages, select **GitHub Actions** as the build
source. Successful pushes to `main` deploy the book only after tests, lint, and
the documentation build pass. Pull requests build docs without deploying them.
The project site is `https://nlr-distribution-suite.github.io/gdm-reduce/`.

For PyPI, register a trusted publisher for repository
`NLR-Distribution-Suite/gdm-reduce`, workflow `publish.yml`, and environment
`pypi`. Create the GitHub `pypi` environment and restrict it to approved releases
as appropriate. No API token is required. Update the package version in
`pyproject.toml`, then publish a GitHub release with a matching tag, such as
`v0.1.0a1`. The publishing workflow verifies the tag, builds and checks the wheel
and source distribution, then publishes those artifacts using OIDC.