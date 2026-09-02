# Installation

## Requirements

The package requires Python 3.10 or later. The main runtime dependencies are standard scientific Python libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `numba`, and `tqdm`. These are installed automatically.

## Installing from PyPI

Install the package from PyPI with:

```bash
pip install bvar
```

## Installing from source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/bank-of-england/bvar.git
cd bvar
pip install -e .
```

An editable install reflects changes under `src/bvar/` immediately, so you need not reinstall the package.

## Optional dependencies

Additional dependency groups are available for development and documentation:

```bash
pip install -e ".[dev]"                # testing and linting
pip install -e ".[docs]"               # Zensical and documentation tools
pip install -e ".[notebooks]"          # Marimo notebook export tools
```

## Verifying the installation

After installing, check the package is importable and inspect the version:

```python
import bvar

print(bvar.__version__)
```
