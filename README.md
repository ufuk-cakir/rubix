# Welcome to virtual-telescope

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ufuk-cakir/virtual-telescope/ci.yml?branch=main)](https://github.com/ufuk-cakir/virtual-telescope/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/virtual-telescope/badge/)](https://virtual-telescope.readthedocs.io/)
[![codecov](https://codecov.io/gh/ufuk-cakir/virtual-telescope/branch/main/graph/badge.svg)](https://codecov.io/gh/ufuk-cakir/virtual-telescope)

## Installation

The Python package `virtual_telescope` can be installed from PyPI:

```
python -m pip install virtual_telescope
```

## Development installation

If you want to contribute to the development of `virtual_telescope`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/ufuk-cakir/virtual-telescope
cd virtual-telescope
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).
