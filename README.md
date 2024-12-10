# Welcome to rubix

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ufuk-cakir/rubix/ci.yml?branch=main)](https://github.com/ufuk-cakir/rubix/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/rubix/badge/)](https://rubix.readthedocs.io/)
[![codecov](https://codecov.io/gh/ufuk-cakir/rubix/branch/main/graph/badge.svg)](https://codecov.io/gh/ufuk-cakir/rubix)
[![All Contributors](https://img.shields.io/github/all-contributors/ufuk-cakir/rubix?color=ee8449&style=flat-square)](#contributors)

## Installation

The Python package `rubix` can be downloades from git and can be installed:

```
git clone https://github.com/ufuk-cakir/rubix
cd rubix
pip install .
```

## Development installation

If you want to contribute to the development of `rubix`, we recommend
the following editable installation from this repository:

```
git clone https://github.com/ufuk-cakir/rubix
cd rubix
python -m pip install --editable .[tests]
```

Having done so, the test suite can be run using `pytest`:

```
python -m pytest
```

This project depends on [jax](https://github.com/google/jax). It only installed for cpu computations with the testing dependencies. For installation instructions with gpu support,
please refer to [here](https://github.com/google/jax?tab=readme-ov-file#installation).


## Documentation
Sphinx Documentation of all the functions is currently available under [this link](https://astro-rubix.web.app/).

## Configuration Generator Tool
A tool to interactively generate a user configuration is available under [this link](https://cakir-ufuk.de/docs/getting-started/configuration/).

## Acknowledgments

This repository was set up using the [SSC Cookiecutter for Python Packages](https://github.com/ssciwr/cookiecutter-python-package).


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/anschaible"><img src="https://avatars.githubusercontent.com/u/131476730?v=4?s=100" width="100px;" alt="anschaible"/><br /><sub><b>anschaible</b></sub></a><br /><a href="#code-anschaible" title="Code">💻</a> <a href="#content-anschaible" title="Content">🖋</a> <a href="#data-anschaible" title="Data">🔣</a> <a href="#doc-anschaible" title="Documentation">📖</a> <a href="#design-anschaible" title="Design">🎨</a> <a href="#example-anschaible" title="Examples">💡</a> <a href="#ideas-anschaible" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-anschaible" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-anschaible" title="Maintenance">🚧</a> <a href="#plugin-anschaible" title="Plugin/utility libraries">🔌</a> <a href="#projectManagement-anschaible" title="Project Management">📆</a> <a href="#question-anschaible" title="Answering Questions">💬</a> <a href="#research-anschaible" title="Research">🔬</a> <a href="#review-anschaible" title="Reviewed Pull Requests">👀</a> <a href="#tool-anschaible" title="Tools">🔧</a> <a href="#test-anschaible" title="Tests">⚠️</a> <a href="#talk-anschaible" title="Talks">📢</a> <a href="#userTesting-anschaible" title="User Testing">📓</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://tobibu.github.io"><img src="https://avatars.githubusercontent.com/u/7574273?v=4?s=100" width="100px;" alt="Tobias Buck"/><br /><sub><b>Tobias Buck</b></sub></a><br /><a href="#code-TobiBu" title="Code">💻</a> <a href="#content-TobiBu" title="Content">🖋</a> <a href="#data-TobiBu" title="Data">🔣</a> <a href="#doc-TobiBu" title="Documentation">📖</a> <a href="#design-TobiBu" title="Design">🎨</a> <a href="#example-TobiBu" title="Examples">💡</a> <a href="#ideas-TobiBu" title="Ideas, Planning, & Feedback">🤔</a> <a href="#infra-TobiBu" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#maintenance-TobiBu" title="Maintenance">🚧</a> <a href="#plugin-TobiBu" title="Plugin/utility libraries">🔌</a> <a href="#projectManagement-TobiBu" title="Project Management">📆</a> <a href="#question-TobiBu" title="Answering Questions">💬</a> <a href="#research-TobiBu" title="Research">🔬</a> <a href="#review-TobiBu" title="Reviewed Pull Requests">👀</a> <a href="#tool-TobiBu" title="Tools">🔧</a> <a href="#test-TobiBu" title="Tests">⚠️</a> <a href="#talk-TobiBu" title="Talks">📢</a> <a href="#userTesting-TobiBu" title="User Testing">📓</a> <a href="#mentoring-TobiBu" title="Mentoring">🧑‍🏫</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
