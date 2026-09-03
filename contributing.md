# Contributing to C4DYNAMICS

Thank you for contributing. This guide explains environment requirements, repository structure, and contribution expectations.

---

## Supported Environment

* **Python:** Python ≥ 3.8
* **OS:** Linux, macOS, Windows
* **GPU:** Not required

### Optional Dependencies

Some modules (e.g. vision-related functionality) require additional heavy dependencies such as OpenCV.

If you do not use those modules, they are not required.

> Note: Python 3.12 may fail due to upstream OpenCV wheel availability.

---

## Installation (Development Setup)

Clone the repository and install development dependencies:

```bash
pip install -r requirements-dev.txt
```

For additional setup details, see the setup notebook:
`c4dynamics_setup.ipynb`

Documentation:
[https://c4dynamics.github.io/c4dynamics/](https://c4dynamics.github.io/c4dynamics/)

---

## Repository Structure

```text
c4dynamics/
├── c4dynamics/
│   ├── states/      # Core objects (the state, datapoint, rigidbody, pixelpoint)
│   ├── modules/     # Functional modules (sensors, control, estimation, vision)
│   └── utils/       # Shared utilities
├── docs/            # Concepts, demonstrations, API reference
└── tests/           # Unit tests and doctests
```

Core logic must live inside the package.
Notebooks are for demonstrations only.

---

## Contribution Types

We accept contributions in two main areas:

1. **Use Case Notebooks**
   Practical demonstrations of modeling, control, estimation, or simulation workflows.

2. **New Features / Improvements**
   Extensions or modifications to the framework core.

---

## Contribution Workflow

1. Open an issue for significant changes.
2. Fork the repository.
3. Create a feature branch.
4. Implement and test locally.
5. Submit a Pull Request.

Reviews typically take a few days.
If blocked, tag the maintainer in the PR.

---

## Testing

All contributions must pass tests.

Run locally:

```bash
python tests/run_doctests.py
python tests/run_unittests.py
```

Fix failures before submitting a PR.

---

## Pull Request Quality Bar

A PR is expected to:

* Pass all tests
* Follow PEP8-style formatting
* Include docstrings for public APIs
* Include tests for new functionality
* Avoid breaking backward compatibility unless discussed

Clear, focused PRs are preferred over large, mixed changes.

---

## Guidance for Notebook Contributions

* Keep examples focused and reproducible
* Add explanatory comments
* Avoid embedding core logic inside notebooks
* Use framework primitives (state objects, modules) properly

---

If you’re unsure whether something belongs in core or in a notebook, open an issue first.
