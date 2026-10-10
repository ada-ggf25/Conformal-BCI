# Tests

The complete test suite: `src/` unit tests, documentation checks, and the LaTeX checks for the paper.

## Structure

```text
tests/
├── analyses/            # Tests for analyses/run_mazes.py (the seed x maze sweep)
├── chain/               # Tests for src/chain/
├── conformal/           # Tests for src/conformal/
├── models/              # Tests for src/models/
├── utils/               # Tests for src/utils/
├── data/                # Tests for src/data/
├── db/                  # Tests for src/db/
├── docs/                # Documentation checks (MkDocs/README.md/WORKFLOWS.md)
└── paper/               # Tests for paper/main.tex
```

## Setup

Setup is tiered, each tier builds on the one above it.

```bash
make setup-python    # src/ unit tests, linters, type checks, MkDocs build
make setup-docs      # adds the docs spelling suite
make setup-latex     # adds the paper LaTeX suite

make setup           # alias for setup-python
make setup-all       # all three tiers at once
```

## Running tests

All commands run from the repo root:

```bash
make src-test        # every src/ unit test in one pytest process (what CI runs)
make data-test       # src/data/ only
make utils-test      # src/utils/ only
make models-test     # src/models/ only
make conformal-test  # src/conformal/ only
make db-test         # src/db/ only (offline, the MongoDB client is mocked)
make chain-test      # src/chain/ only
make src-coverage    # every src/ unit test, with line and branch coverage of src/

make python-lint       # ruff check + ruff format --check over src/, tests/, analyses/
make python-typecheck  # mypy over src/, the typed tests/ subpackages, and analyses/*.py

make docs-test       # README, WORKFLOWS.md and MkDocs page lint + spelling
make mkdocs-test     # MkDocs site build check (mkdocs build --strict)

make paper-test   # full paper LaTeX suite

make ci-test   # everything above
```
