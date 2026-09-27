# .github

This directory contains the GitHub Actions workflows.

## Workflow Overview

Every workflow runs through `uv` and delegates to a Makefile target. All of them run on pull requests only (path-filtered, with superseded runs canceled) and can also be started manually.

| Workflow | Trigger | What it checks |
|---|---|---|
| `src-tests.yml` | PR to `dev` or `main` | Unit tests for `src/data/`, `src/utils/`, `src/models/`, `src/conformal/`, `src/chain/` and `src/db/`, plus the `analyses/run_mazes.py` sweep bookkeeping (`make src-test`). |
| `python-checks.yml` | PR to `dev` or `main` | Ruff lint + format and mypy type checks over `src/`, `tests/` and `analyses/`. |
| `docs.yml` | PR to `dev` or `main` | Markdown lint and American English spelling over the `README.md` files, this `WORKFLOWS.md` file, and every MkDocs page under `docs/`. |
| `mkdocs.yml` | PR to `dev` or `main` | Verifies the site builds with `mkdocs build --strict` (`make mkdocs-test`). |
| `paper-checks.yml` | PR to `main` | Compiles the paper LaTeX under `paper/`, checks word and float counts, American English spelling of `main.tex` and the files it inputs, and bibliography and figure hygiene (every entry cited, no duplicate keys, every figure used and committed). |

For the test suite itself, see [`tests/README.md`](../tests/README.md).
