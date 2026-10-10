# .github

This directory contains the GitHub Actions workflows.

## Workflow Overview

Every workflow runs through `uv` and delegates to a Makefile target. All checks run on pull requests only (path-filtered, with superseded runs canceled) and can also be started manually. The one exception is `docs-deploy.yml`, which publishes the documentation when a pull request is merged into `main`: GitHub Pages only accepts deployments from the default branch, so it runs on the resulting push.

| Workflow | Trigger | What it checks |
|---|---|---|
| `src-tests.yml` | PR to `dev` or `main` | Unit tests for `src/data/`, `src/utils/`, `src/models/`, `src/conformal/`, `src/chain/` and `src/db/`, plus the `analyses/run_mazes.py` sweep bookkeeping (`make src-test`). |
| `python-checks.yml` | PR to `dev` or `main` | Ruff lint + format and mypy type checks over `src/`, `tests/` and `analyses/`. |
| `docs.yml` | PR to `dev` or `main` | Markdown lint and American English spelling over the `README.md` files, this `WORKFLOWS.md` file, and every MkDocs page under `docs/`. |
| `mkdocs.yml` | PR to `dev` or `main` | Verifies the site builds with `mkdocs build --strict` (`make mkdocs-test`). |
| `docs-deploy.yml` | push to `main` (a merged PR) | Builds the MkDocs site the same way and publishes it to GitHub Pages at <https://ada-ggf25.github.io/ML-Decoding/>. |
| `paper-checks.yml` | PR to `main` | Compiles the paper LaTeX under `paper/` with warnings as errors, checks American English spelling of `main.tex` and the files it inputs, and bibliography and figure hygiene (every entry cited, no duplicate keys, every figure used and committed). |

For the test suite itself, see [`tests/README.md`](../tests/README.md).
