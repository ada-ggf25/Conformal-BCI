PAPER := paper

.PHONY: setup setup-all setup-python setup-docs setup-latex \
	paper-test \
	data-test utils-test models-test conformal-test chain-test \
	src-test docs-test \
	mkdocs-test mkdocs-build mkdocs-serve \
	db-test python-lint python-typecheck ci-test \
	maze maze-quick cross-maze

# First-time setup, in three tiers.
#
#   setup-python    python-checks.yml, src-tests.yml, mkdocs.yml
#   setup-docs      docs.yml
#   setup-latex     paper-checks.yml
#
#   make setup      the Python environment only. Run the code, its unit
#                   tests, the linters and the MkDocs build.
#   make setup-all  every tier.
setup: setup-python

setup-all: setup-python setup-docs setup-latex

setup-python:
	uv sync

setup-docs: setup-python
	sudo apt-get update
	sudo apt-get install -y --no-install-recommends \
		aspell \
		aspell-en

setup-latex: setup-docs
	sudo apt-get install -y --no-install-recommends \
		cm-super \
		latexmk \
		lmodern \
		texlive-bibtex-extra \
		texlive-extra-utils \
		texlive-fonts-recommended \
		texlive-latex-base \
		texlive-latex-extra \
		texlive-latex-recommended \
		texlive-pictures \
		texlive-science

# Paper LaTeX checks.
paper-test:
	$(MAKE) -C $(PAPER) ci-test

# Documentation tests (lint + spelling over README files, WORKFLOWS.md and docs/).
docs-test:
	uv run pytest tests/docs/ -v

# MkDocs site: CI build check, local static build, live-reload server.
mkdocs-test:
	uv run mkdocs build --strict

mkdocs-build: mkdocs-test

mkdocs-serve:
	uv run mkdocs serve

# Source unit tests.
src-test:
	uv run pytest tests/data/ tests/utils/ tests/models/ \
	  tests/conformal/ tests/chain/ tests/db/ -v -m "not integration"

data-test:
	uv run pytest tests/data/ -v

utils-test:
	uv run pytest tests/utils/ -v

models-test:
	uv run pytest tests/models/ -v

conformal-test:
	uv run pytest tests/conformal/ -v

chain-test:
	uv run pytest tests/chain/ -v

db-test:
	uv run pytest tests/db/ -v -m "not integration"

# Python linting and type checking.
python-lint:
	uv run ruff check src/ tests/ analyses/
	uv run ruff format --check src/ tests/ analyses/

python-typecheck:
	uv run mypy src/ tests/data tests/utils tests/models tests/conformal tests/chain tests/db tests/docs analyses/

# Experimental analysis run.
# These need .env + a live MongoDB connection and train models:
#   make maze                          — train the pooled model for all 3 mazes x 5 seeds
#   make maze PARADIGM=Y_Maze SEEDS=42 — one maze, one seed
#   make maze-quick                    — 2-epoch smoke run of the plumbing (one seed)
#   make cross-maze                    — compare every maze that has a saved results bundle
# per_maze.ipynb pools every usable mouse in a paradigm into one model and writes one bundle
# per seed to analyses/artifacts/<Paradigm>/ALL_MICE/seed_<k>/. Finished bundles are skipped,
# so an interrupted sweep resumes. cross_maze.ipynb reads every seed's bundle and reports
# mean +- sd across seeds to analyses/artifacts/cross_maze.ipynb.
PARADIGM ?=
SEEDS ?=
_paradigm_name = $(if $(PARADIGM),$(PARADIGM),More_Water Y_Maze Nested_All)
_seed_args = $(if $(SEEDS),--seeds $(SEEDS),)

maze:
	uv run python analyses/run_mazes.py --paradigm $(_paradigm_name) $(_seed_args)

maze-quick:
	uv run python analyses/run_mazes.py --quick --paradigm $(_paradigm_name) $(_seed_args)

cross-maze:
	uv run python -m nbconvert --to notebook --execute analyses/cross_maze.ipynb \
	  --output-dir analyses/artifacts --output cross_maze.ipynb

# Full CI test suite.
ci-test:
	$(MAKE) python-lint
	$(MAKE) python-typecheck
	$(MAKE) src-test
	$(MAKE) docs-test
	$(MAKE) mkdocs-test
	$(MAKE) paper-test
