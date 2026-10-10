STEM      := paper
REPO_ROOT := ..
TESTS     := tests/paper

PYTHON := uv run python
PYTEST  = cd $(REPO_ROOT) && $(PYTHON) -m pytest -vv

.PHONY: all strict latex-test spelling-test hygiene-test ci-test clean

# .latexmkrc sets $aux_dir and $out_dir to build/, so the PDF is build/paper.pdf.
all:
	latexmk -jobname=$(STEM) -interaction=nonstopmode -halt-on-error main.tex

strict:
	latexmk -gg -Werror -emulate-aux-dir -jobname=$(STEM) -interaction=nonstopmode -halt-on-error main.tex

latex-test:
	$(PYTEST) $(TESTS)/test_paper_latex.py

spelling-test:
	$(PYTEST) $(TESTS)/test_paper_spelling.py

hygiene-test:
	$(PYTEST) $(TESTS)/test_paper_hygiene.py

ci-test:
	$(MAKE) latex-test
	$(MAKE) spelling-test
	$(MAKE) hygiene-test

# Remove all build artifacts, including the PDF.
clean:
	latexmk -CA -jobname=$(STEM) main.tex
	rm -rf build/
