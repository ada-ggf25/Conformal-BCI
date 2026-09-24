STEM      := paper
REPO_ROOT := ..
TESTS     := tests/paper

PYTHON := uv run python
PYTEST  = cd $(REPO_ROOT) && $(PYTHON) -m pytest -vv

.PHONY: all strict latex-test limits-test limits spelling-test ci-test clean wordcount

# .latexmkrc sets $aux_dir and $out_dir to build/, so the PDF is build/paper.pdf.
all:
	latexmk -jobname=$(STEM) -interaction=nonstopmode -halt-on-error main.tex

strict:
	latexmk -gg -Werror -emulate-aux-dir -jobname=$(STEM) -interaction=nonstopmode -halt-on-error main.tex

latex-test:
	$(PYTEST) $(TESTS)/test_paper_latex.py

limits-test:
	$(PYTEST) $(TESTS)/test_paper_limits.py

limits: limits-test

spelling-test:
	$(PYTEST) $(TESTS)/test_paper_spelling.py

ci-test:
	$(MAKE) latex-test
	$(MAKE) limits-test
	$(MAKE) spelling-test

# Remove all build artifacts, including the PDF.
clean:
	latexmk -CA -jobname=$(STEM) main.tex
	rm -rf build/

# Word count using the same inclusion/exclusion rules as the CI test.
wordcount:
	cd $(REPO_ROOT) && $(PYTHON) $(TESTS)/test_paper_limits.py
