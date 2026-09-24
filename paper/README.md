# Paper

LaTeX source of the paper. It starts from the master's thesis report and is being refactored into the manuscript. It uses the official NeurIPS 2026 style (`neurips_2026.sty` and `checklist.tex`, unmodified from the NeurIPS author kit).

The style mode is set in the `\usepackage{neurips_2026}` line of `main.tex`: no option gives the anonymous submission (line numbers, author hidden, acknowledgments hidden), `[preprint]` a non-anonymous preprint, and `[main, final]` the camera-ready version, which is only for accepted papers.

## Structure

```text
paper/
├── main.tex          # manuscript source
├── references.bib    # bibliography
├── figures/          # figures (the report figures are written here by analyses/cross_maze.ipynb)
├── neurips_2026.sty  # official NeurIPS 2026 style file (do not edit)
├── checklist.tex     # NeurIPS paper checklist, required at submission; answers still TODO
├── paper.wordlist    # accepted words for the spelling check
├── Makefile          # build and check targets
└── build/            # latexmk output, including build/paper.pdf (not committed)
```

## Building

```bash
make -C paper all      # build/paper.pdf
make -C paper strict   # same, with LaTeX warnings treated as errors
make -C paper clean    # remove build/
```

## Checks

`make paper-test` from the repository root runs the LaTeX compile, word and float limits, and American English spelling checks. The same suite runs in CI as the `paper-checks` workflow on pull requests to `main`. The word and float limits are still the thesis limits (5,000 words, 10 figures and tables). NeurIPS instead limits the main text to nine pages; references, acknowledgments, the checklist, and appendices do not count.
