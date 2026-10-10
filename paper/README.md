# Paper

LaTeX source of the paper *Keeping Neural Decoders Calibrated Session by Session with Learned Class-Conditional Conformal Prediction*, an arXiv preprint. It uses the official NeurIPS 2026 style (`neurips_2026.sty`, unmodified from the NeurIPS author kit) in preprint mode. Two small overrides in `main.tex` rename the acknowledgments heading and remove the first-page preprint notice, so the style file itself stays untouched.

The style mode is set in the `\usepackage{neurips_2026}` line of `main.tex`: no option gives the anonymous submission (line numbers, author hidden, acknowledgments hidden), `[preprint]` a non-anonymous preprint, and `[main, final]` the camera-ready version, which is only for accepted papers.

## Structure

```text
paper/
├── main.tex          # manuscript source
├── references.bib    # bibliography
├── figures/          # figures (written here by analyses/cross_maze.ipynb, except the setup photograph)
├── neurips_2026.sty  # official NeurIPS 2026 style file (do not edit)
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

`make paper-test` from the repository root runs the LaTeX compile, American English spelling of `main.tex` and every file it inputs, and bibliography and figure hygiene: every `references.bib` entry is cited and defined once, and every image in `figures/` is used by the paper and committed. The same suite runs in CI as the `paper-checks` workflow on pull requests to `main`. The paper is an arXiv preprint, so no word or page limit is checked.
