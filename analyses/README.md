# analyses/

One template, regenerated per maze. Edit one notebook, rerun per maze, every maze's run updates consistently.

For reproducibility purposes, [`analyses/per_maze.ipynb`](analyses/per_maze.ipynb) contains a complete run for the **Linear Maze** (**More_Water**), while [`analyses/cross_maze.ipynb`](analyses/cross_maze.ipynb) contains the full comparison across all mazes used in this study.

## How it works

`run_mazes.py` uses papermill to inject `TARGET_PARADIGM` and `SEED` into `per_maze.ipynb` and execute it once per (seed, maze). The template pools every usable mouse in the maze into one pipeline, and every seed reruns the whole pipeline. Each run writes a self-contained bundle (models, metrics, per-window arrays, and the rendered notebook) under `artifacts/<paradigm_slug>/ALL_MICE/seed_<k>/`. Runs go seed by seed, so a partial sweep already covers every maze, and a (seed, maze) whose bundle exists is skipped, so an interrupted sweep resumes where it stopped. `cross_maze.ipynb` then reads every seed's bundle and reports each result as the mean ± standard deviation across seeds.

| Path | Role | Tracked? |
|------|------|----------|
| `per_maze.ipynb` | the whole pipeline | yes |
| `run_mazes.py` | papermill driver, swept over seeds | yes |
| `cross_maze.ipynb` | reads every seed's bundle and compares the mazes | yes |
| `artifacts/<paradigm_slug>/ALL_MICE/seed_<k>/` | one run's bundle and its rendered notebook | no (heavy regenerable) |
| `artifacts/_quick/` | smoke-run bundles (`--quick`), never read by `cross_maze.ipynb` | no |

## Usage

```bash
uv sync                                                          # installs papermill
make maze-quick                                                  # 2/2-epoch smoke test, one seed
make maze                                                        # all 3 mazes x seeds 42-46 (resumable)
make maze PARADIGM=Nested_All SEEDS=42                           # one maze, one seed (a single-seed check)
uv run python analyses/run_mazes.py --paradigm Y_Maze --seeds 42 43 --force  # rerun finished runs

make cross-maze                                                  # compare every maze using the saved bundles
```
