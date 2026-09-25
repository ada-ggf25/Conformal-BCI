# analyses/

One template, regenerated per maze. Edit one notebook, rerun per maze, every maze's run updates consistently.

For reproducibility purposes, [`analyses/per_maze.ipynb`](analyses/per_maze.ipynb) contains a complete run for the **Linear Maze** (**More_Water**), while [`analyses/cross_maze.ipynb`](analyses/cross_maze.ipynb) contains the full comparison across all mazes used in this study.

## How it works

`run_mazes.py` uses papermill to inject `TARGET_PARADIGM` into `per_maze.ipynb` and execute it once per maze. The template pools every usable mouse in the maze into one pipeline. Each maze gets a single self-contained bundle of results under `artifacts/<paradigm_slug>/ALL_MICE/` with the models, metrics and with the rendered notebook written at `artifacts/<paradigm_slug>/per_maze.ipynb`. `cross_maze.ipynb` then reads every maze's saved bundle and compares them.

| Path | Role | Tracked? |
|------|------|----------|
| `per_maze.ipynb` | the whole pipeline | yes |
| `run_mazes.py` | papermill driver | yes |
| `cross_maze.ipynb` | reads every maze's saved bundle and compares results | yes |
| `artifacts/<paradigm_slug>/per_maze.ipynb` | the rendered notebook papermill writes | no (duplicate heavy regenerable) |
| `artifacts/<paradigm_slug>/ALL_MICE/` | the saved results bundle the notebook's own save cell writes | no (heavy regenerable) |

## Usage

```bash
uv sync                                                          # installs papermill
make maze                                                        # pooled model, all 3 analyzed mazes (default)
make maze PARADIGM=Y_Maze                                        # pooled model, a single maze
uv run python analyses/run_mazes.py --paradigm More_Water Y_Maze # several mazes, in order
uv run python analyses/run_mazes.py --quick                      # 2/2-epoch smoke test

make cross-maze                                                  # compare every maze using the saved bundles
```
