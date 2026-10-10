# analyses/

One template, regenerated per maze. Edit one notebook, rerun per maze, every maze's run updates consistently.

Both committed notebooks are executed, so their outputs can be read without the data: [`per_maze.ipynb`](per_maze.ipynb) holds the complete run for the **Linear Maze** (`More_Water`) with seed 42, and [`cross_maze.ipynb`](cross_maze.ipynb) holds the full comparison across all mazes and seeds reported in the paper. They are copies of the rendered notebooks under `artifacts/`, with absolute local paths removed.

## How it works

`run_mazes.py` uses papermill to inject `TARGET_PARADIGM` and `SEED` into `per_maze.ipynb` and execute it once per (seed, maze). The template runs every usable session of every mouse in the maze as one session-by-session chain, and every seed reruns the whole chain. Each run writes a self-contained bundle (models, metrics, per-window arrays, and the rendered notebook) under `artifacts/<paradigm_slug>/ALL_MICE/seed_<k>/`. Runs go seed by seed, so a partial sweep already covers every maze, and a (seed, maze) whose bundle exists is skipped, so an interrupted sweep resumes where it stopped. `cross_maze.ipynb` then reads every seed's bundle and reports each result as the mean ± standard deviation across seeds.

| Path | Role | Tracked? |
|------|------|----------|
| `per_maze.ipynb` | the whole pipeline (executed: Linear Maze, seed 42) | yes |
| `run_mazes.py` | papermill driver, swept over seeds | yes |
| `cross_maze.ipynb` | reads every seed's bundle, compares the mazes, writes the paper figures | yes (executed) |
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
