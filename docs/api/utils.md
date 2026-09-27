# Utilities

The `src.utils` package sits at both ends of a run: it assembles the data before training and writes and reads the results after.

**Before training.** `discovery` finds a mouse's sessions and picks the paradigm it is analyzed under, and `cohort` pools every usable mouse of a maze into one cohort. `pipeline_prep` then assembles each session (extraction, windows, labels, filtering), normalizes it, and checks its split, which `splits` cuts chronologically over task windows. `production_config` is the frozen recipe the production run uses.

**After training.** `artifacts` builds the per-run bundle (`metrics.json`, `arrays.npz`, `sigma_nets.pt`) from the chain's rows, and `bundle_readers` loads bundles back and summarizes them per maze and across seeds. The readers are what `analyses/cross_maze.ipynb` plots, so a figure and the table beside it always read the same number.

## Session discovery

::: src.utils.discovery

## Cohort discovery

::: src.utils.cohort

## Pipeline preparation

::: src.utils.pipeline_prep

## Chronological split

::: src.utils.splits

## Bundle layout and writing

::: src.utils.artifacts

## Bundle readers

::: src.utils.bundle_readers

## Production recipe

The tuned-and-frozen settings of the production run. They override the model defaults in [`src.models.config`](models.md#hyperparameters).

::: src.utils.production_config
    options:
      show_if_no_docstring: true
