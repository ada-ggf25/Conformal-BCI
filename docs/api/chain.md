# Chain

The `src.chain` package runs the session chain. One model per maze is carried through every usable session of the maze, across all mice, in calendar order. Each session fine-tunes the model carried from the previous one on its training block, calibrates on its calibration block, and is tested on its test block. The first sessions are a warm-up that is trained on but never tested.

The modules follow a run's order:

1. `order` sorts the sessions by date, re-splits the warm-up sessions, and then z-scores every session on its own training block (`prepare_chain`).
2. `stages` holds `ChainModel`, the carried encoder, alignment decoder and learned scale. It fine-tunes all three on a session (`fit`), and scores any block of windows.
3. `pool` picks each region's threshold from the session's own calibration block, borrowing from earlier sessions only for a region with too few windows.
4. `evaluate` applies the thresholds to a test block and records coverage, set size, error and threshold level per window.
5. `runner` drives the loop. It evaluates the deployed method and its comparisons, runs the coverage sweep, and checkpoints after every session so that an interrupted run resumes where it stopped.

A session is only ever evaluated by the model right after its own fine-tune, so no result depends on a session that comes later in the chain. [The method](../pipeline.md#6-what-each-tested-session-reports) lists every method a tested session reports.

## Chain order and warm-up

::: src.chain.order

## The carried model

::: src.chain.stages

## Per-region thresholds

::: src.chain.pool

## Test-block evaluation

::: src.chain.evaluate

## Running a chain

::: src.chain.runner
