# Chain

The `src.chain` package trains one model per maze session by session: every usable session of the maze, all mice, in calendar order. Each session fine-tunes the model carried from the previous one on its training block, calibrates on its calibration block, and is tested on its test block; the first sessions are a warm-up that is never tested.

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
