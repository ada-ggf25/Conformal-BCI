# Data

The `src.data` package turns one session's raw trials into model-ready arrays and wraps those arrays as PyTorch datasets.

`hybrid_io` does the extraction. `build_hybrid_neuron_mapping` fixes every session's column layout, and `extract_neural_and_position_hybrid` bins spikes and positions onto a common 0.1 s axis. `create_windows_with_regions` cuts that axis into 1 s windows and labels each window with a maze region, while `window_task_mask` marks the windows that lie inside a task trial. The mask and the extraction skip exactly the same trials, so the mask lines up with the windows one for one. `regions` holds the per-maze geometry that maps a position to a region, and `datasets` feeds Step 1 (single windows grouped by session) and Step 2 (short sequences of latents).

The whole-session assembly that calls these functions, with its filtering and the chronological split, lives in [`src.utils.pipeline_prep`](utils.md#pipeline-preparation).

## Extraction and windowing

::: src.data.hybrid_io

## Maze regions

::: src.data.regions

## Datasets

::: src.data.datasets

## Recording constants

These constants describe the data, not the model. `WINDOW_SIZE * BIN_WIDTH_SEC` is the time one window spans, 1 s.

::: src.data.config
    options:
      show_if_no_docstring: true
