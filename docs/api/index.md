# API Reference

The `src` package is organized into five subpackages, each covering a distinct layer of the pipeline.

| Subpackage | Description |
|---|---|
| [`src.models`](models.md) | BC-VAE encoder, Transformer alignment decoder, training loops, losses, decoder metrics |
| [`src.data`](data.md) | Neural and position extraction, windowing, maze-region labeling, PyTorch datasets and collation, recording geometry |
| [`src.db`](db.md) | MongoDB client, loaders, session ID utilities |
| [`src.utils`](utils.md) | Session discovery, pooled-cohort discovery, pipeline assembly, chronological within-session splitting, the production recipe, bundle I/O |
| [`src.conformal`](conformal.md) | The learned scale sigma(x), class-conditional calibration and prediction sets, the per-window uncertainty signals, and conformal performance |
