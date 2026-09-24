# src

Source package for the scientific code. All research modules live here and are importable as `from src.<package>.<module> import ...` after `uv sync`.

## Structure

```text
src/
├── data/        # Datasets, neuron mapping, region labeling, data constants
├── db/          # MongoDB connection
├── models/      # BC-VAE encoder, Transformer alignment decoder, training loops, decoder metrics
├── utils/       # data discovery, chronological splits, pipeline assembly, results bundles
└── conformal/   # Learned class-conditional split-conformal layer
```

## CI tests

```bash
make python-lint   # ruff lint + format check over src/, tests/ and analyses/ (both .py and .ipynb)
make src-test       # all five suites below, in one pytest process (what CI runs)
make data-test      # tests/data/
make utils-test     # tests/utils/
make models-test    # tests/models/
make conformal-test # tests/conformal/
make db-test        # tests/db/ — offline, the MongoDB client is mocked
```
