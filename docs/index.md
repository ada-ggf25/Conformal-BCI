# Uncertainty-Aware ML for Neural BCIs

This site documents the code of the [ML-Decoding repository](https://github.com/ada-ggf25/ML-Decoding) for the paper *Keeping Neural Decoders Calibrated Session by Session with Learned Class-Conditional Conformal Prediction*, by Guilherme Grancho Duarte Fernandes, Hao Shen, Tianshu Tan, Tom Davison and Jia Liu (Harvard John A. Paulson School of Engineering and Applied Sciences and Imperial College London).

---

Neural drift degrades brain–computer interface decoders over time, and at inference time the decoder's predictions arrive without labels, so that decline cannot be measured as it happens. This project studies a label-free reliability measure. A decoder predicts which maze region a mouse occupies from its neural activity, and a learned conformal layer, calibrated on labeled data recorded earlier in the session, turns each prediction into a *prediction set*: a set of plausible regions, sized to contain the true one 90% of the time. The size of the set then measures the decoder's reliability at inference time, without new labels.

The data are electrophysiological recordings from flexible brain probes in mice navigating virtual-reality mazes. This site documents the `src` package that implements the method.

## What the package does

For each maze, one model is carried across every usable session of every mouse, in calendar order:

1. **Data.** Spikes and positions are read from MongoDB, binned at 0.1 s, cut into 1 s windows, and each window is labeled with the maze region the animal was in.
2. **Decoder.** A behavior-constrained variational autoencoder (BC-VAE) encodes each window into a latent. A Transformer alignment decoder reads a short sequence of latents and predicts the region.
3. **Uncertainty signals.** Four per-window signals describe how uncertain the pipeline is. They are the prediction's entropy, the encoder's posterior variance, and the sensitivity (Jacobian norm) of the encoder and of the alignment map.
4. **Conformal layer.** A small network turns the signals into a scale \(\sigma_\theta(x)\) that loosens or tightens each window's threshold. Per-region thresholds calibrated on held-out windows then produce the prediction set.
5. **Session chain.** Each session fine-tunes the carried model on its own first 60%, calibrates on the next 30%, and is tested on its final 10%.

[The method](pipeline.md) walks through every step with its equations and settings. The [API reference](api/index.md) documents every module, class and function.

## Quick start

The package is installed in editable mode with [`uv`](https://docs.astral.sh/uv/) and imported as `src`:

```bash
make setup          # create the Python environment (uv sync)
cp .env.example .env  # then fill in MONGODB_URI, MONGODB_DATABASE and, if needed, SSH_*
```

Runs read live data, so they need the database credentials and network access to it:

```bash
make maze-quick                     # 2-epoch smoke run of the whole pipeline, one seed
make maze                           # every maze x seeds 42-46, resumable
make maze PARADIGM=Y_Maze SEEDS=42  # one maze, one seed
make cross-maze                     # compare the saved bundles across mazes and seeds
```

The notebooks in `analyses/` orchestrate these runs. The reusable code they import lives in `src/`, which this site documents. The unit tests run offline, with the database client mocked:

```bash
make src-test       # every src/ unit test
make mkdocs-serve   # preview this site locally
```

## Conventions

- **Sessions** are named `M<mouse id>_<YYMMDD>`, e.g. `M19_240918`. The name alone gives the mouse and the recording date, and the chain is ordered by it.
- **Imports** go to the defining module, e.g. `from src.conformal.calibrate import compute_quantile`. Package `__init__` files hold a docstring and nothing else.
- **Shapes** in docstrings follow NumPy order: `(n_windows, T, n_neurons)` for windows, where `T` is the number of bins per window (10).
- **Constants** that describe the data live in `src.data.config`, model hyperparameters in `src.models.config`, and the frozen production recipe in `src.utils.production_config`.

## Where to go next

<div class="grid cards" markdown>

- **[The method](pipeline.md)**: the protocol end to end: windows, the two-step decoder, the signals, the conformal layer, and what each tested session reports.
- **[API reference](api/index.md)**: every package and module, with the data structures they pass between each other.

</div>
