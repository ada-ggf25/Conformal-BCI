# Uncertainty-Aware Machine Learning for Long-Term Neural Recordings in Flexible Brain–Computer Interfaces

<div align="justify">

A major challenge for the lasting reliability of brain-computer interfaces in real world conditions is neural drift, which degrades decoder performance and makes them unreliable over time. To evaluate the decline in accuracy, it's necessary to have recent labeled data, which is expensive and impractical in real life settings. This paper studies a learned class-conditional split-conformal layer to measure uncertainty in neural decoding models, without requiring new labeled data. A behavior-constrained variational autoencoder and a Transformer alignment decoder predict the maze region occupied by a mouse, then the conformal layer outputs a trust region of plausible maze regions. The approach was tested using electrophysiological data from flexible brain probes in 13 mice as they went through 3 virtual-reality mazes. For a targeted 90\% trust region accuracy, the method achieved 81.4\% to 86.5\% test accuracy when constructing trust regions that included the true maze region. The trust region size correlated positively with decoder error in two of the three mazes and negatively in the third. These results show that the proposed model gives a signal for decoding uncertainty, which is promising for long-term reliability in brain-computer interfaces. Nevertheless, a more thorough comparison with different methods and additional decoding models is necessary before broad implementation.

## Authors

| Author | Affiliations |
|---|---|
| Guilherme Grancho | Imperial College London; Liu Lab, Harvard John A. Paulson School of Engineering and Applied Sciences |

<!-- TODO: finalize the author list and affiliations for the paper. -->

## Method

The method starts from virtual-reality navigation sessions recorded with flexible neural probes.

<p align="center">
  <img src="docs/readme-images/set-up.jpeg" alt="Data collection setup" width="760">
</p>

Figure 1: Photograph of the Liu Lab's setup, which includes a mouse mounted on an omnidirectional ball, 6 monitors showing the maze environment, a flexible brain probe implanted in the animal, and an automated water delivery system.

The recorded spike data is converted into short spike-count windows. Each window is the input unit used by the encoder-decoder pipeline.

<p align="center">
  <img src="docs/readme-images/encoder-input-example.png" alt="Example neural input" width="900">
</p>

Figure 2: The first 30 seconds of a Linear Maze session from Mouse 19 on 2024-09-18. The highlighted region corresponds to a 1 s interval input for the BC-VAE.

Each input window is paired with a discrete maze-region label derived from the animal's position in one of the three maze layouts.

<p align="center">
  <img src="docs/readme-images/maze-geometry.png" alt="The three maze layouts with discrete region labels" width="980">
</p>

Figure 3: This schematic shows the 3 maze layouts with discrete region labels. The black dot marks the start of the trial.

The labeled windows then feed the train/calibration/test split, the BC-VAE and Transformer decoder, and the learned conformal layer.

<p align="center">
  <img src="docs/readme-images/pipeline-image.png" alt="Protocol summary" width="980">
</p>

Figure 4: Data split protocol and full conformal pipeline. Top: Sessions 80/10/10 split. The first block, aggregated across all sessions, fits the BC-VAE, the Transformer decoder, and the MLP. The middle validation/calibration block, aggregated across all sessions, supports early stopping and fixes the final thresholds $\hat q_k$ per region. The final test block, aggregated across all sessions, is held out for later evaluation. Bottom: The encoding-decoding pipeline maps each spike-count window to maze region probabilities $\hat p(x)$. The four uncertainty signals come together as $u(x)$, which is passed to the conformal MLP, which then outputs the scale $\sigma_\theta(x)$. Scores $S(x,k)$ are compared with the fixed thresholds $\hat q_k$ per region to produce the trust region $\widehat{\mathcal{C}}(x)$. The diagram was crafted using the [`TikZMaker editor`](https://tikzmaker.com/editor).

## Results

The results are reported at the maze level first, using the test set. Coverage is evaluated against the 90% coverage target, while sharpness is summarized by the mean trust-region size normalized by the number of maze regions.

| Maze | Mice | Sessions | Regions | Decoder Accuracy | Coverage | \|Ĉ\|/K |
|---|---|---|---|---|---|---|
| Linear Maze | 13 | 95 | 8 | 0.374 | 0.856 | 0.467 |
| Y Maze | 7 | 23 | 8 | 0.290 | 0.814 | 0.745 |
| Loop Maze | 2 | 14 | 9 | 0.270 | 0.865 | 0.712 |

Table 1: Summary of decoder and conformal performance. $K$ is the total number of maze regions. Decoder accuracy is the proportion of windows in the test set for which the decoder’s top probability predicted the true region. Coverage is the fraction of test windows for which the trust region contains the true maze region at the 90\% coverage target. The final column reports the mean trust region size normalized by $K$.

The results at session level:

<p align="center">
  <img src="docs/readme-images/coverage-drift-tax.png" alt="Calibration cost under the chronological split-conformal protocol" width="980">
</p>

Figure 5: Calibration performance on test data. Left: Empirical test coverage per maze against the 90\% target, with sessions as points displaying the spread per maze. Right: Achieved test coverage after retraining the conformal layer for 20 coverage targets, equally distributed from 1\% to 99\%. The gap below the diagonal shows the empirical calibration cost between the validation set and the later test data.

The next view shows the coverage-sharpness trade-off directly: pushing for higher achieved coverage generally requires larger trust regions.

<p align="center">
  <img src="docs/readme-images/coverage-sharpness-frontier.png" alt="Coverage-sharpness frontier across all three mazes" width="820">
</p>

Figure 6: Coverage-sharpness trade-off. Achieved test coverage is shown on the horizontal axis, and mean trust region size normalized by total maze region count is shown on the vertical axis. Each curve retrains the conformal layer from scratch for 20 coverage targets, equally distributed from 1\% to 99\%. The star marks the deployed 90\% target models reported in Table 1.

Because maze-region labels are imbalanced, aggregate coverage alone is not enough. The per-region analysis checks whether coverage and trust-region size vary across common and rare regions.

<p align="center">
  <img src="docs/readme-images/per-region-coverage.png" alt="Per-region conformal performance and label distribution" width="980">
</p>

Figure 7: Conformal performance per region and label distribution. Each subplot corresponds to one maze. For inputs whose true label is the corresponding maze region, red bars show empirical test coverage, and green bars show the mean trust region size normalized by the number of maze regions. Blue bars show the region's share of data points for the entire dataset. The dashed line marks the 90\% coverage target. Region 5 in the Loop Maze has no test coverage or trust region size estimate because no test data points were present in that region.

The next result checks whether the conformal layer behaves like a useful uncertainty signal. Positive correlations indicate that decoder errors tend to occur when the trust regions widen or when the learned MLP scale increases.

| Maze | corr(decoder error, `\|C\|/K`) | corr(decoder error, `σθ(x)`) |
|---|---|---|
| Linear Maze | +0.86 | +0.99 |
| Y Maze | +0.61 | +0.68 |
| Loop Maze | -0.78 | -0.86 |

Table 2: Decoder error Pearson correlations over grouped test data points. The first correlation groups by normalized trust region size $|\widehat{\mathcal{C}}|/K$, one group per size level. The second groups by the MLP uncertainty scale $\sigma_\theta(x)$, in 10 equal-count bins. Each group contributes its error rate and is weighted by its number of data points. Positive values indicate that decoder errors tend to coincide with larger trust regions or larger MLP scales.

To interpret the learned scale, the last figure varies each uncertainty input and measures how strongly the MLP output responds.

<p align="center">
  <img src="docs/readme-images/signal-confound.png" alt="Sensitivity of the learned conformal scale to each uncertainty signal" width="820">
</p>

Figure 8: Sensitivity of the learned conformal scale to each uncertainty signal. Bars show the mean relative change of $\sigma_\theta(x)$, $|\partial \log \sigma_\theta(x) / \partial z|$, per standard deviation $z$ of each input uncertainty signal, averaged over the test set for each maze. The relative change does not depend on the overall level of $\sigma_\theta(x)$, so the mazes can be compared directly.

The paper's LaTeX source is in [`paper/`](paper/README.md).

## Repository Structure

```text
ML-Decoding/                 # repository root
├── Makefile                 # setup, test, documentation, and analysis targets
├── LICENSE                  # Apache License 2.0
├── NOTICE                   # copyright notice and what the license does not cover
├── pyproject.toml           # Python package metadata and tool dependencies
├── uv.lock                  # locked uv environment
├── paper/                   # LaTeX source of the paper
├── src/                     # Python research package installed by uv
│   ├── data/                # datasets, neuron mapping, region labeling, recording geometry
│   ├── db/                  # MongoDB access and offline-testable connection helpers
│   ├── models/              # BC-VAE, Transformer decoder, training, and decoder metrics
│   ├── conformal/           # learned class-conditional split-conformal layer
│   └── utils/               # discovery, chronological splits, pipeline assembly, bundles
├── analyses/                # pooled per-maze analysis notebooks and runner
│   ├── per_maze.ipynb       # pooled per-maze training and evaluation notebook
│   ├── cross_maze.ipynb     # cross-maze comparison and final table/figure generation
│   ├── run_mazes.py         # papermill driver for executing per_maze.ipynb
│   └── artifacts/           # generated model and result bundles, not committed (gitignored)
├── docs/                    # MkDocs source pages and API documentation
├── tests/                   # extended pytest suite and documentation checks
└── .github/                 # GitHub Actions workflows
```

## Setup

Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/) first. The Makefile uses it to create and run the Python environment, but it does not install `uv` itself.

Choose the setup target that matches what you want to run:

```bash
make setup-python    # python setup
make setup-docs      # documentation and linting
make setup-latex     # LateX setup

make setup           # alias for setup-python
make setup-all       # all of the above
```

More details about setup in [`Makefile`](Makefile).

To access the database, copy `.env.example` to `.env` and fill in `MONGODB_URI`, `MONGODB_DATABASE`, and the `SSH_*` variables if you need an SSH tunnel. A dump of the database restored into a local MongoDB server works the same way: point `MONGODB_URI` at it and leave `SSH_HOST` empty. Restore it with the same MongoDB major version the dump came from.

## Running Code

The reusable research code lives in `src/`. The notebooks in `analyses/` are orchestration layers: they import from `src/`, choose the cohort and maze, run training/evaluation, and save the results.

The main workflow is per-maze. `analyses/per_maze.ipynb` pools every usable mouse for one maze into a single BC-VAE -> Transformer decoder -> conformal pipeline. `analyses/run_mazes.py` runs that notebook through Papermill, injecting the target maze. Each run writes a bundle under `analyses/artifacts/<paradigm>/ALL_MICE/`.

After the per-maze bundles exist, `analyses/cross_maze.ipynb` reads them back and produces the cross-maze comparison tables and figures.

```bash
make maze-quick                   # 2-epoch smoke test of the whole pipeline, one seed
make maze                         # all analyzed mazes (More_Water, Y_Maze, Nested_All) x seeds 42-46
make maze PARADIGM=Y_Maze SEEDS=42  # one maze, one seed
make cross-maze                   # compare saved bundles, mean ± sd across seeds
```

Expected generated outputs:

```text
analyses/artifacts/<paradigm>/ALL_MICE/  # per-maze metrics, arrays, and models
analyses/artifacts/cross_maze.ipynb      # executed cross-maze comparison notebook
```

These runs need the `.env` database settings and a running MongoDB server, either the remote one or a local copy. More details are in [`analyses/README.md`](analyses/README.md) and [`src/README.md`](src/README.md).

For reproducibility purposes, [`analyses/per_maze.ipynb`](analyses/per_maze.ipynb) contains a complete run for the Linear Maze (More_Water), while [`analyses/cross_maze.ipynb`](analyses/cross_maze.ipynb) contains the full comparison across all mazes used in this study.

## Documentation

The project documentation is an MkDocs site built from `docs/` and configured by `mkdocs.yml`.

```bash
make mkdocs-serve   # preview the documentation locally
make mkdocs-test    # build the site in strict mode, as CI does
make docs-test      # lint and spell-check README files, WORKFLOWS.md, and docs/
```

## Running Tests

```bash
make ci-test   # every available test
```

See [`tests/README.md`](tests/README.md) for the detailed test layout. See [`Makefile`](Makefile) for the command definitions, and [`.github/WORKFLOWS.md`](.github/WORKFLOWS.md) for the CI workflow map.

## Data Availability

The data used in this paper is owned by the Liu Lab for Bioelectronics at the Harvard John A. Paulson School of Engineering and Applied Sciences. Database access instructions are available in [`.env.example`](.env.example), but access requires private credentials.

## Code Availability and Reproducibility

The code used to run the entire analyses is in this repository. A fixed random seed of 42 was used throughout. The per-maze training and evaluation runs are defined in [`analyses/per_maze.ipynb`](analyses/per_maze.ipynb) and executed by [`analyses/run_mazes.py`](analyses/run_mazes.py). Unless otherwise stated in their captions, the figures and tables in the report were generated with [`analyses/cross_maze.ipynb`](analyses/cross_maze.ipynb).

## Citation

<!-- TODO: add the BibTeX entry once the paper is available. -->

A citation entry will be added once the paper is published.

## License

The code is released under the [Apache License 2.0](LICENSE), copyright 2026 Liu Lab, Harvard John A. Paulson School of Engineering and Applied Sciences. The license does not cover the neural recordings, the NeurIPS style files in `paper/`, or the photograph of the recording setup; see [`NOTICE`](NOTICE).

## Acknowledgments

I thank Professor Jia Liu for the opportunity to conduct this project in the Liu Lab and for his guidance. I am grateful to Dr. Tom Davison for his supervision, and to Hao Shen and Tianshu Tan for their support. I also thank the rest of the Liu Lab members for their insightful discussions throughout the project.

</div>
