# Models

The `src.models` package implements the two-step decoder.

**Step 1** is the BC-VAE encoder, `SharedTemporalClassifier`. It encodes a 1 s window of one session's neurons into a Gaussian posterior over a 16-dimensional latent. It is trained to reconstruct the window's spikes (Poisson likelihood), to classify its region, and to keep the posterior close to the prior (KL term). Only its input projection and read-out are per session, so every session lands in one shared latent space. `add_session` creates them when a session first arrives.

**Step 2** is the `AlignmentDecoder`. With the encoder frozen, it reads the posterior means of a window and the four windows before it, and predicts the region. Its training adds a contrastive term that pulls same-region windows together, across sessions most strongly.

`training` holds both training loops, with early stopping and a best-checkpoint restore. `losses` holds the Poisson likelihood and the contrastive loss, and `latents` runs the trained models over a session's windows. The conformal layer derives all four of its uncertainty signals from these models' outputs and gradients (see [`src.conformal`](conformal.md)).

## Encoder

::: src.models.encoder

## Alignment decoder

::: src.models.alignment

## Losses

::: src.models.losses

## Training

::: src.models.training

## Latent extraction

::: src.models.latents

## Hyperparameters

The defaults of the model constructors and training loops. The production run overrides some of them from [`src.utils.production_config`](utils.md#production-recipe): pass values as arguments rather than editing this module.

::: src.models.config
    options:
      show_if_no_docstring: true
