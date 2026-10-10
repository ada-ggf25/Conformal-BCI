"""Hyperparameters for the BC-VAE encoder and Transformer alignment decoder."""

# Model dimensions
LATENT_DIM = 16
"""BC-VAE latent dimension (mu / logvar)."""

SHARED_DIM = 64
"""Per-neuron projection output dimension."""

FACTOR_DIM = 32
"""Generator factor dimension (intermediate for spike reconstruction)."""

HIDDEN_DIM = 128
"""BiGRU hidden dimension."""

# Step 1 loss weights (BC-VAE)
RECON_WEIGHT = 1.0
"""Poisson NLL reconstruction weight."""

KL_WEIGHT = 0.1
"""KL divergence weight. A ``train_step1`` fallback only: the session chain passes
``STEP1_KL_WEIGHT`` from ``utils/production_config.py``."""

KL_WARMUP_EPOCHS = 10
"""Epochs over which the KL weight ramps linearly to its ceiling. Early stopping only starts
once the ramp is done, since validation losses under a changing KL weight are not comparable."""

REGION_WEIGHT = 1.0
"""Region classification CrossEntropy weight. A ``train_step1`` fallback only: the
session chain passes ``STEP1_REGION_WEIGHT`` from ``utils/production_config.py``."""

# Step 2 (Transformer alignment)
TEMPERATURE = 0.1
"""NT-Xent contrastive loss temperature."""

STEP2_SEQ_LEN = 5
"""Sliding window length (consecutive latents fed to the Transformer)."""

# Training
BATCH_SIZE = 256
"""DataLoader batch size for both training steps."""

STEP1_LEARNING_RATE = 1e-3
"""Adam learning rate for the BC-VAE encoder."""

STEP2_LEARNING_RATE = 3e-4
"""Adam learning rate for the alignment decoder."""

MAIN_EARLY_STOP_PATIENCE = 20
"""Epochs without validation improvement before stopping."""

MAIN_EARLY_STOP_MIN_DELTA = 1e-4
"""Minimum improvement to reset the patience counter."""
