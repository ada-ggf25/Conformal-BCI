"""The pooled per-maze pipeline's tuned-and-frozen production recipe.

Adapted from the Liu Lab's cross-session alignment codebase: ``Step 1: BC-VAE encoder``,
``Step 2: alignment decoder`` and ``windowing and labels`` follow the lab's implementations.
Original to this project: ``conformal learned scale sigma(x)``.
"""

# Step 1: BC-VAE encoder
STEP1_REGION_WEIGHT = 0.2
"""Weight on the Step-1 region CrossEntropy term."""

STEP1_KL_WEIGHT = 0.5
"""Weight on the Step-1 KL term."""

STEP1_KL_FREE_BITS = 0.0
"""Per-dimension KL floor in nats (0 = off), guards against ignored latent dims."""

STEP1_CLASS_WEIGHT_POWER = 1.0
"""Exponent on the inverse-frequency class weights (1.0 = full)."""

STEP1_WEIGHT_DECAY = 1e-2
"""AdamW L2 on the region-classifier head only, to curb its overfitting."""

# Step 2: alignment decoder
STEP2_CONTRASTIVE_WEIGHT = 2.0
"""Weight on the NT-Xent contrastive term."""

STEP2_DROPOUT = 0.1
"""Dropout in the alignment Transformer. The classifier head keeps its own fixed rates."""

STEP2_CLASS_WEIGHT_POWER = 1.0
"""Exponent on the inverse-frequency class weights."""

# windowing and labels
LABEL_REDUCE = "mode"
"""Per-window region label: the modal bin, less boundary noise than the window's last."""

DROP_STRADDLE_WINDOWS = True
"""Drop windows that straddle a region boundary."""

# conformal learned scale sigma(x)
MLP_HIDDEN = 32
"""Width of each hidden layer in the scale MLP."""

MLP_DEPTH = 2
"""Number of hidden layers in the scale MLP."""

MLP_TAU = 0.05
"""Sigmoid temperature for soft set membership (smaller = closer to the true objective)."""

MLP_KAPPA = 20.0
"""Weight on the coverage shortfall in the relaxed loss."""

MLP_SIZE_WEIGHT = 1.0
"""Weight on the sharpness term relative to the coverage term."""
