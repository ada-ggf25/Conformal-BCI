"""The pooled per-maze pipeline's tuned-and-frozen production recipe."""

# Step 1: BC-VAE encoder
STEP1_REGION_WEIGHT = 0.2
"""Weight on the Step-1 region CrossEntropy term."""

STEP1_KL_WEIGHT = 0.5
"""Weight on the Step-1 KL term."""

STEP1_KL_FREE_BITS = 0.0
"""Per-dimension KL floor in nats (0 = off), guards against ignored latent dims."""

CLASS_WEIGHTS = False
"""Weight Step 1's and Step 2's region losses by inverse region frequency. Off: every window counts
equally, so training optimizes raw accuracy, the reported metric. On: the powers below apply."""

STEP1_CLASS_WEIGHT_POWER = 1.0
"""Exponent on the inverse-frequency class weights when ``CLASS_WEIGHTS`` is on (1.0 = full)."""

STEP1_WEIGHT_DECAY = 1e-2
"""AdamW L2 on the region-classifier head only, to curb its overfitting."""

# Step 2: alignment decoder
STEP2_CONTRASTIVE_WEIGHT = 2.0
"""Weight on the NT-Xent contrastive term."""

STEP2_DROPOUT = 0.1
"""Dropout in the alignment Transformer. The classifier head keeps its own fixed rates."""

STEP2_CLASS_WEIGHT_POWER = 1.0
"""Exponent on the inverse-frequency class weights when ``CLASS_WEIGHTS`` is on."""

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

MLP_CAL_SPLIT = "chronological"
"""How each training batch is halved into quantile and scored rows (``"random"`` or
``"chronological"``, earlier half of each session's rows vs later half)."""

# session chain protocol
CHAIN_WARMUP_SESSIONS = 3
"""The chain's first sessions: trained and calibrated, never tested."""

CHAIN_SPLIT = (0.60, 0.30, 0.10)
"""Every tested session's chronological train / calibration / test fractions."""

WARMUP_TRAIN_FRAC = 0.70
"""A warm-up session's train fraction (the rest is its calibration block; no test block)."""

POOL_RULE = "B3"
"""Per-region q_hat: own calibration block, a starved region borrows the last 3 sessions'."""

POOL_SENSITIVITY = ("B1", "B5")
"""Pool rules reported alongside ``POOL_RULE`` as a sensitivity analysis."""
