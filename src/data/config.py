"""Recording geometry and session-eligibility constants.

``WINDOW_SIZE * BIN_WIDTH_SEC`` is the real time one window spans: 1.0 s at these
values, which is the conversion every windows-to-seconds caller needs.
"""

BIN_WIDTH_SEC = 0.1
"""Spike count bin width in seconds."""

WINDOW_SIZE = 10
"""Number of bins per time window fed to the encoder."""

MIN_NEURONS_PER_SESSION = 5
"""A session with fewer resolvable neurons is discarded."""

LINEAR_N_REGIONS = 8
"""Linear maze: the run from the start to the trial's water is cut into this many equal regions."""

LINEAR_WATER_MIN_SHARE = 0.8
"""A Linear trial keeps its water position only if the modal value (merged with its +-100 unit
neighbors) covers at least this share of the trial's samples; otherwise the trial is dropped."""
