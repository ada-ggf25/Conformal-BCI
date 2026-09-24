"""Recording geometry and session-eligibility constants.

All three are carried over at the Liu Lab's values.

``WINDOW_SIZE * BIN_WIDTH_SEC`` is the real time one window spans: 1.0 s at these
values, which is the conversion every windows-to-seconds caller needs.
"""

BIN_WIDTH_SEC = 0.1
"""Spike count bin width in seconds."""

WINDOW_SIZE = 10
"""Number of bins per time window fed to the encoder."""

MIN_NEURONS_PER_SESSION = 5
"""A session with fewer resolvable neurons is discarded."""
