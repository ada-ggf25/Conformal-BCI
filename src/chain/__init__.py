"""The session chain: one model carried across a maze's sessions in calendar order.

``order`` fixes the chain and prepares its sessions, ``stages`` holds the carried model,
``pool`` picks each session's per-region conformal thresholds, ``evaluate`` scores a test
block under them, and ``runner`` drives the whole chain with checkpoints.
"""
