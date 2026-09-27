# Conformal

The `src.conformal` package implements the learned class-conditional split-conformal layer, which turns the decoder's region probabilities \(\hat p(x)\) into a trust region.

Each window's score is \(S(x, k) = (1 - \hat p_k(x)) / \sigma_\theta(x)\), where \(\sigma_\theta\) is a small network (`learned`) on four per-window uncertainty signals (`signals`). Two of those signals are Jacobian norms of the encoder and of the alignment map (`strain`). `calibrate` computes the split-conformal quantile of a set of scores, and `predict` admits every region whose score clears that region's threshold. `error_correlation` checks the result: it measures whether larger sets and larger \(\sigma_\theta\) go with more decoder errors.

The per-region thresholds themselves, and the rule for borrowing calibration data from earlier sessions, belong to the session chain: see [`src.chain.pool`](chain.md#per-region-thresholds). [The method](../pipeline.md#5-the-conformal-layer) gives the loss that trains \(\sigma_\theta\).

## Uncertainty signals

::: src.conformal.signals

## Jacobian strain

::: src.conformal.strain

## Learned scale

::: src.conformal.learned

## Calibration

::: src.conformal.calibrate

## Prediction sets

::: src.conformal.predict

## Error correlations

::: src.conformal.error_correlation
