"""Signal combiner.

Takes the output of multiple Signal.run() calls and produces a single
composite score per commodity per date. This is the first step in
portfolio construction: signals → composite score → positions.

Combination methods
-------------------
equal_weight
    Simple average across signals. Each signal contributes 1/N to the
    composite. This is the default and often hard to beat in practice.

custom_weight
    Weighted average using caller-supplied weights per signal name.
    Weights are normalized to sum to 1.0 internally. Use this when you
    have a view on relative signal quality — e.g. carry gets 40%,
    momentum 30%, mean-reversion 20%, fundamentals 10%.

All methods handle missing data gracefully: if a signal has NaN for a
given (date, commodity), it's excluded from the average and the remaining
signals are re-weighted. This means the composite score is available as
soon as ANY signal produces a value, rather than waiting for all signals
to warm up.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def combine_signals(
    signals: dict[str, pd.DataFrame],
    method: Literal["equal_weight", "custom_weight"] = "equal_weight",
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Combine multiple signal DataFrames into a composite score.

    Parameters
    ----------
    signals
        Mapping of signal_name → wide DataFrame (index=date, cols=symbols).
        Each DataFrame is the output of Signal.run().
    method
        Combination method.
    weights
        Signal weights for custom_weight method. Keys must match signal
        names. Weights are normalized to sum to 1.0.

    Returns
    -------
    Wide DataFrame (index=date, cols=symbols) with composite score.
    """
    if not signals:
        raise ValueError("At least one signal is required")

    if method == "custom_weight":
        if weights is None:
            raise ValueError("weights required for custom_weight method")
        missing = set(signals) - set(weights)
        if missing:
            raise ValueError(f"Missing weights for signals: {missing}")
        total_w = sum(weights[k] for k in signals)
        if total_w <= 0:
            raise ValueError("Weights must sum to a positive value")
        norm_weights = {k: weights[k] / total_w for k in signals}
    else:
        norm_weights = {k: 1.0 / len(signals) for k in signals}

    # Build a union index and column set
    all_dates: set[pd.Timestamp] = set()
    all_symbols: set[str] = set()
    for df in signals.values():
        all_dates |= set(df.index)
        all_symbols |= set(df.columns)

    date_index = pd.DatetimeIndex(sorted(all_dates))
    symbols = sorted(all_symbols)

    # Weighted sum with NaN-aware re-normalization:
    # For each (date, symbol), sum(w_i * score_i) / sum(w_i) where the
    # sums are over non-NaN signals only.
    weighted_sum = pd.DataFrame(0.0, index=date_index, columns=symbols)
    weight_sum = pd.DataFrame(0.0, index=date_index, columns=symbols)

    for name, df in signals.items():
        w = norm_weights[name]
        aligned = df.reindex(index=date_index, columns=symbols)
        mask = aligned.notna()
        weighted_sum += aligned.fillna(0.0) * w
        weight_sum += mask.astype(float) * w

    composite = weighted_sum / weight_sum.where(weight_sum > 0)
    composite.index.name = "date"

    n_signals = len(signals)
    coverage = weight_sum.iloc[-1].mean() / (1.0 / n_signals * n_signals)
    logger.info(
        "Combined %d signals (%s), last-date coverage: %.0f%%",
        n_signals, method, coverage * 100,
    )
    return composite
