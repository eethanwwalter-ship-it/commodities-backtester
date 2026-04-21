"""Momentum signals for commodity futures.

Two distinct strategies sharing a common computation:

1. **Time-series momentum (TSMOM)** — classic Moskowitz, Ooi & Pedersen
   (2012). For each commodity independently, compare the past risk-
   adjusted return to zero. Positive → go long; negative → go short.
   The aggregate book can run net long, net short, or net neutral
   depending on how many commodities are trending in which direction.

2. **Cross-sectional momentum (XSMOM)** — the Jegadeesh–Titman style
   applied to commodities. Rank commodities by past risk-adjusted
   return; go long the top, short the bottom. The aggregate book is
   approximately dollar-neutral.

Both tend to earn positive returns in commodities over long samples,
with somewhat different drawdown patterns. Running both diversifies
signal risk and is a common choice at systematic pods.

Shared computation
------------------
Raw momentum = risk-adjusted cumulative return over a trailing window
ending `skip_days` ago.

    ret_cum(t) = price_cont(t - skip) / price_cont(t - skip - lookback) - 1
    vol(t)     = std of daily returns over the same window × sqrt(252)
    score(t)   = ret_cum(t) / vol(t)     if risk_adjust else ret_cum(t)

Skipping the most recent ~21 business days is standard practice because
1-month returns exhibit short-term REVERSAL, not continuation — including
the last month mixes the signal. The classic literature uses
12-month lookback with 1-month skip (252 / 21 business days).

Why risk-adjust
---------------
Natural gas daily vol is often 2-3× that of crude, so an un-adjusted
12-month return ranks by "which had the biggest moves" rather than
"which trended most reliably." Dividing by realized vol turns the signal
into a Sharpe-like quantity that's comparable across the universe.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import Signal, SignalData

logger = logging.getLogger(__name__)


def _compute_risk_adjusted_return(
    continuous: dict[str, pd.DataFrame],
    lookback_days: int,
    skip_days: int,
    risk_adjust: bool,
) -> pd.DataFrame:
    """Core momentum computation — used by both TSMOM and XSMOM.

    Returns a wide DataFrame (date × symbol) of (risk-adjusted) cumulative
    returns over the trailing [t-skip-lookback, t-skip] window.
    """
    if lookback_days < 5:
        raise ValueError("lookback_days must be >= 5")
    if skip_days < 0:
        raise ValueError("skip_days must be >= 0")

    series_by_symbol: dict[str, pd.Series] = {}
    for symbol, df in continuous.items():
        if not {"date", "return", "price_cont"}.issubset(df.columns):
            logger.warning("%s: missing continuous columns, skipping", symbol)
            continue

        s = (
            df[["date", "return", "price_cont"]]
            .dropna()
            .drop_duplicates(subset="date")
            .set_index("date")
            .sort_index()
        )

        # End-of-window price:   price_cont(t - skip_days)
        # Start-of-window price: price_cont(t - skip_days - lookback_days)
        end_price   = s["price_cont"].shift(skip_days)
        start_price = end_price.shift(lookback_days)
        ret_cum = end_price / start_price - 1.0

        if risk_adjust:
            # Realized vol over the same window. Shift by skip to align
            # with the end of the lookback window, then rolling std.
            window_returns = s["return"].shift(skip_days)
            vol = window_returns.rolling(lookback_days).std() * np.sqrt(252)
            score = ret_cum / vol.where(vol > 0)
        else:
            score = ret_cum

        series_by_symbol[symbol] = score.rename(symbol)

    if not series_by_symbol:
        raise ValueError("No symbols had usable continuous-return data")

    wide = pd.concat(series_by_symbol.values(), axis=1).sort_index()
    wide.index.name = "date"
    return wide


class TSMomentumSignal(Signal):
    preferred_standardize = "none"
    """Time-series momentum signal.

    Output is the raw risk-adjusted past return. Positive → trending up.
    Use `signal.run(..., standardize="none")` for the raw signal; the
    portfolio-construction layer handles position sizing via vol-targeting.

    Parameters
    ----------
    lookback_days : int, default 252
        Length of the trailing return window (business days).
    skip_days : int, default 21
        Business days to skip from the most recent data, to sidestep
        the 1-month reversal effect.
    risk_adjust : bool, default True
        Divide cumulative return by realized vol over the same window.
    """
    name = "tsmom"

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        risk_adjust: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.risk_adjust = risk_adjust

    def compute(self, data: SignalData) -> pd.DataFrame:
        if not data.continuous:
            raise ValueError(
                "TSMomentumSignal requires continuous return series"
            )
        return _compute_risk_adjusted_return(
            data.continuous,
            self.lookback_days,
            self.skip_days,
            self.risk_adjust,
        )


class XSMomentumSignal(Signal):
    """Cross-sectional momentum signal.

    Output is the same risk-adjusted past return as TSMOM. The difference
    is how it's standardized: call `signal.run(..., standardize="zscore")`
    (the default) or `"rank"` to get cross-sectional rankings.

    Parameters are identical to TSMomentumSignal.
    """
    name = "xsmom"

    def __init__(
        self,
        lookback_days: int = 252,
        skip_days: int = 21,
        risk_adjust: bool = True,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.lookback_days = lookback_days
        self.skip_days = skip_days
        self.risk_adjust = risk_adjust

    def compute(self, data: SignalData) -> pd.DataFrame:
        if not data.continuous:
            raise ValueError(
                "XSMomentumSignal requires continuous return series"
            )
        return _compute_risk_adjusted_return(
            data.continuous,
            self.lookback_days,
            self.skip_days,
            self.risk_adjust,
        )
