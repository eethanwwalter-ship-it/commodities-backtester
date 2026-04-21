"""Mean-reversion signal on inter-commodity spreads.

Theory
------
Pairs of economically related commodities trade with persistent
relationships, but those relationships wander: on short horizons one
side gets rich, on longer horizons the spread reverts. Systematic pair
trading harvests this reversion.

Canonical energy pairs:
  * WTI-Brent — geographic arbitrage; widens on US supply gluts (2011-14
    shale boom, 2020 storage crunch) and narrows back toward WAF/pipeline
    economics.
  * Heating oil vs RBOB gasoline — seasonal. HO rich in winter, RB rich
    in summer. The HO-RB spread oscillates on a roughly annual cycle.
  * Gas-oil ratio (WTI ÷ NG in consistent energy units) — used to trade
    the "BTU parity" dislocation; ratios far above/below 10 tend to
    attract physical substitution flows.

Signal construction
-------------------
For each spread defined by (leg_a, leg_b, kind):

    spread_t = leg_a_t - leg_b_t            if kind == "difference"
             = log(leg_a_t / leg_b_t)        if kind == "log_ratio"

Rolling z-score over a configurable window (default 60 business days ≈
3 months):

    z_t = (spread_t - μ) / σ       where μ,σ are trailing stats

A positive z means leg_a is historically rich vs leg_b. Mean-reversion
expects that to narrow, so the signal contributions are:

    score[leg_a] -= z          # short the rich leg
    score[leg_b] += z          # long the cheap leg

Contributions from multiple spreads sharing a commodity are summed —
e.g. if WTI appears in both (WTI, Brent) and (WTI, NG) spreads, its
total score is the sum of both.

Mapping onto the Signal framework
---------------------------------
compute() returns a wide DataFrame of these summed contributions per
commodity per date. The values are already in z-score units (meaningful
on their own), so when combining with other signals you typically want
standardize="none" rather than double-standardizing.

Look-ahead safety
-----------------
Rolling z-score at date t uses only data through t — the rolling window
is right-closed. No centered rolling.

Unit consistency
----------------
`kind="difference"` is economically meaningful only when both legs share
units (HO and RB both in $/gallon; WTI and Brent both in $/bbl). It fails
silently on mixed units (WTI $/bbl vs NG $/MMBtu) — the difference has
no interpretation. Use `kind="log_ratio"` (default) when in doubt.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Signal, SignalData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Spread:
    """Definition of a pairwise commodity spread.

    Attributes
    ----------
    leg_a, leg_b
        Universe symbols. Convention: `leg_a - leg_b` (or `log(leg_a/leg_b)`)
        — signal interpretation is "how rich is leg_a vs leg_b?"
    kind
        "log_ratio"  (default) — unit-free, works for any pair. Uses
                    roll-adjusted continuous prices.
        "difference" — uses raw front-month prices. Economically meaningful
                    only when both legs share units.
    label
        Optional human-readable name for logging/debugging (e.g.
        "WTI-Brent", "HO-RB crack").
    """
    leg_a: str
    leg_b: str
    kind: str = "log_ratio"
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind not in ("log_ratio", "difference"):
            raise ValueError(
                f"Spread.kind must be 'log_ratio' or 'difference', "
                f"got {self.kind!r}"
            )
        if self.leg_a == self.leg_b:
            raise ValueError(
                f"Spread legs must differ, got {self.leg_a} == {self.leg_b}"
            )

    @property
    def display_name(self) -> str:
        return self.label or f"{self.leg_a}-{self.leg_b}"


class SpreadMeanReversionSignal(Signal):
    """Mean-reversion signal on a collection of commodity spreads.

    Parameters
    ----------
    spreads
        List of `Spread` objects defining which pairs to trade.
    lookback
        Window (business days) for the rolling mean/std used to z-score
        each spread. 60 ≈ 3 months is a reasonable default — short enough
        to pick up mid-term dislocations, long enough to be stable.
    min_periods_frac
        Fraction of lookback required before emitting a non-NaN z-score.
        Default 0.5 (allow warm-up at half window; trades off reliability
        for history coverage).
    cap
        Optional cap on |z| to prevent outlier regimes (e.g., April 2020
        WTI crash) from dominating. Default None (no cap).
    """
    name = "spread_mean_reversion"

    def __init__(
        self,
        spreads: list[Spread],
        lookback: int = 60,
        min_periods_frac: float = 0.5,
        cap: float | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        if not spreads:
            raise ValueError("At least one Spread is required")
        if lookback < 5:
            raise ValueError("lookback must be >= 5")
        if not 0.0 < min_periods_frac <= 1.0:
            raise ValueError("min_periods_frac must be in (0, 1]")
        if cap is not None and cap <= 0:
            raise ValueError("cap must be positive if provided")
        self.spreads = list(spreads)
        self.lookback = lookback
        self.min_periods = max(5, int(lookback * min_periods_frac))
        self.cap = cap

    def compute(self, data: SignalData) -> pd.DataFrame:
        # Each spread contributes to two columns (its legs). Build a list
        # of per-spread contribution frames, then sum.
        contributions: list[pd.DataFrame] = []

        for spread in self.spreads:
            try:
                a_price = self._get_price(data, spread.leg_a, spread.kind)
                b_price = self._get_price(data, spread.leg_b, spread.kind)
            except KeyError as exc:
                logger.warning(
                    "Skipping %s: missing data for %s",
                    spread.display_name, exc,
                )
                continue

            # Align on common dates
            aligned = pd.concat(
                {"a": a_price, "b": b_price}, axis=1,
            ).dropna()
            if len(aligned) < self.min_periods:
                logger.warning(
                    "Skipping %s: only %d overlapping obs (need >= %d)",
                    spread.display_name, len(aligned), self.min_periods,
                )
                continue

            if spread.kind == "log_ratio":
                # Guard against non-positive prices
                a, b = aligned["a"], aligned["b"]
                if (a <= 0).any() or (b <= 0).any():
                    logger.warning(
                        "Skipping %s: non-positive prices in log_ratio",
                        spread.display_name,
                    )
                    continue
                spread_series = np.log(a / b)
            else:  # "difference"
                spread_series = aligned["a"] - aligned["b"]

            # Rolling z-score (right-closed window)
            mu = spread_series.rolling(
                self.lookback, min_periods=self.min_periods,
            ).mean()
            sigma = spread_series.rolling(
                self.lookback, min_periods=self.min_periods,
            ).std(ddof=0)
            z = (spread_series - mu) / sigma.where(sigma > 0)

            if self.cap is not None:
                z = z.clip(lower=-self.cap, upper=self.cap)

            # Distribute as contributions: leg_a short rich, leg_b long cheap
            contrib = pd.DataFrame(
                {spread.leg_a: -z, spread.leg_b: +z},
                index=z.index,
            )
            contributions.append(contrib)

        if not contributions:
            raise ValueError(
                "No spreads produced usable contributions — check data"
            )

        # Sum contributions by column (NaN + x = x via fillna(0) on empty cells)
        # Build union of all dates and symbols, then align and sum.
        union_idx = contributions[0].index
        for c in contributions[1:]:
            union_idx = union_idx.union(c.index)
        symbols = sorted({col for c in contributions for col in c.columns})

        total = pd.DataFrame(0.0, index=union_idx, columns=symbols)
        # Track where we had ANY observation — cells with no contributions
        # should remain NaN, not 0.
        observed = pd.DataFrame(False, index=union_idx, columns=symbols)

        for c in contributions:
            aligned = c.reindex(index=union_idx, columns=symbols)
            has_val = aligned.notna()
            total = total.add(aligned.fillna(0.0))
            observed = observed | has_val

        result = total.where(observed)
        result.index.name = "date"
        return result

    @staticmethod
    def _get_price(
        data: SignalData, symbol: str, kind: str,
    ) -> pd.Series:
        """Pull the appropriate price series for the spread kind."""
        if kind == "log_ratio":
            if symbol not in data.continuous:
                raise KeyError(f"continuous[{symbol!r}]")
            df = data.continuous[symbol]
            col = "price_cont"
        else:  # "difference"
            if symbol not in data.raw_futures:
                raise KeyError(f"raw_futures[{symbol!r}]")
            df = data.raw_futures[symbol]
            col = "gen1"

        if col not in df.columns or "date" not in df.columns:
            raise KeyError(f"{symbol}: missing {col}/date")

        return (
            df[["date", col]]
            .dropna()
            .drop_duplicates(subset="date")
            .set_index("date")[col]
            .sort_index()
        )
