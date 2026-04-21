"""Fundamental-surprise signal — EIA weekly petroleum inventory data.

Theory
------
The Weekly Petroleum Status Report (WPSR) from the EIA prints every
Wednesday at 10:30 ET. It covers US crude stocks (ex-SPR), Cushing
stocks, gasoline stocks, distillate stocks, refinery utilization, crude
imports, and crude production. The market moves on the SURPRISE —
the difference between the actual print and what was expected.

A larger-than-expected draw (actual stocks < consensus) is bullish
because it signals tighter supply than the market priced in. A larger-
than-expected build is bearish.

Consensus sources
-----------------
The gold standard is the Bloomberg SURV function, which aggregates
analyst estimates before each print. If you have terminal access:

    SURV DOECRUD Index <GO>     # crude stock consensus
    SURV DOEGAST Index <GO>     # gasoline stock consensus

Without Bloomberg consensus, a simple alternative (implemented here as
the default) is to use a rolling forecast — trailing N-week mean change.
The surprise is then:

    surprise_t = actual_change_t - rolling_mean_change_t

This is noisier than analyst consensus but captures the same core idea:
did inventories move more or less than recent history would predict?

Signal construction
-------------------
For each EIA series:
  1. Compute week-over-week change in the level.
  2. Compute the "expected" change (rolling mean of prior changes).
  3. Surprise = actual change - expected change.
  4. Standardize by rolling std of surprises (z-score).

The z-score fires on the Wednesday print date and decays over the
following days via an exponential half-life.

Mapping to commodities
----------------------
Not every EIA series moves every commodity equally. The `series_weights`
parameter maps (series_name, commodity_symbol) → weight, controlling how
each surprise contributes to each commodity's signal score.

Default mapping (configurable):
  - Crude stocks (ex-SPR)  → CL, CO  (bearish if build, bullish if draw)
  - Cushing stocks         → CL      (Cushing is the WTI delivery point)
  - Gasoline stocks        → XB      (direct supply indicator for RBOB)
  - Distillate stocks      → HO      (direct supply indicator for ULSD)
  - Refinery utilization   → HO, XB  (higher util → more refined product supply)

Sign convention: a DRAW (negative change) is bullish, so we negate the
surprise before contributing to the signal. A negative surprise (actual
change more negative than expected = bigger draw than expected) produces
a positive signal score.

Look-ahead safety
-----------------
The signal is computed using only data available at each Wednesday's
print time. The rolling forecast uses a strictly backward-looking window.
For dates between prints, the signal decays from the last print value —
it does NOT interpolate toward the next print.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .base import Signal, SignalData

logger = logging.getLogger(__name__)


# Default mapping: (eia_series_name, commodity_symbol) → weight.
# Negative weight means "a build in this series is BEARISH for this commodity."
# We negate the surprise internally so a draw (negative change) becomes a
# positive signal — these weights control only the cross-commodity mapping.
#
# Sign convention in the weights:
#   +1 = a surprise draw in this series is bullish for this commodity
#   -1 = a surprise draw is bearish (e.g., high refinery util → more product)
DEFAULT_SERIES_WEIGHTS: dict[tuple[str, str], float] = {
    # Crude stocks ex-SPR: draws bullish for crude
    ("crude_stocks_ex_spr", "CL"): 1.0,
    ("crude_stocks_ex_spr", "CO"): 1.0,
    # Cushing: WTI delivery point, very direct
    ("cushing_stocks",      "CL"): 1.5,
    # Gasoline stocks: draws bullish for RBOB
    ("gasoline_stocks",     "XB"): 1.0,
    # Distillate stocks: draws bullish for heating oil
    ("distillate_stocks",   "HO"): 1.0,
    # Refinery utilization: higher util → more refined product → bearish products
    ("refinery_utilization", "HO"): -0.5,
    ("refinery_utilization", "XB"): -0.5,
}


@dataclass(frozen=True)
class FundamentalSurpriseConfig:
    """Configuration for the fundamental-surprise signal.

    Attributes
    ----------
    forecast_window
        Number of prior weekly observations used to compute the rolling
        mean "expected" change. 8 ≈ 2 months of weekly prints.
    decay_halflife
        Business-day half-life for exponential decay between prints. The
        signal fires at full strength on Wednesday and decays toward zero
        until the next print. 3 days is a reasonable default — most of
        the price impact is absorbed within a week.
    series_weights
        Mapping of (series_name, commodity_symbol) → weight. Controls
        how each EIA series contributes to each commodity's score.
    """
    forecast_window: int = 8
    decay_halflife: float = 3.0
    series_weights: dict[tuple[str, str], float] = field(
        default_factory=lambda: dict(DEFAULT_SERIES_WEIGHTS)
    )


class FundamentalSurpriseSignal(Signal):
    """EIA inventory surprise signal with exponential decay.

    Parameters
    ----------
    config
        A FundamentalSurpriseConfig controlling forecast window, decay,
        and series-to-commodity mapping.
    """
    name = "fundamental_surprise"

    def __init__(
        self,
        config: FundamentalSurpriseConfig | None = None,
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.config = config or FundamentalSurpriseConfig()

    def compute(self, data: SignalData) -> pd.DataFrame:
        if not data.eia:
            raise ValueError(
                "FundamentalSurpriseSignal requires EIA data in SignalData.eia"
            )

        # Step 1: compute per-series z-scored surprises on print dates.
        surprises: dict[str, pd.Series] = {}
        for series_name, df in data.eia.items():
            zs = self._compute_series_surprise(series_name, df)
            if zs is not None:
                surprises[series_name] = zs

        if not surprises:
            raise ValueError("No EIA series produced usable surprises")

        # Step 2: build a union business-day index from all available data.
        all_dates = set()
        for s in surprises.values():
            all_dates |= set(s.index)
        # Also include continuous-series dates so we can decay between prints.
        for df in data.continuous.values():
            if "date" in df.columns:
                all_dates |= set(df["date"].dropna())
        bday_index = pd.DatetimeIndex(sorted(all_dates))

        # Step 3: for each series, expand print-day surprises onto the
        # full business-day grid with exponential decay.
        decay_lambda = np.log(2) / self.config.decay_halflife

        expanded: dict[str, pd.Series] = {}
        for series_name, zs in surprises.items():
            full = pd.Series(np.nan, index=bday_index, dtype=float)
            full.loc[zs.index] = zs.values
            # Forward-fill the print value, then apply exponential decay
            # based on business days since the last print.
            ffilled = full.ffill()
            # Count business days since last non-NaN (print date)
            is_print = full.notna().astype(int)
            cumprints = is_print.cumsum()
            # For each date, the last print index is the max index where
            # cumprints incremented. We compute days-since via groupby.
            groups = cumprints.values
            days_since = np.zeros(len(bday_index))
            current_group = -1
            counter = 0
            for i in range(len(bday_index)):
                if groups[i] != current_group:
                    current_group = groups[i]
                    counter = 0
                else:
                    counter += 1
                days_since[i] = counter

            decayed = ffilled * np.exp(-decay_lambda * days_since)
            # Before the first print, should be NaN
            first_print_idx = full.first_valid_index()
            if first_print_idx is not None:
                decayed.loc[:first_print_idx] = np.nan
                decayed.loc[first_print_idx] = ffilled.loc[first_print_idx]
            expanded[series_name] = decayed

        # Step 4: distribute decayed surprises to commodities via weights.
        symbols = sorted({
            sym for (_, sym) in self.config.series_weights
        })
        result = pd.DataFrame(0.0, index=bday_index, columns=symbols)
        observed = pd.DataFrame(False, index=bday_index, columns=symbols)

        for (series_name, sym), weight in self.config.series_weights.items():
            if series_name not in expanded:
                continue
            if sym not in result.columns:
                continue
            contribution = expanded[series_name] * weight
            has_val = contribution.notna()
            result[sym] = result[sym] + contribution.fillna(0.0)
            observed[sym] = observed[sym] | has_val

        result = result.where(observed)
        result.index.name = "date"
        return result

    def _compute_series_surprise(
        self,
        series_name: str,
        df: pd.DataFrame,
    ) -> pd.Series | None:
        """Compute z-scored surprise for a single EIA series.

        Returns a Series indexed by print date with z-scored surprise
        values. Returns None if insufficient data.
        """
        if not {"date", "value"}.issubset(df.columns):
            logger.warning("%s: missing date/value columns", series_name)
            return None

        s = (
            df[["date", "value"]]
            .dropna()
            .drop_duplicates(subset="date")
            .set_index("date")["value"]
            .sort_index()
        )
        if len(s) < self.config.forecast_window + 2:
            logger.warning(
                "%s: only %d obs, need >= %d",
                series_name, len(s), self.config.forecast_window + 2,
            )
            return None

        # Week-over-week change
        change = s.diff()

        # Rolling forecast: trailing mean of changes
        expected = change.rolling(
            self.config.forecast_window,
            min_periods=max(3, self.config.forecast_window // 2),
        ).mean()

        surprise = change - expected

        # Standardize by rolling std of surprises
        surprise_std = surprise.rolling(
            self.config.forecast_window,
            min_periods=max(3, self.config.forecast_window // 2),
        ).std(ddof=0)

        z = surprise / surprise_std.where(surprise_std > 0)

        # Negate: a draw (negative change) that's bigger than expected
        # (negative surprise) should produce a POSITIVE score (bullish).
        z = -z

        return z.dropna()
