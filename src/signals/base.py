"""Signal base class.

A Signal takes market data and produces a cross-sectional score per
commodity per date. Higher score = more attractive to be long. Positions
are a function of these scores (handled by the portfolio-construction
layer, not here).

Contract
--------
Every concrete signal implements `compute(data)` and returns a wide-format
DataFrame with:
    - index:   business dates
    - columns: commodity symbols (matching the universe)
    - values:  raw signal (no standardization yet)

The base class then provides `.standardize()` to cross-sectionally z-score
or rank the raw signal so signals of different scales can be combined.

Lookahead rule
--------------
A signal value at date T must use ONLY information available by the close
of T. Positions implied by the signal apply to returns from T+1 onward.
The portfolio-construction layer enforces this lag; signals must not
peek forward themselves.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalData:
    """Bundle of inputs a signal may read.

    Each field maps symbol -> DataFrame. Signals pick whichever fields
    they need and ignore the rest. All price DataFrames have a `date`
    column; continuous-series DataFrames also have `return` and
    `price_cont`; raw futures have `gen1`, `gen2`, etc.
    """
    raw_futures: dict[str, pd.DataFrame] = field(default_factory=dict)
    continuous: dict[str, pd.DataFrame] = field(default_factory=dict)
    eia: dict[str, pd.DataFrame] = field(default_factory=dict)

    def symbols(self) -> list[str]:
        return sorted(set(self.raw_futures) | set(self.continuous))


class Signal(ABC):
    """Abstract base for every systematic signal.

    Subclasses override `compute()`. Use `run()` as the public entrypoint
    — it computes, standardizes, and validates the output shape.
    """

    #: Human-readable name for logging and storage.
    name: str = "unnamed"

    #: Preferred standardization when combined with other signals.
    #: Subclasses override this to declare their natural scaling.
    preferred_standardize: str = "zscore"

    def __init__(self, name: str | None = None):
        if name is not None:
            self.name = name

    @abstractmethod
    def compute(self, data: SignalData) -> pd.DataFrame:
        """Return wide-format raw signal (index=date, columns=symbols)."""

    def run(
        self,
        data: SignalData,
        standardize: Literal["zscore", "rank", "none"] = "zscore",
        winsorize: float | None = 3.0,
    ) -> pd.DataFrame:
        """Compute + standardize. This is the method you call from outside."""
        raw = self.compute(data)
        self._validate(raw, data)
        if standardize == "none":
            return raw
        scored = self.standardize(raw, method=standardize)
        if winsorize is not None and standardize == "zscore":
            scored = scored.clip(lower=-winsorize, upper=winsorize)
        return scored

    @staticmethod
    def standardize(
        raw: pd.DataFrame,
        method: Literal["zscore", "rank"] = "zscore",
    ) -> pd.DataFrame:
        """Cross-sectional standardization.

        zscore: (x - mean) / std, row-wise across commodities. Output is
            roughly N(0, 1) within each date.
        rank: scale ranks to [-1, 1] within each date. Robust to outliers
            but throws away magnitude information.
        """
        if method == "zscore":
            mean = raw.mean(axis=1)
            std = raw.std(axis=1, ddof=0)
            z = raw.sub(mean, axis=0).div(std.replace(0, np.nan), axis=0)
            return z
        if method == "rank":
            # Symmetric [-1, +1]: lowest -> -1, highest -> +1, middle -> 0.
            # For N items, rank r in [1..N] maps to (r - 1) / (N - 1) * 2 - 1.
            # Handles NaNs by counting non-null per row.
            ranks = raw.rank(axis=1, method="average")
            n = ranks.notna().sum(axis=1)
            denom = (n - 1).replace(0, np.nan)
            return ranks.sub(1).div(denom, axis=0) * 2.0 - 1.0
        raise ValueError(f"Unknown standardization method: {method}")

    def _validate(self, raw: pd.DataFrame, data: SignalData) -> None:
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(
                f"{self.name}.compute() must return a DataFrame, got {type(raw)}"
            )
        if raw.index.name not in ("date", None):
            # tolerate unnamed index; most common case
            pass
        if not isinstance(raw.index, pd.DatetimeIndex):
            raise TypeError(
                f"{self.name}.compute() must return a DatetimeIndex, "
                f"got {type(raw.index)}"
            )
        unknown = set(raw.columns) - set(data.symbols())
        if unknown:
            logger.warning(
                "%s: signal output contains symbols not in universe: %s",
                self.name, sorted(unknown),
            )
