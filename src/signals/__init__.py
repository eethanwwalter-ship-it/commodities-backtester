"""Signal library.

Public classes:
    Signal                      — abstract base
    SignalData                  — input container
    CarrySignal                 — front/second-month annualized carry
    TSMomentumSignal            — time-series momentum
    XSMomentumSignal            — cross-sectional momentum
    Spread                      — spread definition (leg pair + kind)
    SpreadMeanReversionSignal   — rolling z-score mean-reversion on spreads
    FundamentalSurpriseSignal   — EIA weekly inventory surprises
    FundamentalSurpriseConfig   — config for fundamental signal
"""
from .base import Signal, SignalData
from .carry import CarrySignal
from .fundamental import FundamentalSurpriseConfig, FundamentalSurpriseSignal
from .mean_reversion import Spread, SpreadMeanReversionSignal
from .momentum import TSMomentumSignal, XSMomentumSignal

__all__ = [
    "Signal",
    "SignalData",
    "CarrySignal",
    "TSMomentumSignal",
    "XSMomentumSignal",
    "Spread",
    "SpreadMeanReversionSignal",
    "FundamentalSurpriseSignal",
    "FundamentalSurpriseConfig",
]
