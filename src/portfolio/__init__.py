"""Portfolio construction and backtesting.

Public API:
    combine_signals          — merge multiple signal outputs
    size_positions           — vol-targeting position sizer
    apply_risk_management    — drawdown-based position scaling
    run_backtest             — end-to-end simulation engine
    BacktestConfig           — configuration container
    BacktestResult           — result container
    RiskConfig               — risk management settings
"""
from .backtest import BacktestConfig, BacktestResult, run_backtest
from .combiner import combine_signals
from .risk import RiskConfig, apply_risk_management
from .sizer import size_positions

__all__ = [
    "combine_signals",
    "size_positions",
    "apply_risk_management",
    "run_backtest",
    "BacktestConfig",
    "BacktestResult",
    "RiskConfig",
]
