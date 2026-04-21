"""Run the full backtest on real market data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.storage import ParquetStore
from src.signals.base import SignalData
from src.signals.carry import CarrySignal
from src.signals.momentum import TSMomentumSignal, XSMomentumSignal
from src.signals.mean_reversion import Spread, SpreadMeanReversionSignal
from src.signals.fundamental import FundamentalSurpriseSignal
from src.portfolio.backtest import run_backtest, BacktestConfig, RiskConfig

store = ParquetStore("data")

raw_futures = {}
continuous = {}
for sym in ["CL", "CO", "NG", "HO", "XB"]:
    raw_futures[sym] = store.read("futures_raw", sym)
    continuous[sym] = store.read("futures_continuous", sym)

eia = {}
for name in store.list_names("eia_weekly"):
    eia[name] = store.read("eia_weekly", name)

data = SignalData(raw_futures=raw_futures, continuous=continuous, eia=eia)

signals = [
    CarrySignal(smoothing_window=5),
    TSMomentumSignal(lookback_days=252, skip_days=21, risk_adjust=True),
    XSMomentumSignal(lookback_days=252, skip_days=21, risk_adjust=True),
    SpreadMeanReversionSignal(
        spreads=[
            Spread("CL", "CO", kind="log_ratio", label="WTI-Brent"),
            Spread("HO", "XB", kind="log_ratio", label="HO-RB"),
        ],
        lookback=60,
    ),
    FundamentalSurpriseSignal(),
]

config = BacktestConfig(
    vol_lookback=63,
    vol_floor=0.05,
    risk=RiskConfig(max_drawdown=0.15, buffer_fraction=0.5),
)

print("Running backtest on real data...")
print(f"Signals: {[s.name for s in signals]}")
print()

result = run_backtest(signals, data, config)

print("=" * 60)
print("BACKTEST RESULTS — Real Energy Futures 2015-2026")
print("=" * 60)
for key, val in sorted(result.stats.items()):
    if isinstance(val, float):
        if key in ("annualized_return", "annualized_vol", "max_drawdown", "hit_rate"):
            print(f"  {key:<25s}  {val:>8.2%}")
        else:
            print(f"  {key:<25s}  {val:>8.3f}")
    else:
        print(f"  {key:<25s}  {val:>8}")

print()
print("Annual returns:")
yearly = result.portfolio_returns.groupby(result.portfolio_returns.index.year).sum()
for year, ret in yearly.items():
    print(f"  {year}: {ret:+.2%}")

print()
print("Per-signal average scores (last 252 days):")
for name, df in result.signal_outputs.items():
    tail = df.tail(252).mean()
    print(f"  {name}:")
    for sym, val in tail.items():
        if not str(val) == "nan":
            print(f"    {sym}: {val:+.3f}")
print()
print("Done.")
