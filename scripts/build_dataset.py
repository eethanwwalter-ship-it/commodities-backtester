"""Build the initial historical dataset."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.data.bloomberg import BloombergClient, BLPAPI_AVAILABLE
from src.data.eia import EIAClient, PETROLEUM_SERIES
from src.data.storage import ParquetStore
from src.universe.contracts import load_universe
from src.universe.roll import build_calendar_roll_schedule, build_continuous_returns

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_dataset")


def _wide_prices(df, tickers):
    if df.empty:
        return pd.DataFrame()
    ticker_to_gen = {t: f"gen{i+1}" for i, t in enumerate(tickers)}
    df = df.assign(generic=df["security"].map(ticker_to_gen)).dropna(subset=["generic"])
    px = df.pivot_table(index="date", columns="generic", values="PX_LAST", aggfunc="first")
    parts = [px]
    if "PX_VOLUME" in df.columns:
        vol = df.pivot_table(index="date", columns="generic", values="PX_VOLUME", aggfunc="first")
        vol.columns = [f"{c}_volume" for c in vol.columns]
        parts.append(vol)
    if "OPEN_INT" in df.columns:
        oi = df.pivot_table(index="date", columns="generic", values="OPEN_INT", aggfunc="first")
        oi.columns = [f"{c}_oi" for c in oi.columns]
        parts.append(oi)
    return pd.concat(parts, axis=1).reset_index().sort_values("date").reset_index(drop=True)


def build_bloomberg(universe, store, end, n_generics=3):
    if not BLPAPI_AVAILABLE:
        logger.error("blpapi not installed — skipping Bloomberg pull")
        return
    with BloombergClient() as bbg:
        for spec in universe:
            tickers = [spec.generic_ticker(i) for i in range(1, n_generics + 1)]
            raw = bbg.historical_data(
                securities=tickers,
                fields=["PX_LAST", "PX_VOLUME", "OPEN_INT"],
                start=universe.start_date, end=end,
            )
            wide = _wide_prices(raw, tickers)
            if wide.empty:
                logger.warning("%s: no data returned, skipping", spec.symbol)
                continue
            store.write("futures_raw", spec.symbol, wide)
            roll = build_calendar_roll_schedule(wide, spec.symbol)
            cont = build_continuous_returns(wide, roll)
            store.write("roll_schedules", spec.symbol, roll.schedule)
            store.write("futures_continuous", spec.symbol, cont)
            logger.info("%s: %d obs, %d roll days",
                        spec.symbol, len(cont), int(roll.schedule["is_roll_day"].sum()))


def build_eia(store, start, end):
    key = os.environ.get("EIA_API_KEY")
    if not key:
        logger.error("EIA_API_KEY not set — skipping EIA pull")
        return
    eia = EIAClient(key)
    for name, spec in PETROLEUM_SERIES.items():
        try:
            df = eia.get_series(
                route=spec["route"],
                series_id=spec["series"],
                start=start,
                end=end,
            )
            if df.empty:
                logger.warning("EIA %s: empty response", name)
                continue
            store.write("eia_weekly", name, df)
            logger.info("EIA %s: %d obs", name, len(df))
        except Exception as exc:
            logger.error("EIA %s failed: %s", name, exc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/universe.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--skip-bloomberg", action="store_true")
    parser.add_argument("--skip-eia", action="store_true")
    parser.add_argument("--n-generics", type=int, default=3)
    args = parser.parse_args()

    universe = load_universe(args.config)
    store = ParquetStore(args.data_dir)
    end = args.end_date or date.today().isoformat()

    logger.info("Universe: %s", universe.symbols())
    logger.info("Date range: %s to %s", universe.start_date, end)

    if not args.skip_bloomberg:
        build_bloomberg(universe, store, end, n_generics=args.n_generics)
    if not args.skip_eia:
        build_eia(store, universe.start_date, end)

    logger.info("Build complete. Files under %s/", args.data_dir)


if __name__ == "__main__":
    main()
