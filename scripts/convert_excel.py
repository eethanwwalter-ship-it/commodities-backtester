"""Convert Bloomberg Excel export into Parquet files for the backtester."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.data.storage import ParquetStore
from src.universe.roll import build_calendar_roll_schedule, build_continuous_returns

SHEETS = ["CL", "CO", "NG", "HO", "XB"]

def main():
    xlsx_path = "futures_data.xlsx"
    store = ParquetStore("data")

    for sheet in SHEETS:
        print(f"\nProcessing {sheet}...")
        df = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)

        # Columns A-B = gen1 dates+prices, D-E = gen2 dates+prices
        gen1_dates = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        gen1_px = pd.to_numeric(df.iloc[:, 1], errors="coerce")
        gen2_dates = pd.to_datetime(df.iloc[:, 3], errors="coerce")
        gen2_px = pd.to_numeric(df.iloc[:, 4], errors="coerce")

        gen1 = pd.DataFrame({"date": gen1_dates, "gen1": gen1_px}).dropna()
        gen2 = pd.DataFrame({"date": gen2_dates, "gen2": gen2_px}).dropna()

        wide = gen1.merge(gen2, on="date", how="outer").sort_values("date").reset_index(drop=True)
        print(f"  {len(wide)} days, {wide['date'].min().date()} to {wide['date'].max().date()}")

        store.write("futures_raw", sheet, wide)

        roll = build_calendar_roll_schedule(wide, sheet)
        cont = build_continuous_returns(wide, roll)
        n_rolls = int(roll.schedule["is_roll_day"].sum())

        store.write("roll_schedules", sheet, roll.schedule)
        store.write("futures_continuous", sheet, cont)
        print(f"  {n_rolls} roll dates detected")
        print(f"  Continuous series: {len(cont)} obs")

    print("\nDone! All futures data saved under data/")

if __name__ == "__main__":
    main()
