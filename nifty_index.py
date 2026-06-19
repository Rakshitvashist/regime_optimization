"""
nifty_index.py  ->  Daily OHLC CSVs for the INDICES themselves (not constituents).

Downloads the index level series so the same pipeline (start.py -> analyze /
predict / backtest / optimize) can run on the indices exactly like a stock.

  NIFTY50   <- ^NSEI     (Nifty 50 index)
  NIFTY500  <- ^CRSLDX   (Nifty 500 index)

Output matches nifty_50.py's schema so start.py can consume it unchanged:
  columns: Symbol, Company, Industry, Index, Date, Open, High, Low, Close, Volume, ...

Then:
  python start.py --input nifty_index_host --output processed_indicators_index
  python analyze_indicators.py --input processed_indicators_index --top-k 5
  python predict_selected.py   --input processed_indicators_index
  python backtest_selected.py  --input processed_indicators_index
  python optimize_indicators.py --scope batch --input processed_indicators_index --method dl
"""
import os

import pandas as pd
import yfinance as yf

OUTPUT_FOLDER = "nifty_index_host"
# (symbol_name, yahoo_ticker, label)
INDICES = [
    ("NIFTY50",  "^NSEI",    "Nifty 50 Index"),
    ("NIFTY500", "^CRSLDX",  "Nifty 500 Index"),
]


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("NIFTY INDICES -> Daily OHLC CSVs")
    for symbol, ticker_sym, label in INDICES:
        filepath = os.path.join(OUTPUT_FOLDER, f"{symbol}_1d_max.csv")
        print(f"  {symbol:9s} ({ticker_sym}) ...", end=" ")
        try:
            data = yf.Ticker(ticker_sym).history(period="max", interval="1d")
            if len(data) == 0:
                print("NO DATA")
                continue
            data.index = data.index.strftime("%d-%m-%Y")
            data.reset_index(inplace=True)
            data.rename(columns={"Datetime": "Date", "index": "Date"}, inplace=True)
            meta = pd.DataFrame({
                "Symbol":   [symbol] * len(data),
                "Company":  [label]  * len(data),
                "Industry": ["Index"] * len(data),
                "Index":    [symbol] * len(data),
            })
            final_df = pd.concat([meta, data.reset_index(drop=True)], axis=1)
            final_df.to_csv(filepath, index=False)
            print(f"OK  {len(data):,} days -> {filepath}")
        except Exception as e:
            print(f"ERROR: {str(e)[:60]}")

    print("\nNext: python start.py --input nifty_index_host --output processed_indicators_index")


if __name__ == "__main__":
    main()
