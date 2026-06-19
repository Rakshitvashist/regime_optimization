import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import time

print("NIFTY 50 -> Daily OHLC CSVs (Incremental Mode)")
print("Downloads full history only for NEW stocks.")
print("For existing stocks, only appends rows since the last date.")

input_csv    = "ind_nifty50list.csv"
output_folder = "nifty50_host"
os.makedirs(output_folder, exist_ok=True)

df_list = pd.read_csv(input_csv)
symbols = df_list['Symbol'].dropna().unique().tolist()
print(f"Found {len(symbols)} symbols")

success_count = 0
skipped_count = 0
failed_symbols = []

for i, symbol in enumerate(symbols, 1):
    print(f"[{i:3d}/{len(symbols)}] {symbol:<12}", end=" ")

    full_symbol = f"{symbol}.NS"
    filename    = f"{symbol}_1d_max.csv"
    filepath    = os.path.join(output_folder, filename)

    # Get metadata
    row     = df_list[df_list['Symbol'] == symbol].iloc[0]
    company = row['Company Name']
    industry = row['Industry']

    try:
        if os.path.exists(filepath):
            # --- INCREMENTAL: only fetch rows after last date ---
            existing = pd.read_csv(filepath)
            # Find the date column (handles Date or date)
            date_col = next((c for c in existing.columns if c.lower() == 'date'), None)
            if date_col is None:
                raise ValueError("No date column found in existing file")
            existing[date_col] = pd.to_datetime(existing[date_col], dayfirst=True, errors='coerce')
            last_date = existing[date_col].dropna().max()
            if pd.isnull(last_date):
                raise ValueError("Could not parse last date")

            # Fetch only from the day after last_date
            start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            today = datetime.now()
            tomorrow = (today + timedelta(days=1)).strftime('%Y-%m-%d')
            today_str = today.strftime('%Y-%m-%d')

            if start > today_str:
                print(f"up-to-date ({last_date.strftime('%d-%m-%Y')})")
                skipped_count += 1
                continue

            ticker = yf.Ticker(full_symbol)
            new_data = ticker.history(start=start, end=tomorrow, interval="1d")

            if len(new_data) == 0:
                print(f"up-to-date ({last_date.strftime('%d-%m-%Y')})")
                skipped_count += 1
                continue

            new_data.index = new_data.index.strftime("%d-%m-%Y")
            new_data.reset_index(inplace=True)
            new_data.rename(columns={"Datetime": "Date", "index": "Date"}, inplace=True)

            meta = pd.DataFrame({
                "Symbol":   [symbol]   * len(new_data),
                "Company":  [company]  * len(new_data),
                "Industry": [industry] * len(new_data),
                "Index":    ["NIFTY50"] * len(new_data),
            })
            new_rows = pd.concat([meta, new_data.reset_index(drop=True)], axis=1)

            # Append to existing file (no header)
            new_rows.to_csv(filepath, mode='a', header=False, index=False)
            print(f"+{len(new_data)} new rows  (total: {len(existing)+len(new_data)})")
            success_count += 1

        else:
            # --- FULL DOWNLOAD for new stock ---
            ticker = yf.Ticker(full_symbol)
            data   = ticker.history(period="max", interval="1d")

            if len(data) == 0:
                print("NO DATA")
                continue

            data.index = data.index.strftime("%d-%m-%Y")
            data.reset_index(inplace=True)
            data.rename(columns={"Datetime": "Date"}, inplace=True)

            meta = pd.DataFrame({
                "Symbol":   [symbol]   * len(data),
                "Company":  [company]  * len(data),
                "Industry": [industry] * len(data),
                "Index":    ["NIFTY50"] * len(data),
            })
            final_df = pd.concat([meta, data.reset_index(drop=True)], axis=1)
            final_df.to_csv(filepath, index=False)
            print(f"NEW  {len(data):,} days")
            success_count += 1

    except Exception as e:
        print(f"ERROR: {str(e)[:40]}")
        failed_symbols.append(symbol)

    time.sleep(0.3)

print(f"\nDONE: {success_count} updated | {skipped_count} already up-to-date | {len(failed_symbols)} failed")
if failed_symbols:
    print(f"Failed: {failed_symbols[:10]}")