"""
daily_rv_forecast.py  —  forecast FUTURE realized volatility (the predictable target).

Builds high-quality DAILY realized variance from your 1-min bars (sum of intraday
squared returns — far better than close-to-close), then forecasts the average
realized vol over the next H days. Compares, walk-forward (expanding, causal):

  naive      persistence (today's RV -> future RV) — the benchmark to beat
  HAR        Corsi (2009) HAR-RV: log RV on daily/weekly/monthly components (OLS)
  GBM        HistGradientBoosting on HAR + vol-of-vol + skew/kurt + VIX
  GARCH      GARCH(1,1) H-day forecast (optional; needs `arch`)

Metrics: R^2 (log-variance), RMSE (annualized vol %), QLIKE (the standard vol loss).
This answers honestly whether daily vol is more forecastable than the intraday work,
and whether HAR/GBM beat the naive persistence + GARCH baselines.

Usage:
  python daily_rv_forecast.py --input NIFTY_50.csv --H 20 --vix INDIA_VIX.csv
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

ANN = 252


def _load_1min(path):
    df = pd.read_csv(path)
    dt = next(c for c in df.columns if c.lower() in ("datetime", "date"))
    df[dt] = pd.to_datetime(df[dt], utc=True)
    return df.set_index(dt).sort_index()


def _day_index(idx):
    return (idx.tz_convert("Asia/Kolkata") if idx.tz is not None else idx).normalize()


def daily_realized_var(df):
    """Realized variance per day = sum of squared INTRADAY 1-min log returns."""
    lc = np.log(df["Close"].astype(float))
    day = _day_index(df.index)
    g = pd.DataFrame({"lc": lc.to_numpy(), "day": day})
    g["r"] = g.groupby("day")["lc"].diff()      # intraday returns (overnight excluded)
    rv = g.groupby("day")["r"].apply(lambda x: np.nansum(np.square(x)))
    return rv[rv > 0]                            # variance per day


def daily_close_returns(df):
    day = _day_index(df.index)
    close_day = df["Close"].astype(float).groupby(day).last()
    return np.log(close_day).diff()


def daily_vix(path):
    v = _load_1min(path)["Close"].astype(float)
    v = v.where(v >= 5).ffill()
    return v.groupby(_day_index(v.index)).last()


def build(df, H, vix_path):
    rv = daily_realized_var(df)
    logrv = np.log(rv)
    F = pd.DataFrame(index=rv.index)
    F["d"] = logrv                                              # HAR daily
    F["w"] = np.log(rv.rolling(5, min_periods=3).mean())       # HAR weekly
    F["m"] = np.log(rv.rolling(22, min_periods=10).mean())     # HAR monthly
    F["volofvol"] = logrv.rolling(22, min_periods=10).std()
    F["rv_chg5"] = logrv.diff(5)
    r = daily_close_returns(df).reindex(rv.index)
    F["skew20"] = r.rolling(20, min_periods=10).skew()
    F["kurt20"] = r.rolling(20, min_periods=10).kurt()
    if vix_path:
        F["vix"] = np.log(daily_vix(vix_path).reindex(rv.index).ffill())
    # target: log of AVERAGE realized variance over next H days
    target = np.log(rv.rolling(H).mean().shift(-H))
    F["target"] = target
    F = F.replace([np.inf, -np.inf], np.nan).dropna()
    return rv, F


def qlike(var_true, var_pred):
    z = var_true / np.clip(var_pred, 1e-12, None)
    return float(np.mean(z - np.log(z) - 1))


def annvol(logvar):
    return np.sqrt(np.exp(logvar) * ANN) * 100.0     # annualized vol %


def main(argv=None):
    ap = argparse.ArgumentParser(description="Daily realized-vol forecasting benchmark.")
    ap.add_argument("--input", required=True, help="1-min CSV (e.g. NIFTY_50.csv)")
    ap.add_argument("--H", type=int, default=20, help="forecast horizon in trading days")
    ap.add_argument("--vix", default=None)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    ap.add_argument("--garch", action="store_true", help="add GARCH(1,1) baseline (needs `arch`)")
    a = ap.parse_args(argv)

    df = _load_1min(a.input)
    rv, F = build(df, a.H, a.vix)
    feats = [c for c in F.columns if c != "target"]
    y = F["target"].to_numpy()
    Xh = F[["d", "w", "m"]].to_numpy()      # HAR
    Xg = F[feats].to_numpy()                # GBM
    rets = daily_close_returns(df)          # for GARCH
    n = len(F)
    print(f"=== daily RV forecast  {a.input}  | H={a.H}d | days={n} | "
          f"{F.index[0].date()}..{F.index[-1].date()} ===")

    start = int(n * a.start_frac); step = (n - start) // a.folds
    preds = {"naive": [], "HAR": [], "GBM": []}
    ytrue = []
    if a.garch:
        preds["GARCH"] = []
    for f in range(a.folds):
        lo = start + f * step
        hi = (start + (f + 1) * step) if f < a.folds - 1 else n
        tr = slice(0, lo - a.H)             # embargo H days
        te = slice(lo, hi)
        if (lo - a.H) < 100:
            continue
        ytrue.append(y[te])
        preds["naive"].append(Xh[te, 0])   # persistence: today's daily logRV
        har = LinearRegression().fit(Xh[tr], y[tr]); preds["HAR"].append(har.predict(Xh[te]))
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
             max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True,
             random_state=42).fit(Xg[tr], y[tr])
        preds["GBM"].append(gb.predict(Xg[te]))
        if a.garch:
            preds["GARCH"].append(_garch_fold(rets, F.index, te, a.H))

    yt = np.concatenate(ytrue)
    print(f"\n{'model':8s} {'R2(logvar)':>11s} {'RMSE(vol%)':>11s} {'QLIKE':>9s}")
    for name, ps in preds.items():
        p = np.concatenate(ps)
        r2 = r2_score(yt, p)
        rmse = np.sqrt(mean_squared_error(annvol(yt), annvol(p)))
        ql = qlike(np.exp(yt), np.exp(p))
        print(f"{name:8s} {r2:11.4f} {rmse:11.3f} {ql:9.4f}")
    print("\n(R2>0 beats mean; naive persistence is the bar to beat. "
          "Lower RMSE/QLIKE = better.)")
    return 0


def _garch_fold(rets, index, te, H):
    from arch import arch_model
    out = []
    te_idx = index[te]
    for ts in te_idx:
        hist = (rets[rets.index <= ts].dropna() * 100).to_numpy()[-1500:]
        try:
            res = arch_model(hist, vol="Garch", p=1, q=1, mean="Zero",
                             rescale=False).fit(disp="off")
            fc = res.forecast(horizon=H, reindex=False)
            var_daily = fc.variance.to_numpy().ravel().mean() / (100.0 ** 2)
            out.append(np.log(max(var_daily, 1e-12)))
        except Exception:
            out.append(np.log(rets.var() if rets.var() > 0 else 1e-6))
    return np.array(out)


if __name__ == "__main__":
    raise SystemExit(main())
