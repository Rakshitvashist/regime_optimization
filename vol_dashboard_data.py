"""
vol_dashboard_data.py  —  compute everything the dashboard shows (one dict).

Per instrument:
  forecast_cone  HAR forecast + 1s/1.5s/2s bands across horizons (causal/walk-forward)
  hist_cone      Burghardt historical vol cone: realized-vol percentiles by window
  hist_series    historical 20d realized vol time series (annualized, overnight-adj)
  regime_cones   per daily-HMM-regime (quiet/normal/explosive) forward-vol bands
  garch          (optional) GARCH(1,1) cone median across horizons
  intraday       current 30m HMM regime + spike-imminence gauge

Reuses the validated, causal building blocks. Heavy-ish: compute once, cache.
"""
from __future__ import annotations

import os
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

import daily_rv_forecast as drv
from vol_cone import daily_total_var, har_table, walk_forward, daily_regime, RNAMES

HORIZONS = [5, 10, 20, 40, 60, 120]
SIGMAS = [1.0, 1.5, 2.0]

# dashboard instrument -> (1-min daily file, processed_intraday 30m symbol or None)
INSTRUMENTS = {
    "NIFTY 50":   ("NIFTY_50.csv",  "NIFTY-50-30m"),
    "BANK NIFTY": ("BANKNIFTY.csv", "BANKNIFTY-30m"),
    "SENSEX":     ("SENSEX.csv",    "SENSEX-30m"),
    "RELIANCE":   ("RELIANCE.csv",  "RELIANCE-30m"),
    # MCX commodities (real volume; ~9:00-23:30 session). Main liquid contracts
    # only — the M/MINI/MIC/GUINEA/PETAL/TEN/100 variants are the same underlying.
    "CRUDE OIL":   ("CRUDEOIL.csv",   None),
    "GOLD":        ("GOLD.csv",       None),
    "SILVER":      ("SILVER.csv",     None),
    "NATURAL GAS": ("NATURALGAS.csv", None),
}


def _forecast_cone(rv):
    cone = []
    for H in HORIZONS:
        F, target = har_table(rv, H)
        yt, yp, td = walk_forward(F, target, H, 8, 0.4)
        sigma = float(np.std(yt - yp)); r2 = float(r2_score(yt, yp))
        comp = F.notna().all(axis=1) & target.notna()
        m = LinearRegression().fit(F[comp].to_numpy(), target[comp].to_numpy())
        last = F[F.notna().all(axis=1)].iloc[-1:]
        pt = float(m.predict(last.to_numpy())[0])
        e = {"H": H, "r2": round(r2, 3), "median": round(drv.annvol(pt), 2)}
        for k in SIGMAS:
            # key suffix via %g so 1.0->"1", 1.5->"1.5", 2.0->"2" — matches the
            # React side's JS number stringification ('dn'+k).
            ks = f"{k:g}"
            e[f"up{ks}"] = round(drv.annvol(pt + k * sigma), 2)
            e[f"dn{ks}"] = round(drv.annvol(pt - k * sigma), 2)
        cone.append(e)
    return cone


def _hist_cone(rv):
    """Burghardt cone: distribution of REALIZED vol over rolling H-day windows."""
    out = []
    for H in HORIZONS:
        a = (np.sqrt(rv.rolling(H).mean() * drv.ANN) * 100).dropna()
        if len(a) < 30:
            continue
        out.append({"H": H, "min": round(float(a.min()), 1),
                    "p25": round(float(a.quantile(.25)), 1),
                    "median": round(float(a.median()), 1),
                    "p75": round(float(a.quantile(.75)), 1),
                    "max": round(float(a.max()), 1),
                    "current": round(float(a.iloc[-1]), 1)})
    return out


def _regime_cones(rv):
    reg = daily_regime(rv)
    H = 20
    F, target = har_table(rv, H)
    yt, yp, td = walk_forward(F, target, H, 8, 0.4)
    resid = yt - yp; r = reg.reindex(td).to_numpy()
    out = []
    for k in sorted(RNAMES):
        mk = r == k
        if mk.sum() < 30:
            continue
        med = float(np.median(yp[mk]))
        out.append({"regime": RNAMES[k],
                    "lo": round(drv.annvol(med + np.quantile(resid[mk], .1)), 1),
                    "median": round(drv.annvol(med + np.quantile(resid[mk], .5)), 1),
                    "hi": round(drv.annvol(med + np.quantile(resid[mk], .9)), 1),
                    "n": int(mk.sum())})
    return out, RNAMES.get(int(reg.iloc[-1]), "n/a")


def _garch_cone(df, rv):
    try:
        from arch import arch_model
    except Exception:
        return None
    r = (drv.daily_close_returns(df).dropna() * 100).to_numpy()[-2500:]
    try:
        res = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", rescale=False).fit(disp="off")
    except Exception:
        return None
    out = []
    for H in HORIZONS:
        fc = res.forecast(horizon=H, reindex=False)
        v = fc.variance.to_numpy().ravel().mean() / (100.0 ** 2)
        out.append({"H": H, "median": round(np.sqrt(v * drv.ANN) * 100, 2)})
    return out


def _daily_log_close(df):
    return np.log(df["Close"].astype(float).groupby(drv._day_index(df.index)).last())


def _price_bands(df, rv, horizons=(7, 20)):
    """sigma PRICE ranges: P0 * exp(mu_H +/- k*sigma_H). 'bands' = no drift,
    'bands_drift' = recentered on recent drift (fixes strongly-trending names)."""
    lc = _daily_log_close(df)
    P0 = float(np.exp(lc.iloc[-1]))
    mu_daily = float(lc.diff().tail(120).mean())     # recent drift (causal at latest bar)
    out = {"current_price": round(P0, 2), "drift_daily_pct": round(mu_daily * 100, 3),
           "bands": [], "bands_drift": []}
    for H in horizons:
        F, target = har_table(rv, H)
        comp = F.notna().all(axis=1) & target.notna()
        m = LinearRegression().fit(F[comp].to_numpy(), target[comp].to_numpy())
        last = F[F.notna().all(axis=1)].iloc[-1:]
        sig_h = float(np.sqrt(np.exp(m.predict(last.to_numpy())[0]) * H))
        muH = mu_daily * H
        e = {"H": H, "move_pct": round(sig_h * 100, 2)}
        ed = {"H": H, "move_pct": round(sig_h * 100, 2)}
        for k in (1.0, 1.5, 2.0):
            ks = f"{k:g}"
            e[f"up{ks}"] = round(P0 * np.exp(k * sig_h), 2)
            e[f"dn{ks}"] = round(P0 * np.exp(-k * sig_h), 2)
            ed[f"up{ks}"] = round(P0 * np.exp(muH + k * sig_h), 2)
            ed[f"dn{ks}"] = round(P0 * np.exp(muH - k * sig_h), 2)
        out["bands"].append(e)
        out["bands_drift"].append(ed)
    return out


def _coverage(df, rv, horizons=(7, 20), folds=8, start_frac=0.4):
    """Walk-forward band coverage at 1/1.5/2 sigma: how often realized price landed
    inside, RAW vs DRIFT-adjusted (drift = trailing-120d mean daily return)."""
    lc = _daily_log_close(df)
    dret = lc.diff()
    KS = [1.0, 1.5, 2.0]
    res = {}
    for H in horizons:
        F, target = har_table(rv, H)
        rH = (lc.shift(-H) - lc).reindex(rv.index)
        mur = dret.rolling(120, min_periods=30).mean().reindex(rv.index)
        both = F.notna().all(axis=1) & target.notna()
        Xb = F[both].to_numpy(); yb = target[both].to_numpy()
        rHb = rH[both].to_numpy(); mub = mur[both].to_numpy()
        n = len(Xb); start = int(n * start_frac); step = max(1, (n - start) // folds)
        hr = {k: [] for k in KS}; hd = {k: [] for k in KS}
        for f in range(folds):
            lo = start + f * step
            hi = (start + (f + 1) * step) if f < folds - 1 else n
            if lo - H < 100:
                continue
            m = LinearRegression().fit(Xb[:lo - H], yb[:lo - H])
            sigH = np.sqrt(np.exp(m.predict(Xb[lo:hi])) * H)
            r = rHb[lo:hi]; md = mub[lo:hi] * H; v = np.isfinite(r) & np.isfinite(md)
            for k in KS:
                hr[k].append(np.abs(r[v]) < k * sigH[v])
                hd[k].append(np.abs(r[v] - md[v]) < k * sigH[v])
        res[str(H)] = {
            "raw": {f"{k:g}": round(float(np.mean(np.concatenate(hr[k]))), 3) for k in KS},
            "drift": {f"{k:g}": round(float(np.mean(np.concatenate(hd[k]))), 3) for k in KS},
        }
    return res


def _daily_ret(path):
    df = drv._load_1min(path)
    cl = df["Close"].astype(float).groupby(drv._day_index(df.index)).last()
    s = np.log(cl).diff().dropna()
    s.index = [d.date() for d in s.index]   # align cross-instrument on calendar date
    return s


def _intraday_state(symbol30):
    if not symbol30:
        return None
    try:
        from optimization import data_loader
        from hmm_regime import compute_regime
        from optimization import spike_features as sf
        import analyze_indicators  # noqa (ensures import graph ok)
        path = None
        for s, p in data_loader.list_symbols("processed_intraday"):
            if s == symbol30:
                path = p
        if path is None:
            return None
        df = data_loader.load_symbol_frame(path)
        reg = compute_regime(df, 3, 8, 0.4, None, "30min")
        last = int(reg[-1]) if reg[-1] >= 0 else -1
        spk = sf.spike_features(df)
        volz = spk[:, sf.FEATURE_NAMES.index("spike_volz")]
        score = spk[:, sf.FEATURE_NAMES.index("spike_score")]
        pressure = float((volz < volz[-1]).mean() * 100)
        d = "up" if score[-1] > 0.05 else ("down" if score[-1] < -0.05 else "neutral")
        return {"regime": RNAMES.get(last, "n/a"),
                "spike_pressure": round(pressure), "spike_dir": d,
                "asof": str(df.index[-1])}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def instrument_data(daily_path, symbol30, garch=False):
    df = drv._load_1min(daily_path)
    rv = daily_total_var(df, overnight=True)
    hist_vol = (np.sqrt(rv.rolling(20).mean() * drv.ANN) * 100).dropna()
    reg_cones, cur_daily_regime = _regime_cones(rv)
    return {
        "asof": str(rv.index[-1].date()),
        "current": round(float(hist_vol.iloc[-1]), 2),
        "forecast_cone": _forecast_cone(rv),
        "hist_cone": _hist_cone(rv),
        "regime_cones": reg_cones,
        "daily_regime": cur_daily_regime,
        "garch": _garch_cone(df, rv) if garch else None,
        "intraday": _intraday_state(symbol30),
        "price_bands": _price_bands(df, rv),
        "coverage": _coverage(df, rv),
        "hist_series": {"date": [d.strftime("%Y-%m-%d") for d in hist_vol.index[::3]],
                        "vol": [round(float(v), 2) for v in hist_vol.values[::3]]},
    }


def _correlation(log=print):
    """Daily-return correlation across instruments (aligned on common dates)."""
    rets = {}
    for name, (dpath, _) in INSTRUMENTS.items():
        if os.path.exists(dpath):
            try:
                rets[name] = _daily_ret(dpath)
            except Exception:
                pass
    R = pd.DataFrame(rets).corr()
    L = list(R.columns)
    pairs = []
    for i in range(len(L)):
        for j in range(i + 1, len(L)):
            pairs.append([L[i], L[j], round(float(R.iloc[i, j]), 2)])
    pairs.sort(key=lambda x: -abs(x[2]))
    return {"labels": L,
            "matrix": [[round(float(R.iloc[i, j]), 2) for j in range(R.shape[1])]
                       for i in range(R.shape[0])],
            "top": pairs[:8]}


def compute_all(garch=False, log=print):
    data = {}
    for name, (dpath, sym30) in INSTRUMENTS.items():
        if not os.path.exists(dpath):
            continue
        try:
            data[name] = instrument_data(dpath, sym30, garch=garch)
            if log:
                pb = data[name]["price_bands"]
                log(f"  {name:12s} ok  RV {data[name]['current']}%  "
                    f"regime {data[name]['daily_regime']:9s}  "
                    f"price {pb['current_price']}")
        except Exception as e:
            if log:
                log(f"  {name:12s} FAILED: {type(e).__name__}: {e}")
    corr = _correlation(log)
    if log:
        log(f"  correlation matrix: {len(corr['labels'])} instruments")
    return {"instruments": data, "correlation": corr}


if __name__ == "__main__":
    out = compute_all()
    print("\n--- price bands (7d / 20d) ---")
    for k, v in out["instruments"].items():
        for b in v["price_bands"]["bands"]:
            print(f"  {k:12s} {b['H']:>2}d  -2s {b['dn2']:>10} | -1s {b['dn1']:>10} | "
                  f"now {v['price_bands']['current_price']:>10} | +1s {b['up1']:>10} | "
                  f"+2s {b['up2']:>10}  (+/-{b['move_pct']}%)")
    print("\n--- correlation ---")
    c = out["correlation"]; print("   " + " ".join(f"{l[:6]:>6}" for l in c["labels"]))
    for i, l in enumerate(c["labels"]):
        print(f"{l[:10]:10s} " + " ".join(f"{c['matrix'][i][j]:>6}" for j in range(len(c["labels"]))))
