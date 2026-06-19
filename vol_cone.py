"""
vol_cone.py  —  HAR realized-vol forecast + PROBABILITY CONE.

Multi-horizon (RV20/60/120) HAR forecast turned into a forward VOLATILITY
DISTRIBUTION via out-of-sample residual quantiles. Three refinements:

  --overnight (default)  total realized variance = intraday RV + overnight gap
                         variance -> level matches India VIX (intraday-only
                         understates it by excluding the overnight move).
  --garch                add a GARCH(1,1) H-day forecast for comparison (needs `arch`).
  --regime               condition the cone on the current daily HMM vol regime
                         (quiet / normal / explosive) -> the cone widens & lifts in
                         explosive regimes; residuals are bucketed by regime.

All causal / walk-forward; cone width comes from real OOS errors.

Usage:
  python vol_cone.py --input BANKNIFTY.csv --horizons 20 60 120 --regime
  python vol_cone.py --input NIFTY_50.csv  --horizons 20 --regime --garch
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

import daily_rv_forecast as drv

QS = [0.10, 0.25, 0.50, 0.75, 0.90]
RNAMES = {0: "quiet", 1: "normal", 2: "explosive"}


def daily_total_var(df, overnight=True):
    """Daily realized variance = intraday RV (+ overnight gap variance)."""
    day = drv._day_index(df.index)
    lc = np.log(df["Close"].astype(float))
    g = pd.DataFrame({"lc": lc.to_numpy(), "day": day})
    g["r"] = g.groupby("day")["lc"].diff()
    rv = g.groupby("day")["r"].apply(lambda x: np.nansum(np.square(x)))
    if overnight:
        first_open = df["Open"].astype(float).groupby(day).first()
        last_close = df["Close"].astype(float).groupby(day).last()
        onr = (np.log(first_open) - np.log(last_close.shift(1))) ** 2
        rv = rv.add(onr.reindex(rv.index).fillna(0.0), fill_value=0.0)
    return rv[rv > 0]


def har_table(rv, H):
    logrv = np.log(rv)
    F = pd.DataFrame(index=rv.index)
    F["d"] = logrv
    F["w"] = np.log(rv.rolling(5, min_periods=3).mean())
    F["m"] = np.log(rv.rolling(22, min_periods=10).mean())
    F = F.replace([np.inf, -np.inf], np.nan)
    target = np.log(rv.rolling(H).mean().shift(-H)).replace([np.inf, -np.inf], np.nan)
    return F, target


def daily_regime(rv, states=3, fit_frac=0.4):
    """Causal daily HMM vol regime (quiet/normal/explosive), filtered + sorted by vol."""
    from hmmlearn.hmm import GaussianHMM
    from hmm_regime import filtered_posteriors
    logrv = np.log(rv)
    feats = pd.DataFrame({
        "lrv": logrv,
        "lrv_w": np.log(rv.rolling(5, min_periods=3).mean()),
        "chg": logrv.diff(),
    }).replace([np.inf, -np.inf], np.nan).bfill().fillna(0.0)
    X = feats.to_numpy(); n = len(X); n0 = max(300, int(n * fit_frac))
    mu = X[:n0].mean(0); sd = X[:n0].std(0) + 1e-9; Xs = (X - mu) / sd
    m = GaussianHMM(n_components=states, covariance_type="full",
                    n_iter=80, random_state=42).fit(Xs[:n0])
    tr = m.predict(Xs[:n0])
    volby = [Xs[:n0][tr == k, 0].mean() if (tr == k).any() else 0 for k in range(states)]
    remap = {k: r for r, k in enumerate(np.argsort(volby))}
    reg = np.array([remap[k] for k in filtered_posteriors(Xs, m).argmax(1)])
    return pd.Series(reg, index=rv.index)


def walk_forward(F, target, H, folds, start_frac):
    both = F.notna().all(axis=1) & target.notna()
    sub = F[both]; Xb = sub.to_numpy(); yb = target[both].to_numpy(); dates = sub.index
    n = len(Xb); start = int(n * start_frac); step = max(1, (n - start) // folds)
    oy, op, od = [], [], []
    for f in range(folds):
        lo = start + f * step
        hi = (start + (f + 1) * step) if f < folds - 1 else n
        if lo - H < 100:
            continue
        m = LinearRegression().fit(Xb[:lo - H], yb[:lo - H])
        op.append(m.predict(Xb[lo:hi])); oy.append(yb[lo:hi]); od.append(dates[lo:hi])
    td = od[0]
    for d in od[1:]:
        td = td.append(d)                 # append keeps tz (np.concatenate drops it)
    return np.concatenate(oy), np.concatenate(op), td


def main(argv=None):
    ap = argparse.ArgumentParser(description="HAR vol forecast + probability cone.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[20, 60, 120])
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    ap.add_argument("--no-overnight", action="store_true", help="intraday-only RV")
    ap.add_argument("--garch", action="store_true")
    ap.add_argument("--regime", action="store_true")
    a = ap.parse_args(argv)

    df = drv._load_1min(a.input)
    rv = daily_total_var(df, overnight=not a.no_overnight)
    reg = daily_regime(rv) if a.regime else None
    cur = int(reg.iloc[-1]) if reg is not None else None
    hdr = f"=== {a.input}  HAR vol cone ({'overnight-adj' if not a.no_overnight else 'intraday-only'})"
    if reg is not None:
        hdr += f" | current regime: {RNAMES.get(cur, cur)}"
    print(hdr + " ===")
    cols = "forward-vol cone %:  10%   25%   50%   75%   90%"
    print(f"{'H':>4s} {'R2':>6s} {'RMSE':>6s} {'QLIKE':>6s} | {cols}"
          + ("  | GARCH50" if a.garch else ""))

    for H in a.horizons:
        F, target = har_table(rv, H)
        yt, yp, td = walk_forward(F, target, H, a.folds, a.start_frac)
        r2 = r2_score(yt, yp)
        rmse = np.sqrt(mean_squared_error(drv.annvol(yt), drv.annvol(yp)))
        ql = drv.qlike(np.exp(yt), np.exp(yp))
        resid_all = yt - yp

        comp = F.notna().all(axis=1) & target.notna()
        m = LinearRegression().fit(F[comp].to_numpy(), target[comp].to_numpy())
        last = F[F.notna().all(axis=1)].iloc[-1:]
        pt = float(m.predict(last.to_numpy())[0])

        # regime-condition the residuals (cone width/shape) on the CURRENT regime
        resid = resid_all
        if reg is not None:
            rmask = (reg.reindex(td).to_numpy() == cur)
            if rmask.sum() > 50:
                resid = resid_all[rmask]
        cone = [drv.annvol(pt + np.quantile(resid, q)) for q in QS]
        line = (f"{H:4d} {r2:6.3f} {rmse:6.2f} {ql:6.3f} | "
                + "  ".join(f"{v:5.1f}" for v in cone))
        if a.garch:
            g = _garch_latest(df, rv, H)
            line += f"  | {g:7.1f}"
        print(line)

    if reg is not None:
        _regime_summary(rv, a.horizons[0], a.folds, a.start_frac, reg)
    print(f"\n(level {'includes overnight' if not a.no_overnight else 'intraday-only'}; "
          f"cone width = real OOS HAR errors{' in current regime' if reg is not None else ''})")
    return 0


def _regime_summary(rv, H, folds, start_frac, reg):
    F, target = har_table(rv, H)
    yt, yp, td = walk_forward(F, target, H, folds, start_frac)
    resid = yt - yp
    r = reg.reindex(td).to_numpy()
    print(f"\nRegime cones (H={H}) — typical forward-vol band per regime:")
    for k in sorted(RNAMES):
        mk = r == k
        if mk.sum() < 30:
            continue
        med_pred = np.median(yp[mk])            # median HAR forecast in this regime
        v = [drv.annvol(med_pred + np.quantile(resid[mk], q)) for q in (0.1, 0.5, 0.9)]
        print(f"   {RNAMES[k]:10s} 10/50/90: {v[0]:4.1f} / {v[1]:4.1f} / {v[2]:4.1f}%   (n={int(mk.sum())})")


def _garch_latest(df, rv, H):
    try:
        from arch import arch_model
        r = (drv.daily_close_returns(df).dropna() * 100).to_numpy()[-2000:]
        res = arch_model(r, vol="Garch", p=1, q=1, mean="Zero", rescale=False).fit(disp="off")
        fc = res.forecast(horizon=H, reindex=False)
        var_daily = fc.variance.to_numpy().ravel().mean() / (100.0 ** 2)
        return np.sqrt(var_daily * drv.ANN) * 100.0
    except Exception:
        return float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
