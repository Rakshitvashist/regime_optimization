"""
macro_regime.py  —  cross-asset RISK-ON / RISK-OFF regime (Nifty + Crude + Gold).

Crude and gold encode macro risk state: in risk-OFF, equities fall, gold (hedge)
rises, crude (growth proxy) falls, the rupee weakens. A causal Gaussian-HMM on the
joint daily moves of Nifty / Crude / Gold + equity vol detects that regime — the
macro overlay your correlation work pointed to (negative at monthly/quarterly,
gold inverse in crises).

States are sorted by equity return -> risk_off / neutral / risk_on. Filtered
(forward-algorithm) so the current label uses only past data (no look-ahead).

Usage:  python macro_regime.py
"""
from __future__ import annotations

import numpy as np, pandas as pd
import daily_rv_forecast as drv

NAMES = {0: "risk-off", 1: "neutral", 2: "risk-on"}


def _close(p):
    df = drv._load_1min(p)
    return df["Close"].astype(float).groupby(drv._day_index(df.index)).last()


def macro_features(nifty="NIFTY_50.csv", crude="CRUDEOIL.csv", gold="GOLD.csv"):
    P = pd.DataFrame({"nifty": _close(nifty), "crude": _close(crude),
                      "gold": _close(gold)}).dropna()
    P.index = pd.to_datetime([str(d) for d in P.index])
    R = np.log(P).diff()
    F = pd.DataFrame(index=P.index)
    F["nifty"] = R["nifty"]; F["crude"] = R["crude"]; F["gold"] = R["gold"]
    F["nifty_vol"] = R["nifty"].rolling(10, min_periods=5).std()
    F["gold_minus_eq"] = R["gold"] - R["nifty"]      # hedge proxy (high in risk-off)
    return F.dropna()


def compute(fit_frac=0.5, states=3):
    from hmmlearn.hmm import GaussianHMM
    from hmm_regime import filtered_posteriors
    F = macro_features()
    X = F.to_numpy(); n = len(X); n0 = max(200, int(n * fit_frac))
    mu = X[:n0].mean(0); sd = X[:n0].std(0) + 1e-9; Xs = (X - mu) / sd
    m = GaussianHMM(n_components=states, covariance_type="full", n_iter=80, random_state=42).fit(Xs[:n0])
    tr = m.predict(Xs[:n0])
    eqret = [Xs[:n0][tr == k, 0].mean() if (tr == k).any() else 0 for k in range(states)]
    remap = {k: r for r, k in enumerate(np.argsort(eqret))}   # 0=lowest eq ret = risk-off
    reg = np.array([remap[k] for k in filtered_posteriors(Xs, m).argmax(1)])
    F = F.assign(regime=reg)
    return F


def main():
    F = compute()
    cur = int(F["regime"].iloc[-1])
    print(f"=== Macro risk regime (Nifty+Crude+Gold) — current: {NAMES[cur].upper()} "
          f"(as of {F.index[-1].date()}) ===\n")
    print(f"{'regime':10s} {'share':>6s} {'nifty':>8s} {'crude':>8s} {'gold':>8s} {'eq vol':>8s}")
    for k in range(3):
        m = F["regime"] == k
        if not m.any():
            continue
        print(f"{NAMES[k]:10s} {m.mean():6.1%} {F.nifty[m].mean()*100:+7.3f}% "
              f"{F.crude[m].mean()*100:+7.3f}% {F.gold[m].mean()*100:+7.3f}% {F.nifty_vol[m].mean()*100:7.3f}%")
    print("\n(risk-off should show: nifty down, gold up, crude down — the macro hedge pattern)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
