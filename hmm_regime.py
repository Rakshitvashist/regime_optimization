"""
hmm_regime.py  —  CAUSAL Hidden-Markov-Model regime detector (signal layer).

Unsupervised market-regime labeling with a Gaussian HMM. Regimes are sorted by
volatility -> quiet / normal / explosive (n_states=3) so the labels are stable
and meaningful.

NO LOOK-AHEAD (the trap the standard HMM falls into):
  - Standard `model.predict()` is Viterbi over the WHOLE sequence -> uses the
    future to label bar t. We instead use FILTERED posteriors: the forward
    algorithm gives P(state_t | observations 0..t) using PAST only.
  - Walk-forward: the HMM is refit on an expanding PAST window each fold; the
    state->regime (vol) mapping is taken from TRAIN only.

Features (all causal): log-return, realized vol (rolling std), |return|,
trend spread (EMA20-EMA100)/price, and optional VIX level/change.

Reports, on the walk-forward (out-of-sample) bars, whether the regime actually
separates future behavior: forward |move| and spike-rate per regime (a good
vol-regime detector shows explosive >> quiet). Evaluation uses forward returns;
the regime LABELS never do.

Usage:
  python hmm_regime.py --input processed_intraday --symbol BANKNIFTY-30m --states 3
  python hmm_regime.py --input processed_intraday --symbol NIFTY-50-30m --vix INDIA_VIX.csv
"""
from __future__ import annotations

import argparse
import numpy as np, pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

from optimization import data_loader


def causal_features(df, vix_path=None, rule="30min"):
    c = df["close"].astype(float)
    ret = np.log(c).diff().fillna(0.0)
    rv = ret.rolling(20, min_periods=5).std().bfill()
    absret = ret.abs()
    ema_f = c.ewm(span=20, adjust=False).mean()
    ema_s = c.ewm(span=100, adjust=False).mean()
    trend = ((ema_f - ema_s) / (c + 1e-9))
    feats = pd.DataFrame({"ret": ret, "rv": rv, "absret": absret, "trend": trend})
    if vix_path:
        from vix_features import load_vix_1min, vix_features
        vf = vix_features(df.index, load_vix_1min(vix_path), rule)
        feats["vix"] = vf["vix_level"].to_numpy()
        feats["vix_chg"] = vf["vix_chg_5"].to_numpy()
    return feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def filtered_posteriors(X, model):
    """P(state_t | obs 0..t) — causal forward algorithm (no future)."""
    K = model.n_components
    logB = np.column_stack([
        multivariate_normal.logpdf(X, model.means_[k], model.covars_[k], allow_singular=True)
        for k in range(K)])
    logA = np.log(model.transmat_ + 1e-300)
    logpi = np.log(model.startprob_ + 1e-300)
    T = len(X)
    la = np.empty((T, K)); la[0] = logpi + logB[0]
    for t in range(1, T):
        la[t] = logsumexp(la[t-1][:, None] + logA, axis=0) + logB[t]
    return np.exp(la - logsumexp(la, axis=1, keepdims=True))


def compute_regime(df, states, folds, start_frac, vix_path, rule):
    """Walk-forward CAUSAL HMM regime labels -> np.ndarray (-1 where unlabeled)."""
    from hmmlearn.hmm import GaussianHMM
    F = causal_features(df, vix_path, rule)
    Xall = F.to_numpy(float); T = len(df)
    rvi = F.columns.get_loc("rv")
    start = int(T * start_frac); step = (T - start) // folds
    regime = np.full(T, -1)
    for f in range(folds):
        te_lo = start + f * step
        te_hi = (start + (f + 1) * step) if f < folds - 1 else T
        tr_hi = te_lo
        if tr_hi < 500:
            continue
        mu = Xall[:tr_hi].mean(0); sd = Xall[:tr_hi].std(0) + 1e-9   # train-only scaling
        Xs = (Xall - mu) / sd
        m = GaussianHMM(n_components=states, covariance_type="full",
                        n_iter=80, random_state=42, tol=1e-3)
        try:
            m.fit(Xs[:tr_hi])
        except Exception:
            continue
        # map states -> vol rank using TRAIN labels only
        tr_states = m.predict(Xs[:tr_hi])
        vol_by_state = [Xs[:tr_hi][tr_states == k, rvi].mean() if (tr_states == k).any() else 0
                        for k in range(states)]
        remap = {k: r for r, k in enumerate(np.argsort(vol_by_state))}   # 0=lowest vol
        post = filtered_posteriors(Xs[:te_hi], m)        # causal, past-only
        raw = post[te_lo:te_hi].argmax(1)
        regime[te_lo:te_hi] = [remap[k] for k in raw]
    return regime


def run(df, states, folds, start_frac, vix_path, rule):
    F = causal_features(df, vix_path, rule)
    regime = compute_regime(df, states, folds, start_frac, vix_path, rule)
    close = df["close"].to_numpy(float); T = len(df)
    H = 5
    days = pd.Series(df.index).dt.normalize().to_numpy()
    fwd = np.full(T, np.nan); fwd[:T-H] = (close[H:] - close[:T-H]) / close[:T-H]
    same = np.zeros(T, bool); same[:T-H] = days[H:] == days[:T-H]
    fwd = np.where(same, fwd, np.nan)
    atr = data_loader.compute_atr(df).to_numpy(float)

    names = {0: "quiet", 1: "normal", 2: "explosive"} if states == 3 else \
            {i: f"r{i}" for i in range(states)}
    mask = (regime >= 0) & np.isfinite(fwd)
    print(f"=== HMM regime  {df.attrs.get('sym','')}  | states={states} "
          f"| VIX={'yes' if vix_path else 'no'} | OOS bars={int(mask.sum()):,} ===")
    print(f"{'regime':10s} {'share':>7s} {'fwd|move|%':>11s} {'spike_rate':>11s} {'mean_rv%':>9s}")
    for r in range(states):
        rm = mask & (regime == r)
        if rm.sum() == 0:
            continue
        fm = np.abs(fwd[rm]).mean() * 100
        sr = (np.abs(fwd[rm]) > 1.0 * atr[rm] / close[rm]).mean()
        rvm = F["rv"].to_numpy()[rm].mean() * 100
        print(f"{names.get(r,r):10s} {rm.sum()/mask.sum():7.1%} {fm:11.3f} {sr:11.3f} {rvm:9.3f}")
    # persistence
    rr = regime[regime >= 0]
    if len(rr) > 1:
        stay = (rr[1:] == rr[:-1]).mean()
        print(f"persistence (P stay same regime next bar): {stay:.3f}")
    # separation ratio: explosive vs quiet forward move
    if states == 3:
        q = mask & (regime == 0); e = mask & (regime == 2)
        if q.sum() and e.sum():
            ratio = (np.abs(fwd[e]).mean()) / (np.abs(fwd[q]).mean() + 1e-12)
            print(f"explosive/quiet forward-move ratio: {ratio:.2f}x "
                  f"(higher = regimes separate volatility better)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Causal HMM regime detector.")
    ap.add_argument("--input", default="processed_intraday")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--states", type=int, default=3)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--start-frac", type=float, default=0.4)
    ap.add_argument("--vix", default=None)
    a = ap.parse_args(argv)
    by_sym = {s: pth for s, pth in data_loader.list_symbols(a.input)}
    if a.symbol not in by_sym:
        print(f"ERROR: {a.symbol} not found"); return 1
    df = data_loader.load_symbol_frame(by_sym[a.symbol]); df.attrs["sym"] = a.symbol
    from vix_features import rule_for_symbol
    run(df, a.states, a.folds, a.start_frac, a.vix, rule_for_symbol(a.symbol))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
