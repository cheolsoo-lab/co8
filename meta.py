"""메타 라벨 머신러닝 — BTC·ETH 전용 특징(시장 폭 제외).  방향은 기존 그대로, 크기만 ×(0.5 + p).
p = '지금 포지션이 다음 24시간 이길 확률'을 그래디언트 부스팅으로 예측. 검증은 60일마다 과거만으로 재학습(워크포워드)."""
import math
import numpy as np, pandas as pd
import novel as N, robust as RB, swing_live as SL, indicators as ta

FEATS = ["d", "d1", "d2", "|d|", "daily", "lead", "r6", "r42", "volr", "pnl42", "pnl180"]


def frame(df, other_d=None):
    c = df["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))]); ar = np.exp(r) - 1
    sg = N.sigma(r); cap = np.minimum(N.TV / sg, N.LMAX)
    d1, d2 = RB.tsmom_dir(r), RB.breath_dir(r); d = 0.5 * d1 + 0.5 * d2
    dd = ta.resample_ohlcv(df, "1D").dropna()
    daily = (pd.Series(np.mean([fn(dd) for fn in SL.RULES.values()], axis=0), index=dd.index + pd.Timedelta(days=1)).reindex(df.index, method="ffill").fillna(0).to_numpy()
             if len(dd) >= 220 else np.zeros(len(c)))
    lead = np.zeros(len(c)) if other_d is None else other_d.reindex(df.index).fillna(0).to_numpy()
    base = N.band(d * cap, 0.15); p = np.concatenate([[0.0], base[:-1]]); bar = p * ar
    X = np.column_stack([d, d1, d2, np.abs(d), daily, lead,
                         pd.Series(r).rolling(6).sum() / (sg * math.sqrt(6) + 1e-12), pd.Series(r).rolling(42).sum() / (sg * math.sqrt(42) + 1e-12),
                         N.sigma(r, 0.1) / (N.sigma(r, 0.01) + 1e-12), pd.Series(bar).rolling(42).sum(), pd.Series(bar).rolling(180).sum()])
    fwd = pd.Series(np.concatenate([bar[1:], [0.0]])).rolling(6).sum().shift(-5).to_numpy().copy()
    fwd[-6:] = np.nan
    return {"r": r, "ar": ar, "cap": cap, "d": d, "d1": d1, "d2": d2, "base": base, "X": np.nan_to_num(X), "fwd": fwd, "idx": df.index}


def model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=150, l2_regularization=1.0, random_state=0)


def walk_forward(F, step=360, burn=2190):
    y = (F["fwd"] > 0).astype(int); ok = np.isfinite(F["fwd"]) & (np.abs(F["base"]) > 0); mult = np.ones(len(y))
    for start in range(burn, len(y), step):
        tr = np.arange(0, start - 6); tr = tr[ok[tr]]
        if len(tr) < 500: continue
        m = model().fit(F["X"][tr], y[tr]); te = np.arange(start, min(start + step, len(y)))
        mult[te] = 0.5 + m.predict_proba(F["X"][te])[:, 1]
    return mult


def fit_all(F):
    y = (F["fwd"] > 0).astype(int); ok = np.isfinite(F["fwd"]) & (np.abs(F["base"]) > 0)
    return model().fit(F["X"][ok], y[ok])


def positions(F, mult):
    return N.band(F["d"] * F["cap"] * mult, 0.15)
