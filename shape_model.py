"""두 달(60일) 모양 모델 저장·실시간 판정.  python shape_model.py  → state/shape60_model.json
학습: 비트겟 현물 일봉 2018-07~2023-07(앞 60%), K-평균 20개. 이후 기간은 확인용으로만 사용(shapes4.py)."""
import json, math
import numpy as np, pandas as pd
import data
from shapes import shapes

MODEL = data.STATE / "shape60_model.json"
W, HOLD = 60, 10
TVD = 0.40 / math.sqrt(365)


def build():
    import shapes4 as S4
    D, S, F, P, T, SY = S4.dataset(W, HOLD); km, lab, ok, split, tr = S4.fit(S, F, T, 20)
    te = ok & (T >= split)
    m_tr = [float(F[tr & (lab == j)].mean()) for j in range(20)]
    m_te = [float(F[te & (lab == j)].mean()) if (te & (lab == j)).sum() else None for j in range(20)]
    js = {"w": W, "hold": HOLD, "split": str(split.date()), "centroids": km.cluster_centers_.tolist(), "m_tr": m_tr, "m_te": m_te,
          "n_tr": [int((tr & (lab == j)).sum()) for j in range(20)]}
    with open(MODEL, "w", encoding="utf-8") as fh:
        json.dump(js, fh, ensure_ascii=False)
    return js


def load_model():
    with open(MODEL, encoding="utf-8") as fh:
        return json.load(fh)


def assign(S, C):
    d = ((S[:, None, :] - C[None, :, :]) ** 2).sum(axis=2); return d.argmin(axis=1)


def state(df: pd.DataFrame, M=None) -> dict:
    """일봉 df → 현재 모양 군집, 방향, 크기(연변동성 40% 기준, 10일 유지, 완충 0.15)."""
    import novel as N
    M = M or load_model(); C = np.array(M["centroids"]); m_tr = np.array(M["m_tr"])
    c = df["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))])
    S = shapes(c, W); ok = np.isfinite(S).all(axis=1); lab = np.full(len(c), -1); lab[ok] = assign(S[ok], C)
    sig = np.where(lab >= 0, np.sign(m_tr[np.maximum(lab, 0)]), 0.0)
    sgm = np.sqrt(pd.Series(r ** 2).ewm(alpha=0.06, adjust=False).mean().to_numpy()); cap = np.minimum(TVD / np.maximum(sgm, 1e-6), 3.0)
    pos = np.zeros(len(c)); cur, left, last_change = 0.0, 0, None
    for i, v in enumerate(sig):
        if v != cur and left <= 0: cur, left, last_change = v, HOLD, i
        pos[i] = cur; left -= 1
    pb = N.band(pos * cap, 0.15)
    k = int(lab[-1])
    return {"cluster": k, "side": float(sig[-1]), "held_side": float(pos[-1]), "pos": float(pb[-1]), "m_tr": float(m_tr[k]), "m_te": M["m_te"][k],
            "days_left": int(max(left, 0)), "since": str(df.index[last_change].date()) if last_change is not None else "", "shape": S[-1].tolist(),
            "centroid": C[k].tolist(), "date": str(df.index[-1].date())}


if __name__ == "__main__":
    js = build(); print("저장:", MODEL, "학습 분할", js["split"])
    for s in ("BTCUSDT", "ETHUSDT"):
        df = data.drop_unclosed(data.fetch_candles(s, "1D", 400), "1D"); st_ = state(df)
        print(s, {k: v for k, v in st_.items() if k not in ("shape", "centroid")})
