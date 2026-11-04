"""모양 길이별 정밀 탐색 — '이 모양이면 롱/숏'이 어느 길이에서 통하나, 그리고 추세를 넘어서는가.  python shapes3.py
각 설정마다: 앞 60% 학습(K-평균 20개) → 뒤 40% 확인.
  ρ        = 군집별 '학습 평균 수익' vs '확인 평균 수익' 순위상관 (순열 p값)
  ρ(추세 제거) = 두 평균에서 '군집의 과거 구간 실제 수익(추세)' 효과를 빼고 남은 것끼리의 순위상관 → 모양 고유의 정보
  닮은꼴    = 이전 모양 중 가장 닮은 100개가 60% 이상 한쪽으로 쏠렸을 때 확인 구간 적중률
설정 11개를 전부 보고(여러 번 시험했으니 p<0.05 하나쯤은 우연히 나올 수 있음)."""
import json, math, warnings
import numpy as np, pandas as pd
from scipy.stats import rankdata
import data, indicators as ta, swing as W
from shapes import shapes, fwd

warnings.filterwarnings("ignore")
K, KNN = 20, 100
rng = np.random.default_rng(0)


def srho(a, b):
    ra, rb = rankdata(a), rankdata(b); return float(np.corrcoef(ra, rb)[0, 1])


def perm_p(a, b, n=3000):
    r0 = srho(a, b); rb = rankdata(b); ra = rankdata(a)
    sims = np.array([np.corrcoef(ra, rng.permutation(rb))[0, 1] for _ in range(n)])
    return r0, float((sims >= r0).mean())


def resid(y, x):
    X = np.column_stack([x, np.ones(len(x))]); return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def vol_shape(v, w):
    lv = np.log(np.maximum(v, 1e-9)); out = np.full((len(v), w), np.nan)
    for t in range(w - 1, len(v)):
        p = lv[t - w + 1:t + 1]; sd = p.std(); out[t] = (p - p.mean()) / sd if sd > 0 else 0.0
    return out


def run(scale, w, h, with_vol=False):
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    D = {s: (ta.resample_ohlcv(data.load_history(s, "1H", 1825, refresh=False), "4h").dropna() if scale == "4H" else W.load(s)) for s in ("BTCUSDT", "ETHUSDT")}
    S, F, PR_, T = [], [], [], []
    for s, df in D.items():
        c = df["close"].to_numpy(float); sh = shapes(c, w)
        if with_vol: sh = np.hstack([sh, 0.5 * vol_shape(df["volume"].to_numpy(float), w)])
        S.append(sh); F.append(fwd(c, h)); T.append(df.index.to_numpy())
        past = np.full(len(c), np.nan); past[w - 1:] = np.log(c[w - 1:]) - np.log(c[:len(c) - w + 1]); PR_.append(past)
    S, F, PR_, T = np.vstack(S), np.concatenate(F), np.concatenate(PR_), pd.to_datetime(np.concatenate(T))
    ok = np.isfinite(S).all(axis=1) & np.isfinite(F)
    split = T[ok].sort_values()[int(ok.sum() * 0.6)]
    tr, te = ok & (T < split), ok & (T >= split)
    km = KMeans(n_clusters=K, n_init=6, random_state=0).fit(S[tr]); lab = np.full(len(S), -1); lab[ok] = km.predict(S[ok])
    m_tr = np.array([F[tr & (lab == k)].mean() for k in range(K)]); m_te = np.array([F[te & (lab == k)].mean() if (te & (lab == k)).sum() else 0 for k in range(K)])
    trend = np.array([PR_[tr & (lab == k)].mean() for k in range(K)])
    rho, p = perm_p(m_tr, m_te); rho_r, p_r = perm_p(resid(m_tr, trend), resid(m_te, trend))
    same = int(((m_tr > 0) == (m_te > 0)).sum())
    # 닮은꼴 (월 단위로 그 이전 모양만 색인)
    preds = np.full(len(S), np.nan); gap = pd.Timedelta(hours=4 * h) if scale == "4H" else pd.Timedelta(days=h)
    for t0 in pd.date_range(split, T.max(), freq="30D"):
        hist = ok & (T < t0 - gap); q = np.flatnonzero(te & (T >= t0) & (T < t0 + pd.Timedelta(days=30)))
        if len(q) == 0 or hist.sum() < KNN: continue
        _, ind = NearestNeighbors(n_neighbors=KNN).fit(S[hist]).kneighbors(S[q]); preds[q] = (F[hist][ind] > 0).mean(axis=1)
    L = np.isfinite(preds) & (preds >= 0.6); Sh = np.isfinite(preds) & (preds <= 0.4)
    hitL = float((F[L] > 0).mean()) if L.sum() else float("nan"); hitS = float((F[Sh] < 0).mean()) if Sh.sum() else float("nan")
    base_up = float((F[te] > 0).mean())
    return {"scale": scale, "w": w, "h": h, "vol": with_vol, "split": str(split.date()), "rho": rho, "p": p, "rho_beyond": rho_r, "p_beyond": p_r, "same": same,
            "knn_long_hit": hitL, "knn_long_n": int(L.sum()), "knn_short_hit": hitS, "knn_short_n": int(Sh.sum()), "base_up": base_up}


def main():
    cfgs = [("4H", 12, 6, False), ("4H", 30, 6, False), ("4H", 30, 18, False), ("4H", 60, 18, False), ("4H", 120, 42, False),
            ("1D", 10, 5, False), ("1D", 20, 5, False), ("1D", 30, 5, False), ("1D", 30, 10, False), ("1D", 60, 10, False), ("1D", 90, 20, False),
            ("1D", 30, 5, True)]
    lab = lambda sc, w, h: (f"4시간 {w}봉({w*4//24}일) → {h*4}시간" if sc == "4H" else f"일봉 {w}일 → {h}일")
    out = []
    print(f"{'설정':26s} {'거래량':>4s} | {'ρ':>6s} {'p':>5s} | {'ρ 추세제거':>9s} {'p':>5s} | {'같은방향':>6s} | 닮은꼴 롱쏠림 적중 / 숏쏠림 적중 (그냥 오른 비율)")
    for sc, w, h, v in cfgs:
        r = run(sc, w, h, v); out.append(r)
        print(f"{lab(sc, w, h):26s} {'포함' if v else '-':>4s} | {r['rho']:+6.2f} {r['p']:5.2f} | {r['rho_beyond']:+9.2f} {r['p_beyond']:5.2f} | {r['same']:4d}/20 | "
              f"{r['knn_long_hit']:.1%} (n={r['knn_long_n']}) / {r['knn_short_hit']:.1%} (n={r['knn_short_n']})  ({r['base_up']:.1%})", flush=True)
    with open("state/shapes3_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
