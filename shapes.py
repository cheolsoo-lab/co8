"""규칙 없이 '모양 → 방향' 찾기 (BTC·ETH).  python shapes.py
모양 = 최근 W봉 로그가격 경로에서 마지막 값을 빼고 표준편차로 나눈 것(크기와 가격대 무시, 순수한 모양).
A) 모양 사전: 앞 60% 기간의 모양을 K-평균으로 20개 군집 → 군집별 이후 수익(겹침 보정 t값) → |t|≥2인 군집을 롱/숏 모양으로 지정 → 뒤 40%에서 확인
B) 닮은꼴: 각 시점에서 '그 이전'의 모든 모양 중 가장 닮은 100개 → 그 뒤 방향 다수결(60% 이상 쏠릴 때만)
척도: 4H × 30봉(5일 모양) → 다음 24시간·3일 / 일봉 × 30봉(한 달 모양, 2018~) → 다음 5일·10일. 매매 성적은 지정가 0.03%."""
import json, math, warnings
import numpy as np, pandas as pd
import data, indicators as ta, swing as W

warnings.filterwarnings("ignore")
K, KNN, COST = 20, 100, 0.0003


def shapes(c, w):
    x = np.log(c); n = len(x); S = np.full((n, w), np.nan)
    for t in range(w - 1, n):
        p = x[t - w + 1:t + 1] - x[t]; sd = p.std()
        S[t] = p / sd if sd > 0 else 0.0
    return S


def load(scale):
    out = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        if scale == "4H":
            df = ta.resample_ohlcv(data.load_history(s, "1H", 1825, refresh=False), "4h").dropna()
        else:
            df = W.load(s)
        out[s] = df
    return out


def fwd(c, h):
    x = np.log(c); f = np.full(len(x), np.nan); f[:-h] = x[h:] - x[:-h]; return f


def study(scale, w, hs, labels):
    from sklearn.cluster import KMeans
    from sklearn.neighbors import NearestNeighbors
    D = load(scale); res = {"scale": scale, "w": w}
    S, Fw, T, SYM = [], {h: [] for h in hs}, [], []
    for s, df in D.items():
        c = df["close"].to_numpy(float); sh = shapes(c, w)
        for h in hs: Fw[h].append(fwd(c, h))
        S.append(sh); T.append(df.index.to_numpy()); SYM += [s] * len(c)
    S = np.vstack(S); Fw = {h: np.concatenate(v) for h, v in Fw.items()}; T = np.concatenate(T); SYM = np.array(SYM)
    ok = np.isfinite(S).all(axis=1)
    tt = pd.to_datetime(T); split = tt[ok].sort_values()[int(ok.sum() * 0.6)]
    tr, te = ok & (tt < split), ok & (tt >= split)
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(S[tr])
    lab = np.full(len(S), -1); lab[ok] = km.predict(S[ok])
    h0 = hs[0]
    print(f"\n════ {scale} · 최근 {w}봉 모양 · 분할 {split.date()} (앞 60% 학습 / 뒤 40% 확인) ════")
    print(f"{'군집':>4s} {'앞 n':>6s} {'앞 평균':>8s} {'앞 t':>6s} {'지정':>4s} | {'뒤 n':>6s} {'뒤 평균':>8s} {'뒤 적중':>7s}  ({labels[0]} 기준)")
    clus = []
    for k in range(K):
        a = (lab == k) & tr & np.isfinite(Fw[h0]); b = (lab == k) & te & np.isfinite(Fw[h0])
        ma, sa = Fw[h0][a].mean(), Fw[h0][a].std(); n_eff = a.sum() / h0; t = ma / (sa / math.sqrt(max(n_eff, 1)))
        side = 1 if t >= 2 else (-1 if t <= -2 else 0)
        mb = Fw[h0][b].mean() if b.sum() else np.nan
        hit = (np.sign(Fw[h0][b]) == side).mean() if (b.sum() and side) else np.nan
        clus.append({"k": k, "n_tr": int(a.sum()), "m_tr": float(ma), "t": float(t), "side": side, "n_te": int(b.sum()), "m_te": float(mb), "hit_te": float(hit) if np.isfinite(hit) else None,
                     "shape": km.cluster_centers_[k].tolist()})
        print(f"{k:4d} {a.sum():6d} {ma:+8.2%} {t:+6.1f} {({1: '롱', -1: '숏', 0: '-'})[side]:>4s} | {b.sum():6d} {mb:+8.2%} {(f'{hit:.1%}' if np.isfinite(hit) else '-'):>7s}")
    res["clusters"] = clus
    # 지정 군집 합산 (뒤 구간)
    for h, lb in zip(hs, labels):
        sides = np.array([c_["side"] for c_ in clus]); sig = np.where(lab >= 0, sides[np.maximum(lab, 0)], 0)
        m = te & (sig != 0) & np.isfinite(Fw[h])
        r_dir = sig[m] * Fw[h][m]; hit = (r_dir > 0).mean()
        base_up = (Fw[h][te & np.isfinite(Fw[h])] > 0).mean()
        print(f"  ▶ 롱/숏 모양 전체 (뒤 구간, {lb}): 신호 {m.sum()}개, 적중 {hit:.1%} (그냥 롱 적중 {base_up:.1%}), 방향 평균 수익 {r_dir.mean():+.2%}")
        res[f"A|{lb}"] = {"n": int(m.sum()), "hit": float(hit), "ret": float(r_dir.mean()), "base_up": float(base_up)}
    # B) 닮은꼴 — 월(또는 30봉) 단위로 '그 이전' 모양만 색인
    step = 180 if scale == "4H" else 30
    order = np.argsort(tt.values); test_ix = np.flatnonzero(te)
    preds = np.full(len(S), np.nan)
    times = tt.values
    for t0 in pd.date_range(split, tt.max(), freq=("30D" if scale == "4H" else "30D")):
        t1 = t0 + pd.Timedelta(days=30)
        hist = ok & np.isfinite(Fw[h0]) & (times < np.datetime64(t0 - pd.Timedelta(hours=4 * h0 if scale == "4H" else 24 * h0)))
        q = np.flatnonzero(te & (times >= np.datetime64(t0)) & (times < np.datetime64(t1)))
        if len(q) == 0 or hist.sum() < KNN: continue
        nn = NearestNeighbors(n_neighbors=KNN).fit(S[hist]); _, ind = nn.kneighbors(S[q])
        preds[q] = (Fw[h0][hist][ind] > 0).mean(axis=1)
    print(f"  ▶ 닮은꼴 100개 다수결 (뒤 구간, {labels[0]}):")
    resB = {}
    for lo, hi, nm in ((0.0, 0.4, "숏 쏠림 (오른 비율 ≤40%)"), (0.4, 0.6, "애매 (40~60%)"), (0.6, 1.01, "롱 쏠림 (≥60%)")):
        m = np.isfinite(preds) & (preds >= lo) & (preds < hi) & np.isfinite(Fw[h0])
        side = -1 if hi <= 0.4 else (1 if lo >= 0.6 else 0)
        if m.sum() == 0: continue
        up = (Fw[h0][m] > 0).mean(); rr = Fw[h0][m].mean()
        print(f"     {nm:22s} {m.sum():6d}개 · 실제로 오른 비율 {up:.1%} · 평균 {rr:+.2%}" + (f" · 방향 적중 {(up if side > 0 else 1 - up):.1%}" if side else ""))
        resB[nm] = {"n": int(m.sum()), "up": float(up), "ret": float(rr)}
    res["B"] = resB
    # 매매 성적 (모양 사전 신호, h0 동안 보유, 코인별 → 합산)
    sides = np.array([c_["side"] for c_ in clus])
    pn = []
    for s in ("BTCUSDT", "ETHUSDT"):
        mk = SYM == s; lb_ = lab[mk]; c = D[s]["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))])
        raw = np.where(lb_ >= 0, sides[np.maximum(lb_, 0)], 0).astype(float)
        pos = np.zeros(len(raw)); cur, left = 0.0, 0
        for i, v in enumerate(raw):
            if v != 0 and (v != cur or left <= 0): cur, left = v, h0
            pos[i] = cur if left > 0 else 0.0; left -= 1
        p = np.concatenate([[0.0], pos[:-1]]); x = p * (np.exp(r) - 1) - COST * np.abs(np.diff(p, prepend=0.0))
        tm = pd.to_datetime(D[s].index) >= split; pn.append(x[tm])
    L = min(len(pn[0]), len(pn[1])); xx = 0.5 * (pn[0][-L:] + pn[1][-L:]); ann = 6 * 365 if scale == "4H" else 365
    e = np.cumprod(1 + xx)
    print(f"  ▶ 모양 사전으로 매매 (뒤 구간, BTC·ETH 반반, 1배): 샤프 {xx.mean()/xx.std()*math.sqrt(ann):+.2f}, 연 {e[-1]**(ann/len(xx))-1:+.0%}, 낙폭 {(e/np.maximum.accumulate(e)-1).min():.0%}")
    res["trade"] = {"sharpe": float(xx.mean() / xx.std() * math.sqrt(ann)), "ann": float(e[-1] ** (ann / len(xx)) - 1)}
    # 현재 모양
    now = {}
    for s, df in D.items():
        c = df["close"].to_numpy(float); shp = shapes(c, w)[-1]; k = int(km.predict(shp[None])[0])
        hist = ok & np.isfinite(Fw[h0]); nn = NearestNeighbors(n_neighbors=KNN).fit(S[hist]); _, ind = nn.kneighbors(shp[None])
        up = float((Fw[h0][hist][ind[0]] > 0).mean())
        now[s] = {"cluster": k, "side": clus[k]["side"], "knn_up": up, "shape": shp.tolist()}
        print(f"  지금 {s[:3]}: 모양 군집 {k} ({({1: '롱 모양', -1: '숏 모양', 0: '중립 모양'})[clus[k]['side']]}), 닮은꼴 100개 중 {labels[0]} 뒤 오른 비율 {up:.0%}")
    res["now"] = now
    return res


def main():
    out = {"4H": study("4H", 30, (6, 18), ("24시간", "3일")), "1D": study("1D", 30, (5, 10), ("5일", "10일"))}
    with open("state/shapes_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
