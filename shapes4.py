"""일봉 60일 모양 → 다음 10일: 견고성 확인 + 매매 검증.  python shapes4.py
1) 견고성(미리 고정): 길이 45·60·75일 × 모양 개수 12·20·30 → 추세 제거 순위상관이 대부분 양수·유의한가
2) 매매(확인 구간만): 현재 모양 군집의 학습 평균 부호로 방향, 10일 보유, 연변동성 40% 기준 크기, 지정가 0.03%, 롱 펀딩 연 10%
   단독 / 앱 방식과 자본 반반. 판정: 반반의 합산 샤프 +0.10 이상, BTC·ETH 각각, 앞/뒤 절반 각각 앱 방식 이상."""
import json, math, warnings
import numpy as np, pandas as pd
import data, indicators as ta, swing as W, novel as N, precision as PR
from shapes import shapes, fwd
from shapes3 import perm_p, resid

warnings.filterwarnings("ignore")
TVD = 0.40 / math.sqrt(365)


def dataset(w, h):
    D = {s: W.load(s) for s in ("BTCUSDT", "ETHUSDT")}
    S, F, P, T, SY = [], [], [], [], []
    for s, df in D.items():
        c = df["close"].to_numpy(float); S.append(shapes(c, w)); F.append(fwd(c, h)); T.append(df.index.to_numpy()); SY += [s] * len(c)
        past = np.full(len(c), np.nan); past[w - 1:] = np.log(c[w - 1:]) - np.log(c[:len(c) - w + 1]); P.append(past)
    return D, np.vstack(S), np.concatenate(F), np.concatenate(P), pd.to_datetime(np.concatenate(T)), np.array(SY)


def fit(S, F, T, k):
    from sklearn.cluster import KMeans
    ok = np.isfinite(S).all(axis=1) & np.isfinite(F); split = T[ok].sort_values()[int(ok.sum() * 0.6)]
    tr = ok & (T < split); km = KMeans(n_clusters=k, n_init=6, random_state=0).fit(S[tr])
    lab = np.full(len(S), -1); okS = np.isfinite(S).all(axis=1); lab[okS] = km.predict(S[okS])
    return km, lab, ok, split, tr


def main():
    out = {"robust": []}
    print("══ 1) 견고성: 추세 제거 후 순위상관 (p값) — 길이 × 모양 개수 ══")
    print(f"{'':10s}" + "".join(f"{f'모양 {k}개':>16s}" for k in (12, 20, 30)))
    for w in (45, 60, 75):
        D, S, F, P, T, SY = dataset(w, 10); cells = []
        for k in (12, 20, 30):
            km, lab, ok, split, tr = fit(S, F, T, k); te = ok & (T >= split)
            m_tr = np.array([F[tr & (lab == j)].mean() for j in range(k)]); m_te = np.array([F[te & (lab == j)].mean() if (te & (lab == j)).sum() else 0 for j in range(k)])
            tr_ = np.array([P[tr & (lab == j)].mean() for j in range(k)])
            r, p = perm_p(resid(m_tr, tr_), resid(m_te, tr_), 2000); r0, p0 = perm_p(m_tr, m_te, 2000)
            cells.append(f"{r:+.2f} ({p:.2f})"); out["robust"].append({"w": w, "k": k, "rho": r0, "p": p0, "rho_beyond": r, "p_beyond": p})
        print(f"{f'{w}일':10s}" + "".join(f"{c_:>16s}" for c_ in cells))
    # 2) 매매
    D, S, F, P, T, SY = dataset(60, 10); km, lab, ok, split, tr = fit(S, F, T, 20)
    m_tr = np.array([F[tr & (lab == j)].mean() for j in range(20)])
    sh = lambda x: float(x.mean() / x.std() * math.sqrt(365)) if x.std() > 0 else 0.0
    per = {}
    for s in D:
        df = D[s]; c = df["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))])
        lb = lab[SY == s]; sig = np.where(lb >= 0, np.sign(m_tr[np.maximum(lb, 0)]), 0.0)
        sgm = np.sqrt(pd.Series(r ** 2).ewm(alpha=0.06, adjust=False).mean().to_numpy()); cap = np.minimum(TVD / np.maximum(sgm, 1e-6), 3.0)
        pos = np.zeros(len(c)); cur, left = 0.0, 0
        for i, v in enumerate(sig):
            if v != cur and left <= 0: cur, left = v, 10
            pos[i] = cur; left -= 1
        pos = N.band(pos * cap, 0.15); p = np.concatenate([[0.0], pos[:-1]])
        xs = pd.Series(p * (np.exp(r) - 1) - 0.0003 * np.abs(np.diff(p, prepend=0.0)) - 0.10 / 365 * np.maximum(p, 0), index=df.index)
        H4 = ta.resample_ohlcv(data.load_history(s, "1H", 1825, refresh=False), "4h").dropna(); Fq = PR.coin_frame(H4)
        xa = pd.Series(PR.pnl(N.band(Fq["d"] * Fq["cap"], 0.15), Fq, 0.0003), index=Fq["idx"])
        a_ = (1 + xa).resample("1D", offset="16h").prod() - 1; a_.index = a_.index.floor("1D"); xs.index = xs.index.floor("1D")
        j = pd.concat([a_, xs], axis=1, keys=["app", "shape"]).dropna(); per[s] = j[j.index >= split.floor("1D")]
    print(f"\n══ 2) 매매 — 확인 구간 {per['BTCUSDT'].index[0].date()} ~ {per['BTCUSDT'].index[-1].date()} (모양은 그 이전으로만 학습) ══")
    rows = {}
    for nm, f in (("앱 방식만", lambda j: j["app"]), ("60일 모양만", lambda j: j["shape"]), ("반반 운용", lambda j: 0.5 * j["app"] + 0.5 * j["shape"])):
        xb, xe = f(per["BTCUSDT"]), f(per["ETHUSDT"]); x = (0.5 * (xb + xe)).dropna(); h = len(x) // 2; e = (1 + x).cumprod()
        yr = {int(y): sh(x[x.index.year == y]) for y in sorted(set(x.index.year))}
        rows[nm] = {"sh": sh(x), "h1": sh(x[:h]), "h2": sh(x[h:]), "BTC": sh(xb), "ETH": sh(xe), "ann": float(e.iloc[-1] ** (365 / len(x)) - 1), "mdd": float((e / e.cummax() - 1).min()), "yrs": yr}
        o = rows[nm]
        print(f"  {nm:12s} 샤프 {o['sh']:+.2f} (앞/뒤 {o['h1']:+.2f}/{o['h2']:+.2f}) · BTC {o['BTC']:+.2f} · ETH {o['ETH']:+.2f} · 연 {o['ann']:+.0%} · 낙폭 {o['mdd']:.0%} | " + " ".join(f"{y%100:02d}:{v:+.2f}" for y, v in yr.items()))
    b, c_ = rows["앱 방식만"], rows["반반 운용"]
    ok_ = c_["sh"] - b["sh"] >= 0.10 and c_["BTC"] >= b["BTC"] and c_["ETH"] >= b["ETH"] and c_["h1"] >= b["h1"] and c_["h2"] >= b["h2"]
    rho = (0.5 * (per["BTCUSDT"]["app"] + per["ETHUSDT"]["app"])).corr(0.5 * (per["BTCUSDT"]["shape"] + per["ETHUSDT"]["shape"]))
    print(f"  두 방식 상관 {rho:+.2f} · 판정: {'채택' if ok_ else '기각'}")
    out["trade"] = rows; out["verdict"] = "채택" if ok_ else "기각"; out["rho"] = float(rho)
    # 그림용 군집 정보 + 현재 모양
    te = ok & (T >= split)
    out["clusters"] = [{"k": j, "m_tr": float(m_tr[j]), "m_te": float(F[te & (lab == j)].mean()) if (te & (lab == j)).sum() else None,
                        "n_tr": int((tr & (lab == j)).sum()), "n_te": int((te & (lab == j)).sum()), "shape": km.cluster_centers_[j].tolist()} for j in range(20)]
    out["now"] = {}
    for s in D:
        c = D[s]["close"].to_numpy(float); shp = shapes(c, 60)[-1]; j = int(km.predict(shp[None])[0])
        out["now"][s] = {"cluster": j, "m_tr": float(m_tr[j]), "date": str(D[s].index[-1].date())}
        print(f"  지금 {s[:3]} ({D[s].index[-1].date()} 마감): 60일 모양 군집 {j} → 학습 기간 다음 10일 평균 {m_tr[j]:+.1%} ({'롱' if m_tr[j] > 0 else '숏'} 모양)")
    with open("state/shapes4_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
