"""가짜 전환에 속지 않기 — 진짜/가짜 전환 해부 + 구별 단서 + '확인 후 진입' 필터.  python fakes.py
전환 = 현재 방식 포지션이 새 방향(롱/숏)으로 바뀐 순간. 그 방향이 끝날 때까지의 손익 > 0 이면 진짜, 아니면 가짜.
단서(전환 순간, 그 시점 정보만):
  합의도(1·3·7·14·30·60일 방향 중 같은 방향 비율) · 호흡 길이(같은 방향 연속 봉) · 거래량 배수 · 변동 확대(ATR5/ATR30)
  구조 돌파(직전 확정 고점/저점을 종가로 넘었나) · 일봉 흐름 일치 · BTC·ETH 동반 · 종가 위치(봉 안에서 그 방향 끝에 마감했나)
  신호 세기 · 최근 30봉 전환 횟수(혼조) · 변동성 백분위 · 효율비(경로가 얼마나 곧았나)
필터 = 전환 후 단서가 충족될 때까지 기다렸다 진입(그 방향 구간 안에서 한 번 충족되면 구간 끝까지 보유)."""
import json, math, warnings
import numpy as np, pandas as pd
import data, indicators as ta, novel as N, precision as PR, meta as M
from patterns2 import zigzag

warnings.filterwarnings("ignore")
COST = 0.0003
FEATS = ["합의도", "호흡 길이", "거래량 배수", "변동 확대", "구조 돌파", "일봉 흐름 일치", "BTC·ETH 동반", "종가 위치", "신호 세기", "최근 혼조", "변동성 백분위", "효율비"]


def load():
    D, FR = {}, {}
    for s in ("BTCUSDT", "ETHUSDT"):
        D[s] = ta.resample_ohlcv(data.load_history(s, "1H", 1825, refresh=False), "4h").dropna(); FR[s] = PR.coin_frame(D[s])
        FR[s]["base"] = N.band(FR[s]["d"] * FR[s]["cap"], 0.15)
    return D, FR


def bar_features(df, F, other_sign):
    c = df["close"].to_numpy(float); h = df["high"].to_numpy(float); l = df["low"].to_numpy(float); v = df["volume"].to_numpy(float)
    r = F["r"]; n = len(c); cs = np.cumsum(r)
    agree = np.mean([np.sign(np.concatenate([np.zeros(L), cs[L:] - cs[:-L]])) for L in (6, 18, 42, 84, 180, 360)], axis=0)
    sg = np.sign(r); streak = np.zeros(n)
    for t in range(1, n): streak[t] = streak[t - 1] + 1 if (sg[t] != 0 and sg[t] == sg[t - 1]) else 1
    sstreak = streak * sg                                   # +면 상승 연속, −면 하락 연속
    volr = v / pd.Series(v).rolling(30, min_periods=10).mean().to_numpy()
    atr = ta.atr(df).to_numpy(); atr5 = ta.atr(df, 5).to_numpy(); atr30 = ta.atr(df, 30).to_numpy()
    piv = zigzag(h, l, atr, 2.0); lastH = np.full(n, np.nan); lastL = np.full(n, np.nan); hi_ = lo_ = np.nan; j = 0
    piv = sorted(piv)
    for t in range(n):
        while j < len(piv) and piv[j][0] <= t:
            if piv[j][3] == 1: hi_ = piv[j][2]
            else: lo_ = piv[j][2]
            j += 1
        lastH[t], lastL[t] = hi_, lo_
    dd = ta.resample_ohlcv(df, "1D").dropna()
    import swing_live as SL
    daily = pd.Series(np.mean([fn(dd) for fn in SL.RULES.values()], axis=0), index=dd.index + pd.Timedelta(days=1)).reindex(df.index, method="ffill").fillna(0).to_numpy()
    pos_in = (c - l) / np.maximum(h - l, 1e-12)
    flips = pd.Series((np.diff(np.sign(F["base"]), prepend=0) != 0).astype(float)).rolling(30).sum().to_numpy()
    vp = pd.Series(N.sigma(r)).rolling(2190, min_periods=500).apply(lambda a: (a[-1] >= a).mean(), raw=True).to_numpy()
    rr = pd.Series(r); er = (rr.rolling(42).sum().abs() / rr.abs().rolling(42).sum()).to_numpy()
    return {"agree": agree, "sstreak": sstreak, "volr": volr, "exp": atr5 / atr30, "c": c, "lastH": lastH, "lastL": lastL, "daily": daily,
            "other": other_sign, "posin": pos_in, "score": F["d"], "flips": flips, "vp": vp, "er": er}


def feat_at(B, t, d):
    """방향 d로 전환한 순간 t의 단서 값 (클수록 그 방향에 유리하게 정렬)."""
    return [d * B["agree"][t], max(d * B["sstreak"][t], 0), B["volr"][t], B["exp"][t],
            float((d > 0 and B["c"][t] > B["lastH"][t]) or (d < 0 and B["c"][t] < B["lastL"][t])), d * B["daily"][t], float(B["other"][t] == d),
            B["posin"][t] if d > 0 else 1 - B["posin"][t], d * B["score"][t], B["flips"][t], B["vp"][t], B["er"][t]]


def segments(pos):
    sg = np.sign(pos); out = []; start = None
    for t in range(len(sg)):
        if start is None and sg[t] != 0: start = t
        elif start is not None and sg[t] != sg[start]:
            out.append((int(sg[start]), start, t)); start = t if sg[t] != 0 else None
    if start is not None: out.append((int(sg[start]), start, len(sg)))
    return out


def confirm_filter(pos, ok_fn):
    """구간(같은 방향) 안에서 ok_fn(t, d)가 처음 참이 된 뒤부터 구간 끝까지 보유, 그 전엔 0."""
    out = np.zeros(len(pos))
    for d, a, b in segments(pos):
        for t in range(a, b):
            if ok_fn(t, d):
                out[t:b] = pos[t:b]; break
    return out


def judge(FR, P, base=None, sl=slice(None)):
    x = {s: PR.pnl(P[s], FR[s], COST)[sl] for s in FR}; pool = 0.5 * (x["BTCUSDT"] + x["ETHUSDT"]); h = len(pool) // 2; e = np.cumprod(1 + pool)
    r = {"sh": PR.sh(pool), "BTC": PR.sh(x["BTCUSDT"]), "ETH": PR.sh(x["ETHUSDT"]), "h1": PR.sh(pool[:h]), "h2": PR.sh(pool[h:]),
         "ann": float(e[-1] ** (PR.B4 / len(pool)) - 1), "mdd": float((e / np.maximum.accumulate(e) - 1).min())}
    if base is not None:
        r["verdict"] = "채택" if (r["sh"] - base["sh"] >= 0.10 and r["BTC"] >= base["BTC"] and r["ETH"] >= base["ETH"] and r["h1"] >= base["h1"] and r["h2"] >= base["h2"]) else "기각"
    return r


def show(nm, r):
    print(f"  {nm:30s} 샤프 {r['sh']:+.2f} (앞/뒤 {r['h1']:+.2f}/{r['h2']:+.2f}) · BTC {r['BTC']:+.2f} · ETH {r['ETH']:+.2f} · 연 {r['ann']:+.0%} · 낙폭 {r['mdd']:.0%}" + (f" → {r['verdict']}" if "verdict" in r else ""))


def main():
    D, FR = load(); out = {}
    B = {}
    for s, o in (("BTCUSDT", "ETHUSDT"), ("ETHUSDT", "BTCUSDT")):
        oth = pd.Series(np.sign(FR[o]["base"]), index=FR[o]["idx"]).reindex(FR[s]["idx"]).fillna(0).to_numpy()
        B[s] = bar_features(D[s], FR[s], oth)
    # 1) 해부
    rows = []
    for s in FR:
        x = PR.pnl(FR[s]["base"], FR[s], COST); n = len(x); mid = FR[s]["idx"][n // 2]
        for d, a, b in segments(FR[s]["base"]):
            if a < 600: continue
            ret = float(np.prod(1 + x[a + 1:min(b + 1, n)]) - 1)
            rows.append({"sym": s, "t": a, "d": d, "len": b - a, "ret": ret, "real": ret > 0, "half": 1 if FR[s]["idx"][a] < mid else 2,
                         **dict(zip(FEATS, feat_at(B[s], a, d)))})
    E = pd.DataFrame(rows)
    fake, real = E[~E["real"]], E[E["real"]]
    print("══ 1) 가짜 전환 해부 (5년, BTC·ETH) ══")
    print(f"  전환 {len(E)}번 · 진짜 {len(real)} ({len(real)/len(E):.0%}) / 가짜 {len(fake)} ({len(fake)/len(E):.0%})")
    print(f"  가짜: 평균 {fake['ret'].mean():+.2%}, 평균 {fake['len'].mean()*4/24:.1f}일 만에 끝남 · 가짜 손실 합계 {fake['ret'].sum():+.0%}")
    print(f"  진짜: 평균 {real['ret'].mean():+.2%}, 평균 {real['len'].mean()*4/24:.1f}일 유지 · 진짜 이익 합계 {real['ret'].sum():+.0%}")
    print(f"  → 가짜를 전부 피하면 이론상 순이익 {E['ret'].sum():+.0%} → {real['ret'].sum():+.0%} (가짜는 진짜를 잡기 위한 '입장료')")
    out["anatomy"] = {"n": len(E), "real_rate": float(len(real) / len(E)), "fake_avg": float(fake["ret"].mean()), "real_avg": float(real["ret"].mean()),
                      "fake_days": float(fake["len"].mean() * 4 / 24), "real_days": float(real["len"].mean() * 4 / 24), "fake_sum": float(fake["ret"].sum()), "real_sum": float(real["ret"].sum())}
    # 2) 단서
    print("\n══ 2) 전환 순간의 단서 — 위쪽 절반 vs 아래쪽 절반의 진짜 비율 (앞 기간 기준선, 앞 | 뒤) ══")
    tr, te = E[E["half"] == 1], E[E["half"] == 2]; uni = {}
    for f in FEATS:
        med = tr[f].median()
        def rate(g, hi): x_ = g[g[f] > med] if hi else g[g[f] <= med]; return x_["real"].mean(), x_["ret"].mean(), len(x_)
        a1, a1r, _ = rate(tr, True); b1, b1r, _ = rate(tr, False); a2, a2r, n2 = rate(te, True); b2, b2r, _ = rate(te, False)
        cons = (a1 - b1) * (a2 - b2) > 0
        uni[f] = {"tr_hi": a1, "tr_lo": b1, "te_hi": a2, "te_lo": b2, "tr_ret_diff": a1r - b1r, "te_ret_diff": a2r - b2r, "consistent": bool(cons)}
        print(f"  {f:12s} 높을 때 진짜 {a1:.0%} vs 낮을 때 {b1:.0%} | 뒤: {a2:.0%} vs {b2:.0%} · 평균수익 차 {a1r-b1r:+.2%} | {a2r-b2r:+.2%} {'(일관)' if cons else ''}")
    out["univariate"] = uni
    # 로지스틱 (앞 학습 → 뒤 평가)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    Xtr, ytr = np.nan_to_num(tr[FEATS].to_numpy(float)), tr["real"].to_numpy(int); Xte, yte = np.nan_to_num(te[FEATS].to_numpy(float)), te["real"].to_numpy(int)
    sc = StandardScaler().fit(Xtr); lr = LogisticRegression(C=0.3, max_iter=1000).fit(sc.transform(Xtr), ytr)
    auc_tr = roc_auc_score(ytr, lr.predict_proba(sc.transform(Xtr))[:, 1]); auc_te = roc_auc_score(yte, lr.predict_proba(sc.transform(Xte))[:, 1])
    print(f"  ▶ 단서 12개 결합 모델(앞 학습): 구별력 AUC 앞 {auc_tr:.2f} → 뒤 {auc_te:.2f} (0.5 = 동전, 1.0 = 완벽)")
    print("    모델 가중치(클수록 진짜 쪽):", {f: round(float(w), 2) for f, w in zip(FEATS, lr.coef_[0])})
    out["model"] = {"auc_tr": float(auc_tr), "auc_te": float(auc_te), "coef": dict(zip(FEATS, map(float, lr.coef_[0])))}
    # 3) 필터
    print("\n══ 3) '확인 후 진입' 필터 — 전 기간 엄격 판정 ══")
    base = judge(FR, {s: F["base"] for s, F in FR.items()}); show("기준 (현재 앱)", base); out["base"] = base
    filt = {"합의도 4/6 이상 확인": lambda s: (lambda t, d: d * B[s]["agree"][t] >= 4 / 6 - 1e-9),
            "최근 혼조(30봉 전환 3회+) 피함": lambda s: (lambda t, d: B[s]["flips"][t] <= 2),
            "BTC·ETH 같은 방향 확인": lambda s: (lambda t, d: B[s]["other"][t] == d),
            "직전 고점/저점 돌파 확인": lambda s: (lambda t, d: (d > 0 and B[s]["c"][t] > B[s]["lastH"][t]) or (d < 0 and B[s]["c"][t] < B[s]["lastL"][t])),
            "일봉 흐름과 같은 방향 확인": lambda s: (lambda t, d: d * B[s]["daily"][t] > 0)}
    for nm, mk in filt.items():
        P = {s: confirm_filter(FR[s]["base"], mk(s)) for s in FR}; r = judge(FR, P, base); out[nm] = r; show(nm, r)
    # 모델 필터: 뒤 절반만 (앞에서 학습한 모델로, 확률이 앞 기간 하위 30% 미만이면 거름 — 전환 순간만 판단)
    thr = np.quantile(lr.predict_proba(sc.transform(Xtr))[:, 1], 0.3)
    P = {}
    for s in FR:
        n = len(FR[s]["r"]); mid = n // 2; pos = FR[s]["base"].copy(); outp = pos.copy()
        for d, a, b in segments(pos):
            if a < mid: continue
            pr = lr.predict_proba(sc.transform(np.nan_to_num(np.array([feat_at(B[s], a, d)], float))))[:, 1][0]
            if pr < thr: outp[a:b] = 0.0
        P[s] = outp
    n = len(FR["BTCUSDT"]["r"]); sl = slice(n // 2, n)
    b2 = judge(FR, {s: F["base"] for s, F in FR.items()}, None, sl); m2 = judge(FR, P, b2, sl)
    print("  [뒤 절반만, 앞에서 학습한 모델]"); show("기준 (뒤 절반)", b2); show("모델 필터 (하위 30% 거름)", m2)
    out["model_filter"] = {"base": b2, "filter": m2}
    with open("state/fakes_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
