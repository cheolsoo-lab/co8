"""BTC·ETH 정밀 재분석 — '더 확실한' 수익 극대화.  python precision.py
A) 방향 적중률: 합성 점수 구간 × 보유기간(4H·24H·72H·1주), 겹치지 않는 표본 기준 신뢰구간
B) 개선안(미리 고정) — BTC·ETH에서 개발, 알트 22종(선택에 안 쓴 자산)에서 확인. 펀딩 롱 연 10% 반영.
   채택 규칙(미리 고정): 샤프가 BTC·ETH 전체 +0.10 이상 AND 알트 전체 +0.10 이상 개선, BTC·ETH 앞/뒤 절반 어느 쪽도 0.10 넘게 악화 없음
C) 확실성: 블록 부트스트랩(30일 블록)으로 기간별 수익 확률, 월 적립 경로"""
import json, math, warnings
import numpy as np, pandas as pd
import data, indicators as ta, novel as N, robust as RB

warnings.filterwarnings("ignore")
B4 = 6 * 365
FUND = 0.10 / B4


def coin_frame(df):
    c = df["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))])
    sg = N.sigma(r); cap = np.minimum(N.TV / sg, N.LMAX)
    d1, d2 = RB.tsmom_dir(r), RB.breath_dir(r)
    return {"r": r, "ar": np.exp(r) - 1, "cap": cap, "d1": d1, "d2": d2, "d": 0.5 * d1 + 0.5 * d2, "idx": df.index}


def pnl(pos, F, cost):
    p = np.concatenate([[0.0], pos[:-1]])
    return p * F["ar"] - cost * np.abs(np.diff(p, prepend=0.0)) - FUND * np.maximum(p, 0)


VARIANTS = {
    "기준 (현재 앱)": lambda F: N.band(F["d"] * F["cap"], 0.15),
    "추세·호흡 일치할 때만": lambda F: N.band(np.where((np.sign(F["d1"]) == np.sign(F["d2"])) & (F["d1"] != 0), F["d"], 0.0) * F["cap"], 0.15),
    "완충 넓게 (0.30)": lambda F: N.band(F["d"] * F["cap"], 0.30),
    "숏은 절반 크기": lambda F: N.band(np.where(F["d"] < 0, 0.5 * F["d"], F["d"]) * F["cap"], 0.15),
    "확신 가중 (점수²)": lambda F: N.band(F["d"] * np.abs(F["d"]) * 1.6 * F["cap"], 0.15),
}


def sh(x):
    return float(x.mean() / x.std() * math.sqrt(B4)) if x.std() > 0 else 0.0


def pooled(streams):
    return pd.concat(streams, axis=1).mean(axis=1, skipna=True).fillna(0.0).to_numpy()


def main():
    coins, _ = RB.load_all()
    FR = {s: coin_frame(df) for s, df in coins.items()}
    core, alts = ["BTCUSDT", "ETHUSDT"], [s for s in coins if s not in ("BTCUSDT", "ETHUSDT")]
    out = {}

    # ── A) 방향 적중률 ──
    print("══ A) 방향 예측 적중률 (BTC·ETH 5년, 점수 방향 = 예측 방향) ══")
    bins = [(-1.01, -0.6, "강한 숏 (≤−0.6)"), (-0.6, -0.15, "약한 숏"), (-0.15, 0.15, "관망 구간"), (0.15, 0.6, "약한 롱"), (0.6, 1.01, "강한 롱 (≥0.6)")]
    A = {}
    for H, lab in ((1, "4시간"), (6, "24시간"), (18, "3일"), (42, "1주")):
        rows = []
        for lo, hi, nm in bins:
            hits, rets, n_eff = [], [], 0
            for s in core:
                F = FR[s]; fwd = pd.Series(F["r"]).rolling(H).sum().shift(-H).to_numpy(); d = F["d"]
                m = (d >= lo) & (d < hi) & np.isfinite(fwd) & (np.arange(len(d)) > 600)
                sgn = np.sign(d[m]) if nm != "관망 구간" else np.ones(m.sum())
                hits.append((np.sign(fwd[m]) == sgn)); rets.append(sgn * fwd[m]); n_eff += m.sum() / H
            hits, rets = np.concatenate(hits), np.concatenate(rets)
            p = hits.mean(); se = math.sqrt(p * (1 - p) / max(n_eff, 1))
            rows.append((nm, p, se, rets.mean(), len(hits), n_eff))
            A[f"{lab}|{nm}"] = {"hit": p, "se": se, "ret": float(rets.mean()), "n": int(len(hits))}
        print(f"  [{lab} 뒤 방향]  " + " · ".join(f"{nm} {p:.1%}±{2*se:.1%} (평균 {r:+.2%})" for nm, p, se, r, n, ne in rows if nm != "관망 구간"))
    # 일치 여부
    for H, lab in ((6, "24시간"), (42, "1주")):
        res = {}
        for agree in (True, False):
            hits = []; neff = 0
            for s in core:
                F = FR[s]; fwd = pd.Series(F["r"]).rolling(H).sum().shift(-H).to_numpy()
                ag = (np.sign(F["d1"]) == np.sign(F["d2"])) & (F["d1"] != 0)
                m = (np.abs(F["d"]) >= 0.15) & np.isfinite(fwd) & (ag == agree) & (np.arange(len(fwd)) > 600)
                hits.append(np.sign(fwd[m]) == np.sign(F["d"][m])); neff += m.sum() / H
            h_ = np.concatenate(hits); res[agree] = (h_.mean(), math.sqrt(h_.mean() * (1 - h_.mean()) / neff))
        print(f"  [{lab}] 추세·호흡 같은 방향일 때 {res[True][0]:.1%}±{2*res[True][1]:.1%} / 엇갈릴 때 {res[False][0]:.1%}±{2*res[False][1]:.1%}")
        A[f"일치|{lab}"] = {"agree": res[True][0], "disagree": res[False][0]}
    out["A"] = A

    # ── B) 개선안 ──
    print("\n══ B) 개선안 (펀딩 반영) — BTC·ETH로 개발, 알트 22종으로 확인 ══")
    print(f"{'개선안':22s} | {'BTC·ETH 테이커':>12s} (전반/후반) {'메이커':>6s} | {'알트22 테이커':>11s} {'메이커':>6s} | 판정")
    B = {}; base = None
    for nm, fn in VARIANTS.items():
        P = {s: fn(FR[s]) for s in coins}
        row = {}
        for cn, cost in (("taker", 0.0008), ("maker", 0.0003)):
            cs = pooled({s: pd.Series(pnl(P[s], FR[s], cost), index=FR[s]["idx"]) for s in core})
            al = pooled({s: pd.Series(pnl(P[s], FR[s], cost), index=FR[s]["idx"]) for s in alts})
            h = len(cs) // 2
            row[cn] = {"core": sh(cs), "h1": sh(cs[:h]), "h2": sh(cs[h:]), "alts": sh(al)}
            if cn == "taker":
                e = np.cumprod(1 + cs); row["ann"] = float(e[-1] ** (B4 / len(cs)) - 1); row["mdd"] = float((e / np.maximum.accumulate(e) - 1).min())
        if base is None: base = row; verdict = "기준"
        else:
            t, b = row["taker"], base["taker"]
            ok = (t["core"] - b["core"] >= 0.10) and (t["alts"] - b["alts"] >= 0.10) and (t["h1"] >= b["h1"] - 0.10) and (t["h2"] >= b["h2"] - 0.10)
            verdict = "채택" if ok else "기각"
        row["verdict"] = verdict; B[nm] = row
        t, m = row["taker"], row["maker"]
        print(f"{nm:22s} | {t['core']:+12.2f} ({t['h1']:+.2f}/{t['h2']:+.2f}) {m['core']:+6.2f} | {t['alts']:+11.2f} {m['alts']:+6.2f} | {verdict}  (연 {row['ann']:+.0%}, 낙폭 {row['mdd']:.0%})")
    out["B"] = B

    # ── C) 확실성 ──
    adopted = [k for k, v in B.items() if v["verdict"] == "채택"]
    pick = max(adopted, key=lambda k: B[k]["taker"]["core"]) if adopted else "기준 (현재 앱)"
    print(f"\n══ C) 확실성 — 사용 방식: {pick}, BTC·ETH 반반, 배율 1.5, 테이커+펀딩 ══")
    P = {s: VARIANTS[pick](FR[s]) for s in core}
    x = 1.5 * pooled({s: pd.Series(pnl(P[s], FR[s], 0.0008), index=FR[s]["idx"]) for s in core})[600:]
    rng = np.random.default_rng(11); BL = 6 * 30; C = {}
    for days, lab in ((30, "1개월"), (91, "3개월"), (182, "6개월"), (365, "1년"), (1095, "3년")):
        T = days * 6; nb = math.ceil(T / BL)
        st_ = rng.integers(0, len(x) - BL, size=(5000, nb)); ix = (st_[:, :, None] + np.arange(BL)).reshape(5000, -1)[:, :T]
        paths = np.cumprod(1 + x[ix], axis=1); fin = paths[:, -1] - 1; dd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
        C[lab] = {"p_profit": float((fin > 0).mean()), "median": float(np.median(fin)), "p10": float(np.quantile(fin, 0.1)), "p90": float(np.quantile(fin, 0.9)), "p_dd30": float((dd <= -0.3).mean())}
        print(f"  {lab:4s}: 수익 날 확률 {C[lab]['p_profit']:.0%} · 중앙값 {C[lab]['median']:+.0%} · 나쁜 10% {C[lab]['p10']:+.0%} · 좋은 10% {C[lab]['p90']:+.0%} · 중간에 −30% 이상 낙폭 {C[lab]['p_dd30']:.0%}")
    # 월 적립 경로 (80 USDT ≈ 11만원 시작 + 매달 10만원, 5년)
    mon = pd.Series(x, index=FR["BTCUSDT"]["idx"][600:]).groupby(FR["BTCUSDT"]["idx"][600:].to_period("M")).apply(lambda z: (1 + z).prod() - 1).to_numpy()
    sims = np.zeros((5000, 61)); sims[:, 0] = 11.0
    for t in range(1, 61):
        sims[:, t] = sims[:, t - 1] * (1 + mon[rng.integers(0, len(mon), 5000)]) + 10.0
    dep = 11.0 + 10.0 * np.arange(61)
    PATH = {}
    for yr in (1, 3, 5):
        v = sims[:, 12 * yr]; PATH[yr] = {"deposit": float(dep[12 * yr]), "median": float(np.median(v)), "p10": float(np.quantile(v, 0.1)), "p90": float(np.quantile(v, 0.9))}
        print(f"  적립 {yr}년 (만원): 넣은 돈 {dep[12*yr]:.0f} → 중앙값 {np.median(v):.0f} (나쁜 10% {np.quantile(v,0.1):.0f} / 좋은 10% {np.quantile(v,0.9):.0f})")
    out["C"] = {"pick": pick, "horizons": C, "path": PATH}
    with open("state/precision_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
