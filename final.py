"""최종 분석법: 방향(추세+호흡 앙상블) + 지지·저항(진입 위치·손절) — 규칙은 미리 고정.  python final.py
지지·저항 정의(고정): 4H 확정 스윙 고/저(좌우 5봉), 최근 180봉(30일) 안의 것만. 미래참조 없음(확정 시점부터 사용).
  저항 = 현재가 위 가장 가까운 스윙 고점, 지지 = 현재가 아래 가장 가까운 스윙 저점.
검증 규칙(각각 한 번씩만, 격자 탐색 없음):
  R0 기본    : 방향 앙상블 그대로
  R1 벽 앞 대기: 롱인데 1ATR 안에 저항 / 숏인데 1ATR 안에 지지 → 그 봉은 포지션 0 (돌파하거나 멀어지면 다시)
  R2 벽 앞 절반: 같은 조건에서 크기 절반"""
import json, math, warnings
import numpy as np, pandas as pd
import data, indicators as ta, novel as N, robust as RB

warnings.filterwarnings("ignore")
PIV, LOOK, NEAR_ATR = 5, 180, 1.0


def levels(df: pd.DataFrame):
    """각 봉 마감 시점의 (가장 가까운 저항, 지지). 없으면 nan."""
    sh, sl = ta.pivots(df, PIV, PIV)
    c = df["close"].to_numpy(float); n = len(c)
    shv, slv = sh.to_numpy(), sl.to_numpy()
    res, sup = np.full(n, np.nan), np.full(n, np.nan)
    H, L = [], []
    for t in range(n):
        if np.isfinite(shv[t]): H.append((t, shv[t]))
        if np.isfinite(slv[t]): L.append((t, slv[t]))
        H = [x for x in H if t - x[0] <= LOOK]; L = [x for x in L if t - x[0] <= LOOK]
        up = [v for _, v in H + L if v > c[t]]; dn = [v for _, v in H + L if v < c[t]]   # 고점·저점 모두 레벨(역할 전환)
        if up: res[t] = min(up)
        if dn: sup[t] = max(dn)
    return res, sup


def frame(df: pd.DataFrame) -> dict:
    c = df["close"].to_numpy(float); r = np.concatenate([[0.0], np.diff(np.log(c))])
    sg = N.sigma(r); cap = np.minimum(N.TV / sg, N.LMAX)
    d1, d2 = RB.tsmom_dir(r), RB.breath_dir(r); d = 0.5 * d1 + 0.5 * d2
    atr = ta.atr(df).to_numpy(); res, sup = levels(df)
    wall_long = (res - c) < NEAR_ATR * atr          # nan 비교는 False
    wall_short = (c - sup) < NEAR_ATR * atr
    wall = np.where(d > 0, wall_long, np.where(d < 0, wall_short, False))
    return {"r": r, "c": c, "cap": cap, "d": d, "d1": d1, "d2": d2, "atr": atr, "res": res, "sup": sup, "wall": wall}


def positions(F, rule):
    raw = F["d"] * F["cap"]
    if rule == "R1": raw = np.where(F["wall"], 0.0, raw)
    if rule == "R2": raw = np.where(F["wall"], 0.5 * raw, raw)
    return N.band(raw, 0.15)


def main():
    coins, idx = RB.load_all()
    FR = {s: frame(df) for s, df in coins.items()}
    out = {}
    for cost in (0.0008, 0.0015):
        N.COST = cost; print(f"════ 비용 편도 {cost:.2%} ════")
        for rule in ("R0", "R1", "R2"):
            S = {s: pd.Series(N.pnl_from_pos(positions(F, rule), F["r"]), index=coins[s].index) for s, F in FR.items()}
            for grp, ss in (("BTC·ETH", ["BTCUSDT", "ETHUSDT"]), ("24코인", list(coins))):
                p = RB.port({s: S[s] for s in ss}).to_numpy(); st = N.stats(p); h = len(p) // 2
                a, b = N.stats(p[:h]), N.stats(p[h:])
                print(f"  {rule} {grp:8s} 샤프 {st['sharpe']:+.2f} 연 {st['ann']:+.0%} 낙폭 {st['mdd']:.0%} | 전반 {a['sharpe']:+.2f} 후반 {b['sharpe']:+.2f}")
                out[f"{cost}|{rule}|{grp}"] = {"all": st, "h1": a, "h2": b}
            pos_n = sum(1 for s in S if N.stats(S[s].to_numpy())["sharpe"] > 0)
            print(f"     코인별 플러스 {pos_n}/{len(S)}")
    # 지지·저항 자체의 의미 확인: 롱 신호일 때 벽 가까이 vs 멀리에서 다음 24봉 수익
    print("\n══ 롱/숏 신호 중 '반대편 벽(1ATR 안)' 유무별 다음 6봉(24시간) 방향 수익 (24코인 풀링, 비용 전) ══")
    near, far = [], []
    for s, F in FR.items():
        fwd = pd.Series(F["r"]).rolling(6).sum().shift(-6).to_numpy()
        sgn = np.sign(F["d"]); m = (np.abs(F["d"]) >= 0.15) & np.isfinite(fwd)
        near.append((sgn * fwd)[m & F["wall"]]); far.append((sgn * fwd)[m & ~F["wall"]])
    near, far = np.concatenate(near), np.concatenate(far)
    print(f"  벽 가까움: 평균 {near.mean():+.3%} (n={len(near)}) · 벽 멂: 평균 {far.mean():+.3%} (n={len(far)})")
    out["wall_fwd"] = {"near": float(near.mean()), "far": float(far.mean()), "n_near": len(near), "n_far": len(far)}
    with open("state/final_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
