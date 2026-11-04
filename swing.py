"""매일 거래하지 않고 큰 고점·저점 사이를 길게 먹는 매매 연구 (BTC·ETH 일봉 2018-07~).  python swing.py
1) 이론 상한: 지그재그(사후에 아는 고점·저점)를 완벽히 잡았을 때
2) 실전(미래참조 없음, 일봉 마감 신호 → 다음 날 시가 체결, 편도 0.1%):
   반전 확인: 최고점(최저점)에서 δ% 되돌리면 방향 전환   δ = 10·15·20·25·30%
   돌파: 직전 N일 최고가 위 마감 → 롱 / 최저가 아래 마감 → 숏   N = 20·50·100
   장기 방향: L일 수익 부호   L = 50·100·200
   방향 합의: 20·50·100·200일 수익 부호 중 3개 이상 같은 쪽 → 그 방향, 아니면 기존 유지
   각각 '롱·숏 모두' / '롱만(숏 자리는 현금)'
3) 펀딩비 시나리오: 롱 보유 시 연 −10% (숏은 0으로 보수적)"""
import json, math, warnings
import numpy as np, pandas as pd
import data

warnings.filterwarnings("ignore")
COST = 0.001
FUND_LONG = 0.10 / 365


def load(sym):
    df = pd.read_csv(data.CACHE / f"{sym}_spot_1D.csv", index_col=0, parse_dates=True, encoding="utf-8"); df.index = pd.to_datetime(df.index, utc=True)
    return df


# ───────── 1) 이론 상한 ─────────
def zigzag(c, th):
    """사후 지그재그: th 이상 되돌림으로 확정되는 고점·저점 위치."""
    hi_i = lo_i = 0; trend = 0; piv = []; ext = 0
    for i in range(1, len(c)):
        if trend == 0:
            if c[i] > c[hi_i]: hi_i = i
            if c[i] < c[lo_i]: lo_i = i
            if c[hi_i] >= c[lo_i] * (1 + th) and hi_i > lo_i: piv, trend, ext = [lo_i], 1, hi_i
            elif c[lo_i] <= c[hi_i] * (1 - th) and lo_i > hi_i: piv, trend, ext = [hi_i], -1, lo_i
        elif trend == 1:
            if c[i] > c[ext]: ext = i
            elif c[i] <= c[ext] * (1 - th): piv.append(ext); trend, ext = -1, i
        else:
            if c[i] < c[ext]: ext = i
            elif c[i] >= c[ext] * (1 + th): piv.append(ext); trend, ext = 1, i
    piv.append(ext)
    return piv


def oracle(c, th, yrs):
    p = zigzag(c, th); legs = [c[b] / c[a] - 1 for a, b in zip(p[:-1], p[1:])]
    legs = [x for x in legs if x != 0]
    eq_ls = float(np.prod([1 + abs(x) if x > 0 else 1 + (-x) for x in legs]))      # 숏 수익 = 하락률
    eq_l = float(np.prod([1 + x for x in legs if x > 0]))
    days = np.diff(p)
    return {"n": len(legs), "per_year": len(legs) / yrs, "avg_up": float(np.mean([x for x in legs if x > 0])), "avg_dn": float(np.mean([x for x in legs if x < 0])),
            "avg_days": float(days.mean()), "cagr_ls": eq_ls ** (1 / yrs) - 1, "cagr_l": eq_l ** (1 / yrs) - 1}


# ───────── 2) 실전 규칙: 목표 포지션(+1/−1/0) 배열 ─────────
def rule_reversal(c, d):
    pos = np.zeros(len(c)); side = 0; hi = lo = c[0]
    for i in range(len(c)):
        hi, lo = max(hi, c[i]), min(lo, c[i])
        if side >= 0 and c[i] <= hi * (1 - d) and (side == 1 or c[i] < c[0] * 10):
            if side == 1 or side == 0: side = -1; lo = c[i]
        if side <= 0 and c[i] >= lo * (1 + d):
            side = 1; hi = c[i]
        if side == -1: hi = c[i] if c[i] > hi and False else hi
        if side == 1: lo = lo
        # 진행 중 극값 갱신
        if side == 1: hi = max(hi, c[i])
        if side == -1: lo = min(lo, c[i]); hi = c[i] if c[i] < hi and False else hi
        pos[i] = side
        if side == -1: hi = max(c[i], lo * (1 + 0))   # 숏 중에는 반등 기준을 저점에서 재계산하므로 hi 불필요
    return pos


def rule_reversal2(c, d):
    """깨끗한 구현: 롱이면 진입 후 최고가에서 d 하락 시 숏 전환, 숏이면 최저가에서 d 상승 시 롱 전환."""
    pos = np.zeros(len(c)); side = 0; ext = c[0]; lo, hi = c[0], c[0]
    for i in range(len(c)):
        if side == 0:
            lo, hi = min(lo, c[i]), max(hi, c[i])
            if c[i] >= lo * (1 + d): side, ext = 1, c[i]
            elif c[i] <= hi * (1 - d): side, ext = -1, c[i]
        elif side == 1:
            ext = max(ext, c[i])
            if c[i] <= ext * (1 - d): side, ext = -1, c[i]
        else:
            ext = min(ext, c[i])
            if c[i] >= ext * (1 + d): side, ext = 1, c[i]
        pos[i] = side
    return pos


def rule_breakout(df, n):
    c = df["close"].to_numpy(); hh = df["high"].rolling(n).max().shift(1).to_numpy(); ll = df["low"].rolling(n).min().shift(1).to_numpy()
    pos = np.zeros(len(c)); side = 0
    for i in range(len(c)):
        if np.isfinite(hh[i]) and c[i] > hh[i]: side = 1
        elif np.isfinite(ll[i]) and c[i] < ll[i]: side = -1
        pos[i] = side
    return pos


def rule_tsmom(c, L):
    s = np.zeros(len(c)); s[L:] = np.sign(c[L:] / c[:-L] - 1); return s


def rule_consensus(c, Ls=(20, 50, 100, 200)):
    S = np.array([rule_tsmom(c, L) for L in Ls]); tot = S.sum(axis=0)
    pos = np.zeros(len(c)); side = 0
    for i in range(len(c)):
        if i >= max(Ls):
            if tot[i] >= 2: side = 1          # 4개 중 3개 이상 +  (합 ≥ 2)
            elif tot[i] <= -2: side = -1
        pos[i] = side
    return pos


# ───────── 백테스트 ─────────
def backtest(df, pos, long_only=False, fund=False, lev=1.0):
    o, c, h, l = (df[k].to_numpy(float) for k in ("open", "close", "high", "low"))
    if long_only: pos = np.where(pos > 0, 1.0, 0.0)
    held = np.concatenate([[0.0], pos[:-1]])                 # i일에 보유 = i-1일 마감 신호 → i일 시가 체결
    # i일 수익: 시가 체결분 반영 (전날 보유 → 시가까지 전날 포지션, 시가 이후 새 포지션)
    prev = np.concatenate([[0.0], held[:-1]])
    c_prev = np.concatenate([[c[0]], c[:-1]])
    r = prev * (o / c_prev - 1) + held * (c / o - 1)
    r -= COST * np.abs(held - prev)
    if fund: r -= FUND_LONG * np.maximum(held, 0.0)
    r *= lev
    eq = np.cumprod(1 + np.maximum(r, -1))
    # 거래 목록
    trades = []; i0 = None
    for i in range(1, len(held)):
        if held[i] != held[i - 1]:
            if held[i - 1] != 0 and i0 is not None:
                trades.append((held[i - 1], i0, i, float(eq[i - 1] / eq[i0 - 1] - 1) if i0 > 0 else 0.0))
            i0 = i if held[i] != 0 else None
    if i0 is not None: trades.append((held[-1], i0, len(held) - 1, float(eq[-1] / eq[i0 - 1] - 1)))
    # 보유 중 최대 역행(장중 고저 기준, 레버리지 1 기준 가격 변동)
    mae = []
    for side, a, b, _ in trades:
        ent = o[a]
        mae.append(float((ent - l[a:b + 1].min()) / ent) if side > 0 else float((h[a:b + 1].max() - ent) / ent))
    return eq, r, trades, mae


def stats(df, eq, r, trades, mae, sl=slice(None)):
    e = eq[sl]; rr = r[sl]; e = e / e[0]; yrs = len(e) / 365
    mdd = float((e / np.maximum.accumulate(e) - 1).min())
    tr = [t for t in trades]; wins = [t[3] for t in tr if t[3] > 0]; loss = [t[3] for t in tr if t[3] <= 0]
    return {"cagr": float(e[-1] ** (1 / yrs) - 1), "mdd": mdd, "sharpe": float(rr.mean() / rr.std() * math.sqrt(365)) if rr.std() > 0 else 0,
            "n": len(tr), "per_year": len(tr) / (len(eq) / 365), "hold": float(np.mean([t[2] - t[1] for t in tr])) if tr else 0,
            "win": len(wins) / max(len(tr), 1), "avg_win": float(np.mean(wins)) if wins else 0, "avg_loss": float(np.mean(loss)) if loss else 0,
            "best": float(max([t[3] for t in tr], default=0)), "max_mae": float(max(mae, default=0)), "final": float(e[-1])}


def main():
    D = {s: load(s) for s in ("BTCUSDT", "ETHUSDT")}
    out = {"oracle": {}, "rules": {}}
    print("══ 1) 이론 상한: 사후에 아는 고점·저점을 '완벽히' 잡았다면 (레버리지 1배) ══")
    for s, df in D.items():
        c = df["close"].to_numpy(float); yrs = len(c) / 365
        for th in (0.10, 0.20, 0.30):
            o_ = oracle(c, th, yrs); out["oracle"][f"{s}|{th}"] = o_
            print(f"  {s[:3]} 스윙 {th:.0%}: 연 {o_['per_year']:.1f}번 전환, 평균 {o_['avg_days']:.0f}일 보유, 상승 평균 {o_['avg_up']:+.0%} / 하락 평균 {o_['avg_dn']:+.0%} → "
                  f"롱·숏 완벽 연 {o_['cagr_ls']:+,.0%} (하루 {(1+o_['cagr_ls'])**(1/365)-1:+.2%}) · 롱만 연 {o_['cagr_l']:+.0%}")
    print("  ※ 현실에선 고점·저점을 '지나고 나서야' 압니다. 되돌림 δ%로 확인하면 진입·청산에서 각각 δ씩, 스윙당 약 2δ를 내줍니다.\n")

    rules = {}
    for d in (0.10, 0.15, 0.20, 0.25, 0.30): rules[f"반전확인 {d:.0%}"] = lambda df, d=d: rule_reversal2(df["close"].to_numpy(float), d)
    for n in (20, 50, 100): rules[f"돌파 {n}일"] = lambda df, n=n: rule_breakout(df, n)
    for L in (50, 100, 200): rules[f"장기방향 {L}일"] = lambda df, L=L: rule_tsmom(df["close"].to_numpy(float), L)
    rules["방향합의 20·50·100·200일"] = lambda df: rule_consensus(df["close"].to_numpy(float))

    print("══ 2) 실전 규칙 (미래참조 없음, 2018-07~2026-11, 편도 0.1%, 레버리지 1배) ══")
    print(f"{'규칙':22s} {'방식':5s} | {'BTC 연':>7s} {'낙폭':>5s} | {'ETH 연':>7s} {'낙폭':>5s} | {'반반 연':>7s} {'낙폭':>5s} {'샤프':>5s} | 전반/후반 샤프 | 연거래 보유일 승률 평균익/평균손")
    for nm, fn in rules.items():
        for lo in (False, True):
            res = {}; rs = []
            for s, df in D.items():
                eq, r, tr, mae = backtest(df, fn(df), long_only=lo); res[s] = stats(df, eq, r, tr, mae); res[s]["_tr"] = tr; rs.append(r)
            rb = 0.5 * (rs[0] + rs[1]); eb = np.cumprod(1 + rb); h = len(rb) // 2
            sh = lambda x: float(x.mean() / x.std() * math.sqrt(365)) if x.std() > 0 else 0
            yrs = len(rb) / 365; both = {"cagr": float(eb[-1] ** (1 / yrs) - 1), "mdd": float((eb / np.maximum.accumulate(eb) - 1).min()), "sharpe": sh(rb), "h1": sh(rb[:h]), "h2": sh(rb[h:])}
            b, e = res["BTCUSDT"], res["ETHUSDT"]
            key = f"{nm}|{'롱만' if lo else '양방향'}"
            out["rules"][key] = {"BTC": {k: v for k, v in b.items() if k != "_tr"}, "ETH": {k: v for k, v in e.items() if k != "_tr"}, "both": both}
            print(f"{nm:22s} {'롱만' if lo else '양방향':5s} | {b['cagr']:+7.0%} {b['mdd']:5.0%} | {e['cagr']:+7.0%} {e['mdd']:5.0%} | {both['cagr']:+7.0%} {both['mdd']:5.0%} {both['sharpe']:+5.2f} | "
                  f"{both['h1']:+.2f}/{both['h2']:+.2f} | {(b['per_year']+e['per_year'])/2:4.1f} {(b['hold']+e['hold'])/2:5.0f} {(b['win']+e['win'])/2:4.0%} {(b['avg_win']+e['avg_win'])/2:+.0%}/{(b['avg_loss']+e['avg_loss'])/2:+.0%}")
    # 참고: 단순 보유
    rs = []
    for s, df in D.items():
        eq, r, tr, mae = backtest(df, np.ones(len(df))); rs.append(r); st_ = stats(df, eq, r, tr, mae)
        print(f"참고 {s[:3]} 그냥 보유: 연 {st_['cagr']:+.0%} 낙폭 {st_['mdd']:.0%}")
    with open("state/swing_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
