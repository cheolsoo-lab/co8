"""수익판 (11-16): '앱대로 했다면' 시작일부터 지금까지 — 자산 곡선, 끝난 거래(익절·손절·부분익절), 다음 목표까지.
앱 기본 그대로 다시 계산: 추세+호흡 → 흐름 정렬 역가중 → 파동 국면 → 돈의 흐름 주머니 → 추적 부분익절.
비용: 지정가 0.03% · 롱 펀딩 연 10%(보수적 가정). 수량은 소수점까지 계산(실제 최소 단위 때문에 조금 다를 수 있음)."""
from __future__ import annotations
import numpy as np, pandas as pd
import indicators as ta, novel as N, robust as RB, mtf as MT, patterns2 as P2, wave_live as WL, trade_live as TL

FEE, FUND = 0.0003, 0.10 / (6 * 365)


def positions(df: pd.DataFrame, flow6: pd.Series | None, mtf_on=True, wave_on=True, flow_on=True, trail_on=True, ex: pd.Series | None = None) -> pd.DataFrame:
    idx = df.index; c = df["close"].to_numpy(float); n = len(c); r = np.concatenate([[0.0], np.diff(np.log(c))])
    sg = N.sigma(r); cap = np.minimum(N.TV / sg, N.LMAX); d = 0.5 * RB.tsmom_dir(r) + 0.5 * RB.breath_dir(r); base = N.band(d * cap, 0.15)
    p = base.copy()
    if mtf_on:
        dD, _ = MT.tf_signal(ta.resample_ohlcv(df, "1D"), (5, 10, 20, 40, 60), (2, 3, 4), (3, 5, 10))
        dW, _ = MT.tf_signal(ta.resample_ohlcv(df, "W-MON"), (4, 8, 13, 26), (2, 3), (2, 4, 6))
        dD = MT.to_4h(dD, idx, pd.Timedelta(days=1)); dW = MT.to_4h(dW, idx, pd.Timedelta(days=7)); sb = np.sign(base)
        ag = (np.sign(dD) == sb).astype(int) + (np.sign(dW) == sb).astype(int); p = p * np.array([1.5, 1.0, 0.5])[ag]
    if wave_on:
        h, l = df["high"].to_numpy(float), df["low"].to_numpy(float); atr = ta.atr(df).to_numpy(); piv = P2.zigzag(h, l, atr, WL.K)
        v = [q[2] for q in piv]; t = [q[3] for q in piv]; wd = np.zeros(n)
        for j in range(2, len(piv)):
            if WL.label_at(v, t, j) not in WL.GOOD: continue
            dr = -t[j]; t0 = piv[j][0]; t1 = piv[j + 1][0] if j + 1 < len(piv) else n
            for k in range(t0, min(t1, n)):
                if k > t0 and dr * (c[k] - v[j]) < 0: break
                wd[k] = dr
        sp_ = np.sign(p); p = p * np.where((sp_ != 0) & (sp_ == wd), WL.UP, np.where((sp_ != 0) & (sp_ == -wd), WL.DN, 1.0))
    if flow_on and flow6 is not None and len(flow6):
        z = flow6.copy(); z.index = z.index + pd.Timedelta(hours=6); close4 = idx + pd.Timedelta(hours=4)
        s = z.reindex(z.index.union(close4)).ffill().reindex(close4).to_numpy()
        p = p + 0.5 * np.clip(np.nan_to_num(s) / 2, -1, 1) * cap
    if ex is not None and len(ex):                                             # 거래소 순유출 주머니 (시험): 그날 값은 이틀 뒤부터
        import onchain_live as OC
        z = ex.copy(); z.index = z.index + pd.Timedelta(days=OC.LAG); close4 = idx + pd.Timedelta(hours=4)
        s = z.reindex(z.index.union(close4)).ffill().reindex(close4).to_numpy()
        p = p + OC.W * np.clip(np.nan_to_num(s) / 2, -1, 1) * cap
    cutcol = np.zeros(n, bool)
    if trail_on:
        atr = ta.atr(df).to_numpy(); dr = np.sign(base); st = 0; best = 0.0; cut = False; out = p.copy()
        for k in range(1, n):
            if dr[k] != dr[k - 1] or dr[k] == 0: st = k; best = 0.0; cut = False; continue
            a = atr[st] if np.isfinite(atr[st]) and atr[st] > 0 else atr[k]; mv = dr[k] * (c[k] - c[st]) / a; best = max(best, mv)
            if not cut and best >= TL.K and best - mv >= TL.K: cut = True
            if cut and np.sign(p[k]) == dr[k]: out[k] = TL.KEEP * p[k]; cutcol[k] = True
        p = out
    return pd.DataFrame({"close": c, "p": p, "base": base, "cut": cutcol}, index=idx)


def simulate(P: dict, start, capital: float, monthly: float, mult: float) -> tuple[pd.Series, pd.DataFrame]:
    """P: {코인: positions()} → (자산 곡선, 끝난·진행 중 거래 표). 자본은 코인 수로 나눔, 매달 1일 0시(UTC) 추가 입금."""
    idx = P[next(iter(P))].index; _s = pd.Timestamp(start); _s = _s.tz_localize("UTC") if _s.tzinfo is None else _s.tz_convert("UTC"); idx = idx[idx >= _s]
    if len(idx) < 2: return pd.Series(dtype=float), pd.DataFrame()
    eq = capital; path = [eq]; trades = []; open_ = {}; month = idx[0].month; nc = len(P)
    for i in range(1, len(idx)):
        t0, t1 = idx[i - 1], idx[i]
        if t1.month != month: eq += monthly; month = t1.month
        gain = 0.0
        for coin, df in P.items():
            if t0 not in df.index or t1 not in df.index: continue
            p0 = df.at[t0, "p"]; prev = df["p"].shift(1).get(t0, 0.0); c0, c1 = df.at[t0, "close"], df.at[t1, "close"]
            k = eq * mult / nc
            g = k * (p0 * (c1 / c0 - 1) - FEE * abs(p0 - (prev if np.isfinite(prev) else 0.0)) - FUND * max(p0, 0)); gain += g
            sd = np.sign(p0) if abs(p0) > 1e-9 else 0.0; o = open_.get(coin)
            if o is None or sd != o["dir"]:
                if o is not None and o["dir"] != 0: trades.append({**o, "끝": t0, "청산가": c0})
                open_[coin] = o = {"코인": coin, "dir": sd, "시작": t0, "진입가": c0, "손익": 0.0, "부분익절": False}
            o["손익"] += g
            if bool(df.at[t0, "cut"]): o["부분익절"] = True
        eq += gain; path.append(eq)
    for coin, o in open_.items():
        if o["dir"] != 0: trades.append({**o, "끝": None, "청산가": P[coin].at[idx[-1], "close"]})
    curve = pd.Series(path, index=idx)
    T = pd.DataFrame(trades)
    if not T.empty:
        T["방향"] = np.where(T["dir"] > 0, "롱", "숏"); T["가격 변화"] = T["dir"] * (T["청산가"] / T["진입가"] - 1)
        small = T["손익"].abs() < 0.002 * capital                               # 자본의 0.2% 미만은 본전 (4시간 만에 뒤집힌 가짜 전환)
        T["결과"] = np.where(T["끝"].isna(), "진행 중", np.where(small, "본전", np.where(T["손익"] >= 0, "익절", "손절"))); T = T.sort_values("시작", ascending=False)
    return curve, T


def milestones(eq_usdt: float, krw: float, monthly_usdt: float, g_year: float, g_low: float, goals_krw=(1e6, 5e6, 1e7, 5e7, 1e8, 1e9, 5e9)) -> list[dict]:
    """다음 목표들까지 걸리는 달 수 (중앙 성장 · 보수적 성장), 매달 추가 입금 포함."""
    out = []
    for gk in goals_krw:
        if gk <= eq_usdt * krw: continue
        row = {"목표": gk}
        for key, g in (("중앙", g_year), ("보수", g_low)):
            e = eq_usdt; m = 0; gm = (1 + max(g, -0.9)) ** (1 / 12) - 1
            while e * krw < gk and m < 600: e = e * (1 + gm) + monthly_usdt; m += 1
            row[key] = m if m < 600 else None
        out.append(row)
    return out
