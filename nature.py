"""자연·철학 관점의 코인별 독립 분석/매매.  python nature.py
 용수철(反者道之動) — 가격은 느린 평형(이동 중심)에서 멀어질수록 되돌아가려는 장력을 가진다 → 장력(σ단위)이 임계를 넘으면 되돌림 / 반대로 장력이 추진력이면 추종
 호흡(物極必反)      — 같은 방향 숨(봉)이 k번 이어진 뒤의 반응
 일(에너지)         — 큰 거래량(노력)에도 가격이 거의 안 움직인(결과 없음) 봉 뒤의 방향
 리듬(해와 달)       — 요일·시간대(주 168칸)별 반복 편향, 과거만으로 추정하는 워크포워드
 흐름(해→달)        — 한 코인의 큰 움직임이 다른 코인에 다음 봉으로 번지는가
 균형(음양)         — ETH/BTC 비율이 평형에서 벌어졌다가 돌아오는가 (코인별 다리)
각 코인을 따로 훈련: 앞 60%에서 관점별 최적 설정을 고르고, 뒤 40%에서 한 번만 평가. 비용=포지션 변화×0.08%."""
from __future__ import annotations
import json, math
import numpy as np, pandas as pd
import data, indicators as ta, novel as N

COINS = ("BTCUSDT", "ETHUSDT")
KOR = {"용수철": "용수철(평형 복귀/이탈)", "호흡": "호흡(연속 봉 뒤)", "일": "일(거래량 대비 무결과)", "리듬": "리듬(요일·시간대)",
       "흐름": "흐름(상대 코인 선행)", "균형": "균형(ETH/BTC 비율)"}


def load():
    H1 = {s: data.load_history(s, "1H", 1825, refresh=False) for s in COINS}
    H4 = {s: ta.resample_ohlcv(H1[s], "4h") for s in COINS}
    idx = H4["BTCUSDT"].index.intersection(H4["ETHUSDT"].index)
    H4 = {s: H4[s].reindex(idx) for s in COINS}
    return H1, H4, idx


def hold_machine(dirs: np.ndarray, h: int) -> np.ndarray:
    n = len(dirs); pos = np.zeros(n); cur = 0; hold = 0
    for t in range(n):
        if dirs[t] != 0: cur, hold = dirs[t], h
        else:
            hold -= 1
            if hold <= 0: cur = 0
        pos[t] = cur
    return pos


def zdev(x, A):
    s = pd.Series(x); dev = s - s.ewm(span=A, adjust=False).mean()
    return (dev / dev.rolling(max(3 * A, 180), min_periods=100).std()).to_numpy()


def state_machine(z, k, mode):
    pos = np.zeros(len(z)); cur = 0
    for t, zt in enumerate(z):
        if not np.isfinite(zt): pos[t] = 0; continue
        if abs(zt) < 0.25 * k: cur = 0
        elif zt > k: cur = -1 if mode == "revert" else 1
        elif zt < -k: cur = 1 if mode == "revert" else -1
        pos[t] = cur
    return pos


def scale(pos, sg):
    return N.band(np.nan_to_num(pos) * np.minimum(N.TV / sg, N.LMAX), 0.15)


def fam_positions(s, H1, H4, idx):
    """코인 s의 관점별 설정 격자 → {관점: {설정: 포지션(4H, 단위 -1..1 방향)}}. 크기 조절은 평가 때."""
    o = "ETHUSDT" if s == "BTCUSDT" else "BTCUSDT"
    c = H4[s]["close"].to_numpy(); v = H4[s]["volume"].to_numpy(); x = np.log(c)
    r = np.concatenate([[0.0], np.diff(x)]); sg = N.sigma(r); sgp = np.concatenate([[np.nan], sg[:-1]])
    ro = np.concatenate([[0.0], np.diff(np.log(H4[o]["close"].to_numpy()))]); sgo = np.concatenate([[np.nan], N.sigma(ro)[:-1]])
    G = {}
    G["용수철"] = {f"A={A},k={k},{m}": state_machine(zdev(x, A), k, m) for A in (60, 180, 540) for k in (1.0, 1.5, 2.0) for m in ("revert", "follow")}
    sgn = np.sign(r); G["호흡"] = {}
    for k in (4, 5, 6):
        streak = np.zeros(len(r)); last = 0
        for t in range(len(r)):
            if sgn[t] != 0 and sgn[t] == last: streak[t] = streak[t - 1] + 1
            else: streak[t] = 1 if sgn[t] != 0 else 0
            last = sgn[t]
        for hh in (3, 6, 12):
            for m in ("fade", "follow"):
                d = np.where(streak >= k, -sgn if m == "fade" else sgn, 0.0)
                G["호흡"][f"k={k},h={hh},{m}"] = hold_machine(d, hh)
    vr = v / pd.Series(v).rolling(30, min_periods=10).mean().to_numpy(); prior = N.mom_sign(r, 12); G["일"] = {}
    for thr in (1.5, 2.0, 3.0):
        for hh in (6, 18):
            for m in ("fade", "follow"):
                trig = (vr > thr) & (np.abs(r) < 0.5 * sgp) & np.isfinite(prior) & (prior != 0)
                d = np.where(trig, -prior if m == "fade" else prior, 0.0)
                G["일"][f"vol>{thr},h={hh},{m}"] = hold_machine(np.nan_to_num(d), hh)
    G["흐름"] = {}
    for th in (0.5, 1.0, 1.5):
        for hh in (1, 2, 3):
            for m in ("follow", "revert"):
                big = np.abs(ro) > th * sgo
                d = np.where(big, np.sign(ro) * (1 if m == "follow" else -1), 0.0)
                G["흐름"][f"th={th},h={hh},{m}"] = hold_machine(np.nan_to_num(d), hh)
    xe, xb = np.log(H4["ETHUSDT"]["close"].to_numpy()), np.log(H4["BTCUSDT"]["close"].to_numpy())
    G["균형"] = {}
    for A in (60, 180, 540):
        zr = zdev(xe - xb, A)
        for k in (1.0, 1.5, 2.0):
            for m in ("revert", "follow"):
                sp = state_machine(zr, k, m)           # +1 = ETH 롱·BTC 숏
                G["균형"][f"A={A},k={k},{m}"] = sp if s == "ETHUSDT" else -sp
    return G, r, sg


def rhythm_pos(df1h: pd.DataFrame, tmin=2.0, nmin=52):
    """시간대(주 168칸)별 과거 평균수익의 t값 → 유의하면 그 시간 포지션 (워크포워드)."""
    c = df1h["close"]; r = np.log(c).diff().fillna(0.0)
    how = (df1h.index.dayofweek * 24 + df1h.index.hour).to_numpy()
    d = pd.DataFrame({"r": r.to_numpy(), "how": how}, index=df1h.index)
    g = d.groupby("how")["r"]
    cs, cs2, cnt = g.cumsum() - d["r"], (d["r"] ** 2).groupby(d["how"]).cumsum() - d["r"] ** 2, g.cumcount()
    mean = cs / cnt.replace(0, np.nan); var = (cs2 / cnt.replace(0, np.nan)) - mean ** 2
    t = mean / np.sqrt(var / cnt.replace(0, np.nan))
    sig = np.where((cnt >= nmin) & (t.abs() >= tmin), np.sign(mean), 0.0)
    pos = pd.Series(sig, index=df1h.index).shift(-1).fillna(0.0)          # 다음 시간 봉의 통계로 지금 포지션 결정(그 통계는 과거만 사용)
    return pos, r


def eval_pos(pos, r, sg, lo=None, hi=None, scaled=True):
    p = scale(pos, sg) if scaled else pos
    pn = N.pnl_from_pos(p, r)
    return pn, p


def sh(pn):
    st = N.stats(pn)
    return st["sharpe"] if st else float("nan")


def main():
    H1, H4, idx = load(); n = len(idx); cut = int(n * 0.6); split = idx[cut]
    print("분할:", split.date(), "| 코인별 독립 훈련 · 비용 편도 0.08% · 연변동성 40% 타깃\n")
    out = {}; pnl_store = {}
    for s in COINS:
        G, r, sg = fam_positions(s, H1, H4, idx)
        # 리듬 (1H)
        rp, r1 = rhythm_pos(H1[s].loc[H1[s].index.isin(H1[s].index)], 2.0)
        sg1 = N.sigma(r1.to_numpy(), alpha=0.0075)
        out[s] = {}; print(f"═══ {s} ═══")
        print(f"{'관점':22s} {'설정 수':>5s} {'앞 최고(설정)':>30s} {'앞 샤프':>7s} → {'뒤 샤프':>7s} | 격자 뒤 중앙값  뒤>0 비율")
        chosen = {}
        for fam, grid in G.items():
            rows = []
            for name, pos in grid.items():
                pn, _ = eval_pos(pos, r, sg)
                rows.append((name, sh(pn[:cut]), sh(pn[cut:]), pn))
            rows = [x for x in rows if np.isfinite(x[1]) and np.isfinite(x[2])]
            if not rows: continue
            best = max(rows, key=lambda x: x[1])
            oos = np.array([x[2] for x in rows])
            print(f"{KOR[fam]:22s} {len(rows):5d} {best[0]:>30s} {best[1]:+7.2f} → {best[2]:+7.2f} | {np.median(oos):+6.2f}   {float((oos>0).mean()):.0%}")
            out[s][fam] = {"best": best[0], "is": best[1], "oos": best[2], "grid_oos_median": float(np.median(oos)), "grid_oos_pos": float((oos > 0).mean()), "n": len(rows)}
            chosen[fam] = best
        # 리듬 평가: 1H 포지션에 변동성 타깃 적용 후 4H로 합산
        rows = []
        for tm in (1.5, 2.0, 3.0):
            rp, r1 = rhythm_pos(H1[s], tm)
            tv1 = 0.40 / math.sqrt(24 * 365)
            p1 = np.nan_to_num(rp.to_numpy()) * np.minimum(tv1 / sg1, N.LMAX)
            gross = np.zeros(len(r1)); gross[1:] = p1[:-1] * r1.to_numpy()[1:]
            turn = np.abs(np.diff(p1, prepend=0.0)); tc = np.zeros(len(r1)); tc[1:] = N.COST * turn[:-1]
            pn1 = pd.Series(gross - tc, index=r1.index)
            pn4 = pn1.groupby(pn1.index.floor("4h")).sum().reindex(idx).fillna(0.0).to_numpy()
            rows.append((f"|t|>={tm}", sh(pn4[:cut]), sh(pn4[cut:]), pn4))
        best = max(rows, key=lambda x: x[1]); oos = np.array([x[2] for x in rows])
        print(f"{KOR['리듬']:22s} {len(rows):5d} {best[0]:>30s} {best[1]:+7.2f} → {best[2]:+7.2f} | {np.median(oos):+6.2f}   {float((oos>0).mean()):.0%}")
        out[s]["리듬"] = {"best": best[0], "is": best[1], "oos": best[2], "grid_oos_median": float(np.median(oos)), "grid_oos_pos": float((oos > 0).mean()), "n": len(rows)}
        chosen["리듬"] = best
        # 조립: 앞 샤프 ≥ 0.3 인 관점만 동일비중 (규칙은 뒤를 보기 전에 정함)
        inc = [f for f, b in chosen.items() if b[1] >= 0.3]
        print(f"  → 조립 규칙(앞 샤프 ≥0.3): 포함 = {inc if inc else '없음'}")
        if inc:
            comb = np.mean([chosen[f][3] for f in inc], axis=0)
            a, b = N.stats(comb[:cut]), N.stats(comb[cut:])
            A_ = N.stats(comb)
            print(f"  조립 결과: 앞 샤프 {a['sharpe']:+.2f} / 뒤 샤프 {b['sharpe']:+.2f} | 뒤 구간 수익 {b['total']:+.0%} 낙폭 {b['mdd']:.0%} | 전체 연 {A_['ann']:+.0%}")
            out[s]["조립"] = {"포함": inc, "앞": a, "뒤": b, "설정": {f: chosen[f][0] for f in inc}}
            pnl_store[s] = comb
        bh = N.stats(np.where(np.arange(n) < 1, 0, np.diff(np.log(H4[s]["close"].to_numpy()), prepend=np.log(H4[s]["close"].iloc[0]))))
        print(f"  (단순 보유 참고: 5년 샤프 {bh['sharpe']:+.2f}, 낙폭 {bh['mdd']:.0%})\n")
    with open("state/nature_result.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
