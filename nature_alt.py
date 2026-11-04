"""알트코인에 자연 관점 방법 적용.  python nature_alt.py
A) BTC/ETH에서 고른 설정을 알트에 '그대로' 이식 (재훈련 없음) — 자산이 다르니 뒤 구간(2024-10-29~)은 시간·자산 모두 표본외
B) 알트별로 앞 60%에서 따로 훈련 → 뒤 40% 평가 (BTC/ETH와 같은 절차)
비용: 편도 0.08%(기본) / 0.15%(알트 유동성·스프레드 가정)."""
import json, math, sys
import numpy as np, pandas as pd
import data, nature as NT, novel as N

ALTS = ["SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT"]


def fam_generic(c, v, cb):
    x = np.log(c); r = np.concatenate([[0.0], np.diff(x)]); sg = N.sigma(r); sgp = np.concatenate([[np.nan], sg[:-1]])
    xb = np.log(cb); ro = np.concatenate([[0.0], np.diff(xb)]); sgo = np.concatenate([[np.nan], N.sigma(ro)[:-1]])
    G = {}
    G["용수철"] = {f"A={A},k={k},{m}": NT.state_machine(NT.zdev(x, A), k, m) for A in (60, 180, 540) for k in (1.0, 1.5, 2.0) for m in ("revert", "follow")}
    sgn = np.sign(r); G["호흡"] = {}
    for k in (4, 5, 6):
        streak = np.zeros(len(r)); last = 0
        for t in range(len(r)):
            streak[t] = streak[t - 1] + 1 if (sgn[t] != 0 and sgn[t] == last) else (1 if sgn[t] != 0 else 0)
            last = sgn[t]
        for hh in (3, 6, 12):
            for m in ("fade", "follow"):
                d = np.where(streak >= k, -sgn if m == "fade" else sgn, 0.0)
                G["호흡"][f"k={k},h={hh},{m}"] = NT.hold_machine(d, hh)
    vr = v / pd.Series(v).rolling(30, min_periods=10).mean().to_numpy(); prior = N.mom_sign(r, 12); G["일"] = {}
    for thr in (1.5, 2.0, 3.0):
        for hh in (6, 18):
            for m in ("fade", "follow"):
                trig = (vr > thr) & (np.abs(r) < 0.5 * sgp) & np.isfinite(prior) & (prior != 0)
                d = np.where(trig, -prior if m == "fade" else prior, 0.0)
                G["일"][f"vol>{thr},h={hh},{m}"] = NT.hold_machine(np.nan_to_num(d), hh)
    G["흐름"] = {}
    for th in (0.5, 1.0, 1.5):
        for hh in (1, 2, 3):
            for m in ("follow", "revert"):
                d = np.where(np.abs(ro) > th * sgo, np.sign(ro) * (1 if m == "follow" else -1), 0.0)
                G["흐름"][f"th={th},h={hh},{m}"] = NT.hold_machine(np.nan_to_num(d), hh)
    G["균형"] = {}
    for A in (60, 180, 540):
        zr = NT.zdev(x - xb, A)
        for k in (1.0, 1.5, 2.0):
            for m in ("revert", "follow"):
                G["균형"][f"A={A},k={k},{m}"] = NT.state_machine(zr, k, m)
    return G, r, sg


def pooled(streams, mask_fn=None):
    df = pd.concat(streams, axis=1)
    if mask_fn is not None: df = df[mask_fn(df.index)]
    return df.mean(axis=1, skipna=True).fillna(0.0)


def st_line(pn):
    s = N.stats(pn.to_numpy() if hasattr(pn, "to_numpy") else pn)
    return s


def main():
    with open("state/nature_result.json", encoding="utf-8") as fh:
        res = json.load(fh)
    H1b = data.load_history("BTCUSDT", "1H", 1825, refresh=False)
    import indicators as ta
    hb = ta.resample_ohlcv(H1b, "4h")
    split = hb.index[int(len(hb) * 0.6)]
    alts = {}
    for s in ALTS:
        try:
            df = pd.read_csv(data.CACHE / f"{s}_4H.csv", index_col=0, parse_dates=True, encoding="utf-8"); df.index = pd.to_datetime(df.index, utc=True)
            idx = df.index.intersection(hb.index)
            if len(idx) < 3000: print(s, "데이터 부족", len(idx)); continue
            alts[s] = (df.reindex(idx), hb["close"].reindex(idx), idx)
        except Exception as e:
            print(s, "ERR", e)
    print("분할:", split.date(), "| 알트:", {s: f"{a[2][0].date()}~" for s, a in alts.items()}, "\n")
    out = {"A": {}, "B": {}}
    cfgs = {"BTC설정": res["BTCUSDT"]["조립"]["설정"], "ETH설정": res["ETHUSDT"]["조립"]["설정"]}
    for cost in (0.0008, 0.0015):
        N.COST = cost
        print(f"════ A) 이식 (재훈련 없음) · 편도 비용 {cost:.2%} ════")
        streams = {k: {} for k in cfgs}; breath = {}
        for s, (df, cb, idx) in alts.items():
            G, r, sg = fam_generic(df["close"].to_numpy(), df["volume"].to_numpy(), cb.to_numpy())
            for cn, cfg in cfgs.items():
                pns = [NT.eval_pos(G[f][nm], r, sg)[0] for f, nm in cfg.items()]
                streams[cn][s] = pd.Series(np.mean(pns, axis=0), index=idx)
            nm = cfgs["BTC설정"]["호흡"]; breath[s] = pd.Series(NT.eval_pos(G["호흡"][nm], r, sg)[0], index=idx)
        print(f"{'코인':10s} " + " ".join(f"{cn}: 전체/뒤 샤프".ljust(22) for cn in cfgs) + " 호흡만(BTC설정) 전체/뒤")
        for s in alts:
            cells = []
            for cn in cfgs:
                p = streams[cn][s]; a, b = st_line(p), st_line(p[p.index >= split])
                cells.append(f"{a['sharpe']:+.2f} / {b['sharpe']:+.2f}".ljust(22) if a and b else "-".ljust(22))
            p = breath[s]; a, b = st_line(p), st_line(p[p.index >= split])
            print(f"{s:10s} " + " ".join(cells) + f" {a['sharpe']:+.2f} / {b['sharpe']:+.2f}")
        for cn in cfgs:
            P = pooled(streams[cn]); a, b = st_line(P), st_line(P[P.index >= split])
            pos_n = sum(1 for s in alts if (st_line(streams[cn][s][streams[cn][s].index >= split]) or {"sharpe": 0})["sharpe"] > 0)
            print(f"  ▶ 알트 {len(alts)}개 동일비중 [{cn}]: 전체 샤프 {a['sharpe']:+.2f} · 뒤 구간 샤프 {b['sharpe']:+.2f} (뒤 구간 양수 {pos_n}/{len(alts)}개) · 뒤 구간 수익 {b['total']:+.0%} 낙폭 {b['mdd']:.0%}")
            out["A"][f"{cn}|{cost}"] = {"전체": a, "뒤": b, "뒤양수": pos_n}
        Pb = pooled(breath); a, b = st_line(Pb), st_line(Pb[Pb.index >= split])
        print(f"  ▶ 호흡만 동일비중: 전체 {a['sharpe']:+.2f} · 뒤 {b['sharpe']:+.2f}")
        out["A"][f"호흡|{cost}"] = {"전체": a, "뒤": b}
        print()
    N.COST = 0.0008
    # 참고: 알트 단순 보유
    bh = pd.concat({s: pd.Series(np.diff(np.log(a[0]["close"].to_numpy()), prepend=np.log(a[0]["close"].iloc[0])), index=a[2]) for s, a in alts.items()}, axis=1).mean(axis=1)
    b0 = st_line(bh); b1 = st_line(bh[bh.index >= split])
    print(f"참고) 알트 단순 보유 동일비중(생존 종목만): 전체 샤프 {b0['sharpe']:+.2f} 낙폭 {b0['mdd']:.0%} | 뒤 구간 샤프 {b1['sharpe']:+.2f}\n")

    print("════ B) 알트별 따로 훈련 (앞 60% 선택 → 뒤 40% 평가), 비용 0.08% / 재평가 0.15% ════")
    comb = {0.0008: {}, 0.0015: {}}
    print(f"{'코인':10s} {'포함 관점':28s} {'앞 샤프':>7s} {'뒤 샤프(0.08%)':>14s} {'뒤 샤프(0.15%)':>14s}")
    for s, (df, cb, idx) in alts.items():
        G, r, sg = fam_generic(df["close"].to_numpy(), df["volume"].to_numpy(), cb.to_numpy())
        cut = int(np.searchsorted(idx, split))
        N.COST = 0.0008; chosen = {}
        for fam, grid in G.items():
            best = None
            for name, pos in grid.items():
                pn, _ = NT.eval_pos(pos, r, sg); a, b = NT.sh(pn[:cut]), NT.sh(pn[cut:])
                if np.isfinite(a) and np.isfinite(b) and (best is None or a > best[1]): best = (name, a, b)
            if best: chosen[fam] = best
        inc = [f for f, b in chosen.items() if b[1] >= 0.3]
        if not inc:
            print(f"{s:10s} 포함 없음 (앞 샤프 0.3 이상 관점 없음)"); out["B"][s] = {"inc": []}; continue
        row = {}
        for cost in (0.0008, 0.0015):
            N.COST = cost
            pns = [NT.eval_pos(G[f][chosen[f][0]], r, sg)[0] for f in inc]; cm = np.mean(pns, axis=0)
            row[cost] = (N.stats(cm[:cut]), N.stats(cm[cut:])); comb[cost][s] = pd.Series(cm, index=idx)
        print(f"{s:10s} {','.join(inc):28s} {row[0.0008][0]['sharpe']:+7.2f} {row[0.0008][1]['sharpe']:+14.2f} {row[0.0015][1]['sharpe']:+14.2f}")
        out["B"][s] = {"inc": inc, "앞": row[0.0008][0], "뒤": row[0.0008][1], "뒤_고비용": row[0.0015][1]}
    N.COST = 0.0008
    for cost in (0.0008, 0.0015):
        if comb[cost]:
            P = pooled(comb[cost]); b = st_line(P[P.index >= split])
            pos_n = sum(1 for s, p in comb[cost].items() if (st_line(p[p.index >= split]) or {"sharpe": 0})["sharpe"] > 0)
            print(f"  ▶ 알트별 훈련 동일비중 (비용 {cost:.2%}): 뒤 구간 샤프 {b['sharpe']:+.2f}, 수익 {b['total']:+.0%}, 낙폭 {b['mdd']:.0%}, 양수 {pos_n}/{len(comb[cost])}개")
            out["B"][f"합산|{cost}"] = {"뒤": b, "양수": pos_n, "n": len(comb[cost])}
    with open("state/nature_alt.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
