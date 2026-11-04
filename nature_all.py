"""알트 22종 개별 방법 만들기.  python nature_all.py
코인마다: 자기 데이터 앞 60%에서 관점별 설정 선택 → 앞 샤프≥0.3 관점만 동일비중 조립 → 뒤 40%에서 한 번 평가 (비용 0.08% / 0.15%).
등급(미리 정한 규칙, 뒤 구간을 본 뒤 바꾸지 않음):
  자료부족 = 4H 봉 < 3000 (약 500일)
  A = 뒤 샤프 > 0.3 (비용 0.15%에서도 > 0)   B = 뒤 샤프 > 0 (비용 0.08%만)   C = 그 외(사용 안 함)
호흡만(BTC 설정 k=4,h=12,follow) 이식 결과를 나란히 기록."""
import json, math
import numpy as np, pandas as pd
import data, indicators as ta, nature as NT, novel as N
from nature_alt import fam_generic

OLD = ["SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT", "AVAXUSDT", "LTCUSDT", "DOTUSDT", "TRXUSDT"]
NEW = ["HYPEUSDT", "RENDERUSDT", "TAOUSDT", "FILUSDT", "GRASSUSDT", "TIAUSDT", "STXUSDT", "NEARUSDT", "ARBUSDT", "APTUSDT", "ONDOUSDT", "ZROUSDT"]
BREATH = "k=4,h=12,follow"


def grade(n, oos08, oos15):
    if n < 3000: return "자료부족"
    if oos08 > 0.3 and oos15 > 0: return "A"
    if oos08 > 0: return "B"
    return "C"


def main():
    hb = ta.resample_ohlcv(data.load_history("BTCUSDT", "1H", 1825, refresh=False), "4h")
    out = {}
    print(f"{'코인':10s} {'봉수':>6s} {'시작':>10s} {'등급':>6s} {'포함 관점':24s} {'앞':>6s} {'뒤.08%':>7s} {'뒤.15%':>7s} | 호흡만 뒤.08% / .15%")
    for s in OLD + NEW:
        try:
            df = pd.read_csv(data.CACHE / f"{s}_4H.csv", index_col=0, parse_dates=True, encoding="utf-8"); df.index = pd.to_datetime(df.index, utc=True)
        except Exception as e:
            print(f"{s:10s} 데이터 없음 {e}"); continue
        idx = df.index.intersection(hb.index); df = df.reindex(idx); cb = hb["close"].reindex(idx)
        n = len(idx)
        if n < 800: out[s] = {"n": n, "grade": "자료부족", "start": str(idx[0].date()) if n else ""}; print(f"{s:10s} {n:6d} 자료부족"); continue
        G, r, sg = fam_generic(df["close"].to_numpy(), df["volume"].to_numpy(), cb.to_numpy())
        cut = int(n * 0.6); N.COST = 0.0008; chosen = {}
        for fam, grid in G.items():
            best = None
            for name, pos in grid.items():
                pn, _ = NT.eval_pos(pos, r, sg); a, b = NT.sh(pn[:cut]), NT.sh(pn[cut:])
                if np.isfinite(a) and np.isfinite(b) and (best is None or a > best[1]): best = (name, a, b)
            if best: chosen[fam] = best
        inc = [f for f, b in chosen.items() if b[1] >= 0.3]
        row = {"n": n, "start": str(idx[0].date()), "split": str(idx[cut].date()), "inc": inc, "cfg": {f: chosen[f][0] for f in inc}, "fam": {f: {"is": chosen[f][1], "oos": chosen[f][2]} for f in chosen}}
        if inc:
            res = {}
            for cost in (0.0008, 0.0015):
                N.COST = cost
                cm = np.mean([NT.eval_pos(G[f][chosen[f][0]], r, sg)[0] for f in inc], axis=0)
                res[cost] = (N.stats(cm[:cut]), N.stats(cm[cut:]))
            o08, o15 = res[0.0008][1], res[0.0015][1]
            row.update({"is": res[0.0008][0]["sharpe"], "oos08": o08["sharpe"], "oos15": o15["sharpe"], "oos_total": o08["total"], "oos_mdd": o08["mdd"]})
        else:
            row.update({"is": float("nan"), "oos08": float("nan"), "oos15": float("nan")})
        bres = {}
        for cost in (0.0008, 0.0015):
            N.COST = cost
            pn = NT.eval_pos(G["호흡"][BREATH], r, sg)[0]; st = N.stats(pn[cut:]); bres[cost] = st["sharpe"] if st else float("nan")
        row["breath08"], row["breath15"] = bres[0.0008], bres[0.0015]
        row["grade"] = grade(n, row["oos08"], row["oos15"]) if inc and np.isfinite(row["oos08"]) else ("자료부족" if n < 3000 else "C")
        N.COST = 0.0008
        out[s] = row
        print(f"{s:10s} {n:6d} {row['start']:>10s} {row['grade']:>6s} {','.join(inc)[:24]:24s} {row['is']:+6.2f} {row['oos08']:+7.2f} {row['oos15']:+7.2f} | {row['breath08']:+.2f} / {row['breath15']:+.2f}")
    with open("state/alts_methods.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1, default=float)
    g = pd.Series({s: v.get("grade") for s, v in out.items()}).value_counts().to_dict()
    print("\n등급 분포:", g)


if __name__ == "__main__":
    main()
