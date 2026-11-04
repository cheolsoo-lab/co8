"""베어트랩(아래 돌파 실패 → 롱) — 견고성 확인 + 실시간 판정.  python beartrap.py
정의: 지그재그(ATR k배)로 확정된 직전 저점을 종가로 깨고, W봉 안에 종가로 다시 그 위로 올라오면 → 다음 H봉 롱."""
import math, json, warnings
import numpy as np, pandas as pd
import data, indicators as ta, novel as N, precision as PR
from patterns2 import zigzag

warnings.filterwarnings("ignore")
K, WIN, HOLD = 2.0, 3, 6


def signal(df, k=K, win=WIN, hold=HOLD):
    """반환: (보유 신호 배열 0/1, 이벤트 위치 목록, 마지막 상태 dict)."""
    h, l, c = (df[x].to_numpy(float) for x in ("high", "low", "close")); atr = ta.atr(df).to_numpy(); n = len(c)
    piv = zigzag(h, l, atr, k); by_t = {}
    for p in piv: by_t.setdefault(p[0], []).append(p)
    lastL = None; bo = None; sig = np.zeros(n); events = []
    for t in range(n):
        for p in by_t.get(t, []):
            if p[3] == -1: lastL = p[2]
        if t > 0 and lastL is not None and c[t] < lastL <= c[t - 1]: bo = (lastL, t)
        if bo is not None:
            if t > bo[1] and t - bo[1] <= win and c[t] > bo[0]:
                events.append(t); sig[t:t + hold] = 1.0; bo = None
            elif t - bo[1] > win: bo = None
    state = {"armed_level": float(bo[0]) if bo is not None else None, "armed_left": (win - (n - 1 - bo[1])) if bo is not None else 0,
             "last_low": float(lastL) if lastL is not None else None, "active_left": int(max(0, hold - (n - 1 - events[-1]))) if events else 0,
             "last_event": str(df.index[events[-1]]) if events else ""}
    return sig, events, state


def main():
    FR = {}; D = {}
    for s in ("BTCUSDT", "ETHUSDT"):
        D[s] = ta.resample_ohlcv(data.load_history(s, "1H", 1825, refresh=False), "4h").dropna(); FR[s] = PR.coin_frame(D[s])
    sh = PR.sh
    def run(k, win, hold):
        xs = {}
        for s in FR:
            sg, ev, _ = signal(D[s], k, win, hold); xs[s] = PR.pnl(N.band(sg * FR[s]["cap"], 0.15), FR[s], 0.0003)
        p = 0.5 * (xs["BTCUSDT"] + xs["ETHUSDT"]); h = len(p) // 2
        return sh(p), sh(p[:h]), sh(p[h:]), xs
    print("══ 견고성: 베어트랩 단독 샤프 (앞/뒤) — 지그재그 k × 복귀 허용 봉 × 보유 봉 ══")
    grid = []
    for k in (1.5, 2.0, 2.5):
        for win in (2, 3, 5):
            row = []
            for hold in (3, 6, 12):
                a, b, c_, _ = run(k, win, hold); row.append(f"{a:+.2f}({b:+.1f}/{c_:+.1f})"); grid.append((k, win, hold, a, b, c_))
            print(f"  k={k} 복귀≤{win}봉 | 보유 3·6·12봉: " + "  ".join(row))
    pos = sum(1 for g in grid if g[3] > 0); both = sum(1 for g in grid if g[4] > 0 and g[5] > 0)
    print(f"  → 27개 중 플러스 {pos}개, 앞·뒤 모두 플러스 {both}개")
    # 앱 + 베어트랩 동시 운용(같은 크기 합산)
    base = {s: PR.pnl(N.band(FR[s]["d"] * FR[s]["cap"], 0.15), FR[s], 0.0003) for s in FR}
    _, _, _, tx = run(K, WIN, HOLD)
    for nm, xs in (("앱 방식만", base), ("앱 + 베어트랩 (합산)", {s: base[s] + tx[s] for s in FR})):
        p = 0.5 * (xs["BTCUSDT"] + xs["ETHUSDT"]); h = len(p) // 2; e = np.cumprod(1 + p)
        print(f"  {nm:18s} 샤프 {sh(p):+.2f} (앞/뒤 {sh(p[:h]):+.2f}/{sh(p[h:]):+.2f}) · BTC {sh(xs['BTCUSDT']):+.2f} · ETH {sh(xs['ETHUSDT']):+.2f} · 연 {e[-1]**(PR.B4/len(p))-1:+.0%} · 낙폭 {(e/np.maximum.accumulate(e)-1).min():.0%}")
    json.dump({"grid": grid}, open("state/beartrap_result.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    for s in D:
        _, ev, stt = signal(D[s]); print(s, "최근 상태:", stt)


if __name__ == "__main__":
    main()
