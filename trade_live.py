"""실전 행동 판정 (11-10): 진입 · 추가 · 부분익절 · 부분손절 · 익절 · 손절 · 뒤집기 · 유지 · 관망.
- 앱 기준 평단 = 앱 방향(추세+호흡)이 시작된 4시간 마감가. 내 평단을 넣으면 그것으로 익절/손절을 가름.
- 추적 부분익절: 방향 시작 뒤 유리하게 2ATR 이상 갔다가 최고점에서 2ATR 되돌리면 크기 ½ (방향이 바뀔 때까지).
  검증(research/partial.py · partial2.py): 비트겟 2022-08~ 샤프 1.95→2.15·낙폭 −16%→−12% (엄격 통과), 바이낸스 2017~ 1.62→1.72, 코인베이스 11년 2.04→2.08.
  같은 배율이면 연 수익은 줄고(54%→47%) 낙폭이 더 크게 줄어듦 → 같은 낙폭이면 하루 성장 +0.03~0.05%p.
- 따로 정한 부분손절(불리 1~3ATR → ½)은 모두 효과 없음 → 손절은 앱의 전환 가격이 맡음."""
from __future__ import annotations
import numpy as np, pandas as pd
import indicators as ta, precision as PR, novel as N

K, KEEP = 2.0, 0.5
TONE = {"진입": ("#1d4ed8", "#e8efff"), "추가": ("#1d4ed8", "#e8efff"), "부분익절": ("#15803d", "#e6f6ec"), "익절": ("#15803d", "#e6f6ec"),
        "부분손절": ("#c2410c", "#fdeee4"), "손절": ("#b91c1c", "#fde8e8"), "뒤집기": ("#6d28d9", "#efe9fd"), "유지": ("#374151", "#eef0f3"), "관망": ("#6b7280", "#f3f4f6")}


def trail_state(df: pd.DataFrame) -> dict:
    F = PR.coin_frame(df); base = N.band(F["d"] * F["cap"], 0.15); c = df["close"].to_numpy(float); atr = ta.atr(df).to_numpy()
    d = np.sign(base); n = len(d)
    if d[-1] == 0: return {"dir": 0}
    st = n - 1
    while st > 0 and d[st - 1] == d[-1]: st -= 1
    a0 = atr[st] if np.isfinite(atr[st]) and atr[st] > 0 else atr[-1]; c0 = c[st]
    mv = d[-1] * (c[st:] - c0) / a0; best = np.maximum.accumulate(mv)
    cut = bool(np.any((best[1:] >= K) & (best[1:] - mv[1:] >= K))) if len(mv) > 1 else False
    b = float(best[-1]); dd = float(d[-1])
    return {"dir": dd, "t0": str(df.index[st]), "c0": float(c0), "a0": float(a0), "best": b, "mv": float(mv[-1]), "cut": cut,
            "peak": float(c0 + dd * b * a0), "cut_px": (float(c0 + dd * (b - K) * a0) if (b >= K and not cut) else None), "arm_px": float(c0 + dd * K * a0)}


def factor(st: dict | None, sign_pos: float, price: float | None = None) -> float:
    if not st or st.get("dir", 0) == 0 or np.sign(sign_pos) != st["dir"]: return 1.0
    if st["cut"]: return KEEP
    if price is not None:
        m = st["dir"] * (price - st["c0"]) / st["a0"]; b = max(st["best"], m)
        if b >= K and b - m >= K: return KEEP
    return 1.0


def split_rows(rows: list[dict], st: dict | None) -> list[dict]:
    x = (st or {}).get("cut_px")
    if x is None: return rows
    out = []
    for r in rows:
        lo, hi = r["lo"], r["hi"]
        if (not np.isfinite(lo) or x > lo) and (not np.isfinite(hi) or x < hi): out += [{**r, "hi": x}, {**r, "lo": x}]
        else: out.append(r)
    return out


def classify(ref: float, tgt: float, price: float, entry: float | None, step: float, nm: str) -> tuple[str, str]:
    """지금(ref) → 목표(tgt): (종류, 한 줄 설명)."""
    dq = max(0, -int(np.floor(np.log10(step)))); f = lambda x: f"{abs(x):.{dq}f} {nm}"; side = lambda x: "롱" if x > 0 else "숏"
    if abs(ref - tgt) < step / 2: return ("유지", f"{side(tgt)} {f(tgt)} 그대로") if tgt else ("관망", "포지션 없이 기다림")
    if ref == 0: return "진입", f"{'롱 매수' if tgt > 0 else '숏 매도'} {f(tgt)}"
    pnl = np.sign(ref) * (price / entry - 1) if entry else None
    win = pnl is None or pnl >= 0; ptxt = f" ({pnl:+.1%})" if pnl is not None else ""
    if tgt == 0: return ("익절" if win else "손절"), f"{side(ref)} {f(ref)} 전량 정리{ptxt}"
    if np.sign(ref) != np.sign(tgt): return "뒤집기", f"{side(ref)} {f(ref)} {'익절' if win else '손절'}{ptxt} → {side(tgt)} {f(tgt)} 진입"
    if abs(tgt) > abs(ref): return "추가", f"{side(ref)} +{f(abs(tgt) - abs(ref))} 더 → 합계 {f(tgt)}"
    return ("부분익절" if win else "부분손절"), f"{side(ref)} {f(abs(ref) - abs(tgt))} 정리{ptxt} → {f(tgt)} 남김"


def pill(kind: str, text: str, big: bool = False) -> str:
    fg, bg = TONE.get(kind, TONE["유지"])
    sz = "1.05rem" if big else "0.82rem"
    return (f'<span style="display:inline-block;background:{bg};color:{fg};border:1px solid {fg}33;border-radius:8px;padding:{"5px 10px" if big else "1px 7px"};'
            f'font-size:{sz};font-weight:700;margin-right:6px">{kind}</span><span style="font-size:{sz};color:#1f2937;font-weight:{600 if big else 500}">{text}</span>')
