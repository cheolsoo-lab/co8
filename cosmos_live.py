"""앱용: 직진·횡보 판정(30봉 효율비), 달 위상, 반감기 뒤 날짜.  검증은 cosmos.py · cosmos_sky.py."""
from __future__ import annotations
import numpy as np, pandas as pd

W, THR = 30, 0.3
SYNODIC = 29.530588853
NEW_MOON_REF = pd.Timestamp("2000-01-06 18:14", tz="UTC")
LAST_HALVING = pd.Timestamp("2024-04-20", tz="UTC")


def er_state(df: pd.DataFrame) -> dict:
    """마감된 4H 봉 기준: 효율비 = |30봉 순이동| / 30봉 움직임 합. 0.3 이상 직진(입자), 미만 횡보(파동)."""
    lc = np.log(df["close"].to_numpy(float))
    r = np.diff(lc[-(W + 1):])
    er = float(abs(lc[-1] - lc[-1 - W]) / np.abs(r).sum())
    return {"er": er, "mode": "직진" if er >= THR else "횡보", "dir": float(np.sign(lc[-1] - lc[-1 - W])),
            "anchor": float(lc[-W]), "path29": float(np.abs(r[1:]).sum()), "last": float(lc[-1])}


def er_next(s: dict, price: float) -> float:
    """다음 4H 봉이 price로 마감하면 그때의 효율비."""
    lp = np.log(price)
    return float(abs(lp - s["anchor"]) / (s["path29"] + abs(lp - s["last"])))


def factor(s: dict | None, price: float | None = None) -> float:
    if not s: return 1.0
    er = s["er"] if price is None else er_next(s, price)
    return 0.5 if er < THR else 1.0


def moon(ts: pd.Timestamp | None = None) -> tuple[float, str]:
    ts = ts or pd.Timestamp.now(tz="UTC")
    ph = ((ts - NEW_MOON_REF) / pd.Timedelta(days=1) / SYNODIC) % 1.0
    names = ["신월", "초승", "상현", "차오름", "보름", "기욺", "하현", "그믐"]
    return float(ph), names[int(np.floor(ph * 8)) % 8]


def halving_days(ts: pd.Timestamp | None = None) -> int:
    return int(((ts or pd.Timestamp.now(tz="UTC")) - LAST_HALVING).days)


def crossings(s: dict) -> list[float]:
    """다음 마감가가 이 값을 넘나들면 직진↔횡보가 바뀜 (효율비 = 0.3 이 되는 가격들)."""
    a, l, P = s["anchor"], s["last"], s["path29"]; out = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            den = s1 - THR * s2
            if abs(den) < 1e-12: continue
            x = (s1 * a + THR * P - THR * s2 * l) / den
            if np.sign(x - a or 1e-18) in (s1, 0) and np.sign(x - l or 1e-18) in (s2, 0):
                out.append(float(np.exp(x)))
    return sorted(set(round(p, 8) for p in out))


def split_rows(rows: list[dict], s: dict) -> list[dict]:
    """전환 가격표를 직진/횡보 경계에서 더 나눔 (각 칸 안에서는 크기 배수가 같아짐)."""
    cuts = crossings(s); out = []
    for r in rows:
        lo, hi = r["lo"], r["hi"]; pts = [lo] + [c for c in cuts if (not np.isfinite(lo) or c > lo) and (not np.isfinite(hi) or c < hi)] + [hi]
        for a, b in zip(pts[:-1], pts[1:]):
            out.append({**r, "lo": a, "hi": b})
    return out


def mid(r: dict) -> float:
    lo, hi = r["lo"], r["hi"]
    if not np.isfinite(lo): return hi * 0.999
    if not np.isfinite(hi): return lo * 1.001
    return (lo + hi) / 2
