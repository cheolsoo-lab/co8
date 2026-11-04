"""기술적 지표 (모두 '현재 봉까지의 정보'만 사용 — 미래참조 없음)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = true_range(df).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0)


def macd_hist(close: pd.Series, fast=12, slow=26, sig=9) -> pd.Series:
    m = ema(close, fast) - ema(close, slow)
    return m - ema(m, sig)


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = sma(close, n)
    sd = close.rolling(n).std()
    up, lo = mid + k * sd, mid - k * sd
    pctb = (close - lo) / (up - lo).replace(0, np.nan)
    return pctb, (up - lo) / mid


def ichimoku(df: pd.DataFrame, t=9, k=26, s=52):
    """구름대. 선행스팬은 k봉 뒤로 그려지므로 '현재 봉에 보이는 구름' = k봉 전에 계산한 값 (미래참조 없음)."""
    hh = lambda n: df["high"].rolling(n).max()
    ll = lambda n: df["low"].rolling(n).min()
    tenkan = (hh(t) + ll(t)) / 2
    kijun = (hh(k) + ll(k)) / 2
    span_a = ((tenkan + kijun) / 2).shift(k)
    span_b = ((hh(s) + ll(s)) / 2).shift(k)
    return tenkan, kijun, span_a, span_b


def obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["close"].diff().fillna(0)) * df["volume"]).cumsum()


def cmf(df: pd.DataFrame, n=20) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    mfv = ((2 * df["close"] - df["low"] - df["high"]) / rng) * df["volume"]
    return (mfv.rolling(n).sum() / df["volume"].rolling(n).sum().replace(0, np.nan)).fillna(0)


def candle_patterns(df: pd.DataFrame):
    """망치형/핀바/장악형/도지 → (강세 반전, 약세 반전)."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    hammer = (lower >= 2 * body) & (upper <= body.clip(lower=rng * 0.1)) & (lower >= 0.5 * rng)
    star = (upper >= 2 * body) & (lower <= body.clip(lower=rng * 0.1)) & (upper >= 0.5 * rng)
    bull_eng = (c.shift(1) < o.shift(1)) & (c > o) & (c >= o.shift(1)) & (o <= c.shift(1))
    bear_eng = (c.shift(1) > o.shift(1)) & (c < o) & (c <= o.shift(1)) & (o >= c.shift(1))
    return (hammer | bull_eng).fillna(False), (star | bear_eng).fillna(False)


def pivots(df: pd.DataFrame, left=3, right=3):
    """확정된 스윙 고/저. 값은 '확정 시점(right봉 뒤)'에 나타난다 → 미래참조 없음.
    반환: (swing_high, swing_low, 피벗 캔들의 위치 오프셋 = right)"""
    w = left + right + 1
    hs, ls = df["high"].shift(right), df["low"].shift(right)
    sh = hs.where(hs == df["high"].rolling(w).max())
    sl = ls.where(ls == df["low"].rolling(w).min())
    return sh, sl


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    r = df.resample(rule, label="left", closed="left")
    out = pd.DataFrame({"open": r["open"].first(), "high": r["high"].max(), "low": r["low"].min(),
                        "close": r["close"].last(), "volume": r["volume"].sum()})
    return out.dropna()
