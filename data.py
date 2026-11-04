"""데이터 수집: 비트겟(캔들·펀딩·미결제약정·롱숏비율), CoinGecko(도미넌스)."""
from __future__ import annotations
import os, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import requests

BASE = "https://api.bitget.com"
PT = "USDT-FUTURES"
CACHE = Path(__file__).parent / "data_cache"
CACHE.mkdir(exist_ok=True)
STATE = Path(__file__).parent / "state"
STATE.mkdir(exist_ok=True)

GRAN_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
           "1H": 3_600_000, "4H": 14_400_000, "1D": 86_400_000}


def _get(path: str, params: dict | None = None, retries: int = 4):
    last = None
    for k in range(retries):
        try:
            r = requests.get(BASE + path, params=params, timeout=15)
            j = r.json()
            if j.get("code") == "00000":
                return j["data"]
            last = j
        except Exception as e:  # 네트워크 오류 → 재시도
            last = str(e)
        time.sleep(0.6 * (k + 1))
    raise RuntimeError(f"Bitget 요청 실패 {path}: {last}")


def _to_df(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "quote_vol"])
    df = df.astype({"ts": "int64", "open": float, "high": float, "low": float,
                    "close": float, "volume": float, "quote_vol": float})
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df.index.name = "time"
    return df.drop(columns=["ts"]).sort_index()


def fetch_candles(symbol: str, gran: str = "1H", bars: int = 1000, end_ms: int | None = None) -> pd.DataFrame:
    """최근 bars개 캔들(오래된→최신). history-candles를 200개씩 거슬러 올라가며 수집."""
    frames, end = [], end_ms or int(time.time() * 1000) + GRAN_MS[gran]
    got = 0
    while got < bars:
        rows = _get("/api/v2/mix/market/history-candles",
                    {"symbol": symbol, "productType": PT, "granularity": gran,
                     "endTime": end, "limit": 200})
        if not rows:
            break
        d = _to_df(rows)
        frames.append(d)
        got += len(d)
        new_end = int(d.index[0].timestamp() * 1000) - 1
        if new_end >= end:
            break
        end = new_end
        time.sleep(0.08)
    if not frames:
        raise RuntimeError("캔들 없음")
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.iloc[-bars:]


def drop_unclosed(df: pd.DataFrame, gran: str) -> pd.DataFrame:
    """아직 마감되지 않은 마지막 봉 제거(리페인팅 방지)."""
    now = pd.Timestamp.now(tz="UTC")
    if len(df) and df.index[-1] + pd.Timedelta(milliseconds=GRAN_MS[gran]) > now:
        return df.iloc[:-1]
    return df


def load_history(symbol: str, gran: str = "1H", days: int = 730, refresh: bool = True) -> pd.DataFrame:
    """백테스트용 장기 데이터. 디스크 캐시에 누적."""
    f = CACHE / f"{symbol}_{gran}.csv"
    need = int(days * 86_400_000 / GRAN_MS[gran])
    old = None
    if f.exists():
        old = pd.read_csv(f, index_col=0, parse_dates=True, encoding="utf-8")
        old.index = pd.to_datetime(old.index, utc=True)
    if old is not None and not refresh and len(old) >= need:
        return old.iloc[-need:]
    if old is not None and len(old) > 0:
        gap = int((pd.Timestamp.now(tz="UTC") - old.index[-1]).total_seconds() * 1000 / GRAN_MS[gran]) + 5
        new = fetch_candles(symbol, gran, min(max(gap, 10), need))
        df = pd.concat([old, new]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        if len(df) < need:  # 과거가 부족하면 더 가져옴
            older = fetch_candles(symbol, gran, need, end_ms=int(df.index[0].timestamp() * 1000) - 1)
            df = pd.concat([older, df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
    else:
        df = fetch_candles(symbol, gran, need)
    df.to_csv(f, encoding="utf-8")
    return drop_unclosed(df, gran).iloc[-need:]


# ───────────── 파생상품 지표 ─────────────
def ticker(symbol: str) -> dict:
    return _get("/api/v2/mix/market/ticker", {"symbol": symbol, "productType": PT})[0]


def funding_current(symbol: str) -> float:
    return float(_get("/api/v2/mix/market/current-fund-rate", {"symbol": symbol, "productType": PT})[0]["fundingRate"])


def open_interest(symbol: str) -> float:
    d = _get("/api/v2/mix/market/open-interest", {"symbol": symbol, "productType": PT})
    return float(d["openInterestList"][0]["size"])


def long_short_ratio(symbol: str, period: str = "1h") -> pd.DataFrame:
    rows = _get("/api/v2/mix/market/account-long-short", {"symbol": symbol, "period": period})
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms", utc=True)
    df["ratio"] = df["longShortAccountRatio"].astype(float)
    return df.set_index("ts")[["ratio"]].sort_index()


def contract_info(symbol: str) -> dict:
    d = _get("/api/v2/mix/market/contracts", {"symbol": symbol, "productType": PT})[0]
    return {"min_qty": float(d.get("minTradeNum", 0.001)),
            "step": float(d.get("sizeMultiplier", 0.001)),
            "min_usdt": float(d.get("minTradeUSDT", 5)),
            "max_lev": int(float(d.get("maxLever", 100))),
            "price_place": int(d.get("pricePlace", 1))}


# ───────────── 거시(도미넌스) ─────────────
MACRO_HIST = STATE / "macro_history.csv"


def macro_snapshot() -> dict | None:
    """CoinGecko global → BTC.D, ETH.D, USDT.D, TOTAL, TOTAL2, TOTAL3. 실패 시 None."""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=12)
        g = r.json()["data"]
        mp = g["market_cap_percentage"]
        total = g["total_market_cap"]["usd"]
        snap = {"ts": pd.Timestamp.now(tz="UTC").isoformat(),
                "btc_d": mp.get("btc", np.nan), "eth_d": mp.get("eth", np.nan),
                "usdt_d": mp.get("usdt", np.nan), "total": total,
                "total2": total * (1 - mp.get("btc", 0) / 100),
                "total3": total * (1 - (mp.get("btc", 0) + mp.get("eth", 0)) / 100),
                "chg24": g.get("market_cap_change_percentage_24h_usd", np.nan)}
        row = pd.DataFrame([snap])
        row.to_csv(MACRO_HIST, mode="a", header=not MACRO_HIST.exists(), index=False, encoding="utf-8")
        return snap
    except Exception:
        return None


def macro_history() -> pd.DataFrame:
    if not MACRO_HIST.exists():
        return pd.DataFrame()
    df = pd.read_csv(MACRO_HIST, encoding="utf-8")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts").sort_index()


def funding_history(symbol: str, refresh: bool = False) -> pd.Series:
    """펀딩비 이력(8시간마다) — 캐시. 인덱스=정산시각(UTC)."""
    f = CACHE / f"{symbol}_funding.csv"
    if f.exists() and not refresh:
        s = pd.read_csv(f, index_col=0, parse_dates=True, encoding="utf-8").iloc[:, 0]
        s.index = pd.to_datetime(s.index, utc=True)
        if (pd.Timestamp.now(tz="UTC") - s.index[-1]) < pd.Timedelta(days=2):
            return s
    rows, page = [], 1
    while page <= 80:
        d = _get("/api/v2/mix/market/history-fund-rate",
                 {"symbol": symbol, "productType": PT, "pageSize": 100, "pageNo": page})
        if not d:
            break
        rows += [(int(x["fundingTime"]), float(x["fundingRate"])) for x in d]
        page += 1
        time.sleep(0.08)
    s = pd.Series({pd.to_datetime(t, unit="ms", utc=True): r for t, r in rows}).sort_index()
    s.name = "funding"
    s.to_csv(f, encoding="utf-8")
    return s
