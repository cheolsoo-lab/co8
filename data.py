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


SOURCE = {}          # 진단용: (심볼, 봉) → 실제로 쓴 거래소 / 마지막 오류
OKX = "https://www.okx.com"


def _okx_inst(symbol: str) -> str:
    return symbol.replace("USDT", "") + "-USDT-SWAP"


def _okx_get(path: str, params: dict):
    r = requests.get(OKX + path, params=params, timeout=15); j = r.json()
    if str(j.get("code")) != "0":
        raise RuntimeError(f"OKX 요청 실패 {path}: {j}")
    return j["data"]


def _fetch_okx(symbol: str, gran: str, bars: int) -> pd.DataFrame:
    """비트겟이 막혔을 때 대체: OKX 무기한 선물 캔들 (4H·1D는 비트겟과 같은 시각 경계)."""
    rows, after = [], None
    while len(rows) < bars:
        p = {"instId": _okx_inst(symbol), "bar": gran, "limit": 100}
        if after: p["after"] = after
        d = _okx_get("/api/v5/market/history-candles", p)
        if not d: break
        rows += d; after = d[-1][0]; time.sleep(0.12)
    if not rows: raise RuntimeError("OKX 캔들 없음")
    df = pd.DataFrame([[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[6]), float(x[7])] for x in rows],
                      columns=["ts", "open", "high", "low", "close", "volume", "quote_vol"])
    df.index = pd.to_datetime(df["ts"], unit="ms", utc=True); df.index.name = "time"
    df = df.drop(columns=["ts"]).sort_index(); df = df[~df.index.duplicated(keep="last")]
    return df.iloc[-bars:]


CB = "https://api.exchange.coinbase.com"
GRAN_H = {"1H": 1, "4H": 4, "1D": 24}


def _cb_prod(symbol: str) -> str:
    return symbol.replace("USDT", "") + "-USD"


def _fetch_coinbase(symbol: str, gran: str, bars: int) -> pd.DataFrame:
    """두 번째 대체: 코인베이스 현물(USD) 1시간 봉을 모아 4H(UTC 0·4·8시…)·1D(16시 UTC 시작, 비트겟과 같음)로 합침."""
    gh = GRAN_H[gran]; t1 = pd.Timestamp.now(tz="UTC").floor("h") + pd.Timedelta(hours=1)
    cur = t1 - pd.Timedelta(hours=gh * (bars + 2)); rows = []
    while cur < t1:
        end = min(cur + pd.Timedelta(hours=300), t1)
        for k in range(3):
            r = requests.get(f"{CB}/products/{_cb_prod(symbol)}/candles", params={"granularity": 3600, "start": cur.isoformat(), "end": end.isoformat()},
                             timeout=15, headers={"User-Agent": "btceth-app"})
            if r.status_code == 200: rows += r.json(); break
            time.sleep(0.8 * (k + 1))
        else:
            raise RuntimeError(f"코인베이스 요청 실패 {r.status_code}")
        cur = end; time.sleep(0.12)
    if not rows: raise RuntimeError("코인베이스 캔들 없음")
    h = pd.DataFrame(rows, columns=["ts", "low", "high", "open", "close", "volume"]).astype(float)
    h.index = pd.to_datetime(h["ts"].astype("int64"), unit="s", utc=True); h = h.drop(columns="ts").sort_index(); h = h[~h.index.duplicated()]
    h["quote_vol"] = h["volume"] * h["close"]
    if gh == 1:
        df = h
    else:
        rs = h.resample(f"{gh}h", origin="epoch", offset="16h" if gh == 24 else "0h", label="left", closed="left")
        df = rs.agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "quote_vol": "sum"})
        cnt = rs["close"].count(); df = df[(cnt >= gh) | (df.index == df.index[-1])].dropna()
    df.index.name = "time"
    return df[["open", "high", "low", "close", "volume", "quote_vol"]].iloc[-bars:]


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
    """최근 bars개 캔들. 비트겟 우선, 막히면 OKX로 자동 대체 (과거 구간 지정 end_ms는 비트겟만)."""
    try:
        df = _fetch_bitget(symbol, gran, bars, end_ms)
        SOURCE[(symbol, gran)] = "Bitget"
        return df
    except Exception as e:
        if end_ms is not None:
            raise
        e1 = str(e)[:120]
    try:
        df = _fetch_okx(symbol, gran, bars)
        SOURCE[(symbol, gran)] = f"OKX (비트겟 실패: {e1})"
        return df
    except Exception as e:
        e2 = str(e)[:120]
    try:
        df = _fetch_coinbase(symbol, gran, bars)
        SOURCE[(symbol, gran)] = f"코인베이스 현물 (비트겟 실패: {e1} / OKX 실패: {e2})"
        return df
    except Exception as e:
        SOURCE[(symbol, gran)] = f"모두 실패 — 비트겟: {e1} / OKX: {e2} / 코인베이스: {str(e)[:120]}"
        raise RuntimeError(SOURCE[(symbol, gran)])


def _fetch_bitget(symbol: str, gran: str = "1H", bars: int = 1000, end_ms: int | None = None) -> pd.DataFrame:
    """history-candles를 200개씩 거슬러 올라가며 수집."""
    frames, end = [], end_ms or int(time.time() * 1000) + GRAN_MS[gran]
    first = True
    got = 0
    while got < bars:
        rows = _get("/api/v2/mix/market/history-candles",
                    {"symbol": symbol, "productType": PT, "granularity": gran,
                     "endTime": end, "limit": 200}, retries=2 if first else 4)
        first = False
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
    try:
        return _get("/api/v2/mix/market/ticker", {"symbol": symbol, "productType": PT}, retries=2)[0]
    except Exception:
        pass
    try:
        d = _okx_get("/api/v5/market/ticker", {"instId": _okx_inst(symbol)})[0]
        return {"lastPr": d["last"], "bidPr": d["bidPx"], "askPr": d["askPx"], "src": "OKX"}
    except Exception:
        d = requests.get(f"{CB}/products/{_cb_prod(symbol)}/ticker", timeout=10, headers={"User-Agent": "btceth-app"}).json()
        return {"lastPr": d["price"], "bidPr": d["bid"], "askPr": d["ask"], "src": "코인베이스"}


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
