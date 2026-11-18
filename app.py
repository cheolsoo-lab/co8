"""BTC·ETH 최종 분석 (단일 화면).  실행: streamlit run app.py   ·  예전 전체 화면: streamlit run app_full.py
방향 = 추세(7~90일) + 호흡(4H 연속봉) 앙상블, 크기 = 변동성 맞춤 × 배율, 지지·저항 = 참고 표시(검증상 필터로 쓰면 손해)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data, indicators as ta, robust_live as RL, final as FN

if not hasattr(RL, "triggers"):   # 파일 일부만 바뀐 경우 안내
    st.error("robust_live.py가 예전 버전입니다. 최신 zip의 파일 전체를 폴더에 덮어쓰고, Streamlit을 껐다가(Ctrl+C) 다시 실행하세요.")
    st.stop()

st.set_page_config(page_title="△", layout="wide", initial_sidebar_state="collapsed")
st.markdown("<style>.block-container{padding-top:1rem;padding-bottom:3rem;max-width:1000px}"
            "[data-testid=stMetricValue]{font-size:1.3rem}</style>", unsafe_allow_html=True)

COINS = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
STATE = data.STATE
JOURNAL = STATE / "journal.csv"
JCOLS = ["date", "symbol", "side", "entry", "exit", "qty", "pnl_usdt", "note"]
MIN_NOTIONAL = 5.0
LEV_SET = 3   # 거래소 레버리지 설정값(격리). 실제 위험은 '명목'이 결정, 레버리지는 증거금만 줄임

with st.expander("△ 설정", expanded=False):
    c1, c2 = st.columns(2)
    equity = c1.number_input("선물 계좌 자본 (USDT)", 10.0, 10_000_000.0, 80.0, 10.0, key="equity")
    mult = c2.select_slider("배율 (1.5 권장 · 4~5는 성장 최대지만 낙폭 −40~−50%)", options=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0], value=1.5, key="mult")
    order = st.radio("주문 방식", ["지정가 (Post-only, 메이커) — 권장", "시장가 (테이커)"], horizontal=True, key="order")
    mtf_on = st.checkbox("흐름 정렬 역가중 (기본) — 주봉·일봉·4시간이 모두 같으면 ½, 4시간이 둘 다 거스르면 1.5배 · 11년 확인: 최근 5년은 좋았고(샤프 0.63→0.96) 2015~2021엔 나빴음(2.31→2.02), 낙폭은 줄어듦", True, key="mtf_on")
    shape_on = st.checkbox("두 달 모양 반반 운용 (실험) — 4시간 방식 절반 + 일봉 60일 모양 절반을 한 수량으로 합침", False, key="shape_on")
    trap_on = st.checkbox("베어트랩 추가 (엄격 기준 통과 · 설정값 민감) — 직전 저점을 깼다가 3봉 안에 되돌아오면 24시간 롱 추가", False, key="trap_on")
    flow_on = st.checkbox("돈의 흐름 주머니 (기본·권장) — 테더·코인베이스 프리미엄이 높으면 롱 쪽, 낮으면 숏 쪽으로 최대 변동성 목표의 ½ 추가 · 비트겟 2022-08~ 샤프 1.77→1.95", True, key="flow_on")
    ex_on = st.checkbox("거래소 순유출 주머니 (시험 · 엄격 기준 미달) — 최근 30일 코인이 거래소에서 빠져나가면 롱 쪽, 들어오면 숏 쪽으로 변동성 목표의 ¼ 추가 · 비트겟 2022-08~ 샤프 2.15→2.23 (½이면 2.30), 바이낸스 2018~22 1.50→1.71", False, key="ex_on")
    trail_on = st.checkbox("추적 부분익절 (기본·권장) — 방향 시작 뒤 유리하게 2ATR 이상 갔다가 최고점에서 2ATR 되돌리면 절반 익절 · 비트겟 2022-08~ 샤프 1.95→2.15, 낙폭 −16%→−12% (같은 배율이면 연 수익은 조금 줄고, 같은 낙폭이면 성장 ↑)", True, key="trail_on")
    wave_on = st.checkbox("파동 국면 크기 (기본·권장) — 중파동이 높은 저점(2파)·4파 확인 뒤 새로 시작하면 그 방향 1.5배·반대 ½ · 11년 샤프 1.89→2.04, 비트겟 5년 1.36→1.55", True, key="wave_on")
    mode = st.radio("매매 방식 (하나만 고르기 — 둘은 보는 기간이 달라 서로 다른 말을 할 수 있음)",
                    ["4시간 추세 — 롱·숏 모두 (기본·권장, 낙폭 작음)", "스윙 — 길게 먹기 (롱만, 최근 5년 성적 약함, 낙폭 큼)"], key="mode")
    _qp = st.query_params
    def _qf(k_):
        try:
            return float(_qp.get(k_, 0) or 0)
        except Exception:
            return 0.0
    my_on = st.checkbox("내 실제 포지션 입력 (선택) — 넣으면 '지금 할 일'과 익절·손절을 내 수량·평단 기준으로 판정 (안 넣으면 앱을 그대로 따랐다고 가정)", _qp.get("my") == "1", key="my_on")
    MY = {}
    if my_on:
        _c = st.columns(4)
        bq = _c[0].number_input("BTC 수량 (+롱 −숏)", -100.0, 100.0, _qf("bq"), 0.0001, format="%.4f", key="bq")
        be = _c[1].number_input("BTC 평단", 0.0, 10_000_000.0, _qf("be"), 1.0, format="%.1f", key="be")
        eq_ = _c[2].number_input("ETH 수량 (+롱 −숏)", -10000.0, 10000.0, _qf("eq"), 0.01, format="%.2f", key="eq")
        ee = _c[3].number_input("ETH 평단", 0.0, 1_000_000.0, _qf("ee"), 0.1, format="%.2f", key="ee")
        MY = {"BTCUSDT": (float(bq), float(be) or None), "ETHUSDT": (float(eq_), float(ee) or None)}
        _new = {"my": "1", "bq": f"{bq:g}", "be": f"{be:g}", "eq": f"{eq_:g}", "ee": f"{ee:g}"}
        st.caption("주소창 주소를 즐겨찾기하면 입력값이 남습니다. 수량·평단은 거래소 포지션 화면 그대로.")
    else:
        _new = {"my": "0"}
    if any(_qp.get(k_) != v_ for k_, v_ in _new.items()):
        try:
            _qp.update(_new)
        except Exception:
            pass
    st.markdown("**수익판** — 이 날부터 앱대로 했다면")
    _c2 = st.columns(3)
    _today = pd.Timestamp.now(tz="Asia/Seoul").normalize()
    try:
        _sd0 = pd.Timestamp(_qp.get("sd")).date() if _qp.get("sd") else (_today - pd.Timedelta(days=30)).date()
    except Exception:
        _sd0 = (_today - pd.Timedelta(days=30)).date()
    _sdmin = (_today - pd.Timedelta(days=170)).date()
    s_date = _c2[0].date_input("시작일 (최근 170일 안)", max(_sd0, _sdmin), min_value=_sdmin, max_value=_today.date(), key="s_date")
    s_cap = _c2[1].number_input("시작 자본 (USDT)", 1.0, 10_000_000.0, (_qf("sc") or float(equity)), 10.0, key="s_cap")
    s_mon = _c2[2].number_input("매달 추가 입금 (USDT)", 0.0, 1_000_000.0, (_qf("sm") if _qp.get("sm") is not None else 70.0), 10.0, key="s_mon")
    _new2 = {"sd": str(s_date), "sc": f"{s_cap:g}", "sm": f"{s_mon:g}"}
    if any(_qp.get(k_) != v_ for k_, v_ in _new2.items()):
        try:
            _qp.update(_new2)
        except Exception:
            pass
SWING = mode.startswith("스윙")
MAKER = order.startswith("지정가")
MTF = bool(mtf_on)
SHAPE = bool(shape_on) and not SWING
TRAP = bool(trap_on) and not SWING
WAVE = bool(wave_on) and not SWING
FLOW = bool(flow_on) and not SWING
TRAIL = bool(trail_on) and not SWING
EXCH = bool(ex_on) and not SWING
import cycle_live as CYv
CYS = CYv.state()
WP = False   # 횡보면 ½: 11년 확인(2015~2021 처음 보는 기간)에서 나빠져 옵션 삭제 (cosmos.py · macro.py)


@st.cache_data(ttl=600, show_spinner="4H 봉 읽는 중…")
def load(symbol: str):
    df = data.drop_unclosed(data.fetch_candles(symbol, "4H", 1600), "4H")   # 주봉 26주 추세용 여유
    v = RL.one(df)
    v["src"] = data.SOURCE.get((symbol, "4H"), "Bitget")
    try:
        import wave_live as WL
        v["wave"] = WL.state(df); v["wave_prev"] = WL.state(df.iloc[:-1]); v["f53"] = [f for f in (WL.fate53(df, 4.0), WL.fate53(df, 8.0)) if f]
    except Exception:
        v["wave"] = v["wave_prev"] = None; v["f53"] = []
    try:
        import cosmos_live as CL
        v["er"] = CL.er_state(df); v["er_prev"] = CL.er_state(df.iloc[:-1])
    except Exception:
        v["er"] = v["er_prev"] = None
    v["pos_prev"] = RL.one(df.iloc[:-1])["pos"]
    try:
        import trade_live as TL0
        v["trail"] = TL0.trail_state(df); v["trail_prev"] = TL0.trail_state(df.iloc[:-1])
    except Exception:
        v["trail"] = v["trail_prev"] = None          # 직전 마감 기준 목표 (방금 바뀌었는지 표시용)
    try:
        import mtf_live as ML
        v["mtf"] = ML.state(df)
    except Exception:
        v["mtf"] = None
    try:
        import novel as N0
        r0 = np.concatenate([[0.0], np.diff(np.log(df["close"].to_numpy(float)))]); v["cap"] = float(min(N0.TV / N0.sigma(r0)[-1], N0.LMAX))
    except Exception:
        v["cap"] = 0.0
    F = FN.frame(df)
    v.update({"atr": float(F["atr"][-1]), "res": float(F["res"][-1]), "sup": float(F["sup"][-1]), "trig": RL.triggers(df)})
    try:
        import beartrap as BT, novel as N_
        sg, _, stt = BT.signal(df)
        r_ = np.concatenate([[0.0], np.diff(np.log(df["close"].to_numpy(float)))])
        stt.update({"active": float(sg[-1]), "cap": float(min(N_.TV / N_.sigma(r_)[-1], N_.LMAX))})
        v["trap"] = stt
    except Exception:
        v["trap"] = None
    try:
        import adaptive as AD, precision as PR_, novel as N2, fakes as FK
        F2 = PR_.coin_frame(df)
        g = int(AD.regimes(F2)[-1]); v["regime"] = AD.REG[g] if g >= 0 else None
        base = N2.band(F2["d"] * F2["cap"], 0.15); x = PR_.pnl(base, F2, 0.0003); n_ = len(x)
        segs = FK.segments(base); open_ = segs and segs[-1][2] == n_ and np.sign(base[-1]) != 0
        done = segs[:-1] if open_ else segs
        rets = [float(np.prod(1 + x[a + 1:b + 1]) - 1) for _, a, b in done]
        streak = 0
        for r_ in reversed(rets):
            if r_ <= 0: streak += 1
            else: break
        v["fake_streak"] = streak
        v["open_ret"] = float(np.prod(1 + x[segs[-1][1] + 1:]) - 1) if open_ else None
    except Exception:
        v["regime"] = v.get("regime"); v["fake_streak"] = None; v["open_ret"] = None
    return df.iloc[-120:], v


def load_shape(symbol: str):
    return _load_shape(symbol)


@st.cache_data(ttl=1800, show_spinner="일봉 모양 읽는 중…")
def _load_shape(symbol: str):
    import shape_model as SM
    df = data.drop_unclosed(data.fetch_candles(symbol, "1D", 400), "1D")
    return SM.state(df)


@st.cache_data(ttl=3600, show_spinner=False)
def fear_greed():
    try:
        import requests
        d = requests.get("https://api.alternative.me/fng/", params={"limit": 2, "format": "json"}, timeout=10).json()["data"]
        return int(d[0]["value"]), d[0]["value_classification"], int(d[1]["value"])
    except Exception:
        return None


@st.cache_data(ttl=20, show_spinner=False)
def live_px(symbol: str):
    try:
        t = data.ticker(symbol); return float(t["lastPr"]), float(t["bidPr"]), float(t["askPr"])
    except Exception:
        return None


STEP = {"BTCUSDT": 0.0001, "ETHUSDT": 0.01}   # 비트겟 최소 주문 단위(수량)


def qty(notional: float, price: float, step: float) -> float:
    """주문 가능한 수량(부호 포함): 거래소 단위로 반올림, 5 USDT 미만이면 0."""
    q = round(abs(notional) / price / step) * step
    return 0.0 if q * price < MIN_NOTIONAL else float(np.sign(notional) * q)


def todo(cur: float, tgt: float, price: float, step: float, nm: str) -> str:
    """현재 수량 → 목표 수량: 할 일 (수량은 거래소 단위로 반올림된 값)."""
    cur, tgt = qty(cur, price, step), qty(tgt, price, step)
    side = lambda x: "롱" if x > 0 else "숏"
    f = lambda x: f"{abs(x):.{max(0, -int(np.floor(np.log10(step))))}f} {nm}"
    if cur == 0 and tgt == 0: return "없음 (관망)"
    if cur == 0: return f"{side(tgt)} 진입 {f(tgt)}"
    if tgt == 0: return f"{side(cur)} 청산"
    if np.sign(cur) != np.sign(tgt): return f"{side(cur)} 청산 → {side(tgt)} {f(tgt)} 진입"
    if abs(tgt - cur) < step / 2: return "유지"
    return f"{side(cur)} {'늘리기' if abs(tgt) > abs(cur) else '줄이기'} → {f(tgt)}"


def chart(df, v):
    fig = go.Figure(go.Candlestick(x=df.index.tz_convert("Asia/Seoul"), open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                   increasing_line_color="#2e9e5b", decreasing_line_color="#d64545", name=""))
    for lv, nm, col in ((v["res"], "저항", "#d64545"), (v["sup"], "지지", "#2e9e5b")):
        if np.isfinite(lv):
            fig.add_hline(y=lv, line_dash="dot", line_color=col, annotation_text=f"{nm} {lv:,.1f}", annotation_position="top left")
    fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False, showlegend=False)
    return fig


@st.cache_data(ttl=1800, show_spinner="일봉 읽는 중…")
def load_swing(symbol: str):
    import swing_live as SL
    df = data.drop_unclosed(data.fetch_candles(symbol, "1D", 500), "1D")
    return df.iloc[-240:], SL.state(df)


def chart_swing(df, v):
    fig = go.Figure(go.Candlestick(x=df.index.tz_convert("Asia/Seoul"), open=df["open"], high=df["high"], low=df["low"], close=df["close"],
                                   increasing_line_color="#2e9e5b", decreasing_line_color="#d64545", name=""))
    fig.add_hline(y=v["hi20"], line_dash="dot", line_color="#2e9e5b", annotation_text=f"20일 최고 {v['hi20']:,.1f} (넘으면 돌파 롱)", annotation_position="top left")
    fig.add_hline(y=v["lo50"], line_dash="dot", line_color="#d64545", annotation_text=f"50일 최저 {v['lo50']:,.1f}", annotation_position="bottom left")
    fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5), xaxis_rangeslider_visible=False, showlegend=False)
    return fig


def render_journal(box=None):
    with (box or st).expander("△ 매매 일지", expanded=False):
        j = pd.read_csv(JOURNAL, encoding="utf-8") if JOURNAL.exists() else pd.DataFrame(columns=JCOLS)
        with st.form("jform", clear_on_submit=True):
            f1, f2, f3, f4 = st.columns(4)
            s_ = f1.selectbox("코인", list(COINS.values())); sd = f2.selectbox("방향", ["롱", "숏", "모의 롱", "모의 숏"])
            en = f3.number_input("진입가", 0.0, step=1.0); ex = f4.number_input("청산가", 0.0, step=1.0)
            g1, g2 = st.columns(2)
            q = g1.number_input("수량(코인)", 0.0, format="%.4f"); note = g2.text_input("메모")
            if st.form_submit_button("기록"):
                sign = 1 if "롱" in sd else -1
                pnl = sign * (ex - en) * q if ex > 0 else 0.0
                j = pd.concat([j, pd.DataFrame([[pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d %H:%M"), s_, sd, en, ex, q, round(pnl, 4), note]], columns=JCOLS)])
                j.to_csv(JOURNAL, index=False, encoding="utf-8")
        if len(j):
            st.dataframe(j.tail(20), hide_index=True, width="stretch")
            st.caption(f"누적 손익 {j['pnl_usdt'].sum():+,.2f} USDT · 기록 {len(j)}건")


st.title("△")
if SWING:
    st.caption("스윙 모드: 하루 한 번(일봉 마감 새벽 01시 KST, 아침에 확인하면 됨) → 비중이 바뀌었을 때만 조정. 숏은 하지 않음. 수익을 보장하지 않습니다.")
    for sym, nm in COINS.items():
        try:
            dfs, sv = load_swing(sym)
        except Exception as e:
            st.warning(f"{nm} 일봉 오류: {e}"); continue
        w = sv["weight"]; nt = equity * w / len(COINS)
        n_long = sum(sv["votes"].values())
        st.subheader(f"△ {nm}  ${sv['close']:,.1f}")
        a, b, c_ = st.columns(3)
        a.metric("판정", "관망 (현금)" if w == 0 else f"롱 {w:.0%}", f"12개 규칙 중 {n_long}개 롱 → 0·50·100% 중 {w:.0%}", delta_color="off")
        b.metric("지금 들고 있을 포지션", (f"롱 {nt:,.0f} USDT" if nt >= MIN_NOTIONAL else ("없음" if nt == 0 else "5 USDT 미만 → 없음")),
                 f"레버리지 1배 기준 (최대 2배)", delta_color="off")
        c_.metric("50일 / 200일 수익", f"{sv['ret50']:+.0%} / {sv['ret200']:+.0%}", f"마지막 조정 {sv['last_change'][:10]} ({sv['prev_weight']:.0%}→{w:.0%})", delta_color="off")
        st.caption("롱 규칙: " + ", ".join(k for k, x in sv["votes"].items() if x) + " · 현금 규칙: " + (", ".join(k for k, x in sv["votes"].items() if not x) or "없음"))
        st.plotly_chart(chart_swing(dfs, sv), width="stretch", key=f"sw_{sym}")
    with st.expander("△ 스윙 규칙과 검증 (8년, 2018-07~2026-11)", expanded=False):
        st.markdown("""
- **비중** = 아래 12개 규칙 중 '롱'인 비율 → 3단계(25% 미만 0%, 75% 미만 50%, 그 이상 100%). 하나를 고르지 않고 평균 (과최적 방지). BTC·ETH에 자본 반씩. 조정은 코인당 연 16회 정도.
  반전확인 10·15·20·25·30% (저점에서 그만큼 오르면 롱, 고점에서 그만큼 내리면 현금) · 신고가 돌파 20·50·100일 · 장기방향 50·100·200일 · 방향합의
- **숏은 안 함**: 스윙 기간(몇 주~몇 달)으로 숏을 하면 8년 누적 숏 수익 −42%. 코인은 급락 뒤 급반등이 커서, 느린 규칙이 하락을 확인할 즈음엔 이미 많이 빠진 뒤였음. 만장일치 약세일 때만 숏해도 −12%. **숏은 4시간 모드에서** (5년 숏 부분 연 +4%, 2022년 +7%·2026년 +11%).
- **주의 — 최근 5년(2021-10~)만 보면 연 −3%, 낙폭 −58%** (8년 성적은 2019~2021 상승장 덕분). 같은 5년 4시간 모드는 연 +15%, 낙폭 −20% (펀딩비 반영).

| 8년 검증 (BTC·ETH 반반, 1배) | 연 수익 | 최대 낙폭 | 샤프 (전반/후반) |
|---|---|---|---|
| **스윙 롱만 3단계** | **+42%** | **−55%** | 1.01 (1.26 / 0.68) |
| 같은 것, 펀딩비 연 10% 반영 | +34% | −58% | 0.87 (1.14 / 0.49) |
| 그냥 보유 (선물, 펀딩 반영) | +17% | −78% | 0.57 |
| 그냥 보유 (현물) | +29% | −76% | 0.72 |
| 스윙 ×2 레버리지 | 약 +65% | 약 −81% | 같음 |

- 이득의 대부분은 큰 하락장을 피한 것(2018년 −4% vs 보유 −64%, 2022년 −36% vs −66%). 후반 4년만 보면 그냥 보유와 비슷함.
- 고점·저점은 지나고 나서야 확인됨: 20% 되돌림 확인이면 BTC 상승 스윙 평균 +97% 중 실제 +31%, 스윙의 59%는 손실로 끝남 (큰 스윙 몇 개가 전체 수익).
- 한 번 들어간 뒤 최대 33~45%까지 역행한 적 있음 → 레버리지 2.2배를 넘으면 청산됐을 거래가 있었음. **1배 권장, 최대 2배.**
- 롱만 하는 방식이라 **현물로 해도 됨** → 펀딩비 없음, 청산 없음. 선물로 하면 펀딩비만큼(연 약 8%p) 깎임.
""")
    render_journal()
    st.stop()

st.caption("4시간 마감(01·05·09·13·17·21시) 직후 확인 · 마감가 기준 · 수익 보장 없음")

def f53_txt(f, sym):
    """5파·3파 판별 한 줄 (wave_live.fate53, 종가 기준)."""
    nm = "중파동" if f["k"] == 4.0 else "대파동"; u = f["s"] > 0; w, o = ("상승", "하락") if u else ("하락", "상승")
    up, dn = ("위", "아래") if u else ("아래", "위"); q = lambda x: f"{x:,.0f}"; br = "넘음" if u else "깸"
    stt = f["state"]
    if stt == "미결":
        return f"{nm} {w} 5파 끝 {q(f['v5'])}로 셈 · 사실은 3파일 확률 {f['p3']:.0%} · {q(f['v5'])} {up} 마감 = 3파 확정(따라가기) · {q(f['v4'])} {dn}는 약한 신호"
    if stt == "3파 확정":
        return f"{nm} {w} '5파' 끝 {q(f['v5'])} {br}({f['ago']}봉 전) → 사실은 3파 · 11년: 뒤 7일 {w} 쪽 평균 5% (맞음 55%)"
    if stt == "대기":
        return f"{nm} {w} 5파 끝 판정({q(f['v4'])} 이탈) 뒤 대기 · {q(f['v5'])} {up} = {w} 재개 / {q(f['v0'])} {dn} = {o} 전환 · 그 사이 신호는 약함"
    if stt == "추세 재개":
        return f"{nm} {w} 5파 끝 판정 뒤 {q(f['v5'])} 다시 {br}({f['ago']}봉 전) → {w} 재개 · 11년: 뒤 7일 {w} 쪽 평균 7% (맞음 59%)"
    return f"{nm} {w} 5파 시작점 {q(f['v0'])} 이탈({f['ago']}봉 전) → {o} 전환 · 11년: 뒤 7일 {o} 쪽 평균 6% (맞음 62%)"


def render_diag(expanded=False, box=None):
    """접속 진단: 어느 시세 서버가 되는지, 어떤 버전으로 돌고 있는지."""
    with (box or st).expander("△ 접속 진단", expanded=expanded):
        if st.button("지금 다시 검사", key="diag_btn"):
            st.cache_data.clear()
        import sys, platform, time as _t, requests as _rq
        tests = [("비트겟 (기본)", "https://api.bitget.com/api/v2/mix/market/ticker", {"symbol": "BTCUSDT", "productType": "USDT-FUTURES"}),
                 ("OKX (1차 대체)", "https://www.okx.com/api/v5/market/ticker", {"instId": "BTC-USDT-SWAP"}),
                 ("코인베이스 (2차 대체)", "https://api.exchange.coinbase.com/products/BTC-USD/ticker", None),
                 ("공포·탐욕 지수", "https://api.alternative.me/fng/", {"limit": 1})]
        out = []
        for nm_, url, p in tests:
            t0 = _t.time()
            try:
                r = _rq.get(url, params=p, timeout=8, headers={"User-Agent": "btceth-app"})
                out.append((nm_, "되는 중" if r.status_code == 200 else f"막힘 (HTTP {r.status_code})", f"{(_t.time() - t0) * 1000:.0f} ms", r.text[:80].replace("\n", " ")))
            except Exception as e:
                out.append((nm_, "접속 실패", "-", str(e)[:80]))
        st.dataframe(pd.DataFrame(out, columns=["시세 서버", "상태", "응답", "내용 앞부분"]), hide_index=True, width="stretch")
        used = {COINS[s]: (rows[s][1].get("src", "?") if s in rows else "불러오기 실패") for s in COINS}
        st.caption("지금 화면에 쓴 시세: " + " · ".join(f"{k} {v_}" for k, v_ in used.items()))
        import importlib.metadata as _md
        vers = []
        for pk in ("streamlit", "pandas", "numpy", "plotly", "scikit-learn", "scipy", "requests"):
            try: vers.append(f"{pk} {_md.version(pk)}")
            except Exception: vers.append(f"{pk} 없음")
        st.caption(f"파이썬 {platform.python_version()} · " + " · ".join(vers))
        st.caption("셋 다 '막힘'이면 그 서버에서 시세를 못 받는 것 → 내 PC에서 실행. 하나라도 '되는 중'이면 자동으로 그걸 씀.")


rows = {}
errs = {}
for sym, nm in COINS.items():
    try:
        rows[sym] = load(sym)
    except Exception as e:
        errs[nm] = str(e)
        st.warning(f"{nm} 데이터 오류: {e}")
if errs:
    render_diag(expanded=True)
_alt = {COINS[s]: rows[s][1].get("src", "") for s in rows if not str(rows[s][1].get("src", "Bitget")).startswith("Bitget")}
if _alt:
    st.caption("시세 대체 · " + " · ".join(f"{k} {v_.split(' (')[0]}" for k, v_ in _alt.items()) + " (비트겟 막힘)")
tab_trade, tab_gain, tab_info = st.tabs(["△ 매매", "△ 내 수익", "△ 근거·검증"])
tab_info.markdown("**△ 참고 지표**")

@st.cache_data(ttl=3600, show_spinner="일봉(거시) 읽는 중…")
def load_macro(symbol: str):
    import macro_live as MCL
    dd = data.drop_unclosed(data.fetch_candles(symbol, "1D", 700), "1D")
    return MCL.state(dd)


MAC = {}
for _s, _n in COINS.items():
    try:
        MAC[_n] = load_macro(_s)
    except Exception:
        MAC[_n] = None
@st.cache_data(ttl=3600, show_spinner=False)
def load_dom():
    import dominance_live as DL
    return DL.state()


try:
    DOM = load_dom()
except Exception:
    DOM = None
@st.cache_data(ttl=1800, show_spinner="돈의 흐름 읽는 중…")
def load_flow():
    import flow_live as FLv
    return FLv.state()


try:
    FLW = load_flow()
except Exception:
    FLW = None
if FLW:
    import flow_live as FLv
    _sc = np.mean([FLW[c]["score"] for c in ("BTC", "ETH")])
    tab_info.caption(f"돈의 흐름 {FLv.label(_sc)} · 테더 {FLW['usdt']:.4f} ({FLW['usdt_z']:+.1f}σ) · 코인베이스 프리미엄 BTC {FLW['BTC']['prem3']:+.2%} ({FLW['BTC']['prem_z']:+.1f}σ) · "
               f"ETH {FLW['ETH']['prem3']:+.2%} ({FLW['ETH']['prem_z']:+.1f}σ)" + (" — 크기에 반영" if FLOW else " — 설정에서 꺼짐"))
@st.cache_data(ttl=6 * 3600, show_spinner="거래소 입출금 읽는 중…")
def load_onchain():
    import onchain_live as OCv
    return OCv.state()


try:
    OCH = load_onchain()
except Exception:
    OCH = None
if OCH:
    tab_info.caption("거래소 30일 입출금(유통량 대비) · " + " · ".join(f"{c_} {'순유출' if OCH[c_]['pct30'] < 0 else '순입금'} {abs(OCH[c_]['pct30']):.2f}% ({OCH[c_]['z']:+.1f}σ)" for c_ in ("BTC", "ETH"))
                     + f" · {OCH['BTC']['t']} 기준" + (" — 크기에 반영" if EXCH else " — 표시만 (설정에서 켤 수 있음)"))
tab_info.caption(f"반감기 사이클: {CYS['halving']} 반감기 뒤 {CYS['days']}일 · {CYS['name']} · 다음 반감기 {CYS['next']} 무렵 · 이 국면 앱 역사 샤프 {CYv.SHARPE[CYS['phase']]:.2f}"
                 + " — 표시만 (국면별 크기 조절은 교차 검토에서 기각)")
_fg = fear_greed()
_top = []
if any(MAC.values()):
    _top.append("거시 " + " · ".join(f"{n} {m['label']}" for n, m in MAC.items() if m))
if _fg:
    _kr = {"Extreme Fear": "극단 공포", "Fear": "공포", "Neutral": "중립", "Greed": "탐욕", "Extreme Greed": "극단 탐욕"}.get(_fg[1], _fg[1])
    _top.append(f"공포·탐욕 {_fg[0]} {_kr}")
if _top:
    tab_info.caption(" │ ".join(_top) + " — 참고 (매매엔 안 씀)")
# 배지 색: (배경, 테두리, 왼쪽 강조선·글자색, 표시) — 은은한 바탕 + 진한 글자
BADGE = {"롱": ("#f0f8f3", "#d3eadb", "#1f7a45", "롱 ▲"), "숏": ("#fcf2f2", "#f1d4d4", "#b23b3b", "숏 ▼"), "관망": ("#f5f6f8", "#e3e6ea", "#5b6270", "관망 ―")}
top_box = tab_trade.container()
SUMM = []
badge_cols = tab_trade.columns(len(COINS))
exp_box = tab_trade.container(); detail_box = tab_trade.container()
try:
    with open(STATE / "flow_cases.json", encoding="utf-8") as _fh:
        CASES = json.load(_fh)["pool"]
except Exception:
    CASES = {}


try:
    with open(STATE / "fake_runs.json", encoding="utf-8") as _fh:
        FAKE_RUNS = json.load(_fh)
except Exception:
    FAKE_RUNS = {}
try:
    with open(STATE / "topdown_result.json", encoding="utf-8") as _fh:
        TOPDOWN = json.load(_fh)["table"]
except Exception:
    TOPDOWN = {}


@st.cache_data(ttl=600, show_spinner=False)
def funding_rate(sym: str):
    """지금 8시간 펀딩비 (비트겟, 막히면 OKX). +면 롱이 숏에게 냄."""
    import requests
    try:
        d = requests.get("https://api.bitget.com/api/v2/mix/market/current-fund-rate", params={"symbol": sym, "productType": "usdt-futures"}, timeout=10).json()["data"][0]
        return float(d["fundingRate"])
    except Exception:
        try:
            return float(requests.get("https://www.okx.com/api/v5/public/funding-rate", params={"instId": sym.replace("USDT", "-USDT-SWAP")}, timeout=10).json()["data"][0]["fundingRate"])
        except Exception:
            return None


def chip(t, tone=0):
    col = {1: ("#1f7a45", "#e3f2e8"), -1: ("#b23b3b", "#f8e3e3"), 0: ("#4b5563", "#eef0f3")}[int(np.sign(tone))]
    return f'<span style="display:inline-block;margin:3px 4px 0 0;padding:1px 8px;border-radius:10px;background:{col[1]};color:{col[0]};font-size:0.78rem;white-space:nowrap">{t}</span>'


def badge(nm, price, word, strength, qty_line, action, sub, lines, next_t, chips="", conf="", entry="", hold=""):
    bg, bd, ac, label = BADGE[word]
    ln = ("".join(f'<div style="margin-top:3px">{x}</div>' for x in lines) + '<div style="margin-top:3px;color:#6b7280;font-size:0.8rem">그 밖의 가격이면 유지</div>') if lines else '<div style="margin-top:3px;color:#6b7280">어느 가격에 마감해도 그대로</div>'
    return (f'<div style="border-radius:12px;padding:12px 14px 10px 14px;margin-bottom:12px;background:{bg};border:1px solid {bd};border-left:5px solid {ac};color:#374151;line-height:1.45">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px">'
            f'<span style="font-size:1.45rem;font-weight:700;color:{ac}">{label}<span style="font-size:0.85rem;font-weight:500;color:#6b7280"> {strength}</span></span>'
            f'<span style="font-size:0.9rem;color:#374151;font-weight:600;white-space:nowrap">{nm} {price}</span></div>'
            f'<div style="font-size:1.0rem;font-weight:600;color:#1f2937">{qty_line}</div>'
            + (f'<div style="margin-top:4px">{conf}</div>' if conf else "") +
            f'<div style="margin-top:9px;padding:8px 10px;background:#ffffffcc;border-radius:8px;border:1px solid {bd}">'
            f'<div style="font-size:0.74rem;color:#6b7280;margin-bottom:3px">지금 할 일 · {sub}</div>{action}'
            + (f'<div style="font-size:0.78rem;color:#6b7280;margin-top:4px">{entry}</div>' if entry else "") + '</div>'
            + (f'<div style="margin-top:6px;font-size:0.8rem;color:#374151">{hold}</div>' if hold else "") +
            f'<div style="margin-top:8px;font-size:0.74rem;color:#6b7280">다음 {next_t:%H:%M} 마감가가</div><div style="font-size:0.86rem;color:#1f2937">{ln}</div>'
            + (f'<div style="margin-top:8px;padding-top:6px;border-top:1px dashed {bd}">{chips}</div>' if chips else "") + '</div>')


def desc(q, nm, dq):
    return "없음" if q == 0 else f"{'롱' if q > 0 else '숏'} {abs(q):.{dq}f} {nm}"


for i_c, (sym, nm) in enumerate(COINS.items()):
    if sym not in rows:
        continue
    df, v = rows[sym]
    k = equity * mult / len(COINS)
    sv = None
    if SHAPE:
        try:
            sv = load_shape(sym)
        except Exception as e:
            st.warning(f"{nm} 일봉 모양 오류: {e} — 4시간 방식만 사용")
    sp0 = 0.5 * sv["pos"] if sv else 0.0
    fl = (__import__("flow_live").W * FLW[nm]["s"] * v.get("cap", 0.0)) if (FLOW and FLW and nm in FLW) else 0.0   # 돈의 흐름 주머니
    exs = (__import__("onchain_live").W * OCH[nm]["s"] * v.get("cap", 0.0)) if (EXCH and OCH and nm in OCH) else 0.0   # 거래소 순유출 주머니 (시험)
    sp = sp0 + fl + exs
    wt = 0.5 if sv else 1.0
    tr_ = v.get("trap") or {}
    tp = tr_.get("active", 0.0) * tr_.get("cap", 0.0) if TRAP else 0.0                 # 지금 베어트랩 몫
    tp_next = tr_.get("cap", 0.0) if (TRAP and tr_.get("active_left", 0) >= 2) else 0.0   # 다음 마감 뒤에도 남는 몫
    import mtf_live as ML
    mstate = v.get("mtf") if (MTF and not SWING) else None
    fx = (lambda p: ML.factor(float(np.sign(p)), mstate)[0]) if mstate else (lambda p: 1.0)
    import cosmos_live as CL
    ers = v.get("er") if WP else None
    wf = CL.factor(ers)                                                   # 지금 봉 기준 직진 1 / 횡보 ½
    wfx = (lambda r: CL.factor(ers, CL.mid(r))) if ers else (lambda r: 1.0)
    import wave_live as WL
    wvf = (lambda p: WL.factor(v.get("wave"), float(np.sign(p)))) if WAVE else (lambda p: 1.0)
    wvr = (lambda r: WL.factor(v.get("wave"), float(np.sign(r["pos"])), CL.mid(r))) if WAVE else (lambda r: 1.0)   # 그 가격에 마감하면 (꼭짓점 되돌아 넘으면 해제)
    import trade_live as TL
    trl = v.get("trail") if TRAIL else None
    tfx = (lambda raw, price=None: TL.factor(trl, float(np.sign(raw)), price)) if trl else (lambda raw, price=None: 1.0)
    _raw = k * (wt * v["pos"] * fx(v["pos"]) * wf * wvf(v["pos"]) + sp + tp)
    notional = _raw * tfx(_raw)
    px = live_px(sym); last = px[0] if px else v["close"]
    stp = STEP[sym]; dq = max(0, -int(np.floor(np.log10(stp))))
    q_now = qty(notional, last, stp); has = q_now != 0
    side = ("숏" if q_now < 0 else "롱") if has else ""
    close_t = (pd.Timestamp(v["time"]) + pd.Timedelta(hours=4)).tz_convert("Asia/Seoul")
    next_t = close_t + pd.Timedelta(hours=4)
    pf = 2 if sym == "ETHUSDT" else 1
    entry = ""
    if has and px:
        ent = px[2] if q_now < 0 else px[1]
        entry = (f"진입·조정은 {'매도1호가' if q_now < 0 else '매수1호가'} {ent:,.{pf}f} 지정가" if MAKER else f"진입·조정은 시장가 (지금 {last:,.{pf}f})") + f" · {LEV_SET}배 격리 증거금 약 {abs(q_now)*last/LEV_SET:,.0f} USDT"
    elif abs(notional) > 0.5:
        entry = f"목표 {abs(notional):.0f} USDT가 최소 단위 {stp} {nm}(약 {stp*last:,.0f} USDT)보다 작아 관망"
    tab_info.markdown(f"**△ {nm} 참고**")
    # 다음 마감 기준 가장 가까운 '할 일이 생기는' 가격 (위·아래)
    T = CL.split_rows(v["trig"], ers) if ers else v["trig"]
    if WAVE: T = WL.split_rows(T, v.get("wave"))
    if trl: T = TL.split_rows(T, trl)
    raw_of = lambda r: k * (wt * r["pos"] * fx(r["pos"]) * wfx(r) * wvr(r) + sp + tp_next)
    tgt_of = lambda r: raw_of(r) * tfx(raw_of(r), CL.mid(r))
    up = dn = None
    # 기준 포지션: 내 입력 또는 '직전 마감 목표를 그대로 따랐다'
    _rawp = k * (wt * v["pos_prev"] * fx(v["pos_prev"]) * CL.factor(v.get("er_prev") if WP else None) * (WL.factor(v.get("wave_prev"), float(np.sign(v["pos_prev"]))) if WAVE else 1.0) + sp + tp)
    q_prev = qty(_rawp * (TL.factor(v.get("trail_prev"), float(np.sign(_rawp))) if TRAIL else 1.0), last, stp)
    def app_entry(q):
        for st_ in (v.get("trail"), v.get("trail_prev")):
            if st_ and st_.get("dir") and np.sign(q) == st_["dir"]: return st_["c0"], st_["t0"]
        return None, None
    mine = MY.get(sym) if MY else None
    if mine is not None:
        ref_q, ref_e = qty(mine[0] * last, last, stp) if mine[0] else 0.0, mine[1]; ref_src = "내 입력"
    else:
        ref_q = q_prev; ref_e, _t0 = app_entry(q_prev); ref_src = "앱 기준"
    kind, ktxt = TL.classify(ref_q, q_now, last, ref_e, stp, nm)
    if kind in ("부분익절", "익절") and trl and TL.factor(v.get("trail_prev"), float(np.sign(_rawp))) == 1.0 and TL.factor(trl, float(np.sign(_raw))) < 1.0:
        ktxt += " · 최고점 대비 2ATR 되돌림"
    # 진입 뒤 평단 (다음 마감 판정용)
    if mine is not None and mine[1] and np.sign(mine[0]) == np.sign(q_now): e_now, e_src = mine[1], "내 평단"
    else:
        e_now, _t0 = app_entry(q_now); e_src = "앱 기준 평단"
        if e_now is None: e_now, e_src = last, "지금 가격"
    lines = []
    for r in T:
        tg = qty(tgt_of(r), last, stp)
        if abs(tg - q_now) < stp / 2: continue
        if r["lo"] >= v["close"] - 1e-9 and (up is None or r["lo"] < up[0]): up = (r["lo"], tg, r)
        if r["hi"] <= v["close"] + 1e-9 and (dn is None or r["hi"] > dn[0]): dn = (r["hi"], tg, r)
    tf_now = tfx(_raw)
    for edge, arrow, word_ in ((up, "▲", "위"), (dn, "▼", "아래")):
        if edge and np.isfinite(edge[0]):
            kd, tx = TL.classify(q_now, edge[1], edge[0], e_now, stp, nm)
            if kd in ("부분익절", "익절") and trl and np.sign(raw_of(edge[2])) == trl.get("dir") and tfx(raw_of(edge[2]), CL.mid(edge[2])) < tf_now:
                tx += " · 최고점 대비 2ATR 되돌림" + (" (½이 최소 주문 단위보다 작아 전량)" if (edge[1] == 0 and abs(tgt_of(edge[2])) > 0.5) else "")
            lines.append(f"{arrow} <b>{edge[0]:,.{pf}f}</b> {word_} → " + TL.pill(kd, tx))
    # 상단 한눈 배지
    word = "관망" if not has else side
    vd = RL.verdict(v["dir"]); strength = ""
    if has:
        strength = "(강)" if "(강)" in vd else "(약)"
        if not vd.startswith(side): strength = "(신호 약해짐 · 유지)"
    change = q_prev != q_now
    action = TL.pill(kind, ktxt, big=True)
    sub = (f"{close_t:%H:%M} 마감 기준 · {('내 포지션 ' + desc(ref_q, nm, dq)) if mine is not None else ('직전 목표 ' + desc(q_prev, nm, dq) + '를 따랐다면')}")
    hold = ""
    if has and e_now and e_src != "지금 가격":
        _pn = np.sign(q_now) * (last / e_now - 1)
        _t0s = f", {pd.Timestamp(app_entry(q_now)[1]).tz_convert('Asia/Seoul'):%m-%d %H:%M}" if (e_src == "앱 기준 평단" and app_entry(q_now)[1]) else ""
        hold = f"{e_src} {e_now:,.{pf}f}{_t0s} · 지금 <b style='color:{'#15803d' if _pn >= 0 else '#b91c1c'}'>{_pn:+.1%}</b> {'수익 중' if _pn >= 0 else '손실 중'}"
        if trl and np.sign(q_now) == trl.get("dir"):
            if TRAIL and trl.get("cut"): hold += " · 절반 익절 완료(이 방향 끝날 때까지 ½)"
            elif trl.get("cut_px") is not None: hold += f" · 부분익절 감시 중 (최고 {trl['peak']:,.{pf}f})"
            else: hold += f" · {trl['arm_px']:,.{pf}f} {'위' if trl['dir'] > 0 else '아래'}까지 가면 부분익절 감시 시작"
    qline = (f"{abs(q_now):.{dq}f} {nm} · 약 {abs(q_now)*last:,.0f} USDT" if has else "포지션 없음")
    arw = lambda x: "▲" if x > 0 else ("▼" if x < 0 else "―")
    _vd = RL.verdict(v["dir"]); _st = "강" if "(강)" in _vd else ("약" if "(약)" in _vd else "중립")
    chips = chip(f"추세 {arw(v['dir']) if _st != '중립' else '―'} {_st}", v["dir"] if _st != "중립" else 0)
    mood = ""
    if v.get("fake_streak") is not None:
        mx = FAKE_RUNS.get(sym, {}).get("max_run")
        if v.get("open_ret") is not None and has:
            mood = f"이번 방향 {'+' if v['open_ret'] >= 0 else ''}{v['open_ret']:.1%} · "
        mood += f"가짜 전환 연속 {v['fake_streak']}번" + (f" (최장 {mx})" if mx else "")
        if change: mood += " · 새 방향 4번 중 3번은 가짜"
    if v.get("mtf"):
        f_, ag_ = ML.factor(float(np.sign(v["pos"])), v["mtf"])
        tag = {2: "크기 ½", 1: "크기 그대로", 0: "크기 1.5배"}.get(ag_, "")
        chips += chip(f"주{arw(v['mtf']['dW'])} 일{arw(v['mtf']['dD'])} 4H{arw(v['pos'])}" + (f" · {tag}" if (tag and MTF) else ""), 0)
        wsg = float(np.sign(v["mtf"]["dW"]))
        if wsg != 0:
            import topdown as TD
            code = int(wsg > 0) * 4 + int(np.sign(v["mtf"]["dD"]) == wsg) * 2 + int(np.sign(v["dir"]) == wsg)
            tdi = TOPDOWN.get(str(code))
            if tdi:
                mood = f"{TD.label(code).replace(' · ', '·')} · 4주 내 전환 {tdi['flip']:.0%}<br>" + mood
    wv = v.get("wave")
    if wv and "label" in wv:
        _a = lambda x: "▲" if x > 0 else ("▼" if x < 0 else "―")
        _lab = (wv["label"].replace("완성", "끝").replace(" (0-3)", "").replace(" (0-4 유효)", "").replace("·", " ")
                .replace(" (높은 저점)", "(높은 저점)" if wv["dir"] > 0 else "(낮은 고점)"))
        _txt = f"파동 소{_a(wv['dirs'][2.0])} 중{_a(wv['dirs'][4.0])} 대{_a(wv['dirs'][8.0])} · 중파동 {_lab}"
        if wv.get("active"):
            _txt += f" → {'롱' if wv['dir'] > 0 else '숏'} 1.5배 · {wv['stop']:,.0f} {'아래' if wv['dir'] > 0 else '위'}면 해제" if WAVE else " (설정에서 켜야 반영)"
        mood += ("<br>" if mood else "") + _txt
        chips += chip(f"파동 {_lab}" + (f" → {'롱' if wv['dir'] > 0 else '숏'} 1.5배" if (wv.get("active") and WAVE) else ""), wv["dir"] if wv.get("active") else 0)
    case = ""
    if OCH and nm in OCH:
        _oz = OCH[nm]["z"]
        chips += chip(f"거래소 30일 {'순유출' if OCH[nm]['pct30'] < 0 else '순입금'} {_oz:+.1f}σ" + (f" → {'롱' if exs > 0 else '숏'} 쪽 +{abs(k * exs):,.0f}" if (EXCH and abs(k * exs) >= 0.5) else ""), _oz if abs(_oz) >= 0.5 else 0)
    if FLW and nm in FLW:
        import flow_live as FLv
        _sc = FLW[nm]["score"]
        _nd = FLW[nm].get("neg_days", 0)
        if _nd >= 3: chips += chip(f"코인베이스 프리미엄 음수 {_nd:.0f}일째 (미국 수요 약함)", -1)
        elif FLW[nm].get("pos_days", 0) >= 3: chips += chip(f"코인베이스 프리미엄 양수 {FLW[nm]['pos_days']:.0f}일째 (미국 수요 강함)", 1)
        chips += chip(f"돈의 흐름 {FLv.label(_sc)} {_sc:+.1f}σ" + (f" → {'롱' if fl > 0 else '숏'} 쪽 +{abs(k * fl):,.0f}" if (FLOW and abs(k * fl) >= 0.5) else ""), _sc if abs(_sc) >= 0.5 else 0)
        if has and CASES:
            _ag = float(np.sign(q_now)) * _sc
            _key = "흐름 강하게 같은 쪽 (≥1σ)" if _ag >= 1 else ("흐름 약하게 같은 쪽" if _ag > 0 else ("흐름 약하게 반대" if _ag > -1 else "흐름 강하게 반대 (≤−1σ)"))
            _c = CASES.get(_key)
            if _c:
                _tier, _col = ("높음", "#15803d") if _ag >= 1 else (("낮음", "#b91c1c") if _ag <= -1 else ("보통", "#4b5563"))
                case = (f"<span style='display:inline-block;border:1px solid {_col}55;color:{_col};border-radius:10px;padding:1px 8px;font-size:0.8rem;font-weight:700'>신뢰도 {_tier}</span>"
                        f"<span style='font-size:0.78rem;color:#4b5563'> 과거 같은 경우({side} + {_key.split(' (')[0]}) 7일 뒤 이 포지션 맞음 {_c['hit']:.0%} · 평균 {_c['mean']:+.1%}</span>")
    _fr = funding_rate(sym)
    if _fr is not None and has:
        _pay = np.sign(q_now) * _fr > 0; _day = abs(_fr) * 3 * abs(q_now) * last
        chips += chip(f"펀딩 {_fr * 100:+.4f}%/8h · {side} {'냄' if _pay else '받음'} 하루 약 {_day:.2f} USDT" + (" · 과열" if abs(_fr) >= 0.0005 else ""), (-1 if _pay else 1) if (abs(_fr) >= 0.0005 or _day >= 0.01) else 0)
    if DOM:
        import dominance_live as DL
        mood += "<br>" if mood else ""
        if sym == "BTCUSDT":
            mood += (f"USDT.D {DOM['usdt_d']:.2f}% · 7일 {DL.label(DOM['u_z7'])}({DOM['u_z7']:+.1f}σ) · 1년 범위 {DOM['u_lvl']:.0%} · BTC.D {DOM['btc_d']:.1f}% {DL.arrow(DOM['btc_d_tr'])}"
                     if "u_z7" in DOM else f"USDT.D {DOM['usdt_d']:.2f}% {DL.arrow(DOM['usdt_d_tr'])} · BTC.D {DOM['btc_d']:.1f}% {DL.arrow(DOM['btc_d_tr'])}")
        else:
            mood += (f"TOTAL3 7일 {DL.label(DOM['t3_z7'])}({DOM['t3_z7']:+.1f}σ) · 30일 {DL.label(DOM['t3_z30'])}({DOM['t3_z30']:+.1f}σ) · 알트 비중 {DL.arrow(DOM['alt_tr'])}"
                     if "t3_z7" in DOM else f"TOTAL3 {DL.arrow(DOM['t3_tr'])} · 알트 비중 {DL.arrow(DOM['alt_tr'])}")
    _pn_now = (np.sign(q_now) * (last / e_now - 1)) if (has and e_now and e_src != "지금 가격") else None
    SUMM.append({"nm": nm, "side": side, "has": has, "pnl": _pn_now, "kind": kind, "txt": ktxt, "next": next_t, "line": lines[0] if lines else ""})
    if kind in ("익절", "부분익절"):
        _tk = f"toast_{sym}_{close_t:%m%d%H}"
        if not st.session_state.get(_tk):
            st.session_state[_tk] = True
            st.toast(f"{nm} {kind} — {ktxt}. 규칙대로 수익을 챙길 차례입니다.")
    with badge_cols[i_c]:
        st.markdown(badge(nm, f"${last:,.{pf}f}", word, strength, qline, action, sub, lines, next_t, chips, case, entry, hold), unsafe_allow_html=True)
    if mood:
        tab_info.markdown(f'<div style="font-size:0.85rem;color:#4b5563;line-height:1.5">{mood}</div>', unsafe_allow_html=True)
    tb = []; seg = []
    for r in reversed(T):                                   # 위 가격대 → 아래, 같은 결과가 이어지면 한 줄로
        tgt = qty(tgt_of(r), last, stp)
        _px = r["lo"] if np.isfinite(r["lo"]) and r["lo"] >= v["close"] else (r["hi"] if np.isfinite(r["hi"]) else last)
        _kd, _tx = TL.classify(q_now, tgt, _px, e_now, stp, nm); act = f"{_kd} · {_tx}"
        if seg and seg[-1][2] == tgt and seg[-1][3] == act: seg[-1][0] = r["lo"]
        else: seg.append([r["lo"], r["hi"], tgt, act])
    for lo_, hi_, tgt, act in seg:
        lo = "" if not np.isfinite(lo_) else f"{lo_:,.1f}"; hi = "" if not np.isfinite(hi_) else f"{hi_:,.1f}"
        rng = "어느 가격이든" if not (lo or hi) else (f"{lo} 위" if not hi else (f"{hi} 아래" if not lo else f"{lo} ~ {hi}")); here = lo_ <= last < hi_
        tb.append({"다음 4H 마감가": rng + ("  ← 지금" if here else ""), "그때 포지션": (f"{'숏' if tgt < 0 else '롱'} {abs(tgt):.{dq}f} {nm} (약 {abs(tgt)*last:,.0f} USDT)" if tgt else "없음"),
                   "할 일": act})
    _det = detail_box.expander(f"△ {nm} 전환 가격표 · 차트 ({next_t:%H:%M} 마감 기준)", expanded=False)
    _det.dataframe(pd.DataFrame(tb), hide_index=True, width="stretch")
    sr = []
    if np.isfinite(v["res"]): sr.append(f"저항 {v['res']:,.1f}")
    if np.isfinite(v["sup"]): sr.append(f"지지 {v['sup']:,.1f}")
    tab_info.caption(f"{RL.verdict(v['dir'])} · 점수 {v['dir']:+.2f} (추세 {v['tsmom']:+.2f} / 호흡 {v['breath']:+.2f}) · " + " · ".join(sr)
               + f" · {close_t:%m-%d %H:%M} 마감 기준" + (" · 지정가(30분 미체결 시 시장가)" if MAKER else ""))
    if v.get("er"):
        e_ = v["er"]
        tab_info.caption(f"결 · {e_['mode']} (효율비 {e_['er']:.2f}) — " + ("수익이 나는 구간" if e_["mode"] == "직진" else "본전 구간, 그대로 따르기"))
    if v.get("regime"):
        tab_info.caption(f"상황 · {v['regime']} — 방법은 고정, 크기만 변동성에 맞춤")
    if v.get("f53"):
        tab_info.caption("5파·3파 판별 · " + " │ ".join(f53_txt(f, sym) for f in v["f53"]) + " — 참고 (크기엔 안 씀)")
    if tr_:
        if tr_.get("active", 0) > 0:
            ttxt = f"발생 ({pd.Timestamp(tr_['last_event']).tz_convert('Asia/Seoul'):%m-%d %H:%M} 마감) → 롱 유지 남은 {tr_['active_left']}봉"
        elif tr_.get("armed_level"):
            ttxt = f"대기 — {tr_['armed_level']:,.1f} 위로 {tr_['armed_left']}봉 안에 마감하면 롱" + (f" (+{k*tr_['cap']:,.0f} USDT)" if TRAP else "")
        else:
            ttxt = f"없음 · 감시 저점 {tr_['last_low']:,.1f}" if tr_.get("last_low") else "없음"
        tab_info.caption(f"베어트랩 · {ttxt}" + ("" if TRAP else " (설정에서 켜야 반영)"))
    if sv:
        side_s = "롱" if sv["side"] > 0 else "숏"; held = "롱" if sv["held_side"] > 0 else ("숏" if sv["held_side"] < 0 else "없음")
        mte = f"{sv['m_te']:+.1%}" if sv["m_te"] is not None else "-"
        tab_info.caption(f"두 달 모양: 지금 모양은 '{side_s} 모양' (모양 {sv['cluster']}번 · 학습기간 다음 10일 평균 {sv['m_tr']:+.1%}, 확인기간 {mte}) · "
                   f"모양 몫은 {sv['since']}부터 {held} 유지 중 (남은 {sv['days_left']}일, 10일 단위로만 바꿈) · 모양 몫 {k*sp0:+,.0f} + 4시간 몫 {k*wt*v['pos']:+,.0f} = 합계 {notional:+,.0f} USDT (+롱 / −숏)")
        with tab_info.expander(f"△ {nm} 두 달 모양 보기", expanded=False):
            fg = go.Figure()
            fg.add_trace(go.Scatter(y=sv["shape"], mode="lines", name="지금 60일 모양", line=dict(color="#1f4fbf", width=3)))
            fg.add_trace(go.Scatter(y=sv["centroid"], mode="lines", name=f"모양 {sv['cluster']}번 대표 모양", line=dict(color="#9a9a9a", width=2, dash="dot")))
            fg.update_layout(height=240, margin=dict(l=5, r=5, t=10, b=5), xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False), legend=dict(orientation="h"))
            st.plotly_chart(fg, width="stretch", key=f"shp_{sym}")
    _det.plotly_chart(chart(df, v), width="stretch", key=f"ch_{sym}")

try:
    with open(STATE / ("certainty_flow.json" if (MTF and WAVE and FLOW) else ("certainty_wave.json" if (MTF and WAVE) else ("certainty_mtf.json" if MTF else "certainty.json"))), encoding="utf-8") as _fh:
        _cert = json.load(_fh)
    _tb = _cert["table"].get(f"{'maker' if MAKER else 'taker'}|{float(mult)}")
    if _tb:
        _y, _y3 = _tb["1년"], _tb["3년"]
        exp_box.caption(f"이 설정(배율 {mult}, {'지정가' if MAKER else '시장가'})의 지난 성적을 섞어 본 기대: 1년 수익 날 확률 {_y['p']:.0%} · 중앙 {_y['med']:+.0%} · 나쁜 10% {_y['p10']:+.0%}"
                        f" · 3년 안에 −30% 낙폭 겪을 확률 {_y3['dd30']:.0%}" + (f" ({_cert['period'][:7]}~ 기준, 좋았던 시기라 낙관적일 수 있음)" if 'period' in _cert else "") + " — 한 달은 동전 던지기, 신호를 건너뛰지 말 것")
        with tab_info.expander(f"△ 확실성 — 지금 설정(배율 {mult}, {'지정가' if MAKER else '시장가'})으로 하면", expanded=False):
            _rows = [{"기간": k, "수익 날 확률": f"{v['p']:.0%}", "중앙값": f"{v['med']:+.0%}", "나쁜 10%": f"{v['p10']:+.0%}", "좋은 10%": f"{v['p90']:+.0%}",
                      "중간에 −30% 낙폭 겪을 확률": f"{v['dd30']:.0%}"} for k, v in _tb.items()]
            st.dataframe(pd.DataFrame(_rows), hide_index=True, width="stretch")
            _a = _cert.get(f"asym|{'메이커' if MAKER else '테이커'}", {})
            if _a:
                st.caption(f"{'비트겟 ' + _cert['period'] + ' (돈의 흐름 자료가 있는 기간, 좋았던 시기라 낙관적일 수 있음)' if 'period' in _cert else '5년'} BTC·ETH 실제 손익을 30일 단위로 섞어 4천 번 다시 뽑은 결과(펀딩비 반영). 한 달은 동전 던지기에 가깝고 시간이 갈수록 확실해짐. "
                           f"방향 적중률은 약 50%지만 맞을 때 더 크게 먹어서 수익이 남 (이긴 달 {_a['mon_win']:.0%}, 평균 {_a['mon_avg_w']:+.1%} / 진 달 평균 {_a['mon_avg_l']:+.1%}). "
                           "한두 달 성적으로 방법을 바꾸지 마세요.")
except Exception:
    pass

with tab_info.expander("△ ETF 전후 — 시장이 어떻게 바뀌었나 · 역사 속 같은 변화 · 2032년까지 경우의 수 (11-18)", expanded=False):
    st.markdown("""
**① 숫자로 본 변화** (코인베이스 1시간 봉 · 옛날 2015~19 · ETF 전 2020~2024-01 · ETF 후 2024-01-11~)

| | 옛날 | ETF 전 | ETF 후 |
|---|---|---|---|
| BTC 연 변동성 (2022~24 이미 55% → ETF 이전부터 이어진 하락 추세) | 75% | 69% | **47%** |
| BTC 하루 ±5% 넘는 날 | 55일/년 | 44일/년 | **19일/년** |
| 미국 정규장(23~06시 KST) 변동 비중 | 25% | 26% | **39%** |
| 코인베이스 거래대금 비중 (바이낸스 현물 대비) | 19~27% | 10~26% | **27~36%** (계속 증가) |
| 나스닥 하루 상관 (주말 보정) | +0.03 | +0.41 | +0.43 (2025~26 +0.5) |
| 앱 샤프 BTC · ETH (배율 1, 흐름 주머니 없음) | 2.15 · 2.26 | 1.58 · 1.44 | **1.21 · 1.94** |
| 코인베이스 프리미엄 → 1주 수익 상관 BTC · ETH | +0.10 · +0.13 | +0.05 · +0.06 | **+0.10 · +0.14** |
| 테더 프리미엄 → 1주 수익 상관 BTC · ETH | — | +0.08 · +0.15 | **+0.12 · +0.16** |

- 변동은 줄고 미국 시간·미국 거래소로 몰림 → **미국 현물 수요(코인베이스·테더 프리미엄)의 예측력이 ETF 뒤 오히려 커짐**. 앱 '돈의 흐름'이 ETF 시대 핵심 재료.
- BTC는 추세가 약해지고(60~180일 방향 → 다음 1주 적중 48~53%) ETH는 여전히 추세형(60일 분산비 1.43). 두 코인을 따로 보는 이유.
- 교차 검토로 뺀 것: 첨도 비교(2020-03-12 하루 −49%가 만든 차이), '미국 정규장 수익 0%'(표준오차 ±17%/년, 우연 범위).

**② 2024-01 → 2026-11 흐름**
- ETF 출시 46.7k → 2025-10-06 **126k** (ETF 자산 정점 1,695억 달러) → 10-10 사상 최대 청산 190억 달러 → 2026년 ETF 순유출(연초 이후 −20억 달러, 6월 −45억 달러 사상 최악), 헤지펀드 베이시스 청산(13F 보유 35.5만→30.7만 BTC), Strategy 매수 급감(분기 8.5만→0.7만 BTC)·첫 매도 → 2026-07-01 **57.7k (−54%)** → 3분기 +43% 반등(86k) → 11월 65k.
- 지금 배경: 연준 12월 인상 확률 91%, 미 10년물 5%대, 달러 19개월 고점, ETF 2주 연속 −10억 달러, IBIT 보유자 평균 손실, Strategy 평균 단가 75.5k(시세 아래), 코인베이스 프리미엄 음수 70일 넘게 이어짐.

**③ 역사 속 같은 변화** — '접근성·제도 혁명 → 새 돈이 몇 년 몰림 → 새 돈이 멈추면 깊은 하락 → 긴 바닥 → 다른 매수자가 다음 시대를 엶'

| 사건 | 붐 | 붕괴 | 다시 신고가 |
|---|---|---|---|
| 브레턴우즈 종료 1971 → 금 | 41→850달러 (20배, 9년) | 명목 −60%·실질 −80% (19년) | 실질 기준 2025년 |
| 원자재 지수 붐 2000~08 → 유가 | 147달러 | −75% (5개월) | — |
| 닷컴 2000 → 나스닥 | 5,048 | −78% (31개월) | 15년 뒤 (2015) |
| 금 ETF(GLD) 2004 | +341% (7년) | −45% (4년, ETF 1,000톤 매도) | 9년 뒤 (2020) |
| **BTC 현물 ETF 2024** | **2.7배 (21개월)** | **−54% (9개월)** | ? |

- BTC는 같은 길을 훨씬 빨리 지나는 중. 블룸버그 발추나스(2026-07): "금처럼 큰 상승 뒤 긴 침체 가능 — GLD는 2011년 뒤 8년 박스".
- 다른 점: 공급이 정해져 있고 반감기 사이클이 아직 살아 있음, 그리고 하락 폭이 사이클마다 얕아짐.

**④ 반감기 사이클** (고점·저점은 지나고 나서 찾은 값)

| 반감기 | 고점까지 | 고점 배수(반감기 가격 대비) | 고점→저점 | 저점까지(반감기 뒤) |
|---|---|---|---|---|
| 2016-07 | 526일 (2017-12) | 30배 | −84% (363일) | 889일 |
| 2020-05 | 547일 (2021-11) | 7.9배 | −77% (377일) | 924일 |
| 2024-04 | 534일 (2025-10) | **2.0배** | **−54%** (267일, 지금까지) | 801일 (지금까지) |

- 고점 시기는 세 번 다 18개월 전후로 맞았고, 오름폭·하락폭은 크게 줄어듦. **지금은 반감기 뒤 942일 — 앞 두 번의 바닥 시기(889·924일)를 막 지남.**
- 분석가 의견은 갈림: 사이클 유지파(힐버트·갤럭시) 10월 전후 40~46k 바닥 예상 / 사이클 변화파(비트와이즈·SC) 6~7월이 바닥.

**⑤ 2026-11 → 2032 경우의 수** (앱은 롱·숏 추세 추종이라 하나를 맞힐 필요는 없음 — 경우마다 앱이 어떻게 되는지가 중요)

| 경우 | 줄거리 | 알아차리는 신호 | 앱은 |
|---|---|---|---|
| A 사이클 유지·진폭 감소 | 2027 회복 → 2028-04 반감기 → 2029 하반기 고점 → 2030 하락장 | 프리미엄 양수 전환, ETF 순유입 재개 | 2027~29 롱 추세, 2030 숏 추세에서 수익 |
| B 금 2011형 긴 박스 | 수년간 50~100k 오르내림 | ETF 유출 지속, 반등마다 프리미엄 음수, Strategy 매도 | **가장 불리** (2021·22년처럼 샤프 0.1~0.4) → 배율 낮추기 |
| C 통화 전환 가속 | 달러 약세·국부펀드·중앙은행 매수로 이른 신고가 | 달러 지수 하락, 금·BTC 동반 강세 | 롱 추세에서 큰 수익 |
| D 한 번 더 하락 | 매파 연준·고금리·강제 매도로 57.7k 이탈, 40~46k | 57.7k 종가 이탈, 프리미엄 음수 지속 | 숏 추세에서 수익 (2018·22 숏 몫 +0.23~+0.30) |

**⑥ 어느 시기에 비중?** 앱 국면별 샤프(BTC 2015~2026): 반감기~1년 1.73 · 1년~540일 **2.47** · 540~900일(하락장) **0.90** · 900일~다음 반감기 1.99.
- 국면별로 크기를 바꾸는 규칙(1.25/1.25/0.75/1)은 다섯 구간 모두 +0.05~+0.15였지만 **교차 검토에서 기각**: 경계(540·900일)가 같은 고점·저점에 맞춘 값이고, 경계를 450/810·620/1000일로 옮기면 ETF 뒤 성적이 −로 뒤집힘. 사이클도 3번뿐.
- 대신 '위험 달력'으로만 사용: **2029-10 ~ 2030-10 (역사적 고점 뒤 하락장 구간)** = 앱이 가장 약했던 시기이자 목표 시점(2030~32) 바로 앞 → 그때 자산이 목표 근처면 이익 일부 인출·배율 낮춤을 미리 정해 두기.

**⑦ ETF 시대에 통하는 차트 읽기** (일봉 신호 뒤 10·30일 수익, 변동성으로 나눈 값 · BTC·ETH × 세 구간)
- **55일 고점 종가 돌파 → 계속 오름** (BTC ETF 뒤 30일 +0.50) · **20일 저점 종가 이탈 → 계속 내림** (ETH ETF 뒤 30일 −0.61).
- **20일 저점 근처에서의 반등 롱(꼬리 회복·스프링 포함)은 6칸 모두 −** — 저점 근처는 반등보다 한 번 더 빠지는 쪽이 많았음. 교차 검토: 스프링만의 효과가 아니라 '저점 근처 약세' 한 가지 효과이고, 겹치는 신호를 빼면 표본 20~25개라 크기는 불확실. → **지지선 반등 롱은 추세가 돌아선 걸 확인한 뒤**(4시간·일봉 방향 전환, 20일 고점 회복).
- 과열(20일선 +1.5σ 위)은 되돌림보다 이어짐 → 과열 숏 금지. 꼬리만 넘은 가짜 돌파 숏도 효과 없음.
- 변동성이 47%로 줄어 하루 ±5% 날이 19일/년 → **% 손절·목표 대신 ATR(평균 변동폭) 기준**. 큰 움직임은 한국 새벽(23~06시)에 몰림.

**⑧ 시험했지만 넣지 않은 것** (ETF 전 −/+ 와 ETF 후가 엇갈림): 나스닥 20·60·200일 추세 문, 주말 크기 ±25%, 숏 크기 ½, 최근 성적 따라 코인 비중, 반감기 국면 크기.
- ETF 자금 흐름 일별 자료(파사이드 등)는 자동으로 받을 수 없어 직접 검증하지 못함 → 코인베이스·테더 프리미엄이 대신 보는 값(앱 카드 칩 '코인베이스 프리미엄 음수 N일째').
""")
    try:
        st.dataframe(pd.DataFrame(CYS["calendar"]).drop(columns=["시험 크기"]), hide_index=True, width="stretch")
        st.caption("반감기 사이클 달력 (2028·2032 반감기 날짜는 추정, 국면 경계는 과거 세 번에서 본 값이라 앞으로 어긋날 수 있음)")
    except Exception:
        pass
with tab_info.expander("△ 설정값 조율 — 숫자 20종·조합 500개를 세 구간에서 (11-17)", expanded=False):
    st.markdown("""
지금 방식의 모든 숫자(추세·호흡 비중, 보는 기간, 변동성 계산, 완충 폭, 흐름 정렬 가중, 파동 차수·배수, 돈의 흐름 비중·한계, 추적 부분익절 거리·몫·시작점, 숏 크기)를 바꿔 봄.
**고르기** = 코인베이스 2015~2022-07 · 바이낸스 2018-06~2022-07 / **확인** = 비트겟 무기한 2022-08~. 채택은 세 구간 모두 좋아지고 이웃 값도 같은 방향일 때만.

**① 하나씩 74가지** — 세 구간 모두 +0.05 이상은 0개. 세 구간 모두 +인 것은 2개(완충 폭 0.2, 숏 절반 크기 — 뒤쪽은 수익이 줄어 제외).
- 추세·호흡 비중, 보는 기간, 파동 차수(ATR×4), 돈의 흐름 비중(½), 추적 부분익절(2ATR·절반)은 지금 값이 확인 구간에서 가장 좋거나 이웃과 차이 없음.
- 흐름 정렬 역가중은 구간마다 정반대: 끄면 2015~22는 +0.5, 2022-08~는 −0.4. 최근 장세에 기대는 규칙이라는 뜻 → 그대로 두되 가장 조심할 부분.

**② 조합 500개** (2~5개 값을 한꺼번에 바꿈)
- 고르기 구간 상위 25개(평균 +0.37) → 확인 구간 평균 **−0.47**. 과거에 맞출수록 최근엔 나빠짐(두 구간 개선의 상관 −0.10).
- 확인 구간에서 지금 기본보다 나은 조합은 9%, +0.10 이상은 2% → 지금 값은 이미 꼭대기 근처이고, 더 맞추는 건 우연을 고르는 일.

**③ 채택 1개 — 완충 폭 0.15 → 0.175** (목표 크기가 이만큼 넘게 달라질 때만 수량 변경)

| 완충 폭 | 코인베이스 2015~22 | 바이낸스 2018~22 | 비트겟 2022-08~ | 수량 바뀌는 횟수 (비트겟) |
|---|---|---|---|---|
| 0.16 | +0.09 | +0.12 | +0.08 | 374→363번/년 |
| **0.175** | **+0.16** | **+0.13** | **+0.11** | 374→349번/년 |
| 0.19 | +0.18 | +0.15 | +0.08 | 374→335번/년 |
| 0.20 | +0.11 | +0.19 | +0.04 | 374→323번/년 |

이웃 네 값 × 세 구간 12칸이 모두 +, 이유도 분명(자잘한 수량 변경이 줄어 수수료 절약). 비트겟 구간 연 +47%→+49%, 낙폭 −12% 그대로. 작은 개선이며 수익을 크게 바꾸지는 않음.
""")
with tab_info.expander("△ 수익 재검토 3차 — 온체인·거시 재료 28개와 목표 계획 (11-17)", expanded=False):
    st.markdown("""
**① 아직 안 써 본 재료 28개** — 온체인(활성 주소·거래 수·MVRV·거래소 입출금·해시레이트), 거시(나스닥·VIX·금리·달러·순유동성·신용 스프레드), 공포탐욕.
판정은 전과 같음: 1주·3주 뒤 수익과의 순위상관, 과거 수익을 뺀 부분 상관, 시간 어긋낸 우주 p, BTC·ETH와 앞뒤 절반이 같은 부호인지.

| 재료 | 1주 상관 | p | 판정 |
|---|---|---|---|
| 거래소 순입금 7일·30일 (코인이 거래소로 들어옴) | −0.09 · −0.07 | 0.00 · 0.03 | 두 코인·앞뒤 모두 같은 부호 → 새 정보 |
| 장단기 금리차 (10년−2년) | +0.11 | 0.00 | 통과하지만 9년에 큰 굴곡 몇 번뿐인 느린 값 |
| MVRV 30일 변화 · 가격−활성주소 | +0.08 · +0.09 | 0.03 · 0.01 | 과거 수익을 빼면 0 → 추세의 다른 표현 |
| 나스닥·VIX·달러·실질금리·순유동성·활성 주소·거래 수·공포탐욕 | −0.03~+0.05 | 0.2~0.9 | 우연 범위 |

**② 주머니로 넣어 보면** (지금 기본 = 흐름 주머니 + 추적 부분익절 위에)
- 거래소 30일 순유출 ×½: 바이낸스 2018-06~2022-07 샤프 1.50→1.92 (우주 p=0.00) · 비트겟 2022-08~ 2.15→2.30, 연 +47%→+55%, 낙폭 −12% 그대로 (p=0.03). 주머니 단독은 BTC·ETH 9년 중 8년이 +.
- 그러나 창을 14·21·45·60·90일로 바꾸면 비트겟에선 −0.02~+0.04, BTC 단독은 1.93→1.88 → **엄격 기준 미달**. 과거 온체인 자료는 거래소 주소가 나중에 보정된 값이라 실시간보다 좋게 보일 수 있음.
- 장단기 금리차 ×½: 2.15→2.28이지만 최근 1년 1.71→1.26 · 2026년 손실 → 넣지 않음. 달러 60일: +0.06.
- 고르는 절차까지 검증(매 분기 그때까지 자료로만 재료 선택): 앱 위에 +0.11, 지금 기본 위에 ×¼ +0.06 · ×½ −0.06 → **새 재료의 실제 몫은 샤프 +0.05 안팎**.
- 결론: 카드에 '거래소 30일 순유출/순입금' 칩으로 **표시**, 크기 반영은 설정의 시험 옵션(×¼, 기본 끔).

**③ 목표까지 — 수익을 좌우하는 건 재료가 아니라 배율과 '검증 성적이 실전에서 얼마나 남느냐'** (내 수익 탭 '목표 계획')
- 지금 기본의 4시간 손익을 한 달 덩어리로 다시 뽑아 2032년 말까지 4,000번 (시작 164 USDT + 매달 70):

| 배율 | 과거 그대로: 중앙 · 50억 확률 | 수익 절반: 중앙 | 우위 없음: 중앙 | 도중 낙폭 (보통 → 절반 세계) |
|---|---|---|---|---|
| ×1.5 | 0.7억 · 0% | 0.19억 | 0.07억 | −20% → −28% |
| ×2 | 1.5억 · 0% | 0.25억 | 0.06억 | −27% → −38% |
| ×3 | 7.7억 · 10% | 0.44억 | 0.05억 | −40% → −55% |
| ×4 | 36억 · 43% | 0.68억 | 0.04억 | −51% → −69% |
| ×6 | 530억 · 80% | 1.15억 | 0.03억 | −69% → −85% |

- 넣는 돈 합계는 약 730만원. 50억은 '검증 성적이 그대로 이어지고 + 배율 4 이상 + −50~−70% 낙폭을 버팀'이 모두 맞아야 나옴. 수익이 절반만 나오면 어떤 배율도 50억은 1~8%.
- 시험한 배율 규칙: 입금 반영 배율(자산이 작을 때만 높게) → 같은 낙폭의 고정 배율과 거의 같음. 성적 확인 뒤 한 칸씩 올리는 사다리 → 좋은 세계에서 ×4의 절반(18억), 나쁜 세계에선 ×2와 비슷하지만 낙폭은 더 큼 → 고정 ×3보다 낫지 않음. 둘 다 채택 안 함.
""")
with tab_info.expander("△ 재분석 — 가격 모양 말고 '돈이 실제로 들어오나' (BTC·ETH, 11-09)", expanded=False):
    st.markdown("""
**왜 다시 봤나**: 지금까지의 분석(추세·호흡·파동·패턴·거시·지배력·우주 리듬·심리)은 모두 **같은 4시간 가격**에서 뽑은 값이었고, 결국 '추세 추종 하나'로 수렴함.
그래서 이번엔 가격이 아닌 재료로 다시 봄 — 바이낸스 체결 기록(시장가 매수·매도), 선물 펀딩비·프리미엄·미결제약정·롱숏 비율(2021-12~), 코인베이스·테더 가격 차이, 옵션 내재 변동성(DVOL), 시각·요일, ETH/BTC 짝.
판정도 바꿈: 규칙 샤프 대신 **정보계수(IC) = 값과 다음 1·3·7일 수익의 순위상관**, 시간만 어긋낸 우주 200번으로 우연 기준, 앞·뒤 절반과 BTC·ETH 모두 같은 부호인지.

**① 무엇이 다음 1주(42봉)를 맞히나** (BTC·ETH 평균 IC, 0.05 넘으면 쓸 만한 수준)

| 재료 | IC (1주) | 우연 p | 앞·뒤·두 코인 일관 | 성격 |
|---|---|---|---|---|
| **테더 프리미엄** (코인베이스 USDT−1) | **+0.139** | 0.00 | ○ | 코인 사려는 달러 수요 |
| **코인베이스 프리미엄** (미국 현물 − 해외) | **+0.104** | 0.00 | ○ | 미국 현물 매수세 |
| 선물/현물 거래량 비 (바이낸스) | +0.082 | 0.01 | ○ | 빚 낸 참여 (비트겟 자료로는 재현 안 됨) |
| 전체 계정 롱숏 비율 (개미 쏠림) | −0.075 | 0.02 | ○ | 개미가 롱에 쏠리면 하락 (실시간 장기 자료 없음) |
| 펀딩비 · 선물 프리미엄 · 미결제약정 · 고래 롱숏 · 옵션 DVOL | −0.05~+0.06 | 0.1~0.6 | 일부 | 약하거나 들쭉날쭉 |
| 시장가 순매수 · 거래량 급증 · 체결 크기 | ±0.02 | | | 거의 없음 |
| 시각·요일 (9년, 1시간 봉) | — | | 62개 중 1개만 | 없음 |
| ETH/BTC 짝 매매 | 샤프 앞 2.09 → 뒤 0.23 | | ✕ | 최근 무너짐 |

- 두 프리미엄은 **과거 수익을 빼도 IC가 그대로**(0.055~0.16) → 지금 앱(추세)과 다른 정보. 앱 수익과의 상관 0.3 수준.
- 연도별로는 2019~2020·2022·2025에 약하거나 반대 → 늘 맞는 건 아님.

**② 앱에 붙이기 — 주머니(따로 더함)가 기울이기(곱함)보다 나음**
- 바이낸스 2022-06~ (흐름 4종): 1.58 → 1.89 · 고르는 절차까지 매 분기 '그때까지 자료'로 다시 한 걷기 검증: 1.58 → 1.63~1.73 (선택 편향을 뺀 현실값).
- **실시간으로 구할 수 있는 재료만**(코인베이스·비트겟, 6시간 봉)으로, 앱이 쓰는 **비트겟 무기한 2022-08~**:

| | 지금 앱 | + 돈의 흐름 주머니 ×0.5 |
|---|---|---|
| 샤프 (앞/뒤 · 3등분) | 1.77 (1.73/1.82 · 1.67/1.91/1.71) | **1.95** (2.02/1.89 · 1.81/2.04/2.01) |
| BTC · ETH | 1.58 · 1.51 | 1.73 · 1.79 |
| 지난 1년 | 1.02 | 1.71 |
| 연 수익 · 최대 낙폭 | +36% · −11% | **+54%** · −16% |
| 시간 어긋낸 우주 300번 | | p = 0.027 |

- 주머니 ×0.25 · ×0.75는 엄격 판정 한 칸씩 미달(1.87 · 1.86) → 가운데 0.5.
- 주머니 단독 연도별 샤프: 2022 −1.5 · 2023 +1.6 · 2024 +2.1 · 2025 −0.4 · 2026 +1.5.

**③ 하루 기하 성장 (비트겟 2022-08~, 봉 안 반대쪽 끝 청산 반영)**

| 배율 | 지금 앱 | + 돈의 흐름 |
|---|---|---|
| 1.5 | 0.12% (낙폭 −17%) | **0.17%** (−23%) |
| 3 | 0.23% (−32%) | **0.31%** (−41%) |
| 4 | 0.29% (−41%) | **0.39%** (−51%) |
| 5 | 0.34% (−49%) | 0.45% (−60%) |
| 8 | 0.43% (−69%) | 파산 |

- 같은 낙폭에서 비교하면 하루 +0.03~0.05%p. 1%에는 여전히 못 미침(현실적 상한 0.3~0.4%).
- 주의: 테더·코인베이스 자료가 2021-10부터라 검증 기간이 4.3년뿐, 그중 좋은 시기 비중이 큼.

**⑤ 신뢰도와 익절·손절 다듬기 (11-10)**
- **신뢰도 모형**: 점수 크기·추세/호흡 성분·주봉·일봉 일치·파동 국면·방향 나이·시작 뒤 유리하게 간 정도·효율비·변동성으로 '지금 방향이 7일 뒤 수익일 확률'을 매년 그 전 해까지로 배워 다음 해를 맞힘.
  바이낸스 2019~ 표본 밖 구별력(AUC) **0.50 = 못 맞힘** · 비트겟 2023~에 돈의 흐름 일치를 더하면 0.56 (2024 0.56 · 2025 0.50 · 2026 0.64). 신뢰도로 크기를 바꾸면 오히려 나빠짐(1.65→1.51 등).
  → 카드의 **신뢰도 = 돈의 흐름이 내 포지션과 같은 쪽인지** (강하게 같은 쪽 높음: 맞음 59%·+2.5% / 보통 45~48% / 강하게 반대 낮음: 37%·−2.0%). 크기에는 안 씀.
- **부분손절 따로 두기**(불리 1·1.5·2·3ATR → ½): 세 데이터 모두 효과 없음 → 손절은 앱의 전환 가격이 맡음.
- **고정 부분익절**(유리 3·5·8ATR → ½): 모두 나빠짐 — 크게 가는 추세를 일찍 줄임.
- **추적 부분익절**(유리 2ATR 이상 간 뒤 최고점에서 2ATR 되돌리면 ½) → **채택 (기본 켬)**:

| | 지금 | + 추적 부분익절 |
|---|---|---|
| 비트겟 2022-08~ | 1.95 · 연 +54% · 낙폭 −16% | **2.15** · +47% · **−12%** (엄격 통과) |
| 바이낸스 2017-08~ | 1.62 · +31% · −15% | 1.72 · +27% · −12% |
| 코인베이스 11년 | 2.04 · +41% · −14% | 2.08 · +33% · −12% |
| 하루 성장 (비트겟, 낙폭 −50% 근처) | 배율 4: 0.39% (−51%) | 배율 5: **0.44%** (−50%) |

  이웃: 1.5·2.5ATR, 남기는 몫 ⅓·⅔ 모두 비슷하게 좋아짐. 같은 배율이면 연 수익은 줄고 낙폭이 더 크게 줄어듦 → **같은 낙폭이면 배율을 한 단계 올릴 수 있음** (1.5 대신 2가 예전 1.5와 같은 낙폭).
- **펀딩비 (11-16 확인)**: 앱은 거의 늘 포지션이 있지만 롱 ~55%·숏 ~45%로 갈리고 평균 크기는 자본의 ⅓ 안팎. 펀딩이 +일 때 롱은 내고 숏은 받아 상쇄됨.
  실제 8시간 펀딩 기록(바이낸스)으로 다시 계산: 비트겟 2022-08~ 배율 1.5에서 연 −1% 안팎 (연 +81% → +79%, 샤프 2.19 → 2.16) · 2020~21 과열장은 연 −4~−6%.
  지금까지 모든 검증은 '롱만 연 10% 냄, 숏은 못 받음'으로 더 불리하게 계산해 왔음 → 실제 펀딩이 가정보다 조금 덜 깎음. 카드에 지금 펀딩과 하루 금액 표시.
- 카드의 말: **진입 · 추가 · 부분익절 · 부분손절 · 익절 · 손절 · 뒤집기 · 유지 · 관망**. 익절/손절은 평단(앱이 그 방향을 시작한 마감가, 또는 설정에서 넣은 내 평단) 대비 그 가격에서 수익이냐 손실이냐로 가름.

**④ 경우의 수 재검토 (2차, 11-09 오후) — 더 나은 길이 있나**

| 시험한 것 | 결과 |
|---|---|
| 김치 프리미엄 (업비트 ÷ 코인베이스×환율, 2017~) | IC +0.02 (우연 p 0.4~0.7) — 정보 없음 |
| 스테이블코인 발행 증가 7·30일 (2017~) | IC +0.04 (p 0.3~0.4) — 약함 |
| 미결제약정 × 가격 4분면 (새 돈 / 청산) | BTC·ETH가 서로 다른 방향 — 기각 |
| 앱이 방향을 바꿀 때 흐름이 같은 쪽이면 믿기 | 전환 뒤 구간 손익 차이 없음 — 기각 |
| 흐름 쓰는 법 10가지 (강한 반대면 앱 0·½, 같은 쪽 1.5배, tanh, 계단, 기울이기, 1일 프리미엄, 7일 변화 섞기) | 바이낸스 2018-06~2022-07에서 고르고 비트겟 2022-08~에서 확인 → **지금 기본(주머니 0.5)을 이긴 것 없음** |
| CME 주말 갭 메우기 | 5일 안에 60~64% 메워지지만 메우는 쪽으로 걸면 손실(BTC 72시간 −0.26%) — 변동성 때문이지 돈이 되는 패턴 아님 |
| 옵션 만기(매달 마지막 금요일) 전후 | 만기 뒤 1주 +1.1~2.0% vs 평소 +0.4~0.6%, 표준오차 범위 안 — 참고만 |
| 변동성 예측으로 크기 (HAR · 느린 σ · DVOL 섞기) | DVOL 반반이 +0.07, 나머지 나빠짐 — 기각 |
| 펀딩비 받기 (현물 롱 + 무기한 숏) | 2020~21 연 17~38% → 2025~26 연 2~5% — 지금은 할 가치 낮음 |

- **조심할 점**: 테더 자료가 없던 2018-06~2022-07에 코인베이스 프리미엄만으로 주머니를 붙이면 앱이 1.73 → 1.62로 오히려 나빠짐. 지금의 효과는 대부분 **테더 프리미엄**과 2022년 이후(현물 ETF·기관 시대)에서 나옴 → 시대가 바뀌면 약해질 수 있음.
- **실전용 경우의 수** (비트겟 2022-08~, 앱 포지션 × 돈의 흐름 → 다음 7일 그 포지션 손익): 강하게 같은 쪽(시간의 18%) **+2.5% · 맞음 59%** / 약하게 같은 쪽 +0.2% · 48% / 약하게 반대 −0.1% · 45% / 강하게 반대(6%) **−2.0% · 37%**. 카드의 '지금 경우' 줄이 이 표.
""")

with tab_info.expander("△ 엘리어트 파동 다시 세기 — 5파·3파 판별법 · ABC·플랫 경우의 수 (BTC·ETH 11년, 11-09)", expanded=False):
    st.markdown("""
**바로잡기**: 지난번 '5파는 약하다와 반대'는 **같은 방향 꼭짓점이 다섯 번 이상 이어진 것**을 센 값이라 엘리어트 5파가 아니었음. 이번엔 규칙(2파는 시작점 안 깸 · 3파가 가장 짧지 않음 · 4파는 1파와 안 겹침)을 통과한 충격파만 5파로 셈.
파동 크기 = 변동성 단위 (4시간 ATR의 2배 소 · 4배 중 · 8배 대).

**① 규칙대로 센 5파 — 정말 약한가** (실제 vs 방향만 무작위로 섞은 거울 우주 60개)

| | BTC | ETH | 무작위 | 판단 |
|---|---|---|---|---|
| 5파가 3파보다 짧음 (중파동) | 73% | 76% | 5파/3파 0.59 (실제 0.65·0.68) | 규칙(3파가 가장 짧으면 안 됨)이 만든 모양 |
| 5파 연장 (다른 파동의 1.618배 이상) | 13% | 22% | 5~6% | **진짜 특징** (무작위 60개 모두보다 높음) |
| 4파 뒤 5파가 3파 끝 못 넘음 (절단) | 45% | 50% | 38% | 약한 특징 |
| 큰 파동 안에 작은 파동 4·8개 (겹침 구조) | 29% | 36% | | 3분의 1만 교과서대로 |

**② 그 '5파'가 사실은 3파였나** (5파 완성 150번, 그 뒤 가격이 어디를 먼저 넘었나)
- 5파 끝을 먼저 넘음(추세 연장) **48%** — 무작위도 48%. 그중 68%는 다시 세면 '0·1·2·3(=옛 5)·4·5' 충격파 → **전체의 약 3분의 1은 사실 3파(또는 연장 5파)였음**.
- 4파 끝을 먼저 깸(교과서대로 조정) 52% → 그 뒤 5파 끝을 다시 넘음 53% / 시작점까지 다 되돌림 45% — 무작위 57% / 41%.

**②-1 5파냐 3파냐 판별법 (11-09, 5개 차수 442번 · 거울 우주 100개)**
- 확인 순간 단서 19개(파동 비율 7 · 시간·속도 · 5파 안 하위 파동 수 · RSI 다이버전스 · 거래량 · 채널 스로오버 · 상위 흐름) + 되돌림이 진행되는 동안 단서 7개(깊이 · 하위 파동 수 · 되돌림이 5파 충격파 모양인지 · 겹침 · 속도) + 공포·탐욕 + 한 단계 큰 차수 국면:
  앞 절반으로 배우고 뒤 절반·다른 코인 맞히기 → **거리 비율 위에 더해 준 정보 0** (최고 +0.26비트/100건, 무작위 우주 상위 5%보다 작음).
- 맞히는 건 **거리 비율** 하나: 지금 가격에서 4파 끝까지 vs 5파 끝까지 → 시험 AUC 0.69. 실제 보정 — 거리 확률 10%→19% · 30%→37% · 50%→51% · 70%→64% · 90%→81% (끝으로 갈수록 덜 극단).
- **판정이 난 뒤는 강하게 갈림** (뒤 7일 = 42봉, 판정 방향으로):

| 판정 | 횟수 | 뒤 7일 평균 / 중앙 | 맞음 | 무작위 우주 |
|---|---|---|---|---|
| 5파 끝 돌파 = **사실은 3파 확정** | 223 | **+5.4% / +1.4%** | 55% | 0 (p=0.00) |
| 4파 끝 이탈 = '5파 끝' 판정 → 꺾인 쪽 따라가기 | 218 | −1.3% / −0.8% | 46% | 손해 |
| 판정 뒤 5파 끝 다시 돌파 = **추세 재개** | 113 | **+7.0% / +6.5%** | 59% | 0 (p=0.00) |
| 판정 뒤 시작점 이탈 = **추세 전환** | 104 | **+6.2% / +3.6%** | 62% | 0 (p=0.00) |

- 상승·하락, BTC·ETH, 앞·뒤 절반, 차수 5개 모두 같은 방향. → **극점(5파 끝·시작점) 돌파만 믿고, 중간(4파 끝) 이탈은 믿지 않기.**
- 앱 크기 규칙(판정 뒤 7일 그 방향 1.5배·반대 ½)으로 넣으면 11년 2.04 → 2.12 (앞 +0.10 · 뒤 +0.05 · BTC +0.07 · ETH +0.08), 비트겟 5년 1.55 → 1.57 — 엄격 기준(+0.10) 미달 → 크기엔 안 쓰고 차트 위 '5파·3파 판별' 줄로만 표시.

**③ 조정 파동 (ABC 지그재그 · 정규/확장/러닝 플랫 · 삼각형)**
- 비율: 러닝 플랫 29~33% · 기타 조정 31~37% · 지그재그 12~19% · 확장 플랫 12~14% · 정규 플랫 6~8% · 삼각형 거의 없음 → **무작위와 거의 같음** (30·31·18·16·4%).
- 조정이 끝난 뒤 추세 재개 비율도 무작위와 비슷 (러닝 플랫 76~100% vs 93% · 지그재그 0~10% vs 15%).
- 패턴별 매매(지그재그·플랫·삼각형 끝 재진입, 교대 원칙 5파, 절단 5파 뒤 역추세): 대부분 샤프 0 근처, 앞·뒤 절반 뒤집힘. 앱에 더하면 12개 모두 기각.
- **거래량까지 본 조정 돌파 (11-10, 바이낸스 현물 2017-08~, 447번)**: 추세 뒤 조정(지그재그·플랫·삼각형)이 끝나고 범위를 처음 벗어나는 봉.
  추세 쪽으로 돌파한 비율 43% — 무작위 44%와 같음 (플랫 61% vs 64%, 지그재그 35% vs 36%) → **모양으로 방향은 못 맞힘**.
  하지만 '조정 동안 거래량이 줄고(뒤 절반/앞 절반 < 0.85) 돌파 봉 거래량이 1.5배 이상'인 117번은 돌파 쪽으로 7일 평균 +0.54σ(+2.3%), 무작위 +0.01 (p=0.00), BTC·ETH·앞뒤 절반 모두 + · 거래량 확인 없는 330번은 +0.13σ(우연 범위).
  맞은 비율은 39~57%라 평균은 가끔 크게 가는 쪽에서 나옴. 앱 크기에 넣으면 바이낸스 1.62→1.69 · 비트겟 1.80→1.77 → 기각 (추세 신호가 이미 그 돌파를 타고 있음).
- 이전 시험: 삼각형·박스·깃발 돌파(4H 5년) 적중 48~52%, 대칭삼각형 아래 돌파는 오히려 반대(35%) · 변동성 수축(볼린저 폭 하위 20%) 뒤 변동성은 1.2배로 커지지만 첫 돌파 방향은 46~48%.

**④ 경우의 수 엔진 — 꼭짓점이 확인된 순간 국면별로 다음 파동이 직전 꼭짓점을 넘을 확률 (중파동)**

| 국면 | 횟수 | 돌파 (앞 절반 / 뒤 절반) |
|---|---|---|
| 2파 완성 (높은 저점 · 하락이면 낮은 고점) | 615 | **61% / 58%** |
| 4파 완성 (0-4 유효) | 106 | **68% / 60%** |
| A·B파 완성 | 148 | 37~48% |
| 3파 완성 · 5파 완성 | 466 | 20~32% |
| 전체 평균 | | 44% / 45% |

- 앞 절반에서 고른 두 국면(2파·4파 완성)을 그대로 크기 조절에 씀: **새 파동과 앱 방향이 같으면 1.5배, 반대면 ½**, 그 꼭짓점을 종가로 되돌아 넘으면 해제.
- 확인: 같은 개수의 꼭짓점을 무작위로 고른 200번 → 개선 평균 +0.02, 상위 10% +0.09 vs 채택 **+0.16** (무작위가 이 이상 0%). BTC p=0.06 · ETH p=0.01.
- 힘의 출처: '2파 완성만' +0.17 ≈ 채택(+0.16), '4파만' −0.02, 모든 꼭짓점 +0.06(낙폭 −19%), 나머지 이름표 −0.13, 5파 −0.01, 3파 −0.09. → **실제로 일하는 건 높은 저점·낮은 고점 확인 뒤 새 파동** (다우 이론 = 엘리어트 2파).
- 돌파 확률 자체는 무작위 우주에서도 비슷하게 높음 (꼭짓점 정의가 만든 부분). 차이는 터진 뒤 길이: 2파·4파 뒤 새 파동 길이 2.3~2.5 vs 3파·5파 뒤 1.8.

**⑤ 채택 → 설정의 '파동 국면 크기' 기본 켬** (엄격 판정 통과)

| | 지금 앱 | + 파동 국면 |
|---|---|---|
| 코인베이스 11년 샤프 | 1.89 | **2.04** (BTC +0.08 p=0.02 · ETH +0.19 p=0.00) |
| 앞 / 뒤 절반 · 3등분 | 2.66 / 1.16 · 2.69 / 1.58 / 1.43 | 2.81 / 1.32 · 2.86 / 1.72 / 1.59 |
| 지난 1년 | 0.19 | 0.70 |
| 비트겟 무기한 5년 (앱이 쓰는 시세) | 1.36 · 연 +20% · 낙폭 −13% | **1.55** · 연 +30% · 낙폭 −12% |
| 이웃 설정 | | 차수 ATR×3~6 모두 ≥ 1.89, 배수 1.25/0.75~2.0/0.5 모두 ≥ 1.93 |

- 지금 중파동 국면은 카드의 '파동' 줄, 5파·3파 판별은 차트 위 글로 표시.

**⑥ 하루 1%까지** (배율 1.5 → 낙폭 −50% 안쪽 최고 배율, 봉 안 청산 반영)
- 2022~2026: 하루 0.10% → **0.21%** (배율 4, 낙폭 −44%) · 베어트랩까지 0.25%.
- 최근 2년: 0.12% → **0.28%** (배율 4, −43%) · 베어트랩까지 0.31%.
- 가장 좋았던 2016~2018: 배율 10에서 1.03% (−69%) — 1%는 그 시대에만 닿음. 지금 시대 샤프 1.4~1.8 < 필요 2.7.
""")

with tab_info.expander("△ 하루 1% 기하 복리 — 배율별 성장 지도 (11-09 · 파동 국면·돈의 흐름 포함)", expanded=False):
    st.markdown("""
**목표**: 수익일·손실일이 섞인 실제 결과를 복리로 쌓아 하루 평균 기하 성장 1% = 1년 37.8배.
**조건**: 로그 성장의 최대치는 샤프²/2 → 연 샤프 **2.70 이상**(켈리 배율, 낙폭 매우 큼) 또는 3.11(켈리 ½)이 몇 년간 유지돼야 함.

| 배율 | 2022~2026 하루 / 1년 | 최근 2년 하루 / 1년 | 최대 낙폭 |
|---|---|---|---|
| 1 | 0.07% / 1.27배 | 0.08% / 1.34배 | −13% |
| 1.5 (기본) | 0.10% / 1.42배 | 0.12% / 1.54배 | −19% |
| 2 | 0.12% / 1.57배 | 0.15% / 1.75배 | −24% |
| 3 | 0.17% / 1.87배 | 0.22% / 2.22배 | −35% |
| 4 | 0.21% / 2.17배 | 0.28% / 2.73배 | −44% |
| 5 | 0.24% / 2.43배 | 0.33% / 3.28배 | −52% |
| 8~12 (한계) | 0.29% | 0.47% | −72~−88% |
| 12 이상 | 2022~ 파산 (FTX 같은 4시간 급락 한 번) | | |

- 지금 앱 기본(흐름 정렬 역가중 + 파동 국면 크기) 그대로, 비용·롱 펀딩·봉 안 최저가 청산까지 반영한 실제 복리 (BTC·ETH 코인베이스 4H).
- 돈의 흐름 주머니를 더하면(비트겟 2022-08~): 배율 1.5 하루 0.12% → 0.17%, 배율 4 0.29% → 0.39% (낙폭 −41% → −51%) — '재분석' 펼침 참고.
- 가장 좋았던 2016~2018년(샤프 3.14)은 배율 10에서 하루 1.03% (낙폭 −69%) — 1%에 닿은 유일한 시대.
- 지금 시대 샤프 1.4~1.8 → 현실적 최대 하루 0.2~0.3% (1년 2~3배). 베어트랩을 더하면 +0.03~0.04%p.
- 지난 1년 성적으로 배율을 정하는 방식(켈리 ½, 최대 10)도 2022~ 하루 0.12%, 최근 2년 0.21% (낙폭 −54%).
- 1%로 가려면 서로 독립적인 수익원(샤프 1짜리 7~8개)이 필요한데, BTC·ETH(상관 0.9)의 추세 추종은 사실상 하나.
- 표의 배율은 지나고 나서 본 값 — 앞으로도 같다는 보장 없음. 4~5는 자본이 절반까지 줄어드는 구간을 견딜 수 있을 때만.
""")

with tab_info.expander("△ 4원칙 — 이 매매법이 따르는 원리 (5년 검증)", expanded=False):
    st.markdown("""
**① 손실은 짧게, 이익은 길게 — 참. 이 방식의 본질**
- 거래 900개 중 이긴 건 26%뿐. 하지만 평균 이익 +2.5%(8.6일 보유) vs 평균 손실 −0.6%(2.4일) → 손익비 4.4.
- 상위 10% 거래가 순이익 전체(254%)를 만듦. → **작은 손실이 연달아 나도 신호를 건너뛰지 말 것.** 큰 추세 하나를 놓치면 1년 수익이 사라짐.
- 출구를 손봐도(2ATR 조기 손절, 3ATR 추적 연장) 더 좋아지지 않음 — 이미 '틀리면 금방 방향이 바뀌어 작게, 맞으면 길게' 구조.

**② 힘이 다하면 꺾인다 — 위쪽은 거짓, 아래쪽은 반쪽**
- 위로 과열(EMA20보다 3ATR 위)되면 24시간 뒤 더 오른 비율 54~58%. **과열이라고 숏 치면 손해** (과열 역매매 샤프 −1.12).
- 아래로 과열되면 58~59%가 반등하지만, 틀릴 때 훨씬 크게 빠져 9가지 변형 모두 손해. **떨어지는 칼은 잡지 않음.**
- 포스터의 반전 패턴(쌍봉·H&S·쐐기·원형·하모닉) 전체도 손해(샤프 −0.69). 추세는 생각보다 쉽게 안 꺾임.

**③ 모였다가 터진다 — 변동성은 참, 방향은 거짓**
- 변동성이 눌린 뒤(볼린저 폭 하위 20%) 다음 4일 변동성이 1.2배로 커짐.
- 하지만 처음 터진 방향은 46~48%로 오히려 반대가 많음. **첫 돌파를 쫓지 말고, 추세 신호가 확인된 뒤 따라감** (이 방식이 자동으로 그렇게 함).

**⑨ 위에서 아래로 — 큰 흐름의 '전환'은 일봉·4시간으로 미리 보이지만, 다음 주 '방향'은 안 보인다**
- 주봉 방향 × 일봉·4시간이 따르는지 거스르는지 = 8가지 상태. 일봉·4시간이 모두 주봉을 거스르는 '깊은 조정'이면 4주 안에 주봉 흐름이 뒤집힌 비율 70~80%, 세 시간대가 같은 '순풍'이면 30%대 (앞뒤 일관, 구별력 0.65~0.67).
- 하지만 다음 1주 가격 방향은 8개 상태 모두 통계적으로 의미 없음 → 상태만 보고 롱·숏을 정하면 안 됨. 배지의 '상태'는 큰 흐름이 바뀔 가능성을 가늠하는 참고 정보.
- 약한 경향(두 기간 같은 방향, 확정은 아님): 상승 주봉 속 깊은 조정은 다시 오르는 편(+0.9%/주), 하락 주봉 속 짧은 반등 뒤엔 다시 내리는 편.

**⑧ 흐름 정렬의 역설 — 주봉·일봉·4시간이 다 맞을 때가 가장 나쁘고, 4시간이 큰 흐름을 거스를 때가 가장 좋다** (기본 적용)
- 4시간 포지션 성과(5년): 세 시간대 모두 같은 방향(보유 54%) 샤프 0.33|0.60 · 하나만 같음 1.47|1.80 · 일봉·주봉 모두 반대(11%) 1.85|2.66 (앞|뒤).
- 해석: 모두 같은 방향 = 모두가 아는 늙은 추세. 새 흐름은 언제나 낡은 큰 흐름을 거슬러 시작함. 주봉을 따르는 엘더식 필터는 오히려 손해(0.50).
- 그래서 크기를 거꾸로: 모두 같음 ½ · 하나만 같음 1 · 모두 반대 1.5배 → 샤프 1.04 → 1.36, 낙폭 −19% → −13%. 일봉·주봉 정의 5가지로 바꿔도 모두 개선(1.16~1.36).
- 평균 노출이 0.72배로 줄어서 같은 수익을 원하면 배율을 올림: 지정가 기준 배율 1.5 → 연 +31%·낙폭 −17%, 2.0 → 연 +42%·낙폭 −22%.

**⑦ 차트 속 사람 심리 — 남는 건 '쏠림'과 '손절', 반대매매 심리는 안 통함** (일봉 2018~2026 + 공포·탐욕 지수)
- 쏠림·과소반응은 참: 하루 +2σ 급등 뒤 5일 +2.4%(60% 상승), −2σ 급락 뒤 −1.2% — 두 기간 모두, 추세 통제 후 t=3.8. 이 방식의 추세·호흡이 바로 이 심리를 따라감 (더해도 개선 없음, 상관 0.38).
- '공포에 사고 탐욕에 팔기'는 거짓: 극단 공포 뒤 20일 수익은 평소와 같음(+1.7%), 탐욕일수록 오히려 더 오름(최근 4년은 효과 사라짐).
- 처분 효과(평균단가 위·아래), 52주 고점 기준점: 추세를 빼면 효과 없음.
- 손절 사냥(베어트랩)만 역행이 통함 — 모두의 손절이 몰린 직전 저점.

**⑥ 가짜 전환에 속지 않는다 — 들어가기 전엔 못 가린다. 대신 '입장료'로 작게 낸다**
- 5년 전환 860번 중 74%가 가짜(평균 −0.6%, 2.4일 만에 끝), 26%가 진짜(평균 +2.6%, 8.4일). 가짜 손실 합계 −361% vs 진짜 이익 합계 +577%.
- 전환 순간 단서 12가지(기간 일치도·연속봉·거래량·변동 확대·고점/저점 돌파·일봉 흐름·BTC/ETH 동반·종가 위치·신호 세기·혼조·변동성·경로 곧음)를 다 합친 모델도 뒤 기간 구별력 AUC 0.54 (동전 수준).
- '확인되면 진입' 필터 5종(0.60~1.05), 분할 진입 6종(0.89~1.04) 모두 기준(1.04)보다 낫지 않음 — 가짜를 거르면 진짜의 앞부분도 같이 놓침.
- 그래서 속지 않는 법 = **가짜가 연달아 와도 계속 따라가기.** 연속 가짜 평균 4번, 10번 중 1번은 7~9번, 5년 최장 BTC 18번(그 코인 몫 −12.6%)·ETH 15번. 배지에 지금 연속 횟수를 표시.

**⑤ 상황마다 방법을 바꾼다 — 거짓. 바꾸는 건 '방법'이 아니라 '크기와 반응 속도'**
- 기존 7종(이평선·돈치안·MACD·RSI·볼린저·일목·슈퍼트렌드) + 자체 + 새 관점 4종, 15가지 대회에서 이 방식이 1등 (샤프 1.04, 2등 호흡 0.90, 기존 최고 MACD 0.66, RSI·볼린저 역추세 −1.2).
- 상승·하락·횡보 × 변동성 6가지 상황별 1등이 다음 기간엔 이어지지 않음 (앞·뒤 상관 −0.03). 매달 상황별 최고 방법으로 갈아타면 샤프 0.04~0.56 (그대로 유지 1.37).
- 이 방식은 이미 상황에 맞춰 연속적으로 변함: 변동성 커지면 크기 자동 축소, 호흡(4H 연속봉)이 새 흐름에 빨리 반응, 추세는 7~90일을 섞어 빠름·느림을 같이 봄.

**④ 모두가 아는 자리는 잘 안 온다 — 둥근 숫자는 거짓, '손절 자리'는 참**
- BTC 80,000·ETH 2,500 같은 둥근 숫자도 다른 가격대와 똑같이 93~96% 도달함.
- 진짜는 **모두가 손절을 걸어둔 직전 저점 아래를 쓸고 3봉 안에 되돌아오는 베어트랩** → 롱 (적중 58%, 앱과 합산 샤프 1.04 → 1.26). 반대(불트랩)는 안 통함. 설정에서 '베어트랩 추가'로 켬 (설정값을 바꾸면 약해지는 점은 주의).
""")

with tab_info.expander("△ 코인별 독립 분석 — BTC × 테더 도미넌스 · ETH × TOTAL2·3 · 강도별 흐름 (11-05)", expanded=False):
    if DOM:
        import dominance_live as DL
        st.caption(f"지금 ({DOM['date']}) · USDT.D {DOM['usdt_d']:.2f}% {DL.arrow(DOM['usdt_d_tr'])} · USDT 발행량 30일 {DOM['usdt_30']:+.1%} · BTC.D {DOM['btc_d']:.1f}% {DL.arrow(DOM['btc_d_tr'])} · ETH.D {DOM['eth_d']:.1f}%"
                   f" · TOTAL3(스테이블 제외) {DL.arrow(DOM['t3_tr'])} 30일 {DOM['t3_30']:+.0%} · 알트 비중 {DL.arrow(DOM['alt_tr'])} · ETH/알트 {DL.arrow(DOM['eva_tr'])} (화살표 = 20·50·100일 방향)")
    st.markdown("""
**같은 규칙, 각자의 데이터** — BTC와 ETH는 각자 가격으로 신호·크기·전환 가격·흐름 정렬을 따로 계산합니다.
- ETH 변동이 BTC의 1.33배 → 같은 금액이면 ETH 크기가 자동으로 작아짐. 일별 상관 0.71 (2016년 0.26 → 2026년 0.91로 점점 같이 움직임).
- 같은 시각 앱 방향이 같았던 시간 74%, 반대 25%.
- 코인마다 설정을 따로 골라 보니(18가지 중 앞 절반 1등) 뒤 절반에서 BTC 18위(0.62 vs 지금 0.95), ETH 13위(0.95 vs 1.11) → 따로 맞추면 과거에만 맞음. **규칙은 공유, 데이터는 독립**이 가장 좋았음.

**BTC × 테더 도미넌스(USDT.D, 2020~)**
- USDT.D 방향은 BTC 자기 추세와 79% 같음(상관 0.77) — BTC 가격이 분모에 들어 있어 거의 같은 정보.
- 'USDT.D가 30일 오르면(공포) 다음 30일 BTC가 더 오름' 경향(t=3.0)은 앞 절반에만 있고 뒤 절반엔 없음. USDT 발행 속도는 관계없음.
- 규칙 4가지(USDT.D 반대로 매매, 발행 속도로 매매, 어긋나면 ½ 두 가지) 모두 기각.

**ETH × 알트 시장(TOTAL2·TOTAL3)**
- TOTAL3 방향도 ETH 자기 추세와 79% 같음, 앞서거나 뒤서지 않음(같은 날 함께 움직임).
- 규칙 7가지(TOTAL3·TOTAL2 추세대로, 어긋나면 ½, 알트 비중, ETH/알트 강약, 스테이블 제외) 모두 기각 — 가장 나은 'TOTAL2와 어긋나면 ½'도 +0.08로 기준 미달.
- ETH는 최근 1년 앱 성적이 마이너스(샤프 −0.43, 코인베이스 기준)였고 알트 지표로 걸러도 나아지지 않음.

**강도별 흐름 (7일 변화 ÷ 지난 1년 보통 크기 = σ, 5단계)**

| 테더 도미 7일 | 시간 | 그 주 BTC / ETH | 다음 30일 BTC / ETH (오른 비율) | 다음 주 흔들림 |
|---|---|---|---|---|
| 강한 하락 (≤ −1.5σ) | 5% | +17% / +20% | +7.5% (68%) / +11.3% (63%) | 보통 |
| 하락 | 22% | +8% / +11% | +4.0% / +6.0% | 보통 |
| 보합 | 47% | +1% / +1% | +4.4% / +5.8% | 보통 |
| 상승 | 20% | −6% / −8% | +4.5% / +3.3% | 보통 |
| 강한 상승 (≥ +1.5σ) | 6% | −14% / −17% | −0.4% (47%) / −2.1% (39%) | **1.4배** |

| TOTAL3 7일 (2021-05~) | 그 주 ETH / BTC | 다음 30일 ETH / BTC (오른 비율) |
|---|---|---|
| 강한 하락 | −18% / −13% | −3.7% (38%) / −0.3% (52%) |
| 보합 | 0% / 0% | +2.5% / +2.2% |
| 강한 상승 | +16% / +12% | +2.0% / +2.2% |

- **같은 주**: 테더 도미와 코인은 거울처럼 반대, TOTAL3와 코인은 같은 방향. ETH가 BTC보다 1.15~1.4배 크게 움직임 (TOTAL3 1% → ETH 0.93% · BTC 0.67%).
- **다음 주**: 강도 5단계 모두 평소와 차이 없음 → 예측 안 됨.
- **다음 30일**: 센 주간의 방향이 조금 더 이어짐 (테더 도미 강한 하락 → 더 오름, 강한 상승 → 덜 오름, 알트 강한 하락 → 약함, 앞·뒤 절반 모두). 단 이것을 규칙(센 주간 뒤 30일 반대 포지션 ½)으로 만들면 BTC +0.01·ETH +0.05로 개선 없음 — 앱의 추세가 이미 같은 성질을 잡고 있음.
- **관찰 중**: 'TOTAL3 강한 주간 뒤 30일 반대 ½'이 BTC에서 +0.25 (우연 범위 밖)였지만 최근 1년은 나빠졌고 원래 대상인 ETH에선 −0.07 → 채택 안 함.
- **둘이 따로 놀 때**: 테더 도미만 오르고 TOTAL3는 버티는 주 → 다음 주 BTC −1.5%·ETH −2.2% (앞·뒤 절반 같은 방향), 단 겹치지 않는 표본 13주라 참고만.

→ 두 지표는 **카드 아래 참고 줄**로만 표시 (매매·크기엔 안 씀).
""")

with tab_info.expander("△ 거시 → 세부 → 경우의 수 (BTC 11년 · ETH 10년, 11-05)", expanded=False):
    for _n, _m in MAC.items():
        if _m:
            arw = lambda x: "▲" if x > 0 else ("▼" if x < 0 else "―")
            st.caption(f"{_n} 거시 {_m['label']} (합의 {_m['mval']:+.2f}) · " + " · ".join(f"{k} {arw(v)}" for k, v in _m["parts"].items())
                       + f" · 200일선 {_m['s200']:,.0f} (지금 {_m['dist200']:+.0%}) · 1년 고점 대비 {_m['dd1y']:+.0%} · 기준 {_m['date']} 마감")
    st.markdown("""
**1. 거시 방향은 판단할 수 있지만 늘 늦다 (불확정성)**
- 6가지 거시 정의를 BTC 11년에 시험: 거시 상승장일 때 90일 뒤 평균 +14~21%, 하락장일 때 −2~+5%. 차이는 있지만 **방향 적중 57~63%로 '그냥 늘 롱'(59%)과 비슷**하고, 뒤 절반에선 차이가 줄어듦.
- 사이클 고점(2017-12·2021-11·2025-10)·바닥(2018-12·2022-11)을 알아채는 데: 가장 빠른 200일선 위/아래도 고점 뒤 평균 32일(이미 −15~−65%), 대신 1년에 6.8번 뒤집힘. 헛신호를 줄인 200일선 기울기는 연 1.6번이지만 평균 112일 늦음. **빨리 알면 헛신호, 확실하면 늦음** — 둘 다 가질 수 없음.

**2. 세부(4시간)는 거시와 상관없이 양쪽에서 번다**
- 지금 앱을 거시 상태 × 롱·숏 6칸으로 쪼개면 BTC·ETH 모두 6칸 전부 플러스 (상승장 속 숏 샤프 +0.8~1.1, 하락장 속 롱 +0.7~2.1).
- 그래서 '거시와 같은 방향만', '반대는 절반', '같은 쪽 1.5배', '중립이면 절반' × 거시 정의 3가지 = 12가지 모두 기각 (가장 좋은 것도 +0.07, 우연 범위).
- 4시간 추세는 이미 거시를 품고 있음: 큰 흐름이 이어지면 같은 방향으로 오래 머물고, 큰 흐름 안의 반대 파도(상승장 속 급락, 하락장 속 반등)도 따로 먹음.

**3. 경우의 수 — 다음 1주는 5가지 중 하나 (변동성 기준, 11년 빈도)**

| 다음 1주 | 빈도 | 그때 지금 앱의 1주 손익 (배율 1) |
|---|---|---|
| 곧장 상승 | 약 22% | **+2.5%** |
| 눌림 후 상승 | 약 2% | +1.3% |
| 횡보 | **약 58%** | −0.4% |
| 반등 후 하락 | 약 2% | +0.6% |
| 곧장 하락 | 약 16% | +0.8% |

- 가장 흔한 횡보에서 조금 잃고, 드문 직진에서 크게 범 → 1주 기대값 약 +0.5%. **예측이 아니라 경우의 수에 대한 대응**으로 돈이 남.
- 거시·주봉·일봉·4시간·직진도·변동성·200일선 거리 7가지로 '어느 경우가 올지' 학습 → 뒤 5.5년에서 평소 비율보다 0.2~0.4% 나아짐(사실상 없음). 오를지 내릴지 구별 0.51~0.52, 횡보일지 구별 0.59.

**4. 예측이 되는 것과 안 되는 것**
- 안 됨: 다음 4시간·1주 방향 (0.49~0.52), 가짜 전환 구별 (0.54).
- 됨: 다음 하루 움직임 크기 (0.75 — 그래서 앱이 변동성에 맞춰 크기를 줄이고 늘림), 4주 안 주봉 흐름 전환 가능성 (0.66 — 카드의 '상태' 줄).

**5. 11년으로 다시 보니**
- 추세 추종의 힘이 시대가 갈수록 약해짐: BTC 4시간 기본 샤프 2015~2021년 2.31 → 2021~2026년 0.63. 시장이 성숙하면서 쉬운 추세가 줄어듦.
- 지난번 채택한 흐름 정렬 역가중은 처음 보는 2015~2021년에서 2.31 → 2.02로 오히려 나빴고, 최근 5년에서만 0.63 → 0.96으로 좋았음. 11년 전체 샤프는 비슷(1.95 vs 1.89), 최대 낙폭은 −21% → −14%로 줄어듦 → 기본으로 두되 '수익 개선'은 확실하지 않음.
""")

with tab_info.expander("△ 우주·물리 렌즈 17개 — 평행우주 검증 (11-05)", expanded=False):
    try:
        import cosmos_live as CL
        _ph, _pn = CL.moon(); st.caption(f"오늘 · 달 {_pn} (위상 {_ph:.2f}) · 2024-04-20 반감기 뒤 {CL.halving_days()}일 — 둘 다 판단에 쓰지 않음 (아래 ⑤·⑦)")
    except Exception:
        pass
    st.markdown("""
**새 검증법 '평행우주'** — 실제 BTC 성적을 수천 개의 가짜 우주와 비교: ① 포지션을 시간축에서 통째로 민 우주 1000개(롱 편향·매매 횟수는 그대로, 타이밍만 무작위) ② 매 봉 움직임 크기는 그대로 두고 방향만 무작위로 뒤집은 '거울 우주' 200개 ③ 시간을 거꾸로 돌린 우주 ④ ETH 쌍둥이 ⑤ 마지막 1년 ⑥ 설정값을 바꾼 이웃 우주. 실제가 가짜들의 상위 5% 밖이면 '우연'.

| 렌즈 | 관점 | BTC 5년 | 결과 |
|---|---|---|---|
| 현재 앱 (추세+호흡) | 기준 | 샤프 +0.87 · 가짜 우주 상위 1% | **통과** — 새 방식으로도 우연 아님 확인 |
| 오를 때 롱·내릴 때 숏 (5일 효율비 ≥ 0.3일 때) | 양자: 입자 | +1.15 · ETH +1.02 | **통과** — 단 뒤 절반 +0.29로 약해짐, 앱에 더하면 개선 +0.09로 기준(+0.10) 미달 |
| 횡보 때 박스 하단 롱·상단 숏 | 양자: 파동 | −0.21 · ETH −0.41 | 탈락 — 이걸 붙이면 위 규칙이 1.15 → 0.58로 나빠짐 |
| 수축 뒤 첫 이탈 따라가기 | 양자: 측정 붕괴 | −0.15 | 탈락 |
| 변동의 양으로 잰 시간(고유시간) 추세 | 상대성 | +0.31 | 탈락 (우연 범위) |
| 광속 근접 시 크기 축소 | 상대성: 로렌츠 인자 | −0.13 | 탈락 |
| 닮은 과거 궤적 다수결 | 혼돈: 로렌츠 아날로그 | −0.96 | 탈락 |
| 칼만 입자 추적 | 물리: 상태추정 | +0.16 | 탈락 |
| 티핑 포인트 조기경보(자기상관·분산 상승) | 열역학·생태학 | −0.03 · 전환 구별력 0.45~0.54 | 탈락 |
| BTC·ETH 상관 붕괴 시 ½ | 양자: 얽힘 | +0.13 | 탈락 (보정 후 우연 범위, 최근 1년 나빠짐) |
| 렌즈 합의만큼 키움 | 노자: 무위 | −0.02 | 탈락 |
| 최근 1년 잘 맞은 쪽에 가중 | 헤라클레이토스 | +0.51 | 탈락 (앱 단독보다 못함) |
| 횡보면 ½ | 상황 적용 | 5년 개선 +0.22, 앱 1.36→1.53 | 거울 우주(p=0.11)·이웃 설정(67%) 미통과, 11년 확인 때 처음 보는 6년(2015~2021)에서 나빠짐 → 탈락·옵션 삭제 |
| 달: 신월 롱·보름 숏 (11년 일봉) | 우주 | +0.18 (뒤 절반 −0.16) | 탈락 |
| 지구 자기장 폭풍 뒤 6일 | 우주 | 오히려 폭풍 뒤가 더 오름 | 탈락 |
| 태양 활동 / 계절(월) | 우주 | +0.36 / +0.19 (뒤 절반 무너짐) | 탈락 |
| 황금비 0.382·0.5·0.618·0.786 | 자연의 수 | 0.45·0.55·0.70과 똑같음 | 특별하지 않음 |

- **시간의 화살**: 가격을 거꾸로 돌리면 같은 앱 방법이 손실(BTC +1.13 → −0.59). 그런데 '선형 기억'(자기상관)은 0.006으로 정·역방향이 같음 → 앱이 먹는 것은 반복 패턴이 아니라 **되돌릴 수 없는 흐름**(사람들이 늦게 알아채고 따라붙는 과정).
- **되돌림 진입**: 어느 깊이(0.3~0.8)에서 추세 방향으로 들어가도 무작위보다 스윙 크기의 +5% 안팎 유리 — 추세가 이어지는 성질 때문이고 황금비 자리만의 힘은 없음.
- **반감기**: 세 번 모두 반감기 뒤 525~546일에 고점, 이후 약 1년 −53~−84%. 지금 929일째. 표본 3개 → 통계 검증 불가, 참고만.
""")

with tab_info.expander("△ 규칙 (한 번 읽어두기)", expanded=False):
    st.markdown("""
- **방향**: 추세(7·14·30·60·90일 수익 방향 평균) 50% + 호흡(같은 방향 4H 봉이 3~6개 이어지면 따라가기) 50%. 점수 ±0.15 안이면 관망.
- **크기**: 변동성이 크면 작게, 작으면 크게(코인 연변동성 40% 기준) × 배율. 표시된 금액이 '지금 들고 있어야 할 포지션'.
- **매매 순서**: ① 4H 마감 직후 앱 확인 → ② '지금 들고 있을 포지션'과 내 포지션이 다르면 맞춤 (진입 가격 = 그때 현재가, 지정가) → ③ 다음 마감까지는 '전환 가격'표만 보고 대기.
- **'판정' 글자 말고 '포지션 금액'을 따를 것**: 롱/숏 글자만 보고 크로스로 뒤집으면 5년 샤프 1.08 → 0.47~0.64, 방향 전환 연 92회 → 222회로 크게 나빠짐. 앱 금액은 작은 흔들림엔 그대로 있도록 계산돼 있음.
- **바꾸는 때**: 4H 봉 마감 때만. 앱 금액이 바뀌지 않았으면 그대로 둠 — 잦은 매매는 수수료로 수익을 깎음.
- **지지·저항**: 차트 참고용. "벽 앞에서 기다리기"는 5년 검증에서 수익을 거의 0으로 만들어서 판단에 쓰지 않음.
- **손절 = 전환 가격**: 따로 두지 않음 — 다음 마감가가 전환 가격을 넘으면 그때 정리. 2ATR·지지저항 손절은 검증에서 오히려 나빴음. 대신 크기 자체가 작게 잡혀 있음. 전환 가격은 마감 때마다 새로 계산됨.
- **레버리지**: 거래소 설정은 3배 격리면 충분. 실제 위험은 레버리지가 아니라 '명목 금액'이 정함.
- **롱·숏 모두**: 5년 검증에서 롱 부분 연 +11%, 숏 부분 연 +4%. 숏은 하락장에서 계좌를 지켜줌 (2022년 롱 −17% / 숏 +7% → 합계 −11%).
- **적중률의 진실**: 강한 신호도 다음 4시간~1주 방향 적중률은 48~54% (동전 수준). 돈이 되는 이유는 '맞을 때 크게, 틀릴 때 작게' — 추세가 길게 이어질 때 크게 먹고, 아니면 금방 방향이 바뀌어 작게 잃음.
- **주문은 지정가(Post-only)**: 수수료가 편도 0.08% → 약 0.03%로 줄어 5년 샤프 0.79 → 1.04. 검증한 개선안 중 BTC·ETH와 알트 22종 모두에서 확실히 좋아진 건 이것 하나.
- **바꾸지 않은 것**: 추세·호흡 일치 필터, 완충 확대, 숏 절반, 확신 가중 — BTC·ETH에선 좋아 보였지만 알트 22종에선 나빠져서(과최적 신호) 기각.
- **추가로 기각한 정보** (11-03): 시장 폭(다른 23개 코인 방향), 일봉 스윙 투표, BTC 선행, 머신러닝 메타 라벨(BTC·ETH 1.04→1.16이지만 알트 1.03→0.98), 코인베이스 프리미엄 — 크기 가중 적중률은 모두 50% 근처 그대로.
""")

with tab_info.expander("△ 검증 성적 (5년, 과장 없이)", expanded=False):
    st.markdown("""
| 방식 (BTC·ETH) | 연 수익 | 최대 낙폭 | 비고 |
|---|---|---|---|
| **이 앱 방식 (배율 1)** | **+18%** | **−19%** | 수수료 0.15%면 +11% / −26%, 펀딩비 연 10% 반영 시 +15% / −20% |
| **이 앱 방식 + 지정가 주문 (배율 1, 펀딩 반영)** | **+21%** | **−19%** | 샤프 1.04, 1년 수익 확률 82% · 배율 1.5면 +31% / −27% |
| **+ 흐름 정렬 역가중 (기본, 지정가·펀딩, 배율 1)** | **+20%** | **−11%** | 샤프 1.36, 1년 수익 확률 89% · 배율 1.5면 +31%/−17%, 2.0이면 +42%/−22% |
| 11년 재확인 (코인베이스 2015-07~, 지정가·펀딩) | — | — | 4H 기본 BTC 샤프 2.31 (2015~2021) → 0.63 (2021~) · 흐름 정렬 역가중 2.02 → 0.96 · 추세의 힘이 시대가 갈수록 약해짐, 역가중은 최근 시대에서만 보탬 |
| 스윙 롱만 (같은 5년) | −3% | −58% | 최근 5년 약함 |
| 앱 + 베어트랩 (5년, 같은 크기 합산) | +27% | −19% | 샤프 1.04 → 1.26, 그러나 뒤 절반은 1.25 → 1.25 (최근엔 보탬 없음), 설정을 바꾸면 약해짐 (실험 옵션) |
| 포스터 반전 패턴 전체 (쌍봉·H&S·쐐기·원형·하모닉) | −12% | −55% | 샤프 −0.69 — 4H에서 포스터 방향대로 매매하면 손실 |
| 캔들 패턴 전체 (약 30종) | −42% | −96% | 샤프 −1.46 |
| 두 달 모양 반반 (2023-07~ 확인 구간) | +40% | −19% | 샤프 1.47 → 1.72, BTC 1.15 → 1.35 / ETH 1.45 → 1.42 · 2025년 약함 (실험 옵션) |
| 이 앱 방식 (배율 1.5) | 약 +27% | 약 −28% | 권장 |
| 지지저항 벽 앞 대기 추가 | +1% | −19% | 버림 |
| 1회 2% 손실 한도 + 2ATR 손절 | +40% | −61% | 낙폭 너무 큼, 최장 17연패 |
| 1회 2% + 지지저항 뒤 손절 | +59% | −70% | 수수료 0.15%면 BTC 연 +3%, 낙폭 −75% |
| 150배 · 증거금 2% | 파산 | — | 0.21% 역행에 청산, 청산율 77~94% |
""")
    st.caption("한계: 펀딩비는 맨 위 줄 비고만 반영, 미래에도 같다는 보장 없음, 하루 1% 복리 수준과는 거리가 멂. 소액일 때는 실제 주문 전 일지에 모의로 먼저 기록 권장.")

# 매매 탭 맨 위: 오늘 한 줄
if SUMM:
    _todo = [x for x in SUMM if x["kind"] not in ("유지", "관망")]
    _bits = []
    for x in SUMM:
        if x["has"] and x["pnl"] is not None:
            _bits.append(f"{x['nm']} {x['side']} <b style='color:{'#15803d' if x['pnl'] >= 0 else '#b91c1c'}'>{x['pnl']:+.1%}</b>")
        elif x["has"]: _bits.append(f"{x['nm']} {x['side']} 새로 시작")
        else: _bits.append(f"{x['nm']} 관망")
    _head = ("<b>할 일 있음</b> · " + " · ".join(f"{x['nm']} {x['kind']}" for x in _todo)) if _todo else "<b>지금 할 일 없음</b> · 들고 기다리면 됩니다"
    top_box.markdown(f'<div style="padding:9px 12px;border-radius:10px;background:{"#fff7e6" if _todo else "#f1f7f3"};border:1px solid {"#f3d29b" if _todo else "#cfe5d7"};'
                     f'font-size:0.92rem;color:#1f2937;margin-bottom:10px">{_head} │ {" · ".join(_bits)} │ 다음 확인 {SUMM[0]["next"]:%H:%M}</div>', unsafe_allow_html=True)


@st.cache_data(ttl=600, show_spinner="수익판 계산 중…")
def load_pos(sym, last_time, flags):
    import score_live as SC
    df = data.drop_unclosed(data.fetch_candles(sym, "4H", 1600), "4H")
    try:
        fl = load_flow()
        ser = fl[COINS[sym]]["series"] if (fl and flags[2]) else None
    except Exception:
        ser = None
    try:
        exz = load_onchain()[COINS[sym]]["series"] if flags[4] else None
    except Exception:
        exz = None
    return SC.positions(df, ser, *flags[:4], ex=exz)


@st.cache_data(ttl=3600, show_spinner=False)
def krw_rate():
    import requests
    try:
        return float(requests.get("https://api.upbit.com/v1/ticker", params={"markets": "KRW-USDT"}, timeout=10).json()[0]["trade_price"])
    except Exception:
        return 1400.0


with tab_gain:
    try:
        import score_live as SC
        _flags = (MTF, WAVE, FLOW, TRAIL, EXCH)
        P_ = {COINS[s_]: load_pos(s_, str(rows[s_][0].index[-1]), _flags) for s_ in COINS if s_ in rows}
        curve, TR = SC.simulate(P_, pd.Timestamp(s_date).tz_localize("Asia/Seoul").tz_convert("UTC"), float(s_cap), float(s_mon), float(mult))
        if curve.empty:
            st.info("시작일이 너무 최근입니다 — 하루 이상 지난 날을 고르세요.")
        else:
            krw = krw_rate()
            dep = float(s_cap) + float(s_mon) * (len(set(curve.index.tz_convert("Asia/Seoul").to_period("M"))) - 1)
            now_e = float(curve.iloc[-1]); prof = now_e - dep
            wk0 = pd.Timestamp.now(tz="Asia/Seoul").normalize() - pd.Timedelta(days=pd.Timestamp.now(tz="Asia/Seoul").dayofweek)
            _cw = curve[curve.index >= wk0.tz_convert("UTC")]
            wk = float(curve.iloc[-1] - (_cw.iloc[0] if len(_cw) else curve.iloc[-1]))
            st.caption(f"{s_date} 부터 앱 카드대로 (배율 {mult}, 지정가, 롱 펀딩 연 10% 가정) 했다면 · 소수점 수량으로 계산 — 실제 최소 단위 때문에 조금 다를 수 있음")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("지금 자산", f"{now_e:,.1f} USDT", f"약 {now_e * krw / 1e4:,.0f}만원", delta_color="off")
            m2.metric("넣은 돈", f"{dep:,.0f} USDT", f"시작 {s_cap:,.0f} + 매달 {s_mon:,.0f}", delta_color="off")
            m3.metric("순이익", f"{prof:+,.1f} USDT", f"{prof / dep:+.1%}")
            m4.metric("이번 주", f"{wk:+,.1f} USDT", f"{wk / max(now_e - wk, 1e-9):+.1%}")
            depl = pd.Series([float(s_cap) + float(s_mon) * (len(set(curve.index[:i + 1].tz_convert("Asia/Seoul").to_period("M"))) - 1) for i in range(len(curve))], index=curve.index)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=curve.index.tz_convert("Asia/Seoul"), y=depl, name="넣은 돈", line=dict(color="#9ca3af", width=1.5, dash="dot")))
            fig.add_trace(go.Scatter(x=curve.index.tz_convert("Asia/Seoul"), y=curve, name="자산", line=dict(color="#2563eb", width=2.4), fill="tonexty", fillcolor="rgba(37,99,235,0.08)"))
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08, x=0), yaxis_title="USDT", plot_bgcolor="white")
            st.plotly_chart(fig, width="stretch", key="gain_curve")
            if not TR.empty:
                done = TR[TR["결과"] != "진행 중"]; wins = done[done["결과"] == "익절"]; loss = done[done["결과"] == "손절"]
                if len(done):
                    st.markdown(f"**끝난 거래 {len(done)}번** · 익절 {len(wins)} · 손절 {len(loss)} · 본전 {(done['결과'] == '본전').sum()} · "
                                f"평균 익절 <span style='color:#15803d'>{wins['손익'].mean() if len(wins) else 0:+.2f}</span> USDT vs 평균 손절 <span style='color:#b91c1c'>{loss['손익'].mean() if len(loss) else 0:+.2f}</span> USDT"
                                + (f" · 가장 큰 익절 {wins['손익'].max():+.2f} ({wins.loc[wins['손익'].idxmax(), '코인']} {wins.loc[wins['손익'].idxmax(), '방향']})" if len(wins) else ""),
                                unsafe_allow_html=True)
                    st.caption("손절이 더 자주 나와도 정상입니다 — 이 방식은 '작게 자주 지고, 가끔 크게 이기는' 구조 (11년 기준 맞는 비율 26~50%).")
                kst = lambda t_: "" if pd.isna(t_) else pd.Timestamp(t_).tz_convert("Asia/Seoul").strftime("%m-%d %H시")
                show = pd.DataFrame({"거래": [f"{c_} {d_} · {kst(a_)}~{kst(b_) or '지금'}" for c_, d_, a_, b_ in zip(TR["코인"], TR["방향"], TR["시작"], TR["끝"])],
                                     "결과": [f"{r_} {g_:+.2f}" + (" · 부분익절" if pt_ else "") for r_, g_, pt_ in zip(TR["결과"], TR["손익"], TR["부분익절"])],
                                     "가격": [f"{a_:,.0f}→{b_:,.0f}" for a_, b_ in zip(TR["진입가"], TR["청산가"])]}).head(15)
                st.dataframe(show, hide_index=True, width="stretch")
            # 다음 목표
            try:
                with open(STATE / ("certainty_flow.json" if (MTF and WAVE and FLOW) else ("certainty_wave.json" if (MTF and WAVE) else ("certainty_mtf.json" if MTF else "certainty.json"))), encoding="utf-8") as _fh:
                    _ct = json.load(_fh)["table"].get(f"{'maker' if MAKER else 'taker'}|{float(mult)}")
                g_med, g_low = (_ct["1년"]["med"], max(_ct["1년"]["p10"], 0.0)) if _ct else (0.4, 0.0)
            except Exception:
                g_med, g_low = 0.4, 0.0
            MS = SC.milestones(now_e, krw, float(s_mon), g_med, g_low)
            if MS:
                now_m = pd.Timestamp.now(tz="Asia/Seoul")
                fmt_k = lambda x: f"{x / 1e8:g}억" if x >= 1e8 else f"{x / 1e4:,.0f}만"
                fmt_m = lambda m_: "—" if m_ is None else (now_m + pd.DateOffset(months=int(m_))).strftime("%Y-%m") + f" ({m_}개월)"
                st.markdown("**△ 다음 목표까지** (매달 입금 포함, 지금 배율의 과거 1년 성적 기준)")
                st.dataframe(pd.DataFrame({"목표": [fmt_k(r_["목표"]) + "원" for r_ in MS], "보통이면": [fmt_m(r_["중앙"]) for r_ in MS], "나쁜 해가 이어지면": [fmt_m(r_["보수"]) for r_ in MS]}).head(7), hide_index=True, width="stretch")
                st.caption(f"보통 = 연 {g_med:+.0%}, 나쁜 해 = 연 {g_low:+.0%} (그보다 나쁘면 입금만 쌓임). 과거 성적이라 앞으로를 약속하지 않습니다 — 그래도 큰 목표보다 바로 다음 칸을 보는 게 오래 갑니다.")
            try:
                import plan_live as PLv
                st.markdown("**△ 목표 계획 — 배율별 경우의 수** (지금 자산·매달 입금으로 목표 시점까지 3,000번 다시 돌려 본 결과)")
                g1, g2 = st.columns(2)
                _goal = g1.selectbox("목표 금액", ["1억원", "10억원", "50억원"], index=2, key="plan_goal")
                _year = g2.selectbox("목표 시점 (그해 12월)", [2028, 2029, 2030, 2031, 2032, 2033, 2034], index=4, key="plan_year")
                _gk = {"1억원": 1e8, "10억원": 1e9, "50억원": 5e9}[_goal]
                _nowk = pd.Timestamp.now(tz="Asia/Seoul"); _mo = max((int(_year) - _nowk.year) * 12 + 12 - _nowk.month, 1)
                _ml = sorted(set([1.5, 2.0, 3.0, 4.0, 5.0] + [float(mult)]))
                PT = st.cache_data(ttl=3600, show_spinner="목표 계획 계산 중…")(PLv.table)(str(STATE / "plan_base.json"), round(now_e, 0), float(s_mon), int(_mo), tuple(_ml), _gk / krw)
                fk = lambda u_: (f"{u_ * krw / 1e8:,.1f}억" if u_ * krw >= 1e8 else f"{u_ * krw / 1e4:,.0f}만")
                st.dataframe(pd.DataFrame({"배율": [("▶ " if r_["배율"] == float(mult) else "") + f"×{r_['배율']:g}" for r_ in PT],
                                           "과거 그대로면": [f"{fk(r_['그대로']['med'])} · 달성 {r_['그대로']['p']:.0%}" for r_ in PT],
                                           "절반만 나오면": [f"{fk(r_['절반']['med'])} · 달성 {r_['절반']['p']:.0%}" for r_ in PT],
                                           "안 통하면": [fk(r_["없음"]["med"]) for r_ in PT],
                                           "도중 낙폭 (보통~나쁨)": [f"{r_['그대로']['dd']:.0%} ~ {r_['절반']['dd10']:.0%}" for r_ in PT]}), hide_index=True, width="stretch")
                st.caption(f"지금 {now_e:,.0f} USDT + 매달 {s_mon:,.0f} USDT · {_mo}달 동안 · 지금 자산과 앞으로 넣을 돈을 합치면 약 {(now_e + s_mon * _mo) * krw / 1e4:,.0f}만원 · 금액은 중앙값(절반은 그보다 아래). "
                           "'과거 그대로' = 비트겟 2022-08~ 검증 성적이 계속됨(가장 낙관적), '절반' = 평균 수익이 절반(검증 성적은 보통 실전에서 깎임), '안 통하면' = 우위 없음. "
                           "낙폭은 월말 기준이라 실제 도중엔 더 깊을 수 있음. 배율을 올리면 목표 확률과 낙폭이 함께 커지고, 안 통하는 세계에서는 더 빨리 잃습니다.")
            except Exception as e_:
                st.caption(f"목표 계획 계산 오류: {e_}")
            st.markdown("""**△ 즐겁게 오래 하는 법**
- **수익은 주·월 단위로 보기** — 4시간마다 보면 절반은 손실이라 지칩니다. 카드는 할 일만 확인하고, 이 탭은 일요일 저녁에 한 번.
- **손절 날은 '규칙을 지킨 날'** — 작게 잃는 게 이 방식의 정상 운영이고, 큰 익절은 그걸 버틴 사람에게만 옵니다.
- **익절이 나오면 일부는 기념으로 현물·저축으로 옮기기** — 수익이 '숫자'가 아니라 '내 것'이 되는 느낌이 오래 하게 만듭니다.
- **배율은 낙폭을 견딜 만큼만** — 잠을 설치는 배율이면 즐거움이 먼저 사라집니다.""")
    except Exception as e:
        st.warning(f"수익판 계산 오류: {e}")

render_journal(tab_trade)
if not errs:
    render_diag(expanded=False, box=tab_info)
