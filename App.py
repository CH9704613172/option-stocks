import warnings
import time
from datetime import datetime
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta_classic as ta
import streamlit as st

# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nifty Options Screener v5",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #00d4ff; margin: 0; font-size: 2rem; }
    .main-header p  { color: #adb5bd; margin: 0.3rem 0 0; font-size: 0.9rem; }

    .metric-card {
        background: #1e1e2e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card .label { color: #adb5bd; font-size: 0.75rem; text-transform: uppercase; }
    .metric-card .value { color: #00d4ff; font-size: 1.8rem; font-weight: bold; }

    .signal-buy     { background:#0d3321; border-left: 4px solid #00c853; }
    .signal-sell    { background:#3b0d0d; border-left: 4px solid #ff1744; }
    .signal-bull    { background:#0d2b1e; border-left: 4px solid #69f0ae; }
    .signal-bear    { background:#2e1f00; border-left: 4px solid #ffc107; }

    .badge-high   { background:#00c853; color:#000; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:bold; }
    .badge-medium { background:#ffc107; color:#000; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:bold; }
    .badge-low    { background:#607d8b; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:bold; }
    .badge-long   { background:#00c853; color:#000; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:bold; }
    .badge-short  { background:#ff1744; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:bold; }
    .badge-wait   { background:#607d8b; color:#fff; padding:2px 8px; border-radius:4px; font-size:0.75rem; font-weight:bold; }

    div[data-testid="stDataFrame"] table { font-size: 0.82rem; }
    .stProgress > div > div { background-color: #00d4ff; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
#  UNIVERSE
# ══════════════════════════════════════════════════════════════════
NIFTY_OPTION_STOCKS = [
    "RELIANCE.NS",  "TCS.NS",        "HDFCBANK.NS",   "BHARTIARTL.NS", "ICICIBANK.NS",
    "INFY.NS",      "SBIN.NS",       "HINDUNILVR.NS", "ITC.NS",        "LT.NS",
    "KOTAKBANK.NS", "AXISBANK.NS",   "BAJFINANCE.NS", "WIPRO.NS",      "HCLTECH.NS",
    "ASIANPAINT.NS","MARUTI.NS",     "SUNPHARMA.NS",  "TITAN.NS",      "NTPC.NS",
    "POWERGRID.NS", "ULTRACEMCO.NS", "TATAMOTORS.NS", "NESTLEIND.NS",  "ADANIENT.NS",
    "BAJAJFINSV.NS","JSWSTEEL.NS",   "TATASTEEL.NS",  "M&M.NS",        "HINDALCO.NS",
    "ONGC.NS",      "COALINDIA.NS",  "DIVISLAB.NS",   "CIPLA.NS",      "DRREDDY.NS",
    "TECHM.NS",     "GRASIM.NS",     "EICHERMOT.NS",  "BRITANNIA.NS",  "APOLLOHOSP.NS",
    "BPCL.NS",      "TATACONSUM.NS", "HEROMOTOCO.NS", "INDUSINDBK.NS", "SHRIRAMFIN.NS",
    "ADANIPORTS.NS","SBILIFE.NS",    "HDFCLIFE.NS",   "BAJAJ-AUTO.NS", "LTIM.NS",
    "SIEMENS.NS",   "PIDILITIND.NS", "HAVELLS.NS",    "DABUR.NS",      "GODREJCP.NS",
    "BERGEPAINT.NS","MUTHOOTFIN.NS", "BANKBARODA.NS", "INDHOTEL.NS",   "ZOMATO.NS",
    "DMART.NS",     "IRCTC.NS",      "HAL.NS",        "BEL.NS",        "IRFC.NS",
    "NHPC.NS",      "RECLTD.NS",     "PFC.NS",
]

# ══════════════════════════════════════════════════════════════════
#  SIDEBAR — CONFIG
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Conditions")

    MA_FAST           = st.number_input("MA Fast (days)", value=50, min_value=5, max_value=200)
    MA_SLOW           = st.number_input("MA Slow (days)", value=252, min_value=50, max_value=500)
    RSI_PERIOD        = st.number_input("RSI Period", value=22, min_value=5, max_value=50)
    RSI_BUY_LEVEL     = st.number_input("RSI Buy Level", value=55, min_value=40, max_value=80)
    RSI_SEL_LEVEL     = st.number_input("RSI Sell Level", value=45, min_value=20, max_value=60)
    MACD_FAST         = st.number_input("MACD Fast", value=26, min_value=5, max_value=50)
    MACD_SLOW         = st.number_input("MACD Slow", value=44, min_value=10, max_value=100)
    MACD_SIGNAL       = st.number_input("MACD Signal", value=9, min_value=3, max_value=20)
    VOLUME_MULTIPLIER = st.number_input("Volume Multiplier (×)", value=2.5, min_value=1.0, max_value=10.0, step=0.5)
    ATR_PERIOD        = st.number_input("ATR Period", value=14, min_value=5, max_value=30)
    TARGET_ATR_MULT   = st.number_input("Target ATR ×", value=1.5, min_value=0.5, max_value=5.0, step=0.25)
    SL_ATR_MULT       = st.number_input("Stop-Loss ATR ×", value=0.75, min_value=0.25, max_value=3.0, step=0.25)
    LOOKBACK_DAYS     = st.number_input("Lookback Days", value=400, min_value=300, max_value=600)
    SLEEP_BETWEEN     = st.number_input("Sleep between fetches (s)", value=0.2, min_value=0.0, max_value=2.0, step=0.1)

    st.markdown("---")
    st.markdown("### 🎯 Stock Selection")
    select_all = st.checkbox("Select All Stocks", value=True)
    if not select_all:
        chosen = st.multiselect(
            "Pick stocks to scan",
            options=[t.replace(".NS", "") for t in NIFTY_OPTION_STOCKS],
            default=[t.replace(".NS", "") for t in NIFTY_OPTION_STOCKS[:10]],
        )
        STOCKS = [f"{s}.NS" for s in chosen] if chosen else NIFTY_OPTION_STOCKS
    else:
        STOCKS = NIFTY_OPTION_STOCKS

    st.markdown("---")
    st.markdown(f"**Universe:** {len(STOCKS)} stocks")
    st.markdown("*Data via yfinance (NSE)*")

# ══════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════
ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
st.markdown(f"""
<div class="main-header">
  <h1>📈 Nifty Options Screener v5</h1>
  <p>RSI({RSI_PERIOD}) &nbsp;│&nbsp; MACD({MACD_FAST},{MACD_SLOW},{MACD_SIGNAL})
     &nbsp;│&nbsp; MA{MA_FAST}/MA{MA_SLOW} &nbsp;│&nbsp; Vol≥{VOLUME_MULTIPLIER}×
     &nbsp;│&nbsp; ATR({ATR_PERIOD})
     &nbsp;│&nbsp; {ist_now.strftime("%d %b %Y  %H:%M")} IST</p>
</div>
""", unsafe_allow_html=True)

def fetch_data(ticker: str):
    try:
        df = yf.download(
            ticker,
            period=f"{LOOKBACK_DAYS}d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            multi_level_index=False,
            threads=False,
        )
        if df is None or df.empty:
            return None, f"{ticker}: empty data"

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).strip().title() for c in df.columns]
        required = ["Open", "High", "Low", "Close", "Volume"]

        missing = [c for c in required if c not in df.columns]
        if missing:
            return None, f"{ticker}: missing columns {missing}"

        if len(df) < MA_SLOW + 10:
            return None, f"{ticker}: insufficient rows ({len(df)})"

        df = df[required].copy()
        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)

        if len(df) < MA_SLOW + 10:
            return None, f"{ticker}: insufficient clean rows ({len(df)})"

        return df, None

    except Exception as e:
        return None, f"{ticker}: {type(e).__name__}: {e}"

def compute_indicators(df):
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    df["MA_FAST"] = close.rolling(MA_FAST).mean()
    df["MA_SLOW"] = close.rolling(MA_SLOW).mean()
    df["RSI"] = ta.rsi(close, length=RSI_PERIOD)

    atr_df = ta.atr(high, low, close, length=ATR_PERIOD)
    if atr_df is not None:
        df["ATR"] = atr_df

    macd_df = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd_df is not None and not macd_df.empty:
        df = pd.concat([df, macd_df], axis=1)

    return df

def _f(val) -> float:
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    if hasattr(val, "item"):
        return float(val.item())
    return float(val)

def detect_candle_pattern(o, h, l, c) -> str:
    body = abs(c - o)
    full_range = h - l
    if full_range == 0:
        return "Doji"

    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    body_pct = body / full_range
    is_bull = c >= o

    if body_pct < 0.05:
        return "Doji"
    if body_pct >= 0.85:
        return "Bull Marubozu" if is_bull else "Bear Marubozu"
    if lower_wick >= body * 2 and upper_wick <= body * 0.4:
        return "Hammer" if is_bull else "Hanging Man"
    if upper_wick >= body * 2 and lower_wick <= body * 0.4:
        return "Inv Hammer" if is_bull else "Shooting Star"
    if upper_wick >= body * 1.5 and lower_wick >= body * 1.5:
        return "Spinning Top"
    if body_pct >= 0.6:
        return "Bull Candle" if is_bull else "Bear Candle"
    return "Bull Small" if is_bull else "Bear Small"

_PATTERN_SCORE = {
    "Bull Marubozu": (3, 0, "Strong bull body"),
    "Bear Marubozu": (0, 3, "Strong bear body"),
    "Hammer": (2, 0, "Reversal up"),
    "Hanging Man": (0, 2, "Reversal down"),
    "Inv Hammer": (1, 0, "Possible bull rev"),
    "Shooting Star": (0, 2, "Possible bear rev"),
    "Bull Candle": (2, 0, "Trend continuation"),
    "Bear Candle": (0, 2, "Trend continuation"),
    "Bull Small": (1, 0, "Mild bullish"),
    "Bear Small": (0, 1, "Mild bearish"),
    "Spinning Top": (0, 0, "Indecision"),
    "Doji": (0, 0, "Indecision"),
}

def next_day_bias(sig: dict) -> dict:
    bias = dict(
        nd_bias="WAIT", nd_action="—", nd_confidence="—",
        nd_entry=None, nd_target=None, nd_stoploss=None, nd_reason="—"
    )

    close = sig.get("close")
    atr = sig.get("atr")
    pattern = sig.get("candle_pattern", "Doji")
    rsi = sig.get("rsi")
    hist = sig.get("macd_hist")
    ma_fast = sig.get("ma_fast")
    ma_slow = sig.get("ma_slow")
    vol_r = sig.get("vol_ratio", 0)
    candle = sig.get("candle", "—")

    if any(v is None for v in [close, rsi, hist, ma_fast, ma_slow]):
        return bias

    ps = _PATTERN_SCORE.get(pattern, (0, 0, "Unknown"))
    bull_score, bear_score, pat_desc = ps[0], ps[1], ps[2]

    if rsi > 60:
        bull_score += 2
    elif rsi > 55:
        bull_score += 1

    if rsi < 40:
        bear_score += 2
    elif rsi < 45:
        bear_score += 1

    if hist > 0:
        bull_score += 2
    elif hist < 0:
        bear_score += 2

    if ma_fast > ma_slow:
        bull_score += 2
    elif ma_fast < ma_slow:
        bear_score += 2

    if vol_r >= VOLUME_MULTIPLIER:
        if candle == "ABOVE_HIGH":
            bull_score += 1
        if candle == "BELOW_LOW":
            bear_score += 1

    total = bull_score + bear_score
    if total == 0:
        return bias

    if bull_score > bear_score + 1:
        direction = "LONG"
        action = "BUY CE (Call)"
        conf = "HIGH" if bull_score >= 6 else ("MEDIUM" if bull_score >= 4 else "LOW")
        reason = f"{pattern} | RSI {rsi:.1f} | Hist {hist:+.3f} | MA {'above' if ma_fast > ma_slow else 'below'} | Vol {vol_r:.1f}× | {pat_desc}"
    elif bear_score > bull_score + 1:
        direction = "SHORT"
        action = "BUY PE (Put)"
        conf = "HIGH" if bear_score >= 6 else ("MEDIUM" if bear_score >= 4 else "LOW")
        reason = f"{pattern} | RSI {rsi:.1f} | Hist {hist:+.3f} | MA {'above' if ma_fast > ma_slow else 'below'} | Vol {vol_r:.1f}× | {pat_desc}"
    else:
        bias.update(
            nd_bias="WAIT",
            nd_action="Conflicting signals",
            nd_confidence="LOW",
            nd_reason=f"Bull {bull_score} vs Bear {bear_score} — too close"
        )
        return bias

    if atr is not None and atr > 0:
        if direction == "LONG":
            entry = round(close, 2)
            target = round(close + atr * TARGET_ATR_MULT, 2)
            sl = round(close - atr * SL_ATR_MULT, 2)
        else:
            entry = round(close, 2)
            target = round(close - atr * TARGET_ATR_MULT, 2)
            sl = round(close + atr * SL_ATR_MULT, 2)
    else:
        pct = close * 0.015
        if direction == "LONG":
            entry = round(close, 2)
            target = round(close + pct, 2)
            sl = round(close - pct * 0.5, 2)
        else:
            entry = round(close, 2)
            target = round(close - pct, 2)
            sl = round(close + pct * 0.5, 2)

    bias.update(
        nd_bias=direction,
        nd_action=action,
        nd_confidence=conf,
        nd_entry=entry,
        nd_target=target,
        nd_stoploss=sl,
        nd_reason=reason
    )
    return bias

def classify_signal(df):
    macd_col = f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    macds_col = f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"
    macdh_col = f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"

    empty = dict(
        strict_signal="NEUTRAL", relaxed_signal="NEUTRAL", cross="—",
        close=None, prev_high=None, prev_low=None, atr=None,
        vol_today=None, vol_ratio=None, candle="—", candle_pattern="—",
        ma_fast=None, ma_slow=None, rsi=None,
        macd=None, macd_signal=None, macd_hist=None,
        nd_bias="—", nd_action="—", nd_confidence="—",
        nd_entry=None, nd_target=None, nd_stoploss=None, nd_reason="—"
    )

    needed = ["MA_FAST", "MA_SLOW", "RSI", macd_col, macds_col, macdh_col,
              "Open", "Close", "High", "Low", "Volume"]

    if len(df) < 2 or any(c not in df.columns for c in needed):
        return empty

    clean = df.dropna(subset=needed)
    if len(clean) < 2:
        return empty

    today, prev = clean.iloc[-1], clean.iloc[-2]

    ma_fast_now = _f(today["MA_FAST"])
    ma_slow_now = _f(today["MA_SLOW"])
    ma_fast_prev = _f(prev["MA_FAST"])
    ma_slow_prev = _f(prev["MA_SLOW"])
    rsi = _f(today["RSI"])
    macd_val = _f(today[macd_col])
    macd_sig = _f(today[macds_col])
    macd_hist = _f(today[macdh_col])
    close_now = _f(today["Close"])
    open_now = _f(today["Open"])
    high_now = _f(today["High"])
    low_now = _f(today["Low"])
    vol_now = _f(today["Volume"])
    prev_high = _f(prev["High"])
    prev_low = _f(prev["Low"])
    prev_vol = _f(prev["Volume"])

    atr_val = _f(today["ATR"]) if "ATR" in clean.columns and not pd.isna(today.get("ATR")) else None

    vol_ratio = (vol_now / prev_vol) if prev_vol > 0 else 0.0
    high_vol = vol_ratio >= VOLUME_MULTIPLIER
    bull_candle = (close_now > prev_high) and high_vol
    bear_candle = (close_now < prev_low) and high_vol
    candle_lbl = "ABOVE_HIGH" if bull_candle else ("BELOW_LOW" if bear_candle else "—")
    candle_pattern = detect_candle_pattern(open_now, high_now, low_now, close_now)

    golden = (ma_fast_now > ma_slow_now) and (ma_fast_prev <= ma_slow_prev)
    death = (ma_fast_now < ma_slow_now) and (ma_fast_prev >= ma_slow_prev)
    above = ma_fast_now > ma_slow_now
    below = ma_fast_now < ma_slow_now
    macd_bull = (macd_val > macd_sig) and (macd_hist > 0)
    macd_bear = (macd_val < macd_sig) and (macd_hist < 0)

    if golden and rsi > RSI_BUY_LEVEL and macd_bull and bull_candle:
        strict, cross = "BUY", "GOLDEN"
    elif death and rsi < RSI_SEL_LEVEL and macd_bear and bear_candle:
        strict, cross = "SELL", "DEATH"
    elif above and rsi > RSI_BUY_LEVEL and macd_bull and bull_candle:
        strict, cross = "BULLISH", "ABOVE"
    elif below and rsi < RSI_SEL_LEVEL and macd_bear and bear_candle:
        strict, cross = "BEARISH", "BELOW"
    else:
        strict, cross = "NEUTRAL", "—"

    if golden and rsi > RSI_BUY_LEVEL and macd_bull:
        relaxed, cross = "BUY", "GOLDEN"
    elif death and rsi < RSI_SEL_LEVEL and macd_bear:
        relaxed, cross = "SELL", "DEATH"
    elif above and rsi > RSI_BUY_LEVEL and macd_bull:
        relaxed, cross = "BULLISH", "ABOVE"
    elif below and rsi < RSI_SEL_LEVEL and macd_bear:
        relaxed, cross = "BEARISH", "BELOW"
    else:
        relaxed = "NEUTRAL"

    cross_lbl = cross if (strict != "NEUTRAL" or relaxed != "NEUTRAL") else "—"

    sig = dict(
        strict_signal=strict,
        relaxed_signal=relaxed,
        cross=cross_lbl,
        close=round(close_now, 2),
        prev_high=round(prev_high, 2),
        prev_low=round(prev_low, 2),
        atr=round(atr_val, 2) if atr_val is not None else None,
        vol_today=int(vol_now),
        vol_ratio=round(vol_ratio, 2),
        candle=candle_lbl,
        candle_pattern=candle_pattern,
        ma_fast=round(ma_fast_now, 2),
        ma_slow=round(ma_slow_now, 2),
        rsi=round(rsi, 2),
        macd=round(macd_val, 4),
        macd_signal=round(macd_sig, 4),
        macd_hist=round(macd_hist, 4),
    )
    sig.update(next_day_bias(sig))
    return sig

@st.cache_data(ttl=900, show_spinner=False)
def run_screener(stocks_tuple, lookback_days, sleep_between):
    stocks = list(stocks_tuple)
    rows, skipped, errors = [], [], []

    for ticker in stocks:
        df, err = fetch_data(ticker)
        if df is None:
            skipped.append(ticker.replace(".NS", ""))
            if err:
                errors.append(err)
            time.sleep(sleep_between)
            continue

        df = compute_indicators(df)
        sig = classify_signal(df)
        sig["ticker"] = ticker.replace(".NS", "")
        rows.append(sig)
        time.sleep(sleep_between)

    return pd.DataFrame(rows) if rows else pd.DataFrame(), skipped, errors

def _bias_badge(b):
    if b == "LONG":
        return "🟢 LONG"
    if b == "SHORT":
        return "🔴 SHORT"
    return "⚪ WAIT"

def _conf_badge(c):
    if c == "HIGH":
        return "🔥 HIGH"
    if c == "MEDIUM":
        return "🟡 MEDIUM"
    return "⚫ LOW"

def _render_signal_table(df, label, emoji):
    if df.empty:
        st.info(f"No {label} signals found.")
        return

    display_cols = ["ticker", "close", "prev_high", "prev_low", "candle", "candle_pattern",
                    "vol_ratio", "rsi", "macd_hist", "ma_fast", "ma_slow"]
    display_cols = [c for c in display_cols if c in df.columns]

    rename = {
        "ticker": "Stock",
        "close": "Close",
        "prev_high": "Prev High",
        "prev_low": "Prev Low",
        "candle": "Breakout",
        "candle_pattern": "Pattern",
        "vol_ratio": "Vol ×",
        "rsi": "RSI",
        "macd_hist": "MACD Hist",
        "ma_fast": f"MA{MA_FAST}",
        "ma_slow": f"MA{MA_SLOW}",
    }

    styled = df[display_cols].rename(columns=rename)
    st.dataframe(styled, use_container_width=True, hide_index=True)

def _render_nd_table(df):
    if df.empty:
        return

    nd_cols = ["ticker", "close", "nd_bias", "nd_confidence", "nd_action",
               "nd_entry", "nd_target", "nd_stoploss", "atr", "nd_reason"]
    nd_cols = [c for c in nd_cols if c in df.columns]

    rename = {
        "ticker": "Stock",
        "close": "Close",
        "nd_bias": "Next Day",
        "nd_confidence": "Confidence",
        "nd_action": "Action",
        "nd_entry": "Entry",
        "nd_target": "Target",
        "nd_stoploss": "Stop Loss",
        "atr": "ATR",
        "nd_reason": "Reason"
    }

    view = df[nd_cols].rename(columns=rename).copy()
    view["Next Day"] = view["Next Day"].apply(_bias_badge)
    view["Confidence"] = view["Confidence"].apply(_conf_badge)

    st.dataframe(view, use_container_width=True, hide_index=True)

col_run, col_refresh = st.columns([1, 5])
with col_run:
    run_btn = st.button("🚀  Run Screener", type="primary", use_container_width=True)
with col_refresh:
    clear_btn = st.button("🔄  Clear Cache & Re-scan", use_container_width=False)

if clear_btn:
    st.cache_data.clear()
    st.session_state.pop("results", None)
    st.session_state.pop("skipped", None)
    st.session_state.pop("errors", None)
    st.session_state.pop("scan_time", None)
    st.rerun()

if run_btn or "results" in st.session_state:
    if run_btn or "results" not in st.session_state:
        with st.spinner("Scanning stocks..."):
            results, skipped, errors = run_screener(tuple(STOCKS), LOOKBACK_DAYS, SLEEP_BETWEEN)

        st.session_state["results"] = results
        st.session_state["skipped"] = skipped
        st.session_state["errors"] = errors
        st.session_state["scan_time"] = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d %b %Y  %H:%M:%S IST")

    results = st.session_state["results"]
    skipped = st.session_state.get("skipped", [])
    errors = st.session_state.get("errors", [])
    scan_time = st.session_state.get("scan_time", "—")

    if skipped:
        st.warning(f"⚠️ Skipped {len(skipped)} stock(s): {', '.join(skipped)}")

    if errors:
        with st.expander("Debug details for skipped stocks"):
            for err in errors:
                st.text(err)

    if results.empty:
        st.error("No data returned. Check internet connection, symbol availability, or package installation.")
        st.stop()

    s_buy = results[results["strict_signal"] == "BUY"].copy()
    s_sell = results[results["strict_signal"] == "SELL"].copy()
    s_bull = results[results["strict_signal"] == "BULLISH"].copy()
    s_bear = results[results["strict_signal"] == "BEARISH"].copy()

    relaxed = results[(results["strict_signal"] == "NEUTRAL") & (results["relaxed_signal"] != "NEUTRAL")].copy()
    r_buy = relaxed[relaxed["relaxed_signal"] == "BUY"].copy()
    r_sell = relaxed[relaxed["relaxed_signal"] == "SELL"].copy()
    r_bull = relaxed[relaxed["relaxed_signal"] == "BULLISH"].copy()
    r_bear = relaxed[relaxed["relaxed_signal"] == "BEARISH"].copy()

    all_signals = pd.concat([s_buy, s_sell, s_bull, s_bear, r_buy, r_sell, r_bull, r_bear], ignore_index=True)
    all_signals = all_signals[all_signals["nd_bias"].isin(["LONG", "SHORT"])].copy()

    st.markdown(
        f"<p style='color:#adb5bd; font-size:0.8rem;'>Last scan: {scan_time}  │  Scanned: {len(results)} stocks</p>",
        unsafe_allow_html=True
    )

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    for col, label, val, clr in [
        (c1, "Strict BUY",  len(s_buy),  "#00c853"),
        (c2, "Strict SELL", len(s_sell), "#ff1744"),
        (c3, "Strict BULL", len(s_bull), "#69f0ae"),
        (c4, "Strict BEAR", len(s_bear), "#ffc107"),
        (c5, "Relax BUY",   len(r_buy),  "#00c853"),
        (c6, "Relax SELL",  len(r_sell), "#ff1744"),
        (c7, "Relax BULL",  len(r_bull), "#69f0ae"),
        (c8, "Relax BEAR",  len(r_bear), "#ffc107"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value" style="color:{clr}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tabs = st.tabs([
        "🌟 Master Next-Day",
        "🟢 Strict BUY", "🔴 Strict SELL", "📗 Strict BULLISH", "📙 Strict BEARISH",
        "💚 Relax BUY", "❤️ Relax SELL", "🌿 Relax BULLISH", "🟡 Relax BEARISH",
        "📋 All Results",
    ])

    with tabs[0]:
        st.subheader("📊 Next-Day Master Table — All Actionable Stocks")
        if all_signals.empty:
            st.info("No actionable signals found.")
        else:
            conf_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
            all_signals["_cs"] = all_signals["nd_confidence"].map(conf_order).fillna(3)
            all_signals["_bs"] = (all_signals["nd_bias"] == "SHORT").astype(int)
            all_signals = all_signals.sort_values(["_cs", "_bs"]).drop(columns=["_cs", "_bs"])
            _render_nd_table(all_signals)

    for tab, df_sig, label, emoji in [
        (tabs[1], s_buy,  "Strict BUY",      "🟢"),
        (tabs[2], s_sell, "Strict SELL",     "🔴"),
        (tabs[3], s_bull, "Strict BULLISH",  "📗"),
        (tabs[4], s_bear, "Strict BEARISH",  "📙"),
        (tabs[5], r_buy,  "Relaxed BUY",     "💚"),
        (tabs[6], r_sell, "Relaxed SELL",    "❤️"),
        (tabs[7], r_bull, "Relaxed BULLISH", "🌿"),
        (tabs[8], r_bear, "Relaxed BEARISH", "🟡"),
    ]:
        with tab:
            st.subheader(f"{emoji} {label} Signals")
            _render_signal_table(df_sig, label, emoji)
            if not df_sig.empty:
                st.markdown("**Next-Day Prediction:**")
                _render_nd_table(df_sig)

    with tabs[9]:
        st.subheader("📋 Full Scan Results")
        show_cols = ["ticker", "strict_signal", "relaxed_signal", "cross", "close",
                     "rsi", "macd_hist", "vol_ratio", "candle_pattern", "nd_bias", "nd_confidence"]
        show_cols = [c for c in show_cols if c in results.columns]
        st.dataframe(results[show_cols], use_container_width=True, hide_index=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Full CSV",
            data=csv,
            file_name=f"nifty_signals_{datetime.now(ZoneInfo('Asia/Kolkata')).strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem; color:#adb5bd;">
        <h2>👆 Click <b>Run Screener</b> to start</h2>
        <p>Scans all Nifty F&O stocks using MA, RSI, MACD, Volume & ATR indicators.<br>
        Results are cached for 15 minutes — use <b>Clear Cache</b> to force a fresh scan.</p>
    </div>
    """, unsafe_allow_html=True)
