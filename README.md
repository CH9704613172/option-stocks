📈 Nifty Options Screener 

A high-probability options trading screener built using Python + Streamlit, designed to identify actionable trade setups in Nifty F&O stocks using multi-indicator confirmation and price action logic.

🚀 Overview

Most screeners tell you what happened.
This tool focuses on what to do next.

It combines:

Technical indicators (RSI, MACD, Moving Averages)
Price action (breakouts, candle patterns)
Volume confirmation
Risk management (ATR-based levels)

👉 Output: Next-Day Trading Bias (CE/PE) with Entry, Target & Stop-Loss

🎯 Key Features

✅ Scans Nifty F&O stocks universe
✅ Multi-indicator strategy:

Moving Averages (Trend)
RSI (Momentum)
MACD (Confirmation)
Volume breakout (Strength)
ATR (Risk management)

✅ Detects:

Breakouts above previous high / below previous low
Candlestick patterns (Hammer, Doji, Marubozu, etc.)

✅ Generates:

Strict Signals (High-confidence setups)
Relaxed Signals (Developing setups)

✅ Next-Day Prediction Engine:

LONG → Buy Call (CE)
SHORT → Buy Put (PE)
Confidence levels: HIGH / MEDIUM / LOW

✅ Risk Framework:

ATR-based Target & Stop-Loss
Position bias with reasoning
📊 Strategy Logic (Simplified)

Signal is generated when:

Trend → MA Fast > MA Slow
Momentum → RSI above threshold
Confirmation → MACD bullish
Strength → Volume spike + breakout

👉 Only when confluence exists, signal is triggered

🧠 Next-Day Bias Engine

The model scores:

RSI strength
MACD histogram
Trend alignment
Volume breakout
Candlestick pattern

Then outputs:
Bias	Action
🟢 LONG	BUY CE (Call)
🔴 SHORT	BUY PE (Put)
⚪ WAIT	No trade


🛠 Tech Stack
Python
Streamlit
yFinance (Market data)
pandas / numpy
pandas-ta
