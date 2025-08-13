# app.py — Streamlit app for price-only (with event scaffolding) predictions

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from utils import compute_indicators

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Prediction App")
st.caption("Educational demo -- not financial advice.")

# ----------------- Robust Yahoo Finance loader -----------------
def load_prices_yf(ticker: str, period: str = "2y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten to simple string columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1).astype(str)
    else:
        df.columns = df.columns.astype(str)

    # Standardize common names
    df = df.rename(columns={
        "AdjClose":"Adj Close", "adj close":"Adj Close", "adjclose":"Adj Close",
        "open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume",
    })

    # Deduplicate (keep first)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure fields
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # Coerce to numeric, force 1-D if needed
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        out[c] = pd.to_numeric(s.squeeze(), errors="coerce")

    cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in out.columns]
    if not cols:
        cols = out.select_dtypes(include="number").columns.tolist()
    return out[cols]

# ----------------- UI -----------------
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker", "TSLA").strip().upper()
with col2:
    period = st.selectbox("History window", ["1y","2y","5y","max"], index=1)

if st.button("Predict"):
    # 1) Load & preprocess prices
    df = load_prices_yf(ticker, period=period)
    if df.empty:
        st.error("No price data returned for that ticker/period.")
        st.stop()

    df = compute_indicators(df)  # robust 1-D safe indicators from utils.py

    # 2) Load artifacts produced by training_text.py
    try:
        model = joblib.load("artifacts/model.pkl")
        features = joblib.load("artifacts/features.pkl")
    except Exception as e:
        st.error(f"Could not load artifacts: {e}")
        st.stop()

    # 3) Recreate training's simple event features (placeholders if no news)
    for c in ("Daily_Sentiment", "News_Volume"):
        if c not in df.columns:
            df[c] = 0.0

    def zscore(s, w=20):
        return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)

    df["news_vol_z20"]  = zscore(df["News_Volume"])
    df["sent_jump"]     = (df["Daily_Sentiment"] - df["Daily_Sentiment"].shift(1)).fillna(0.0)
    df["sent_jump_z20"] = zscore(df["sent_jump"])
    df["is_event_day"]  = ((df["news_vol_z20"] > 1.5) | (df["sent_jump_z20"] > 1.5)).astype(int)

    # 4) Ensure feature columns; fill NaNs/±inf (NO hard warm-up trim)
    for c in features:
        if c not in df.columns:
            df[c] = 0.0

    X = (df[features]
         .astype(float)
         .replace([np.inf, -np.inf], np.nan)
         .fillna(method="ffill")
         .fillna(0.0))

    if X.empty:
        st.error("No rows with complete feature data to predict. Try a longer history (e.g., 2y or 5y).")
        st.stop()

    # 5) Predict next-day returns
    preds = model.predict(X)

    # 6) Chart
    price_col = next((c for c in ["Adj Close","Close","Open"] if c in df.columns), None)
    chart = pd.DataFrame(
        {"Price": df.loc[X.index, price_col], "Predicted next-day return": preds},
        index=X.index
    )
    st.subheader(f"Prediction for {ticker}")
    st.line_chart(chart)

    latest_dt = X.index[-1]
    latest_pred = float(preds[-1])
    st.metric(
        label=f"Latest predicted next-day return ({getattr(latest_dt, 'date', lambda: latest_dt)()})",
        value=f"{latest_pred:.2%}"
    )
