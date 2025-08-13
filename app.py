import streamlit as st
import pandas as pd
import joblib, yfinance as yf
from utils import compute_indicators

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Prediction App")
st.caption("Educational demo — not financial advice.")

ticker = st.text_input("Ticker", "TSLA")

def load_prices_yf(ticker: str):
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Make sure columns are simple strings
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    # If Adj Close missing, mirror Close
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    # Ensure numeric dtypes
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

if st.button("Predict"):
    df = load_prices_yf(ticker)
    if df.empty:
        st.error("No price data returned for that ticker.")
        st.stop()

    # Robust indicators (your utils.py already patched to handle odd shapes)
    df = compute_indicators(df)

    # Load artifacts saved by training_text.py
    model = joblib.load("artifacts/model.pkl")          # ← pkl, not joblib
    features = joblib.load("artifacts/features.pkl")

    # Ensure all expected feature cols exist (fill missing with 0)
    for c in features:
        if c not in df.columns:
            df[c] = 0.0

    X = df[features].dropna()
    if X.empty:
        st.error("No rows with complete feature data to predict.")
        st.stop()

    preds = model.predict(X)

    # Pick a price series for the chart (Adj Close preferred)
    price_cols = [c for c in ["Adj Close","Close","Open"] if c in df.columns]
    price = df.loc[X.index, price_cols[0]]

    st.subheader("Prediction")
    st.line_chart(
        pd.DataFrame({"Price": price, "Predicted return (next-day)": preds},
                     index=X.index)
    )
    st.success("Done!")
