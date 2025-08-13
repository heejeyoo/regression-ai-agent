import streamlit as st
import pandas as pd
import joblib, yfinance as yf
from utils import compute_indicators

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Prediction App")
st.caption("Educational demo — not financial advice.")

def load_prices_yf(ticker: str):
    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)
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

    # Deduplicate (keep first occurrence)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure fields
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # Coerce to numeric (force 1-D if anything slipped through)
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

st.write("Cols:", list(df.columns))
st.write(df.dtypes)
st.write(df.head())

ticker = st.text_input("Ticker", "TSLA")

if st.button("Predict"):
    df = load_prices_yf(ticker)
    if df.empty:
        st.error("No price data returned for that ticker.")
        st.stop()

    df = compute_indicators(df)  # now guaranteed 1-D series per column

    # Load artifacts produced by training_text.py
    model = joblib.load("artifacts/model.pkl")     # <-- pkl, not joblib
    features = joblib.load("artifacts/features.pkl")

    # Ensure all expected features exist
    for c in features:
        if c not in df.columns:
            df[c] = 0.0

    X = df[features].dropna()
    if X.empty:
        st.error("No rows with complete feature data to predict.")
        st.stop()

    preds = model.predict(X)

    price_col = next((c for c in ["Adj Close","Close","Open"] if c in df.columns), None)
    st.subheader("Prediction")
    st.line_chart(pd.DataFrame(
        {"Price": df.loc[X.index, price_col], "Predicted next-day return": preds},
        index=X.index
    ))
