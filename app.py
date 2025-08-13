import streamlit as st
import pandas as pd
import joblib, yfinance as yf
from utils import compute_indicators

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Prediction App")
st.caption("Educational demo — not financial advice.")

ticker = st.text_input("Ticker", "TSLA")

def load_prices_yf(ticker: str):
    import pandas as pd, yfinance as yf

    df = yf.download(ticker, period="1y", interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # 1) Flatten to simple string columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1).astype(str)
    else:
        df.columns = df.columns.astype(str)

    # 2) Standardize common names
    rename = {
        "AdjClose": "Adj Close",
        "adj close": "Adj Close",
        "adjclose": "Adj Close",
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    }
    df = df.rename(columns=rename)

    # 3) Deduplicate columns (keep first occurrence)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # 4) Ensure required fields exist
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # 5) Coerce to numeric SAFELY (guard against any lingering 2D)
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]          # take first if duplicates slipped through
        out[c] = pd.to_numeric(s.squeeze(), errors="coerce")

    # 6) Keep just the canonical set, in order (if present)
    cols = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in out.columns]
    if not cols:
        # last resort: keep any numeric columns
        cols = out.select_dtypes(include="number").columns.tolist()
    return out[cols]

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
