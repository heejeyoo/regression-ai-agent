# app.py — correctness-first version with rich diagnostics

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from utils import compute_indicators  # 1-D safe technicals you patched

st.set_page_config(page_title="AI Stock Predictor (Diagnostics)", layout="wide")
st.title("📈 AI Stock Predictor — Diagnostics")
st.caption("Educational demo — not financial advice.")

# ----------------- Helpers -----------------
def _to_date_index(idx):
    dt = pd.to_datetime(idx, errors="coerce", utc=True)
    return dt.tz_convert("UTC").tz_localize(None).normalize()

def _as_date_index(df):
    out = df.copy()
    out.index = _to_date_index(out.index)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()

def load_prices_yf(ticker: str, period: str = "2y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1).astype(str)
    else:
        df.columns = df.columns.astype(str)

    # Standardize
    df = df.rename(columns={
        "AdjClose":"Adj Close", "adj close":"Adj Close", "adjclose":"Adj Close",
        "open":"Open", "high":"High", "low":"Low", "close":"Close", "volume":"Volume",
    })

    # Deduplicate
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # Ensure fields
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # Coerce numeric (force 1-D)
    out = pd.Da
