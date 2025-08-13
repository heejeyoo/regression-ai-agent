# app.py — Streamlit app with lenient indicators (no-flat features) + robust alignment

import streamlit as st
import pandas as pd
import numpy as np
import joblib, yfinance as yf
from utils import compute_indicators  # strict version (used first)

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("AI Stock Prediction App")
st.caption("Educational demo -- not financial advice.")

# ---------- helpers ----------
def _to_date_index(idx):
    dt = pd.to_datetime(idx, errors="coerce", utc=True)
    return dt.tz_convert("UTC").tz_localize(None).normalize()

def _as_date_index(df):
    out = df.copy()
    out.index = _to_date_index(out.index)
    out = out[~out.index.duplicated(keep="last")]
    return out.sort_index()

def load_prices_yf(ticker: str, period: str = "5y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1).astype(str)
    else:
        df.columns = df.columns.astype(str)
    df = df.rename(columns={
        "AdjClose":"Adj Close","adj close":"Adj Close","adjclose":"Adj Close",
        "open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume",
    })
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        s = df[c]
        if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
        out[c] = pd.to_numeric(s.squeeze(), errors="coerce")
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in out.columns]
    if not keep: keep = out.select_dtypes(include="number").columns.tolist()
    return _as_date_index(out[keep])

def build_indicators_lenient(raw: pd.DataFrame) -> pd.DataFrame:
    """Always returns varying technicals; no full-drop. Uses min_periods=1 + ewm smoothing."""
    df = raw.copy()
    px = df.get("Adj Close")
    if px is None:
        px = df.select_dtypes(include="number").iloc[:, 0]
    px = pd.to_numeric(px, errors="coerce").fillna(method="ffill").fillna(method="bfill")

    # Basic return
    df["ret_1d"] = px.pct_change().fillna(0.0)

    # SMAs (lenient)
    sma10 = px.rolling(10, min_periods=2).mean()
    sma20 = px.rolling(20, min_periods=2).mean()
    df["sma_ratio_10_20"] = (sma10 / (sma20 + 1e-12)).fillna(1.0)

    # Volatility (lenient)
    df["vol_20"] = df["ret_1d"].rolling(20, min_periods=2).std().fillna(0.0) * np.sqrt(252)

    # RSI(14) with Wilder-style EWM
    delta = px.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    roll_up = gain.ewm(alpha=1/14, adjust=False, min_periods=2).mean()
    roll_down = loss.ewm(alpha=1/14, adjust=False, min_periods=2).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    # MACD (12,26) + signal(9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26).fillna(0.0)
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean().fillna(0.0)

    # Volume z-score (lenient)
    vol = pd.to_numeric(df.get("Volume", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    vmean = vol.rolling(20, min_periods=2).mean()
    vstd  = vol.rolling(20, min_periods=2).std()
    df["vol_z20"] = ((vol - vmean) / (vstd + 1e-12)).fillna(0.0)

    return df

# ---------- UI ----------
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker", "TSLA").strip().upper()
with col2:
    period = st.selectbox("History window", ["1y","2y","5y","max"], index=2)

if st.button("Predict"):
    # 1) Load prices
    raw = load_prices_yf(ticker, period=period)
    if raw.empty:
        st.error("No price data returned for that ticker/period."); st.stop()

    # 2) Indicators (strict first; lenient fallback)
    feats_df = compute_indicators(raw)
    if feats_df.empty or feats_df[["sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal"]].nunique().sum() <= 5:
        feats_df = build_indicators_lenient(raw)
    feats_df = _as_date_index(feats_df)

    # 3) Load artifacts
    try:
        model = joblib.load("artifacts/model.pkl")
        features = joblib.load("artifacts/features.pkl")
    except Exception as e:
        st.error(f"Could not load artifacts: {e}"); st.stop()

    # 4) Rebuild simple event features (placeholders; only these get zero/ffill)
    for c in ("Daily_Sentiment","News_Volume"):
        if c not in feats_df.columns: feats_df[c] = 0.0
    def zscore(s, w=20): return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)
    feats_df["news_vol_z20"]  = zscore(feats_df["News_Volume"]).fillna(0.0)
    feats_df["sent_jump"]     = (feats_df["Daily_Sentiment"] - feats_df["Daily_Sentiment"].shift(1)).fillna(0.0)
    feats_df["sent_jump_z20"] = zscore(feats_df["sent_jump"]).fillna(0.0)
    feats_df["is_event_day"]  = ((feats_df["news_vol_z20"] > 1.5) | (feats_df["sent_jump_z20"] > 1.5)).astype(int)

    # 5) Ensure all expected feature columns exist
    for c in features:
        if c not in feats_df.columns: feats_df[c] = 0.0

    # Build X: keep technicals “as is”, only clean event/text
    core_feats  = [f for f in ["sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal","vol_z20","ret_1d"] if f in features]
    event_feats = [f for f in ["news_vol_z20","sent_jump_z20","is_event_day"] if f in features]
    text_feats  = [f for f in features if f.startswith("topic_") or f.startswith("kw_")]

    X = feats_df[features].copy()
    if event_feats: X[event_feats] = X[event_feats].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    if text_feats:  X[text_feats]  = X[text_feats].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    # Drop rows missing any core technicals; then forward-fill only remaining holes
    if core_feats:
        X = X.dropna(subset=core_feats)
        if X.isna().any().any():
            X[core_feats] = X[core_feats].fillna(method="ffill")

    if X.empty:
        st.error("No usable rows found. Try a longer history or a different ticker."); st.stop()

    # Warn if features are still constant
    nunq = X.nunique()
    if int(nunq.sum()) <= len(nunq):
        st.warning("Features look constant; predictions may appear flat. Try a longer window or another ticker.")

    # 6) Predict
    preds = model.predict(X)

    # 7) Align price to X index for chart
    price_col = next((c for c in ["Adj Close","Close","Open"] if c in raw.columns), None)
    if price_col:
        price_series = raw[price_col].astype(float).reindex(X.index).fillna(method="ffill").fillna(method="bfill")
    else:
        price_series = pd.Series(index=X.index, dtype=float)

    chart = pd.DataFrame(index=X.index)
    if not price_series.dropna().empty: chart["Price"] = price_series
    chart["Predicted next-day return"] = preds

    st.subheader(f"Prediction for {ticker}")
    st.line_chart(chart)

    latest_dt = X.index[-1]
    latest_pred = float(preds[-1])
    try: label_date = latest_dt.date()
    except Exception: label_date = str(latest_dt)
    st.metric(label=f"Latest predicted next-day return ({label_date})", value=f"{latest_pred:.2%}")

    with st.expander("Debug info"):
        st.write("Feature rows:", X.shape[0], "Feature cols:", X.shape[1])
        st.write("Per-feature nunique (first 15):", nunq.sort_values().head(15))
        st.write("X head:", X.head(3))
