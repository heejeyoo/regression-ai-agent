# app.py — robust alignment + non-flat features

import streamlit as st
import pandas as pd
import numpy as np
import joblib, yfinance as yf
from utils import compute_indicators

st.set_page_config(page_title="AI Stock Predictor", layout="wide")
st.title("📈 AI Stock Prediction App")
st.caption("Educational demo -- not financial advice.")

# ---------- helpers ----------
def _to_date_index(idx):
    dt = pd.to_datetime(idx, errors="coerce", utc=True)
    return dt.tz_convert("UTC").tz_localize(None).normalize()

def _as_date_index(df):
    out = df.copy()
    out.index = _to_date_index(out.index)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    return out

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
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        out[c] = pd.to_numeric(s.squeeze(), errors="coerce")
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in out.columns]
    if not keep:
        keep = out.select_dtypes(include="number").columns.tolist()
    return _as_date_index(out[keep])

# ---------- UI ----------
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker", "TSLA").strip().upper()
with col2:
    period = st.selectbox("History window", ["1y","2y","5y","max"], index=2)

if st.button("Predict"):
    # 1) Load & indicators
    raw = load_prices_yf(ticker, period=period)
    if raw.empty:
        st.error("No price data returned for that ticker/period."); st.stop()

    feats_df = compute_indicators(raw)
    feats_df = _as_date_index(feats_df)

    # If compute_indicators dropped everything (edge case), fallback minimally
    if feats_df.empty:
        px = pd.to_numeric(raw.get("Adj Close", raw.select_dtypes("number").iloc[:,0]), errors="coerce")
        tmp = pd.DataFrame(index=raw.index)
        tmp["ret_1d"] = px.pct_change()
        # placeholders retain some variance via ret_1d
        tmp["sma_ratio_10_20"] = 1.0
        tmp["vol_20"] = 0.0
        tmp["rsi_14"] = 50.0
        tmp["macd"] = 0.0
        tmp["macd_signal"] = 0.0
        tmp["vol_z20"] = 0.0
        feats_df = _as_date_index(tmp.fillna(0.0))

    # 2) Load artifacts
    try:
        model = joblib.load("artifacts/model.pkl")
        features = joblib.load("artifacts/features.pkl")
    except Exception as e:
        st.error(f"Could not load artifacts: {e}"); st.stop()

    # Split features by type (so we only fill event/text)
    core_feats  = [f for f in ["sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal","vol_z20","ret_1d"] if f in features]
    event_feats = [f for f in ["news_vol_z20","sent_jump_z20","is_event_day"] if f in features]
    text_feats  = [f for f in features if f.startswith("topic_") or f.startswith("kw_")]

    # 3) Rebuild simple event features
    for c in ("Daily_Sentiment","News_Volume"):
        if c not in feats_df.columns: feats_df[c] = 0.0
    def zscore(s, w=20): return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)
    feats_df["news_vol_z20"]  = zscore(feats_df["News_Volume"])
    feats_df["sent_jump"]     = (feats_df["Daily_Sentiment"] - feats_df["Daily_Sentiment"].shift(1)).fillna(0.0)
    feats_df["sent_jump_z20"] = zscore(feats_df["sent_jump"])
    feats_df["is_event_day"]  = ((feats_df["news_vol_z20"] > 1.5) | (feats_df["sent_jump_z20"] > 1.5)).astype(int)

    # 4) Ensure all expected columns exist
    for c in features:
        if c not in feats_df.columns: feats_df[c] = 0.0

    # Build X:
    X = feats_df[features].copy()

    # Only fill event/text features; keep technicals unfilled to preserve variance
    if event_feats: X[event_feats] = X[event_feats].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    if text_feats:  X[text_feats]  = X[text_feats].replace([np.inf,-np.inf], np.nan).fillna(0.0)

    # Drop rows missing any core technicals
    if core_feats:
        X = X.dropna(subset=core_feats)

    # If still empty, last fallback: drop only rows where *all* core are NaN, then fill forward
    if X.empty and core_feats:
        core_only = feats_df[core_feats].replace([np.inf,-np.inf], np.nan)
        mask = ~core_only.isna().all(axis=1)
        X = feats_df.loc[mask, features].copy()
        X[core_feats] = X[core_feats].fillna(method="ffill")
        X[event_feats + text_feats] = X[event_feats + text_feats].fillna(0.0)
        X = X.dropna(subset=core_feats)
    # Final guard
    if X.empty:
        st.error("No usable rows found. Try a longer history or a different ticker.")
        st.stop()

    # 5) Sanity check variance (flat features -> flat preds)
    nunq = X.nunique()
    if int(nunq.sum()) <= len(nunq):  # each column has <=1 unique value
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

    # Latest point
    latest_dt = X.index[-1]
    latest_pred = float(preds[-1])
    try: label_date = latest_dt.date()
    except Exception: label_date = str(latest_dt)
    st.metric(label=f"Latest predicted next-day return ({label_date})", value=f"{latest_pred:.2%}")

    # Debug panel
    with st.expander("Debug info"):
        st.write("Feature rows:", X.shape[0], "Feature cols:", X.shape[1])
        st.write("Missing feature columns:", sorted(set(features) - set(feats_df.columns)))
        st.write("Per-feature nunique (first 15):", nunq.sort_values().head(15))
        try:
            coef = getattr(model, "coef_", None)
            if coef is not None and len(coef)==len(features):
                top = pd.Series(coef, index=features).abs().sort_values(ascending=False).head(10)
                st.write("Top |coef| (Ridge):", top)
            else:
                fi = getattr(model, "feature_importances_", None)
                if fi is not None and len(fi)==len(features):
                    top = pd.Series(fi, index=features).sort_values(ascending=False).head(10)
                    st.write("Top feature importances (RF):", top)
        except Exception: pass
