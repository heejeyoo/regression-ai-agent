# app.py — diagnostics-safe version (everything scoped inside Predict)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from utils import compute_indicators  # your 1-D safe version

st.set_page_config(page_title="AI Stock Predictor (Diagnostics)", layout="wide")
st.title("📈 AI Stock Predictor — Diagnostics")
st.caption("Educational demo — not financial advice.")

# ---------- Helpers ----------
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

    # Flatten / normalize columns
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

    # Coerce numeric, force 1-D
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

def add_event_scaffolding(df):
    df = df.copy()
    for c in ("Daily_Sentiment", "News_Volume"):
        if c not in df.columns:
            df[c] = 0.0
    def zscore(s, w=20):
        return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)
    df["news_vol_z20"]  = zscore(df["News_Volume"])
    df["sent_jump"]     = (df["Daily_Sentiment"] - df["Daily_Sentiment"].shift(1)).fillna(0.0)
    df["sent_jump_z20"] = zscore(df["sent_jump"])
    df["is_event_day"]  = ((df["news_vol_z20"] > 1.5) | (df["sent_jump_z20"] > 1.5)).astype(int)
    return df

def realized_next_return(price_series: pd.Series) -> pd.Series:
    return price_series.pct_change().shift(-1)

# ---------- UI ----------
col1, col2, col3 = st.columns([2,1,1])
with col1:
    ticker = st.text_input("Ticker", "TSLA").strip().upper()
with col2:
    period = st.selectbox("History window", ["1y","2y","5y","max"], index=1)
with col3:
    show_debug = st.toggle("Show diagnostics", value=True)

if st.button("Predict"):
    # (1) Pull data
    raw = load_prices_yf(ticker, period=period)
    if raw.empty:
        st.error("No price data returned for that ticker/period.")
        st.stop()

    if show_debug:
        st.subheader("Data Source ✅")
        st.write(f"{ticker} {period} | rows: {len(raw)} | cols: {list(raw.columns)}")
        st.write(f"Dates: {raw.index.min().date()} → {raw.index.max().date()}")
        st.write("Dtypes:", {k:str(v) for k,v in raw.dtypes.items()})

    # (2) Indicators
    ind = compute_indicators(raw)
    ind = _as_date_index(ind)

    if show_debug:
        st.subheader("Indicators ✅")
        core = [c for c in ["ret_1d","sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal","vol_z20"] if c in ind.columns]
        st.write(f"Indicator rows: {len(ind)} | core present: {core}")
        st.write("Head:", ind[core].head(3) if core else ind.head(3))
        st.write("Tail:", ind[core].tail(3) if core else ind.tail(3))

    # (3) Event scaffolding
    ind = add_event_scaffolding(ind)

    # (4) Load artifacts & build X
    try:
        model = joblib.load("artifacts/model.pkl")
        features = joblib.load("artifacts/features.pkl")
    except Exception as e:
        st.error(f"Could not load artifacts: {e}")
        st.stop()

    for c in features:
        if c not in ind.columns:
            ind[c] = 0.0

    X = (ind[features]
         .astype(float, errors="ignore")
         .replace([np.inf, -np.inf], np.nan))

    event_feats = [f for f in ["news_vol_z20","sent_jump_z20","is_event_day"] if f in X.columns]
    text_feats  = [f for f in X.columns if f.startswith("topic_") or f.startswith("kw_")]
    if event_feats:
        X[event_feats] = X[event_feats].fillna(0.0)
    if text_feats:
        X[text_feats]  = X[text_feats].fillna(0.0)

    core_feats = [f for f in ["sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal","vol_z20","ret_1d"] if f in X.columns]
    if core_feats:
        mask = ~X[core_feats].isna().all(axis=1)
        X = X.loc[mask]
    X = X.fillna(method="ffill").fillna(0.0)

    if X.empty:
        st.error("No rows with complete feature data to predict (after alignment/imputing). Try a longer history.")
        st.stop()

    # (5) Predict
    preds = model.predict(X)

    # (6) Align price to X index for display
    price_col = next((c for c in ["Adj Close","Close","Open"] if c in raw.columns), None)
    if price_col:
        price_on_X = raw[price_col].astype(float).reindex(X.index).fillna(method="ffill").fillna(method="bfill")
    else:
        price_on_X = pd.Series(index=X.index, dtype=float)
    realized = realized_next_return(price_on_X)

    # (7) Display
    st.subheader(f"Prediction for {ticker}")
    chart = pd.DataFrame(index=X.index)
    if not price_on_X.dropna().empty:
        chart["Price"] = price_on_X
    chart["Predicted next-day return"] = preds
    st.line_chart(chart)

    latest_dt = X.index[-1]
    latest_pred = float(preds[-1])
    try:
        label_date = latest_dt.date()
    except Exception:
        label_date = str(latest_dt)
    st.metric(label=f"Latest predicted next-day return ({label_date})", value=f"{latest_pred:.2%}")

    # (8) Diagnostics (inside the same scope → no NameError)
    if show_debug:
        st.subheader("Diagnostics 🔍")

        # Alignment summary
        diag_alignment = {
            "raw_rows": int(len(raw)),
            "ind_rows": int(len(ind)),
            "X_rows": int(len(X)),
            "first_X": str(X.index.min().date()) if len(X) else None,
            "last_X": str(X.index.max().date()) if len(X) else None,
            "price_aligned_non_na": int(price_on_X.notna().sum()),
        }
        st.markdown("**Alignment summary**")
        st.write(diag_alignment)

        # Feature coverage
        missing = sorted(set(features) - set(ind.columns))
        extra   = sorted(set(ind.columns) - set(features))
        diag_features = {
            "expected_features": int(len(features)),
            "missing_in_current": int(len(missing)),
            "extra_in_current": int(len(extra)),
        }
        st.markdown("**Feature coverage**")
        st.write(diag_features)
        if missing:
            st.caption("Missing (first 20): " + ", ".join(missing[:20]))

        # NaN report
        st.markdown("**NaN report on X (first 15 with NaNs)**")
        nan_counts = X.isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        if nan_counts.empty:
            st.write("No NaNs in X after imputing.")
        else:
            st.write(nan_counts.head(15))

        # Prediction summary
        st.markdown("**Prediction summary**")
        pred_summary = {
            "pred_mean": float(np.mean(preds)),
            "pred_std": float(np.std(preds)),
            "pred_min": float(np.min(preds)),
            "pred_max": float(np.max(preds)),
        }
        st.write(pred_summary)

        # Debug table & download
        try:
            dbg = pd.DataFrame(
                {"Price": price_on_X, "Pred_next_ret": preds, "Realized_next_ret": realized},
                index=X.index
            )
            st.markdown("**Debug table (last 10 rows)**")
            st.dataframe(dbg.tail(10).style.format({
                "Price": "{:,.2f}",
                "Pred_next_ret": "{:.3%}",
                "Realized_next_ret": "{:.3%}",
            }))
            csv = dbg.to_csv(index=True).encode("utf-8")
            st.download_button("Download debug_export.csv", data=csv,
                               file_name="debug_export.csv", mime="text/csv")
        except Exception as e:
            st.info(f"Could not build debug table: {e}")
