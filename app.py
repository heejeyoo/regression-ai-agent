# app.py — robust non-flat predictions with local calibration fallback

import streamlit as st
import pandas as pd
import numpy as np
import joblib, yfinance as yf
from sklearn.linear_model import Ridge
from utils import compute_indicators  # strict technicals

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
    return out.sort_index()

def load_prices_yf(ticker: str, period: str = "5y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty: return pd.DataFrame()
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
        s = df[c]; 
        if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
        out[c] = pd.to_numeric(s.squeeze(), errors="coerce")
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in out.columns]
    if not keep: keep = out.select_dtypes(include="number").columns.tolist()
    return _as_date_index(out[keep])

def build_indicators_lenient(raw: pd.DataFrame) -> pd.DataFrame:
    """Always produces varying technicals with EWMs/min_periods=2."""
    df = raw.copy()
    px = df.get("Adj Close")
    if px is None: px = df.select_dtypes(include="number").iloc[:,0]
    px = pd.to_numeric(px, errors="coerce").fillna(method="ffill").fillna(method="bfill")

    df["ret_1d"] = px.pct_change().fillna(0.0)

    sma10 = px.rolling(10, min_periods=2).mean()
    sma20 = px.rolling(20, min_periods=2).mean()
    df["sma_ratio_10_20"] = (sma10 / (sma20 + 1e-12)).fillna(1.0)

    df["vol_20"] = df["ret_1d"].rolling(20, min_periods=2).std().fillna(0.0) * np.sqrt(252)

    delta = px.diff().fillna(0.0)
    gain = delta.clip(lower=0.0); loss = (-delta).clip(lower=0.0)
    roll_up = gain.ewm(alpha=1/14, adjust=False, min_periods=2).mean()
    roll_down = loss.ewm(alpha=1/14, adjust=False, min_periods=2).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    df["macd"] = (ema12 - ema26).fillna(0.0)
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean().fillna(0.0)

    vol = pd.to_numeric(df.get("Volume", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0)
    vmean = vol.rolling(20, min_periods=2).mean()
    vstd  = vol.rolling(20, min_periods=2).std()
    df["vol_z20"] = ((vol - vmean) / (vstd + 1e-12)).fillna(0.0)

    return _as_date_index(df)

def choose_more_variable(strict_df: pd.DataFrame, lenient_df: pd.DataFrame) -> pd.DataFrame:
    core = ["sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal","vol_z20","ret_1d"]
    def varscore(df):
        cols = [c for c in core if c in df.columns]
        if not cols or df.empty: return -1.0
        return float(pd.DataFrame({c: df[c].astype(float) for c in cols}).std(ddof=0).sum())
    return strict_df if varscore(strict_df) >= varscore(lenient_df) else lenient_df

# ---------- UI ----------
col1, col2 = st.columns([2,1])
with col1:
    ticker = st.text_input("Ticker", "TSLA").strip().upper()
with col2:
    period = st.selectbox("History window", ["1y","2y","5y","max"], index=2)

if st.button("Predict"):
    raw = load_prices_yf(ticker, period=period)
    if raw.empty:
        st.error("No price data returned for that ticker/period."); st.stop()

    # Strict first, lenient fallback; pick the more variable feature set
    strict = compute_indicators(raw)
    lenient = build_indicators_lenient(raw)
    feats_df = choose_more_variable(strict, lenient)

    # Load artifacts
    try:
        model = joblib.load("artifacts/model.pkl")
        features = joblib.load("artifacts/features.pkl")
    except Exception as e:
        st.error(f"Could not load artifacts: {e}"); st.stop()

    # Rebuild simple event features (only these get zero/ffill)
    for c in ("Daily_Sentiment","News_Volume"):
        if c not in feats_df.columns: feats_df[c] = 0.0
    def zscore(s, w=20): return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)
    feats_df["news_vol_z20"]  = zscore(feats_df["News_Volume"]).fillna(0.0)
    feats_df["sent_jump"]     = (feats_df["Daily_Sentiment"] - feats_df["Daily_Sentiment"].shift(1)).fillna(0.0)
    feats_df["sent_jump_z20"] = zscore(feats_df["sent_jump"]).fillna(0.0)
    feats_df["is_event_day"]  = ((feats_df["news_vol_z20"] > 1.5) | (feats_df["sent_jump_z20"] > 1.5)).astype(int)

    # Ensure expected columns
    for c in features:
        if c not in feats_df.columns: feats_df[c] = 0.0

    # Build X: keep technicals “as is”; only clean event/text
    core_feats  = [f for f in ["sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal","vol_z20","ret_1d"] if f in features]
    event_feats = [f for f in ["news_vol_z20","sent_jump_z20","is_event_day"] if f in features]
    text_feats  = [f for f in features if f.startswith("topic_") or f.startswith("kw_")]

    X = feats_df[features].copy()
    if event_feats: X[event_feats] = X[event_feats].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    if text_feats:  X[text_feats]  = X[text_feats].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    if core_feats:
        # drop rows with missing core, then gently ffill remaining holes
        X = X.dropna(subset=core_feats)
        if X.isna().any().any():
            X[core_feats] = X[core_feats].fillna(method="ffill")
    if X.empty:
        st.error("No usable rows found. Try a longer history or a different ticker."); st.stop()

    # Predict with the saved model
    preds = model.predict(X)
    flat = float(np.ptp(preds)) < 1e-8  # range ~0

    # ----- Local calibration fallback if preds are flat -----
    if flat:
        try:
            # Build a small local training set using the same features
            # Target = next-day return from raw Adj Close
            px = raw.get("Adj Close", raw.select_dtypes(include="number").iloc[:,0]).astype(float)
            ret_1d = px.pct_change()
            target = ret_1d.shift(-1)  # next-day return

            # Align target to X index and drop missing
            y = target.reindex(X.index)
            ok = y.notna()
            Xcal, ycal = X.loc[ok], y.loc[ok]

            # Use recent window to avoid overfitting (e.g., last 504 trading days ~ 2y)
            if len(Xcal) > 504:
                Xcal, ycal = Xcal.iloc[-504:], ycal.iloc[-504:]

            if len(Xcal) >= 50:  # need some data
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(Xcal, ycal)
                preds = ridge.predict(X)
                st.info("Predictions were flat; applied local Ridge calibration on recent data.")
        except Exception:
            pass

    # Align price to X index for chart
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
        nunq = X.nunique()
        st.write("Feature rows:", X.shape[0], "Feature cols:", X.shape[1])
        st.write("Flat-from-model:", flat, "| Pred range:", float(np.ptp(preds)))
        st.write("Per-feature nunique (min 12):", nunq.sort_values().head(12))
        core_in = [c for c in core_feats if c in X.columns]
        if core_in:
            st.write("Core feature std (sum):", float(X[core_in].std(ddof=0).sum()))
        try:
            coef = getattr(model, "coef_", None)
            if coef is not None and len(coef)==len(features):
                top = pd.Series(coef, index=features).abs().sort_values(ascending=False).head(10)
                st.write("Top |coef| (saved model):", top)
        except Exception:
            pass
