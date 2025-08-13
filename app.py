# app.py — correctness-first with robust fallbacks (no empty X)

import streamlit as st
import pandas as pd
import numpy as np
import joblib, yfinance as yf
from utils import compute_indicators  # your 1-D safe version

st.set_page_config(page_title="AI Stock Predictor — Diagnostics", layout="wide")
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

def load_prices_yf(ticker: str, period: str = "5y"):
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()

    # Flatten/normalize columns
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
        if isinstance(s, pd.DataFrame): s = s.iloc[:,0]
        out[c] = pd.to_numeric(s.squeeze(), errors="coerce")

    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in out.columns]
    if not keep: keep = out.select_dtypes(include="number").columns.tolist()
    return _as_date_index(out[keep])

def build_indicators_lenient(raw: pd.DataFrame) -> pd.DataFrame:
    """EWMs + min_periods=2 so we always get varying technicals."""
    df = raw.copy()
    px = df.get("Adj Close")
    if px is None: px = df.select_dtypes(include="number").iloc[:,0]
    px = pd.to_numeric(px, errors="coerce").ffill().bfill()

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

def add_event_scaffolding(df):
    df = df.copy()
    for c in ("Daily_Sentiment","News_Volume"):
        if c not in df.columns: df[c] = 0.0
    def zscore(s, w=20): return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)
    df["news_vol_z20"]  = zscore(df["News_Volume"]).fillna(0.0)
    df["sent_jump"]     = (df["Daily_Sentiment"] - df["Daily_Sentiment"].shift(1)).fillna(0.0)
    df["sent_jump_z20"] = zscore(df["sent_jump"]).fillna(0.0)
    df["is_event_day"]  = ((df["news_vol_z20"] > 1.5) | (df["sent_jump_z20"] > 1.5)).astype(int)
    return df

def realized_next_return(price_series: pd.Series) -> pd.Series:
    return price_series.pct_change().shift(-1)

# ---------- UI ----------
col1, col2, col3 = st.columns([2,1,1])
with col1:
    ticker = st.text_input("Ticker", "TSLA").strip().upper()
with col2:
    period = st.selectbox("History window", ["1y","2y","5y","max"], index=2)
with col3:
    show_debug = st.toggle("Show diagnostics", value=True)

if st.button("Predict"):
    # (1) Pull
    raw = load_prices_yf(ticker, period=period)
    if raw.empty:
        st.error("No price data returned for that ticker/period."); st.stop()

    # (2) Indicators (strict then lenient fallback)
    strict = compute_indicators(raw)
    lenient = build_indicators_lenient(raw)
    # choose more variable core feature set
    core = ["sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal","vol_z20","ret_1d"]
    def varscore(df):
        cols = [c for c in core if c in df.columns]
        return -1.0 if df.empty or not cols else float(pd.DataFrame({c: df[c].astype(float) for c in cols}).std(ddof=0).sum())
    feats_df = strict if varscore(strict) >= varscore(lenient) else lenient

    # (3) Event scaffolding
    feats_df = add_event_scaffolding(feats_df)

    # (4) Artifacts & feature matrix
    try:
        model = joblib.load("artifacts/model.pkl")
        features = joblib.load("artifacts/features.pkl")
    except Exception as e:
        st.error(f"Could not load artifacts: {e}"); st.stop()

    for c in features:
        if c not in feats_df.columns: feats_df[c] = 0.0

    X = feats_df[features].copy().astype(float, errors="ignore").replace([np.inf,-np.inf], np.nan)

    # Only impute event/text; keep technicals as real as possible
    event_feats = [f for f in ["news_vol_z20","sent_jump_z20","is_event_day"] if f in X.columns]
    text_feats  = [f for f in X.columns if f.startswith("topic_") or f.startswith("kw_")]
    if event_feats: X[event_feats] = X[event_feats].fillna(0.0)
    if text_feats:  X[text_feats]  = X[text_feats].fillna(0.0)

    core_feats = [f for f in core if f in X.columns]
    if core_feats:
        # Mild strategy: ffill/bfill only for core, then drop rows still all-NaN
        X[core_feats] = X[core_feats].ffill().bfill()
        mask = ~X[core_feats].isna().all(axis=1)
        X = X.loc[mask]

    # Final guard: if still empty, predict on latest row after full fill
    if X.empty:
        last_row = feats_df.iloc[[-1]][features].copy()
        last_row = last_row.replace([np.inf,-np.inf], np.nan).ffill(axis=1).bfill(axis=1).fillna(0.0)
        X = last_row

    # (5) Predict
    preds = model.predict(X)

    # (6) Align for display — robust as-of + daily resample fallbacks
    price_col = next((c for c in ["Adj Close","Close","Open"] if c in raw.columns), None)
    
    # default: empty series on X's index
    price_on_X = pd.Series(index=X.index, dtype=float)
    
    if price_col:
        # Ensure both sides are clean date indexes
        raw_price = raw[price_col].astype(float).copy()
        raw_price.index = pd.to_datetime(raw_price.index, errors="coerce").tz_localize(None).normalize()
    
        x_idx = pd.to_datetime(X.index, errors="coerce").tz_localize(None).normalize()
    
        # 1) Try simple exact-date reindex + fills
        tmp = raw_price.reindex(x_idx).ffill().bfill()
    
        if tmp.dropna().empty:
            # 2) As-of merge (maps each X date to the most recent raw date <= X date)
            ps = raw_price.reset_index()
            ps.columns = ["Date", "Price"]
            xi = pd.DataFrame({"Date": x_idx})
            asof = pd.merge_asof(
                xi.sort_values("Date"),
                ps.sort_values("Date"),
                on="Date",
                direction="backward",
                tolerance=pd.Timedelta("14D"),
            )
            asof.set_index("Date", inplace=True)
            price_on_X = asof["Price"].reindex(x_idx)
    
            if price_on_X.dropna().empty:
                # 3) Last fallback: daily resample the raw series and align
                daily = raw_price.asfreq("D").ffill().bfill()
                price_on_X = daily.reindex(x_idx).ffill().bfill()
        else:
            price_on_X = tmp
    
    # Realized next-day return from the aligned price series
    realized = price_on_X.pct_change().shift(-1)


    # (7) Display
    st.subheader(f"Prediction for {ticker}")
    chart = pd.DataFrame(index=X.index)
    if not price_on_X.dropna().empty: chart["Price"] = price_on_X
    chart["Predicted next-day return"] = preds
    st.line_chart(chart)

    latest_dt = X.index[-1]
    latest_pred = float(preds[-1])
    try: label_date = latest_dt.date()
    except Exception: label_date = str(latest_dt)
    st.metric(label=f"Latest predicted next-day return ({label_date})", value=f"{latest_pred:.2%}")

    # (8) Diagnostics
    if show_debug:
        st.subheader("Diagnostics 🔍")

        diag_alignment = {
            "raw_rows": int(len(raw)),
            "strict_rows": int(len(strict)),
            "lenient_rows": int(len(lenient)),
            "chosen_rows": int(len(feats_df)),
            "X_rows": int(len(X)),
            "first_X": str(X.index.min().date()) if len(X) else None,
            "last_X": str(X.index.max().date()) if len(X) else None,
            "price_aligned_non_na": int(price_on_X.notna().sum()),
            "varscore_strict": varscore(strict),
            "varscore_lenient": varscore(lenient),
        }
        st.markdown("**Alignment & variance summary**")
        st.write(diag_alignment)

        missing = sorted(set(features) - set(feats_df.columns))
        extra   = sorted(set(feats_df.columns) - set(features))
        st.markdown("**Feature coverage**")
        st.write({
            "expected_features": int(len(features)),
            "missing_in_current": int(len(missing)),
            "extra_in_current": int(len(extra)),
        })
        if missing:
            st.caption("Missing (first 20): " + ", ".join(missing[:20]))

        st.markdown("**NaN report on X (first 15 with NaNs)**")
        nan_counts = X.isna().sum()
        nan_counts = nan_counts[nan_counts > 0].sort_values(ascending=False)
        if nan_counts.empty:
            st.write("No NaNs in X after imputing.")
        else:
            st.write(nan_counts.head(15))

        st.markdown("**Prediction summary**")
        st.write({
            "pred_mean": float(np.mean(preds)),
            "pred_std": float(np.std(preds)),
            "pred_min": float(np.min(preds)),
            "pred_max": float(np.max(preds)),
        })

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
