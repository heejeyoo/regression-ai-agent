import os, joblib, numpy as np, pandas as pd, yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from utils import compute_indicators, train_test_split_chrono, rmse
from utils_text import build_daily_text_features

def load_prices_yf(ticker, start, end):
    """
    Download with yfinance and return a DataFrame with numeric columns:
    Open, High, Low, Close, Adj Close, Volume (any subset that exists).
    Handles MultiIndex, ticker prefixes/suffixes, odd spellings, and coerces to numeric.
    """
    import pandas as _pd, numpy as _np, yfinance as _yf

    df = _yf.download(ticker, start=start, end=end, auto_adjust=False)
    if df is None or df.empty:
        return df

    # If MultiIndex, try to collapse to field names
    if isinstance(df.columns, _pd.MultiIndex):
        # Prefer a level that looks like field names
        fields = {"Open","High","Low","Close","Adj Close","Volume","AdjClose"}
        best_level = None
        for lvl in range(df.columns.nlevels):
            lvl_vals = set(map(str, df.columns.get_level_values(lvl)))
            if len(fields.intersection(lvl_vals)) >= 3:
                best_level = lvl; break
        if best_level is not None:
            df = df.copy()
            df.columns = df.columns.get_level_values(best_level)
        else:
            # Flatten: pick any token that matches a field; else last token
            new_cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    pick = None
                    for tok in c:
                        s = str(tok)
                        if s in fields:
                            pick = s; break
                    new_cols.append(pick if pick else str(c[-1]))
                else:
                    new_cols.append(str(c))
            df = df.copy()
            df.columns = new_cols
    else:
        df = df.copy()
        df.columns = list(map(str, df.columns))

    # Canonicalize names (case/spacing/underscore-insensitive)
    canon = {}
    for c in df.columns:
        cl = c.strip().lower().replace("_"," ").replace("-"," ")
        if cl in {"open"}: canon[c] = "Open"
        elif cl in {"high"}: canon[c] = "High"
        elif cl in {"low"}: canon[c] = "Low"
        elif cl in {"close"}: canon[c] = "Close"
        elif cl in {"adj close","adjclose","adjusted close"}: canon[c] = "Adj Close"
        elif cl in {"volume","vol"}: canon[c] = "Volume"
        else:
            # Try to strip ticker prefixes/suffixes like "TSLA Open" or "Open TSLA"
            tokens = cl.split()
            mapped = None
            for t in tokens:
                if t in {"open","high","low","close","volume"}:
                    mapped = t.capitalize(); break
                if t in {"adj","adjusted"}:
                    mapped = "Adj Close"  # best effort
            canon[c] = mapped if mapped else c  # leave as-is if unknown
    df = df.rename(columns=canon)

    # Keep only the columns we know how to use (but don't crash if few exist)
    wanted = ["Open","High","Low","Close","Adj Close","Volume"]
    have = [c for c in wanted if c in df.columns]
    if not have:
        # last resort: keep any numeric columns
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        df = df[num_cols].copy()
    else:
        df = df[have].copy()

    # Create Adj Close from Close if missing
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Ensure Volume exists (fill zeros) so downstream code is happy
    if "Volume" not in df.columns:
        df["Volume"] = 0.0

    # Coerce all to numeric and drop rows where all prices are NaN
    for c in df.columns:
        df[c] = _pd.to_numeric(df[c], errors="coerce")
    price_cols = [c for c in ["Adj Close","Close","Open","High","Low"] if c in df.columns]
    df = df.dropna(subset=price_cols, how="all")

    # Final order (subset of wanted that exist)
    order = [c for c in wanted if c in df.columns]
    df = df[order]
    return df



def normalize_yf_columns(df):
    """
    Return a DataFrame with columns: Open, High, Low, Close, Adj Close, Volume.
    Handles yfinance returning MultiIndex or ticker-grouped columns.
    """
    import pandas as _pd

    if df is None or df.empty:
        return df

    # If MultiIndex, try to collapse to field names
    if isinstance(df.columns, _pd.MultiIndex):
        # Try: if one level contains the 6 field names, use that level
        fields = {"Open","High","Low","Close","Adj Close","Volume"}
        # Check level 0 and level 1
        for lvl in range(df.columns.nlevels):
            lvl_vals = set(df.columns.get_level_values(lvl))
            if fields.issubset(lvl_vals) or len(fields.intersection(lvl_vals)) >= 4:
                df = df.copy()
                df.columns = df.columns.get_level_values(lvl)
                break
        else:
            # Fallback: flatten by picking the *field* token from each tuple
            new_cols = []
            for c in df.columns:
                if isinstance(c, tuple):
                    # take the token that matches a known field, else last token
                    pick = None
                    for tok in c:
                        if tok in {"Open","High","Low","Close","Adj Close","Volume"}:
                            pick = tok; break
                    new_cols.append(pick if pick else c[-1])
                else:
                    new_cols.append(c)
            df = df.copy()
            df.columns = new_cols

    # If any columns still look like "TSLA Open" or "Open TSLA", strip to the field name
    def _to_field(name):
        s = str(name)
        for f in ["Open","High","Low","Close","Adj Close","Volume","AdjClose","Adj close","Adj_close"]:
            if s == f or s.endswith(" " + f) or s.startswith(f + " ") or f in s.split("/"):
                return "Adj Close" if f.lower().replace(" ","").replace("_","")== "adjclose" else f
        return s
    df = df.copy()
    df.columns = [_to_field(c) for c in df.columns]

    # If 'Adj Close' missing, mirror from 'Close'
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    # Finally, subset to expected columns if present
    needed = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    if len(needed) < 4:
        # As a safety, try lower/variant spellings
        rename_map = {}
        for c in df.columns:
            cl = str(c).lower().replace("_"," ").strip()
            if cl == "adjclose" or cl == "adj close":
                rename_map[c] = "Adj Close"
            elif cl in {"open","high","low","close","volume"}:
                rename_map[c] = cl.capitalize()
        if rename_map:
            df = df.rename(columns=rename_map)
            needed = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]

    # Keep only the needed columns (in order), drop others
    df = df[needed].copy()
    return df



# ---- Params
TICKER = os.getenv("TICKER", "TSLA")
START  = os.getenv("START", "2020-01-01")
END    = os.getenv("END",   "2025-08-12")
AMERICA_TZ = "America/New_York"
USE_TEXT = True
N_TOPICS = 20
NEWS_CSV = os.getenv("NEWS_CSV", "")  # CSV with columns: publishedAt,title,url (optional)
np.random.seed(42)

# ---- Load news (optional)
articles = None
if USE_TEXT and NEWS_CSV and os.path.exists(NEWS_CSV):
    try:
        articles = pd.read_csv(NEWS_CSV)
        print(f"Loaded {len(articles)} articles from {NEWS_CSV}")
    except Exception as e:
        print("Couldn't load NEWS_CSV:", e)

# ---- Prices
print(f"Downloading {TICKER} {START}→{END}…")
px = load_prices_yf(TICKER, START, END)
# Flatten MultiIndex columns if present
if isinstance(px.columns, pd.MultiIndex):
    try:
        px.columns = px.columns.droplevel(0)
    except Exception:
        px.columns = [" ".join([str(c) for c in col if c]) for col in px.columns]

# If 'Adj Close' is missing (some data sources), mirror from 'Close'
if "Adj Close" not in px.columns and "Close" in px.columns:
    px["Adj Close"] = px["Close"]

px.index.name = "Date"
df = compute_indicators(px)

# ---- Text features
vec = None; svd = None
if USE_TEXT and articles is not None and len(articles) > 0:
    text_daily, vec, svd, _ = build_daily_text_features(articles, n_topics=N_TOPICS)
    text_daily.index = pd.to_datetime(text_daily.index).tz_localize(AMERICA_TZ)
    df = df.join(text_daily, how="left")
else:
    print("No text CSV provided — continuing with price-only + event scaffolding.")

# ---- Sentiment/volume placeholders (if absent)
for c in ["Daily_Sentiment","News_Volume"]:
    if c not in df.columns:
        df[c] = 0.0

# ---- Event flags from z-scores
def zscore(s, w=20):
    return (s - s.rolling(w).mean()) / (s.rolling(w).std() + 1e-9)
df["news_vol_z20"] = zscore(df["News_Volume"])
df["sent_jump"]    = (df["Daily_Sentiment"] - df["Daily_Sentiment"].shift(1)).fillna(0)
df["sent_jump_z20"]= zscore(df["sent_jump"])
df["is_event_day"] = ((df["news_vol_z20"] > 1.5) | (df["sent_jump_z20"] > 1.5)).astype(int)

# ---- Target & features
df["ret_1d"] = df["Adj Close"].pct_change()
df["target_next_ret"] = df["ret_1d"].shift(-1)

base_feats = [
    "sma_ratio_10_20","vol_20","rsi_14","macd","macd_signal","vol_z20","ret_1d",
    "news_vol_z20","sent_jump_z20","is_event_day",
]
topic_feats = [c for c in df.columns if c.startswith("topic_")]
kw_feats    = [c for c in df.columns if c.startswith("kw_")]
all_feats   = base_feats + topic_feats + kw_feats

df = df.dropna(subset=["target_next_ret"]).copy()
X = df[all_feats].fillna(0.0).values
y = df["target_next_ret"].values

# ---- CV
tscv = TimeSeriesSplit(n_splits=5)
def eval_cv(model):
    maes, rmses, r2s = [], [], []
    for tr, va in tscv.split(X):
        Xt, Xv, yt, yv = X[tr], X[va], y[tr], y[va]
        model.fit(Xt, yt)
        p = model.predict(Xv)
        maes.append(mean_absolute_error(yv, p))
        from utils import rmse as _rmse
        rmses.append(_rmse(yv, p))
        r2s.append(r2_score(yv, p))
    import numpy as _np
    return float(_np.mean(maes)), float(_np.mean(rmses)), float(_np.mean(r2s))

models = {
    "ridge": Ridge(alpha=1.0, random_state=42),
    "rf":    RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_leaf=2, n_jobs=-1, random_state=42),
}
cv = {name: dict(zip(["MAE","RMSE","R2"], eval_cv(mdl))) for name, mdl in models.items()}
print("CV:", cv)

# ---- Hold-out
train_df, test_df = train_test_split_chrono(df, test_frac=0.2)
Xtr, ytr = train_df[all_feats].fillna(0.0).values, train_df["target_next_ret"].values
Xte, yte = test_df[all_feats].fillna(0.0).values, test_df["target_next_ret"].values

best = min(cv.items(), key=lambda kv: kv[1]["MAE"])[0]
final_model = models[best].fit(Xtr, ytr)
pred = final_model.predict(Xte)

from utils import rmse as RMSE
test_metrics = dict(MAE=float(np.mean(np.abs(yte-pred))), RMSE=RMSE(yte, pred), R2=float(r2_score(yte, pred)))
print("Final:", best, "Hold-out:", test_metrics)

# ---- Save artifacts
os.makedirs("artifacts", exist_ok=True)
joblib.dump(final_model, "artifacts/model.pkl")
joblib.dump(all_feats, "artifacts/features.pkl")
joblib.dump({"vectorizer":vec, "svd":svd}, "artifacts/text_pipeline.pkl")
pd.DataFrame(cv).T.to_csv("artifacts/cv_results.csv")
pd.Series(test_metrics).to_csv("artifacts/test_metrics.csv")
print("Saved artifacts/.")
