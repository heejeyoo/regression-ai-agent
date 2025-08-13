
# utils.py - robust, 1-D safe helpers
import numpy as np
import pandas as pd

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical features on a price DataFrame, tolerating yfinance quirks:
    - Flattens MultiIndex columns
    - Deduplicates columns (keeps first)
    - Picks price/volume as 1-D Series even if duplicates exist
    Returns: ret_1d, sma_10, sma_20, sma_ratio_10_20, vol_20, rsi_14, macd,
             macd_signal, vol_z20 (plus original OHLCV columns).
    """
    df = df.copy()

    # 1) Normalize columns to simple, unique strings
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1).astype(str)
    else:
        df.columns = df.columns.astype(str)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    # 2) Helper: always return a 1-D numeric Series for the chosen column
    def _one_series(frame: pd.DataFrame, name_candidates) -> pd.Series:
        for name in name_candidates:
            if name in frame.columns:
                s = frame.loc[:, [name]].iloc[:, 0]  # force 1-D even if dupes
                s = pd.to_numeric(s.squeeze(), errors="coerce")
                s.name = name
                return s
        # Fallback: first numeric column
        num_cols = frame.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            raise ValueError("No numeric column available.")
        s = frame.loc[:, [num_cols[0]]].iloc[:, 0]
        return pd.to_numeric(s.squeeze(), errors="coerce")

    # 3) Price & Volume (1-D safe)
    px  = _one_series(df, ["Adj Close","AdjClose","adj close","adjclose","Close","close"]).astype(float)
    vol = _one_series(df, ["Volume","volume"]) if "Volume" in df.columns else pd.Series(0.0, index=df.index)

    # 4) Basics
    df["ret_1d"] = px.pct_change()
    df["sma_10"] = px.rolling(10, min_periods=10).mean()
    df["sma_20"] = px.rolling(20, min_periods=20).mean()
    df["sma_ratio_10_20"] = df["sma_10"] / (df["sma_20"] + 1e-12)

    # 5) Volatility (20d, annualized)
    df["vol_20"] = df["ret_1d"].rolling(20, min_periods=20).std() * np.sqrt(252)

    # 6) RSI(14) - use clip instead of np.where to avoid 2-D arrays
    delta = px.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    roll_up = gain.rolling(14, min_periods=14).mean()
    roll_down = loss.rolling(14, min_periods=14).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 7) MACD (12,26) + signal(9)
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # 8) Volume z-score (20d)
    vol_mean = vol.rolling(20, min_periods=20).mean()
    vol_std  = vol.rolling(20, min_periods=20).std()
    df["vol_z20"] = (vol - vol_mean) / (vol_std + 1e-12)

    return df.dropna()

def train_test_split_chrono(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split: last `test_frac` rows become test."""
    n = len(df)
    k = max(1, int(n * test_frac))
    return df.iloc[: n - k].copy(), df.iloc[n - k :].copy()

def rmse(y_true, y_pred) -> float:
    """Root-mean-squared error without sklearn dependency."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def ci_from_residuals(y_true, y_pred, alpha: float = 0.2):
    """
    Empirical prediction interval from residual quantiles.
    Returns (lo_residual, hi_residual) to add to a point prediction.
    """
    resid = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    lo = float(np.nanquantile(resid, alpha / 2))
    hi = float(np.nanquantile(resid, 1 - alpha / 2))
    return lo, hi
