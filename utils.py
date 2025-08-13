import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def _one_series(frame: pd.DataFrame, name: str) -> pd.Series:
        """Return a numeric 1D Series for the given column name, even if duplicates exist."""
        if name not in frame.columns:
            raise KeyError(name)
        obj = frame.loc[:, [name]]
        s = pd.to_numeric(obj.iloc[:, 0], errors="coerce")
        s.name = name
        return s

    # choose price column
    price_col = None
    for candidate in ["Adj Close", "AdjClose", "adj close", "adjclose", "Close", "close"]:
        if candidate in df.columns:
            price_col = candidate
            break
    if price_col is None:
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            raise ValueError("No numeric price column available in DataFrame.")
        price_col = num_cols[0]

    px = _one_series(df, price_col).astype(float)

    # basics
    df["ret_1d"] = px.pct_change()
    df["sma_10"] = px.rolling(10).mean()
    df["sma_20"] = px.rolling(20).mean()
    df["sma_ratio_10_20"] = df["sma_10"] / (df["sma_20"] + 1e-12)

    # volatility
    df["vol_20"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)

    # RSI(14)
    delta = px.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=df.index).rolling(14).mean()
    roll_down = pd.Series(loss, index=df.index).rolling(14).mean()
    rs = roll_up / (roll_down + 1e-12)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = px.ewm(span=12, adjust=False).mean()
    ema26 = px.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # volume z-score
    if "Volume" in df.columns:
        try:
            vol = _one_series(df, "Volume")
        except KeyError:
            vol = pd.Series(0.0, index=df.index)
        vol_mean = vol.rolling(20).mean()
        vol_std  = vol.rolling(20).std()
        df["vol_z20"] = (vol - vol_mean) / (vol_std + 1e-12)
    else:
        df["vol_z20"] = 0.0

    return df.dropna()



def train_test_split_chrono(df, test_frac=0.2):
    k = max(1, int(len(df) * test_frac))
    return df.iloc[:-k].copy(), df.iloc[-k:].copy()

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def ci_from_residuals(y_true, y_pred, alpha=0.2):
    resid = (np.asarray(y_true) - np.asarray(y_pred))
    lo = float(np.quantile(resid, alpha/2))
    hi = float(np.quantile(resid, 1 - alpha/2))
    return lo, hi
