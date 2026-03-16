from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line


def build_feature_dataset(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark_close: pd.Series,
) -> pd.DataFrame:
    """
    Build a panel of features and forward-return targets for all assets.

    Output columns include:
      - asset, date
      - engineered features
      - future_return_1d, future_return_5d, future_return_20d
    """
    if close.empty:
        return pd.DataFrame()

    features: List[pd.DataFrame] = []

    bm_ret = benchmark_close.pct_change().rename("bm_ret")
    bm_vol20 = benchmark_close.pct_change().rolling(20).std().rename("bm_vol20")

    for sym in close.columns:
        px = close[sym].dropna()
        if px.empty:
            continue

        vol = volume.get(sym, pd.Series(index=px.index, dtype=float)).reindex(px.index)

        ret = px.pct_change()

        ma20 = px.rolling(20).mean()
        ma50 = px.rolling(50).mean()
        price_over_ma20 = px / ma20
        price_over_ma50 = px / ma50

        rsi14 = _rsi(px, window=14)
        macd, macd_signal = _macd(px)

        vol20 = ret.rolling(20).std()
        vol60 = ret.rolling(60).std()

        vol_ma20 = vol.rolling(20).mean()
        vol_change = vol.pct_change()

        df = pd.DataFrame(
            {
                "asset": sym,
                "price": px,
                "ret": ret,
                "ma20": ma20,
                "ma50": ma50,
                "price_over_ma20": price_over_ma20,
                "price_over_ma50": price_over_ma50,
                "rsi14": rsi14,
                "macd": macd,
                "macd_signal": macd_signal,
                "vol20": vol20,
                "vol60": vol60,
                "volume": vol,
                "volume_ma20": vol_ma20,
                "volume_change": vol_change,
            }
        )

        # Market context
        df["bm_ret"] = bm_ret.reindex(df.index)
        df["bm_vol20"] = bm_vol20.reindex(df.index)

        # Targets: forward returns
        df["future_return_1d"] = df["ret"].shift(-1)
        df["future_return_5d"] = df["price"].shift(-5) / df["price"] - 1.0
        df["future_return_20d"] = df["price"].shift(-20) / df["price"] - 1.0

        features.append(df)

    if not features:
        return pd.DataFrame()

    full = pd.concat(features).dropna(subset=["future_return_1d", "future_return_5d", "future_return_20d"])
    # Guard the model pipeline against inf values from divisions like pct_change or moving-average ratios.
    full = full.replace([np.inf, -np.inf], np.nan)
    # Drop rows with NaNs in features
    full = full.dropna()
    full = full.sort_index()
    return full

