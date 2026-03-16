import datetime as dt
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False, ttl=60 * 60)
def download_history(
    tickers: Tuple[str, ...],
    benchmark: str,
    years: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Download multi-asset history (Close, Volume) for given tickers and benchmark.

    Returns a dict with keys:
      - 'close': DataFrame of close prices (columns = symbols)
      - 'volume': DataFrame of volumes (columns = symbols)
      - 'benchmark_close': Series of benchmark close prices
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * years + 10)

    symbols = list(dict.fromkeys([*tickers, benchmark]))
    df = yf.download(
        symbols,
        start=start,
        end=end + dt.timedelta(days=1),
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )

    if df.empty:
        return {"close": pd.DataFrame(), "volume": pd.DataFrame(), "benchmark_close": pd.Series(dtype=float)}

    if not isinstance(df.columns, pd.MultiIndex):
        # Single symbol case – normalize to MultiIndex
        df = pd.concat({symbols[0]: df}, axis=1)
        df.columns = pd.MultiIndex.from_product([df.columns.get_level_values(1), [symbols[0]]])

    df = df.sort_index()
    df.index = pd.to_datetime(df.index).tz_localize(None)

    close = df["Close"].copy() if "Close" in df.columns.levels[0] else pd.DataFrame()
    volume = df["Volume"].copy() if "Volume" in df.columns.levels[0] else pd.DataFrame()

    # Reindex to requested symbols
    close = close.reindex(columns=symbols)
    volume = volume.reindex(columns=symbols)

    bm_series = close[benchmark].dropna().ffill() if benchmark in close.columns else pd.Series(dtype=float)

    return {
        "close": close,
        "volume": volume,
        "benchmark_close": bm_series,
    }

