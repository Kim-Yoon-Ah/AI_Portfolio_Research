from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from data_engine import download_history
from feature_engine import build_feature_dataset
from forecast_model import ForecastModels, forecast_for_asset, train_forecast_models


def _get_or_train_models(
    yahoo_tickers: List[str],
    benchmark_symbol: str,
) -> tuple[ForecastModels | None, pd.DataFrame]:
    """
    Download history, build features, and train (or reuse) global models.
    """
    history = download_history(tuple(yahoo_tickers), benchmark_symbol, years=5)
    close = history["close"]
    volume = history["volume"]
    bm_close = history["benchmark_close"]

    features_df = build_feature_dataset(close, volume, bm_close)
    if features_df.empty:
        return None, pd.DataFrame()

    models = train_forecast_models(features_df)
    return models, features_df


def _ensure_forecast_state(
    yahoo_tickers: List[str],
    benchmark_symbol: str,
    *,
    force_refresh: bool = False,
) -> tuple[ForecastModels | None, pd.DataFrame]:
    if "forecast_models" not in st.session_state:
        st.session_state["forecast_models"] = None
    if "forecast_features" not in st.session_state:
        st.session_state["forecast_features"] = pd.DataFrame()
    if "forecast_signature" not in st.session_state:
        st.session_state["forecast_signature"] = None

    current_signature = (tuple(yahoo_tickers), benchmark_symbol)
    signature_changed = st.session_state.get("forecast_signature") != current_signature

    if force_refresh or signature_changed or st.session_state.get("forecast_models") is None:
        with st.spinner("Training market forecast models..."):
            models, features_df = _get_or_train_models(yahoo_tickers, benchmark_symbol)
            if models is None or features_df.empty:
                st.session_state["forecast_models"] = None
                st.session_state["forecast_features"] = pd.DataFrame()
                st.session_state["forecast_signature"] = current_signature
                return None, pd.DataFrame()

            st.session_state["forecast_models"] = models
            st.session_state["forecast_features"] = features_df
            st.session_state["forecast_signature"] = current_signature

    models = st.session_state.get("forecast_models")
    features_df = st.session_state.get("forecast_features", pd.DataFrame())
    return models, features_df


def get_market_forecast_results(
    yahoo_tickers: List[str],
    yahoo_to_display: Dict[str, str],
    benchmark_symbol: str,
    *,
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """
    Return forecast statistics for each available asset using the existing model pipeline.
    """
    if not yahoo_tickers:
        return []

    models, features_df = _ensure_forecast_state(
        yahoo_tickers,
        benchmark_symbol,
        force_refresh=force_refresh,
    )
    if models is None or features_df.empty:
        return []

    latest_rows = (
        features_df.sort_index()
        .groupby("asset", group_keys=False)
        .tail(1)
        .reset_index(drop=True)
        .set_index("asset")
    )

    results: List[Dict[str, Any]] = []
    for yt in yahoo_tickers:
        if yt not in latest_rows.index:
            continue

        row = latest_rows.loc[yt]
        stats = forecast_for_asset(models, row)
        results.append(
            {
                "symbol": yt,
                "display_name": yahoo_to_display.get(yt, yt),
                "stats": stats,
            }
        )

    return results


def render_market_forecast(
    yahoo_tickers: List[str],
    yahoo_to_display: Dict[str, str],
    benchmark_symbol: str,
) -> None:
    """
    Render the Market Forecast section in the dashboard.
    """
    st.divider()
    st.subheader("Market Forecast")
    st.caption("Probabilistic forecasts based on a global Random Forest model trained on 5 years of history.")

    if not yahoo_tickers:
        st.info("Enter at least one asset to see forecasts.")
        return

    run_forecast = st.button("Run Forecast", key="run_forecast")
    results = get_market_forecast_results(
        yahoo_tickers,
        yahoo_to_display,
        benchmark_symbol,
        force_refresh=run_forecast,
    )
    if not results:
        st.info("Click **Run Forecast** to train models and see forecasts.")
        return

    for result in results:
        display_name = result["display_name"]
        stats = result["stats"]
        st.markdown(f"**{display_name}**")

        one = stats["1d"]
        st.write(
            f"Next-day probability of gain: **{one['prob_positive'] * 100.0:.1f}%**  "
            f"(range: {one['low'] * 100.0:.2f}% to {one['high'] * 100.0:.2f}%)"
        )

        wk = stats["5d"]
        mo = stats["20d"]
        st.write(
            f"Weekly outlook (5 trading days): **{wk['exp_return'] * 100.0:.2f}%** expected return  \n"
            f"Monthly outlook (20 trading days): **{mo['exp_return'] * 100.0:.2f}%** expected return"
        )

        st.markdown("---")
