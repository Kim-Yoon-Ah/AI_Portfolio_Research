import datetime as dt
import html
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import yfinance as yf

from forecast_dashboard import get_market_forecast_results


@dataclass(frozen=True)
class PortfolioConfig:
    tickers: list[str]
    start: dt.date
    end: dt.date
    initial_investment: float
    rebalance: str
    use_adj_close: bool
    weights: dict[str, float] | None


def _parse_tickers(raw: str) -> list[str]:
    typo_map = {
        "SLIVER": "SILVER",
    }
    tickers = [t.strip().upper() for t in raw.replace("\n", ",").split(",")]
    tickers = [t for t in tickers if t]
    seen: set[str] = set()
    out: list[str] = []
    for t in tickers:
        t = typo_map.get(t, t)
        key = t.casefold()
        if key not in seen:
            out.append(t)
            seen.add(key)
    return out


def _looks_like_symbol(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.^=-")
    return " " not in stripped and all(ch.upper() in allowed for ch in stripped)


def _missing_price_data_message() -> str:
    return (
        "This asset does not have market price data available through Yahoo Finance.\n\n"
        "Supported asset types include:\n"
        "- Stocks\n"
        "- ETFs\n"
        "- Market indices\n"
        "- Commodities\n\n"
        "Mutual funds typically use NAV data which may not be available through this data source."
    )


@st.cache_data(show_spinner=False, ttl=60 * 60)
def _search_yahoo_symbol(query: str, market: str) -> tuple[str | None, str | None]:
    if not hasattr(yf, "Search"):
        return None, None
    try:
        search = yf.Search(query=query, max_results=12, news_count=0)
        quotes = getattr(search, "quotes", []) or []
    except Exception:
        return None, None

    allowed_types = {"EQUITY", "ETF", "MUTUALFUND", "INDEX"}
    preferred_exchanges = {"India (NSE)": ("NSE", "BSE"), "US (NYSE/NASDAQ)": ("NYQ", "NMS", "NAS", "PCX", "ASE")}
    market_exchanges = preferred_exchanges.get(market, ())

    def _score(quote: dict) -> tuple[int, int, int]:
        quote_type = str(quote.get("quoteType", "")).upper()
        exchange = str(quote.get("exchange", "")).upper()
        symbol = str(quote.get("symbol", ""))
        exact_name = int(query.casefold() == str(quote.get("shortname", "")).casefold())
        exact_symbol = int(query.casefold() == symbol.casefold())
        type_score = int(quote_type in allowed_types)
        exchange_score = int(any(exchange.startswith(prefix) for prefix in market_exchanges)) if market_exchanges else 0
        return exact_symbol + exact_name, type_score, exchange_score

    sorted_quotes = sorted(quotes, key=_score, reverse=True)
    for quote in sorted_quotes:
        symbol = str(quote.get("symbol", "")).strip()
        if not symbol:
            continue
        display = str(quote.get("shortname") or quote.get("longname") or query).strip()
        return symbol, display
    return None, None


def _resolve_yahoo_symbol(asset: str, market: str) -> tuple[str, str]:
    cleaned = asset.strip()
    direct_aliases: dict[str, tuple[str, str]] = {
        "GOLD.C": ("GC=F", "GOLD"),
        "SILVER.C": ("SI=F", "SILVER"),
        "GOLD": ("GC=F", "GOLD"),
        "SILVER": ("SI=F", "SILVER"),
        "NSEI": ("^NSEI", "NSEI"),
        "NIFTY": ("^NSEI", "NSEI"),
        "NIFTY50": ("^NSEI", "NSEI"),
        "BSEI": ("^BSESN", "BSEI"),
        "BSESN": ("^BSESN", "BSEI"),
    }
    upper_cleaned = cleaned.upper()

    if upper_cleaned in direct_aliases:
        return direct_aliases[upper_cleaned]

    if _looks_like_symbol(cleaned):
        if market == "India (NSE)":
            symbol = cleaned.upper()
            if not (symbol.endswith(".NS") or "." in symbol or symbol.startswith("^")):
                symbol = f"{symbol}.NS"
            return symbol, cleaned.upper()
        return cleaned.upper(), cleaned.upper()

    searched_symbol, searched_name = _search_yahoo_symbol(cleaned, market)
    if searched_symbol:
        return searched_symbol, searched_name or cleaned

    if market == "India (NSE)":
        fallback_symbol = upper_cleaned if (upper_cleaned.endswith(".NS") or "." in upper_cleaned or upper_cleaned.startswith("^")) else f"{upper_cleaned}.NS"
        return fallback_symbol, cleaned
    return cleaned.upper(), cleaned


def _to_yahoo_tickers(tickers: list[str], market: str) -> tuple[list[str], dict[str, str]]:
    yahoo: list[str] = []
    yahoo_to_display: dict[str, str] = {}

    for t in tickers:
        yt, display = _resolve_yahoo_symbol(t, market)
        yahoo.append(yt)
        yahoo_to_display[yt] = display

    return yahoo, yahoo_to_display


def _build_resolved_assets_table(input_assets: list[str], resolved_tickers: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"Input Asset": input_assets, "Resolved Ticker": resolved_tickers})


def _to_yahoo_benchmark(raw: str, market: str) -> tuple[str, str]:
    b = raw.strip()
    if not b:
        b = "^NSEI"
    return _resolve_yahoo_symbol(b, market)


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    cleaned = {k: float(v) for k, v in weights.items() if float(v) > 0}
    s = sum(cleaned.values())
    if s <= 0:
        raise ValueError("Weights must sum to a positive number.")
    return {k: v / s for k, v in cleaned.items()}


@st.cache_data(show_spinner=False, ttl=60 * 30)
def _download_prices(tickers: tuple[str, ...], start: dt.date, end: dt.date) -> pd.DataFrame:
    df = yf.download(
        list(tickers),
        start=start,
        end=end + dt.timedelta(days=1),
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        out = df
    else:
        out = pd.concat({c: df[c] for c in df.columns}, axis=1)
        out.columns = pd.MultiIndex.from_product([out.columns, [tickers[0]]])
    out.index = pd.to_datetime(out.index).tz_localize(None)
    return out.sort_index()


def _select_close(prices: pd.DataFrame, tickers: list[str], use_adj_close: bool) -> pd.DataFrame:
    field = "Adj Close" if use_adj_close else "Close"
    if not isinstance(prices.columns, pd.MultiIndex) or field not in prices.columns.levels[0]:
        raise ValueError(_missing_price_data_message())
    close = prices[field].copy().reindex(columns=tickers)
    close = close.sort_index().ffill().dropna(how="all")
    all_nan_cols = [c for c in close.columns if close[c].isna().all()]
    if all_nan_cols:
        raise ValueError(_missing_price_data_message())
    return close


def _find_failed_assets(
    prices: pd.DataFrame,
    tickers: list[str],
    yahoo_to_display: dict[str, str],
    use_adj_close: bool,
) -> list[str]:
    field = "Adj Close" if use_adj_close else "Close"
    if prices.empty or not isinstance(prices.columns, pd.MultiIndex) or field not in prices.columns.levels[0]:
        return [yahoo_to_display.get(t, t) for t in tickers]

    close = prices[field].copy().reindex(columns=tickers)
    failed = [yahoo_to_display.get(t, t) for t in tickers if t not in close.columns or close[t].isna().all()]
    return failed


def _compute_portfolio(close: pd.DataFrame, config: PortfolioConfig) -> tuple[pd.DataFrame, pd.Series]:
    close = close.copy().sort_index().ffill().dropna(how="all")
    if close.empty or len(close) < 2:
        raise ValueError("Not enough price data after cleaning. Try a wider date range.")

    normalized_close = close / close.iloc[0]
    if config.weights is None:
        weights = {t: 1.0 / len(close.columns) for t in close.columns}
    else:
        missing = [t for t in close.columns if t not in config.weights]
        if missing:
            raise ValueError(f"Missing weights for: {', '.join(missing)}")
        weights = config.weights

    w = pd.Series(weights).reindex(close.columns).astype(float)
    w = w / w.sum()
    rets = close.pct_change().fillna(0.0)

    if config.rebalance == "Buy & Hold":
        shares = (config.initial_investment * w) / close.iloc[0]
        portfolio_value = (close * shares).sum(axis=1)
    else:
        rule_map: dict[str, str] = {"Monthly Rebalance": "M", "Quarterly Rebalance": "Q"}
        if config.rebalance not in rule_map:
            raise ValueError(f"Unknown rebalancing frequency: {config.rebalance}")
        rule = rule_map[config.rebalance]
        rebalance_dates = set(close.resample(rule).last().index)
        shares = (config.initial_investment * w) / close.iloc[0]
        values: list[float] = []
        for ts, row in close.iterrows():
            value = float((row * shares).sum())
            values.append(value)
            if ts in rebalance_dates and ts != close.index[0]:
                shares = (value * w) / row
        portfolio_value = pd.Series(values, index=close.index, name="PortfolioValue")

    return normalized_close, portfolio_value


def _performance_summary(portfolio_value: pd.Series) -> dict[str, float]:
    pv = portfolio_value.dropna()
    daily_ret = pv.pct_change().dropna()
    if daily_ret.empty:
        return {"Cumulative return (%)": 0.0, "Annualized vol (%)": 0.0, "Max drawdown (%)": 0.0}
    cum_ret = pv.iloc[-1] / pv.iloc[0] - 1.0
    ann_vol = float(daily_ret.std()) * np.sqrt(252)
    roll_max = pv.cummax()
    dd = pv / roll_max - 1.0
    max_dd = float(dd.min())
    return {
        "Cumulative return (%)": 100.0 * float(cum_ret),
        "Annualized vol (%)": 100.0 * float(ann_vol),
        "Max drawdown (%)": 100.0 * float(max_dd),
    }


def _annualized_return_from_series(value: pd.Series) -> float:
    v = value.dropna()
    if len(v) < 2:
        return 0.0
    n_days = (v.index[-1] - v.index[0]).days
    if n_days <= 0:
        return 0.0
    total = float(v.iloc[-1] / v.iloc[0])
    return total ** (365.25 / n_days) - 1.0


def _max_drawdown(value: pd.Series) -> float:
    v = value.dropna()
    if v.empty:
        return 0.0
    roll_max = v.cummax()
    dd = v / roll_max - 1.0
    return float(dd.min())


def _drawdown_series(value: pd.Series) -> pd.Series:
    v = value.dropna()
    if v.empty:
        return pd.Series(dtype=float)
    roll_max = v.cummax()
    return v / roll_max - 1.0


def _rolling_return(value: pd.Series, window_days: int) -> pd.Series:
    v = value.dropna()
    if len(v) <= window_days:
        return pd.Series(dtype=float)
    return v / v.shift(window_days) - 1.0


def _daily_returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    cleaned = prices.copy().sort_index().ffill().dropna(how="all")
    return cleaned.pct_change().dropna(how="all")


def _asset_class(label: str) -> str:
    u = label.upper()
    if u in {"GOLD", "SILVER"}:
        return "Commodity"
    if "ETF" in u:
        return "ETF"
    if "FUND" in u or "MF" in u:
        return "Mutual Fund"
    return "Equity"


def _historical_var(portfolio_value: pd.Series, confidence: float = 0.95) -> float:
    v = portfolio_value.dropna()
    if len(v) < 2:
        return 0.0
    rets = v.pct_change().dropna()
    if rets.empty:
        return 0.0
    q = np.quantile(rets, 1.0 - confidence)
    current_val = float(v.iloc[-1])
    return max(0.0, -q * current_val)


def _efficient_frontier(
    asset_returns: pd.DataFrame,
    rf_annual: float,
    n_portfolios: int = 2000,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if asset_returns.empty or asset_returns.shape[1] < 2:
        return pd.DataFrame(columns=["vol", "ret", "sharpe"]), pd.Series(dtype=float), pd.Series(dtype=float)

    mu = asset_returns.mean() * 252.0
    cov = asset_returns.cov() * 252.0
    rf = rf_annual
    results: list[tuple[float, float, float]] = []
    weight_list: list[pd.Series] = []

    for _ in range(n_portfolios):
        w = np.random.random(len(mu))
        w = w / w.sum()
        w_series = pd.Series(w, index=mu.index)
        port_ret = float(np.dot(w, mu.values))
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov.values, w))))
        sharpe = (port_ret - rf) / port_vol if port_vol > 0 else 0.0
        results.append((port_vol, port_ret, sharpe))
        weight_list.append(w_series)

    frontier_df = pd.DataFrame(results, columns=["vol", "ret", "sharpe"])
    if frontier_df.empty:
        return frontier_df, pd.Series(dtype=float), pd.Series(dtype=float)
    idx_max_sharpe = int(frontier_df["sharpe"].idxmax())
    idx_min_vol = int(frontier_df["vol"].idxmin())
    return frontier_df, weight_list[idx_max_sharpe], weight_list[idx_min_vol]


def _risk_adjusted_metrics(
    portfolio_value: pd.Series,
    benchmark_value: pd.Series,
    risk_free_rate_pct: float,
) -> dict[str, float]:
    pv = portfolio_value.dropna()
    bv = benchmark_value.dropna()
    df = pd.concat([pv.rename("pv"), bv.rename("bv")], axis=1).dropna()
    if len(df) < 3:
        return {
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Beta vs Benchmark": 0.0,
            "Alpha vs Benchmark (%)": 0.0,
        }

    port_ret = df["pv"].pct_change().dropna()
    bench_ret = df["bv"].pct_change().dropna()
    aligned = pd.concat([port_ret.rename("p"), bench_ret.rename("b")], axis=1).dropna()
    if aligned.empty:
        return {
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "Beta vs Benchmark": 0.0,
            "Alpha vs Benchmark (%)": 0.0,
        }

    rf_annual = float(risk_free_rate_pct) / 100.0
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0
    excess = aligned["p"] - rf_daily
    vol = float(aligned["p"].std()) * np.sqrt(252)
    sharpe = float(excess.mean()) * 252.0 / vol if vol > 0 else 0.0
    downside = aligned["p"][aligned["p"] < rf_daily] - rf_daily
    downside_dev = float(downside.std()) * np.sqrt(252) if len(downside) > 1 else 0.0
    sortino = float(excess.mean()) * 252.0 / downside_dev if downside_dev > 0 else 0.0
    var_b = float(aligned["b"].var())
    beta = float(aligned["p"].cov(aligned["b"]) / var_b) if var_b > 0 else 0.0
    rp = _annualized_return_from_series(df["pv"])
    rb = _annualized_return_from_series(df["bv"])
    alpha = (rp - rf_annual) - beta * (rb - rf_annual)

    return {
        "Sharpe Ratio": float(sharpe),
        "Sortino Ratio": float(sortino),
        "Beta vs Benchmark": float(beta),
        "Alpha vs Benchmark (%)": 100.0 * float(alpha),
    }


def _format_pct(value: float) -> str:
    return f"{value:.2f}%"


def _format_currency(value: float, currency_symbol: str) -> str:
    return f"{currency_symbol}{value:,.2f}"


def _index_from_one(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = out.index + 1
    return out


def _describe_direction(value: float, positive_label: str, negative_label: str, flat_label: str) -> str:
    if value > 0:
        return positive_label
    if value < 0:
        return negative_label
    return flat_label


def _scaled_horizon_return(long_window_return: float, target_days: int, base_days: int = 20) -> float:
    if long_window_return <= -1.0:
        return -1.0
    return (1.0 + long_window_return) ** (target_days / base_days) - 1.0


def _interpret_sharpe_ratio(value: float) -> str:
    if value < 0:
        band = "below 0"
        verdict = "the portfolio has been destroying value after adjusting for total volatility"
    elif value < 0.5:
        band = "between 0 and 0.5"
        verdict = "risk-adjusted returns have been weak"
    elif value < 1.0:
        band = "between 0.5 and 1"
        verdict = "risk-adjusted performance has been moderate"
    elif value < 2.0:
        band = "between 1 and 2"
        verdict = "risk-adjusted performance has been strong"
    else:
        band = "above 2"
        verdict = "risk-adjusted performance has been exceptional"
    return (
        f"Sharpe ratio measures excess return earned for each unit of total volatility. At {value:.2f}, "
        f"it sits {band}, which means {verdict}. For the portfolio, this shows whether return has been efficient "
        f"enough to justify the full amount of volatility investors absorbed."
    )


def _interpret_sortino_ratio(value: float) -> str:
    if value < 0:
        verdict = "the portfolio has not been compensated for downside risk"
    elif value < 0.5:
        verdict = "downside-risk compensation has been weak"
    elif value < 1.0:
        verdict = "downside-risk compensation looks acceptable but not especially strong"
    elif value < 2.0:
        verdict = "the portfolio has been rewarded well for the downside risk taken"
    else:
        verdict = "downside-risk compensation has been unusually strong"
    return (
        f"Sortino ratio focuses only on harmful volatility, so it is useful for judging downside-risk efficiency. "
        f"At {value:.2f}, {verdict}. This matters because investors typically care more about being paid for losses than for upside swings."
    )


def _interpret_volatility(value: float) -> str:
    if value < 10:
        verdict = "low for an equity-oriented portfolio"
    elif value < 20:
        verdict = "moderate and broadly in line with many diversified equity portfolios"
    elif value < 30:
        verdict = "elevated, meaning investors should expect larger swings than average"
    else:
        verdict = "high and more typical of aggressive or concentrated equity exposure"
    return (
        f"Volatility measures how widely returns tend to swing around their average over time. "
        f"An annualized volatility of {_format_pct(value)} is {verdict}. In portfolio terms, that sets expectations for how smooth or turbulent the holding experience is likely to feel."
    )


def _interpret_drawdown(value: float) -> str:
    depth = abs(value)
    if depth < 10:
        verdict = "shallow by equity standards"
    elif depth < 20:
        verdict = "noticeable but still contained"
    elif depth <= 40:
        verdict = "within the range often seen in meaningful equity corrections"
    else:
        verdict = "deeper than the 20% to 40% drawdowns commonly associated with major equity selloffs"
    return (
        f"Maximum drawdown shows the worst peak-to-trough loss an investor would have experienced. "
        f"A drawdown of {_format_pct(value)} is {verdict}. It helps frame the depth of loss an investor needed to tolerate before recovery."
    )


def _interpret_var(value: float, portfolio_value: float, currency_symbol: str) -> str:
    pct = (value / portfolio_value * 100.0) if portfolio_value > 0 else 0.0
    if pct < 1:
        verdict = "a relatively contained one-day loss estimate"
    elif pct < 2.5:
        verdict = "a moderate one-day loss estimate"
    else:
        verdict = "an elevated one-day loss estimate"
    return (
        f"Value at Risk estimates the loss threshold that should not be exceeded on roughly 95% of trading days "
        f"under historical conditions. Here, the 95% one-day VaR is {_format_currency(value, currency_symbol)} "
        f"or about {_format_pct(pct)}, which implies {verdict}. It is best read as normal-day loss exposure rather than a worst-case scenario."
    )


def _interpret_beta(value: float) -> str:
    if value < 0:
        verdict = "the portfolio has tended to move opposite the benchmark, which is unusual and can support diversification"
    elif value < 0.8:
        verdict = "the portfolio has been less sensitive to market moves than the benchmark"
    elif value <= 1.2:
        verdict = "the portfolio has moved broadly in line with the benchmark"
    else:
        verdict = "the portfolio has amplified benchmark moves"
    return (
        f"Beta measures how sensitive the portfolio is to benchmark movements. At {value:.2f}, {verdict}. This helps investors gauge how much broad market shocks are likely to flow through to portfolio returns."
    )


def _interpret_alpha(value: float, benchmark_label: str) -> str:
    if value > 2:
        verdict = f"the portfolio appears to be outperforming {benchmark_label} after adjusting for market risk"
    elif value >= 0:
        verdict = f"the portfolio is modestly ahead of what its benchmark exposure alone would imply versus {benchmark_label}"
    else:
        verdict = f"the portfolio appears to be underperforming {benchmark_label} after adjusting for market risk"
    return (
        f"Alpha estimates return beyond what beta exposure and the risk-free rate would normally explain. "
        f"At {_format_pct(value)}, {verdict}. In other words, it is a rough measure of value added beyond simple market exposure."
    )


def _interpret_benchmark_comparison(portfolio_return_pct: float, benchmark_return_pct: float, benchmark_label: str) -> str:
    delta = portfolio_return_pct - benchmark_return_pct
    if delta > 0:
        verdict = f"the portfolio has added value relative to {benchmark_label}"
    elif delta < 0:
        verdict = f"the portfolio has lagged {benchmark_label}"
    else:
        verdict = f"the portfolio has tracked {benchmark_label} closely"
    return (
        f"Benchmark comparison helps separate stock selection and allocation effects from broad market movement. "
        f"With portfolio return at {_format_pct(portfolio_return_pct)} versus {_format_pct(benchmark_return_pct)} "
        f"for {benchmark_label}, {verdict}. This indicates whether taking active portfolio risk has been worthwhile."
    )


def generate_snapshot_insight(metrics: dict[str, float], currency_symbol: str) -> str:
    total_return = metrics["total_return_pct"]
    gain_loss = metrics["gain_loss"]
    annualized = metrics["annualized_return_pct"]
    direction = "grew" if gain_loss >= 0 else "declined"
    strength = "strong" if annualized >= 12 else "steady" if annualized >= 0 else "weak"
    return (
        f"The portfolio has {direction} to {_format_currency(metrics['portfolio_value'], currency_symbol)}, "
        f"producing a total return of {_format_pct(total_return)} and a net change of "
        f"{_format_currency(gain_loss, currency_symbol)}. Based on the observed history, the annualized pace "
        f"looks {strength} at {_format_pct(annualized)}, which helps frame whether the recent path has been "
        f"compounding efficiently or simply recovering from shorter swings."
    )


def generate_forecast_explanation(forecast_metrics: dict[str, float], subject: str = "asset") -> str:
    next_day = forecast_metrics["next_day_return_pct"]
    week = forecast_metrics["week_return_pct"]
    month = forecast_metrics["month_return_pct"]
    six_month = forecast_metrics["six_month_return_pct"]
    year = forecast_metrics["year_return_pct"]
    prob = forecast_metrics["prob_positive_pct"]

    next_day_signal = "mild upside pressure" if next_day > 0 else "mild downside pressure" if next_day < 0 else "little directional edge"
    if abs(next_day) >= 1.0:
        next_day_signal = "strong upside pressure" if next_day > 0 else "clear downside pressure"
    week_signal = _describe_direction(
        week,
        "short-term momentum looks positive",
        "short-term momentum looks negative",
        "short-term momentum looks neutral",
    )
    year_signal = _describe_direction(
        year,
        "the longer-horizon backdrop is constructive if current conditions persist",
        "the longer-horizon backdrop is cautious if current conditions persist",
        "the longer-horizon backdrop is broadly flat",
    )
    alignment = (
        "Short-term and long-term signals are aligned, which usually makes the outlook easier to interpret."
        if (next_day >= 0 and year >= 0) or (next_day <= 0 and year <= 0)
        else "Short-term and long-term signals differ, so near-term momentum may not match the broader trend."
    )

    return (
        f"Horizon forecasts estimate expected change over specific holding periods rather than restating a raw model number. "
        f"For this {subject}, the next trading day outlook is {_format_pct(next_day)}, which suggests {next_day_signal} in the next session. "
        f"The end-of-this-week horizon points to {_format_pct(week)} over roughly five trading days, while the end-of-this-month view implies {_format_pct(month)} over about 21 sessions. "
        f"The six-month horizon points to {_format_pct(six_month)} over about 126 sessions. "
        f"The end-of-this-year horizon implies {_format_pct(year)} over about 252 trading days, while the probability of a positive next session is {prob:.1f}%. "
        f"Taken together, {year_signal}. {alignment}"
    )


def generate_heatmap_insight(corr: pd.DataFrame) -> str:
    if corr.empty or len(corr) < 2:
        return "A correlation heatmap needs at least two assets. Once multiple holdings are available, this section will explain whether diversification is genuinely broad or mostly cosmetic."

    off_diag = corr.where(~np.eye(len(corr), dtype=bool)).stack()
    avg_corr = float(off_diag.mean()) if not off_diag.empty else 0.0
    strong_pairs = int((off_diag.abs() >= 0.7).sum() / 2) if not off_diag.empty else 0
    negative_pairs = int((off_diag < 0).sum() / 2) if not off_diag.empty else 0

    if avg_corr >= 0.65:
        diversification = "weak because many positions appear to move together"
    elif avg_corr >= 0.35:
        diversification = "moderate, with some diversification benefit but still meaningful co-movement"
    else:
        diversification = "strong because return drivers are more differentiated across holdings"

    pair_note = (
        f"There are {strong_pairs} strongly related asset pairs, which can increase portfolio-wide swings during market stress."
        if strong_pairs > 0
        else "There are few strongly related asset pairs, which supports better diversification."
    )
    hedge_note = (
        f"Negative correlations appear in {negative_pairs} asset pairings, which can help cushion drawdowns."
        if negative_pairs > 0
        else "There are limited negative correlations, so natural hedging inside the portfolio is modest."
    )
    return (
        f"Correlation measures how strongly assets move together, from -1 for opposite movement to +1 for near lockstep behavior. "
        f"The average cross-asset correlation here is {avg_corr:.2f}, so diversification looks {diversification}. "
        f"{pair_note} {hedge_note}"
    )


def _portfolio_stats_from_weights(
    weights: pd.Series,
    mu: pd.Series,
    cov: pd.DataFrame,
    rf_annual: float,
) -> tuple[float, float, float]:
    aligned_weights = weights.reindex(mu.index).fillna(0.0).astype(float)
    if aligned_weights.sum() <= 0:
        return 0.0, 0.0, 0.0
    aligned_weights = aligned_weights / aligned_weights.sum()
    port_ret = float(np.dot(aligned_weights.values, mu.values))
    port_vol = float(np.sqrt(np.dot(aligned_weights.values.T, np.dot(cov.values, aligned_weights.values))))
    sharpe = (port_ret - rf_annual) / port_vol if port_vol > 0 else 0.0
    return port_ret, port_vol, sharpe


def generate_portfolio_improvement_suggestions(
    current_alloc: pd.Series,
    suggested_alloc: pd.Series,
    asset_returns: pd.DataFrame,
    rf_annual: float,
) -> tuple[list[str], pd.DataFrame]:
    if asset_returns.empty or suggested_alloc.empty or current_alloc.empty:
        return [], pd.DataFrame()

    mu = asset_returns.mean() * 252.0
    cov = asset_returns.cov() * 252.0
    _, _, current_sharpe = _portfolio_stats_from_weights(current_alloc, mu, cov, rf_annual)
    _, _, suggested_sharpe = _portfolio_stats_from_weights(suggested_alloc, mu, cov, rf_annual)

    changes = pd.DataFrame(
        {
            "Asset": current_alloc.index,
            "Current Allocation (%)": current_alloc.reindex(current_alloc.index).fillna(0.0).values * 100.0,
            "Suggested Allocation (%)": suggested_alloc.reindex(current_alloc.index).fillna(0.0).values * 100.0,
        }
    )
    changes["Change (%)"] = changes["Suggested Allocation (%)"] - changes["Current Allocation (%)"]
    changes = changes.loc[changes["Change (%)"].abs() >= 2.0].copy()
    changes = changes.sort_values("Change (%)", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    if changes.empty:
        return [], pd.DataFrame()

    increases = changes[changes["Change (%)"] > 0].reset_index(drop=True)
    decreases = changes[changes["Change (%)"] < 0].reset_index(drop=True)
    suggestions: list[str] = []

    for idx in range(min(3, len(increases), len(decreases))):
        inc = increases.iloc[idx]
        dec = decreases.iloc[idx]
        suggestions.append(
            f"Reducing {dec['Asset']} from {dec['Current Allocation (%)']:.1f}% to {dec['Suggested Allocation (%)']:.1f}% "
            f"and increasing {inc['Asset']} from {inc['Current Allocation (%)']:.1f}% to {inc['Suggested Allocation (%)']:.1f}% "
            f"could improve the estimated portfolio Sharpe ratio from {current_sharpe:.2f} to approximately {suggested_sharpe:.2f} "
            f"while maintaining a more efficient risk-return mix."
        )

    summary_df = changes.head(6).copy()
    summary_df.index = summary_df.index + 1
    return suggestions, summary_df


def build_allocation_enhancement_table(
    current_alloc: pd.Series,
    suggested_alloc: pd.Series,
) -> tuple[pd.DataFrame, list[str]]:
    assets = sorted(set(current_alloc.index).union(set(suggested_alloc.index)))
    table_df = pd.DataFrame(
        {
            "Asset": assets,
            "Current Allocation": [100.0 * float(current_alloc.get(asset, 0.0)) for asset in assets],
            "Suggested Allocation": [100.0 * float(suggested_alloc.get(asset, 0.0)) for asset in assets],
        }
    )
    table_df = table_df.sort_values(["Suggested Allocation", "Asset"], ascending=[False, True]).reset_index(drop=True)
    removed_assets = table_df.loc[table_df["Suggested Allocation"] <= 0.0001, "Asset"].tolist()
    table_df.index = table_df.index + 1
    return table_df, removed_assets


def generate_portfolio_summary(
    performance_metrics: dict[str, float],
    risk_metrics: dict[str, float],
    benchmark_label: str,
    structure_insight: str,
    forecast_metrics: dict[str, float] | None = None,
) -> str:
    relative = performance_metrics["relative_cum_return_pct"]
    if relative > 2:
        benchmark_view = f"outperformed {benchmark_label}"
    elif relative < -2:
        benchmark_view = f"underperformed {benchmark_label}"
    else:
        benchmark_view = f"tracked {benchmark_label} closely"

    vol = risk_metrics["volatility_pct"]
    risk_view = "low" if vol < 10 else "moderate" if vol < 20 else "elevated"
    drawdown_view = "contained" if abs(risk_metrics["max_drawdown_pct"]) < 15 else "meaningful"
    forecast_sentence = ""
    if forecast_metrics is not None:
        if forecast_metrics["next_day_return_pct"] < 0 and forecast_metrics["year_return_pct"] > 0:
            forecast_sentence = "Near-term pressure is visible, although the longer-horizon signal still points to recovery potential."
        elif forecast_metrics["year_return_pct"] < 0:
            forecast_sentence = "Forecasts lean cautious across the main horizons."
        elif forecast_metrics["year_return_pct"] > 0:
            forecast_sentence = "Forecasts remain constructive across the main horizons."
        else:
            forecast_sentence = "Forecasts are broadly neutral across the main horizons."

    return (
        f"The portfolio has {benchmark_view} over the selected period, with {risk_view} overall risk and {drawdown_view} downside pressure. "
        f"{structure_insight} {forecast_sentence}"
    )


def calculate_portfolio_risk_score(
    volatility_pct: float,
    max_drawdown_pct: float,
    sharpe_ratio: float,
    alloc: pd.Series,
    corr: pd.DataFrame,
) -> tuple[float, str, str]:
    vol_component = min(max(volatility_pct / 30.0, 0.0), 1.0)
    drawdown_component = min(max(abs(max_drawdown_pct) / 35.0, 0.0), 1.0)
    sharpe_component = min(max((1.5 - sharpe_ratio) / 2.5, 0.0), 1.0)
    concentration_component = min(max(float(alloc.max()) if not alloc.empty else 0.0, 0.0), 1.0)

    if corr.empty or len(corr) < 2:
        diversification_component = 0.6
    else:
        off_diag = corr.where(~np.eye(len(corr), dtype=bool)).stack()
        avg_corr = float(off_diag.mean()) if not off_diag.empty else 0.0
        diversification_component = min(max((avg_corr + 1.0) / 2.0, 0.0), 1.0)

    weighted_score = (
        0.28 * vol_component
        + 0.24 * drawdown_component
        + 0.18 * sharpe_component
        + 0.18 * concentration_component
        + 0.12 * diversification_component
    )
    score = round(10.0 * weighted_score, 1)

    if score < 3:
        band = "Low Risk"
    elif score < 6:
        band = "Moderate Risk"
    elif score < 8:
        band = "High Risk"
    else:
        band = "Very High Risk"

    vol_text = "volatility is contained" if volatility_pct < 12 else "volatility is moderate" if volatility_pct < 20 else "volatility is elevated"
    dd_text = "drawdowns have stayed relatively shallow" if abs(max_drawdown_pct) < 12 else "the drawdown profile shows meaningful downside periods" if abs(max_drawdown_pct) < 25 else "drawdowns have been deep enough to signal significant downside risk"
    sharpe_text = "risk-adjusted returns help offset some of that risk" if sharpe_ratio >= 1 else "risk-adjusted returns only partially offset that risk" if sharpe_ratio >= 0.5 else "risk-adjusted returns have not done much to offset the underlying risk"
    concentration = float(alloc.max()) if not alloc.empty else 0.0
    concentration_text = "The portfolio is well spread across holdings." if concentration < 0.2 else "Position concentration is noticeable but still manageable." if concentration < 0.35 else "Position concentration is high, which increases single-asset exposure."
    diversification_text = (
        "Diversification across assets partially offsets this risk."
        if diversification_component < 0.55
        else "Correlation across holdings is fairly high, so diversification benefits are limited."
    )
    explanation = (
        f"This portfolio carries {band.lower()}. {vol_text.capitalize()}, and {dd_text}. "
        f"{sharpe_text.capitalize()}. {concentration_text} {diversification_text}"
    )
    return score, band, explanation


def _inject_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .portfolio-summary-card {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
            margin-bottom: 1rem;
        }
        .portfolio-summary-card h3 {
            margin: 0 0 0.45rem 0;
            font-size: 1.05rem;
            color: #f8fafc;
        }
        .portfolio-summary-card p {
            margin: 0;
            color: #dbe4f0;
            line-height: 1.5;
        }
        .assistant-shell {
            background: rgba(15, 23, 42, 0.72);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 18px;
            padding: 0.9rem 1rem 1rem 1rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.24);
        }
        .assistant-scroll {
            max-height: 260px;
            overflow-y: auto;
            margin: 0.35rem 0 0.75rem 0;
            padding-right: 0.2rem;
        }
        .assistant-msg {
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            margin-bottom: 0.45rem;
            font-size: 0.92rem;
            line-height: 1.45;
            color: #e2e8f0;
        }
        .assistant-msg.user {
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid rgba(96, 165, 250, 0.22);
        }
        .assistant-msg.assistant {
            background: rgba(17, 24, 39, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.16);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def generate_growth_insight(portfolio_value: pd.Series, currency_symbol: str) -> str:
    pv = portfolio_value.dropna()
    if len(pv) < 2:
        return "There is not enough portfolio history yet to describe the growth path."
    drawdown = 100.0 * _max_drawdown(pv)
    start_value = float(pv.iloc[0])
    end_value = float(pv.iloc[-1])
    ann = 100.0 * _annualized_return_from_series(pv)
    return (
        f"The equity curve moved from {_format_currency(start_value, currency_symbol)} to "
        f"{_format_currency(end_value, currency_symbol)} over the selected window. That translates to an "
        f"annualized return of {_format_pct(ann)} while the deepest peak-to-trough decline reached "
        f"{_format_pct(drawdown)}. Use this section to separate smooth compounding from periods where gains "
        f"were accompanied by meaningful pullbacks."
    )


def generate_performance_insight(metrics: dict[str, float], benchmark_label: str) -> str:
    delta = metrics["relative_cum_return_pct"]
    if delta > 0:
        lead = f"outperformed {benchmark_label} by {_format_pct(delta)} on a cumulative basis"
    elif delta < 0:
        lead = f"trailed {benchmark_label} by {_format_pct(abs(delta))} on a cumulative basis"
    else:
        lead = f"tracked {benchmark_label} almost exactly"
    return (
        f"The portfolio {lead}. Its cumulative return is {_format_pct(metrics['portfolio_cum_return_pct'])} "
        f"versus {_format_pct(metrics['benchmark_cum_return_pct'])} for the benchmark, while the annualized "
        f"return stands at {_format_pct(metrics['portfolio_annualized_return_pct'])}. This helps show whether "
        f"recent gains reflect genuine excess performance or broad market participation."
    )


def generate_risk_explanation(metrics: dict[str, float], currency_symbol: str, benchmark_label: str) -> str:
    return " ".join(
        [
            _interpret_sharpe_ratio(metrics["sharpe_ratio"]),
            _interpret_sortino_ratio(metrics["sortino_ratio"]),
            _interpret_volatility(metrics["volatility_pct"]),
            _interpret_drawdown(metrics["max_drawdown_pct"]),
            _interpret_var(metrics["var_95"], metrics["portfolio_value"], currency_symbol),
            _interpret_beta(metrics["beta"]),
            _interpret_alpha(metrics["alpha_pct"], benchmark_label),
            _interpret_benchmark_comparison(
                metrics["portfolio_cum_return_pct"],
                metrics["benchmark_cum_return_pct"],
                benchmark_label,
            ),
        ]
    )


def generate_structure_insight(alloc: pd.Series, class_series: pd.Series) -> str:
    if alloc.empty:
        return "Portfolio structure insight will appear once asset weights are available."
    top_asset = str(alloc.idxmax())
    top_asset_weight = float(alloc.max()) * 100.0
    top_class = str(class_series.idxmax()) if not class_series.empty else "Asset mix"
    top_class_weight = float(class_series.max()) * 100.0 if not class_series.empty else 0.0
    return (
        f"The largest holding is {top_asset} at {_format_pct(top_asset_weight)}, which is the clearest source "
        f"of concentration risk in the current allocation. At the asset-class level, {top_class} leads at "
        f"{_format_pct(top_class_weight)}. Use this section to judge whether diversification is broad enough "
        f"or whether a small subset of positions is likely driving most of the portfolio behavior."
    )


def generate_advanced_insight(corr: pd.DataFrame, frontier_df: pd.DataFrame, optimal_weights: pd.Series) -> str:
    avg_corr = float(corr.where(~np.eye(len(corr), dtype=bool)).stack().mean()) if not corr.empty and len(corr) > 1 else 0.0
    max_weight = float(optimal_weights.max()) * 100.0 if not optimal_weights.empty else 0.0
    top_asset = str(optimal_weights.idxmax()) if not optimal_weights.empty else "N/A"
    if avg_corr < 0.2:
        diversification_note = (
            "Correlation measures how strongly assets move together. The average relationship here is relatively low, "
            "which improves diversification because positions are less likely to rise and fall in sync."
        )
    elif avg_corr < 0.6:
        diversification_note = (
            "Correlation measures how strongly assets move together. This portfolio sits in a middle range, so there "
            "are some diversification benefits, but several holdings may still react similarly during broad market moves."
        )
    else:
        diversification_note = (
            "Correlation measures how strongly assets move together. High positive correlation means many holdings are "
            "moving in the same direction, which reduces diversification benefits and can amplify portfolio-wide swings."
        )
    frontier_note = (
        "The efficient frontier highlights how different long-only mixes trade return for volatility."
        if not frontier_df.empty
        else "There are not enough assets to estimate an efficient frontier reliably."
    )
    weight_note = (
        f"Within the max-Sharpe mix, {top_asset} receives the highest weight at {_format_pct(max_weight)}."
        if not optimal_weights.empty
        else "Optimal allocation details will appear once the frontier can be estimated."
    )
    return f"{diversification_note} {frontier_note} {weight_note}"


def generate_ai_assistant_response(question: str) -> str:
    q = question.strip().lower()
    if not q:
        return "Ask about Sharpe ratio, drawdown, correlation, efficient frontier, rebalancing, or how to read the forecast horizons."
    unsupported_message = (
        "This assistant currently explains portfolio analytics concepts such as Sharpe ratio, drawdown, volatility, "
        "Value at Risk, beta, alpha, correlation heatmaps, efficient frontiers, forecast horizons, and rebalancing. "
        "For portfolio improvement suggestions, please refer to the 'Portfolio Improvement Suggestions' section above."
    )
    if "sharpe" in q:
        return "Sharpe ratio tells you how much excess return the portfolio earned for each unit of total volatility. A negative Sharpe ratio means the portfolio delivered returns below the risk-free rate relative to its risk, around 1 is generally solid, and above 2 is unusually strong."
    if "sortino" in q:
        return "Sortino ratio is similar to Sharpe ratio, but it only penalizes downside volatility. That makes it useful when investors care more about harmful losses than upside variability. A higher Sortino ratio means the portfolio has been compensated better for downside risk."
    if "drawdown" in q or "maximum drawdown" in q:
        return "Maximum drawdown shows the worst peak-to-trough decline the portfolio went through before recovering. It answers a practical investor question: how painful was the deepest loss period? Around 10% is often manageable, while 20% or more usually feels like a major correction."
    if "efficient frontier" in q:
        return "The efficient frontier is a map of possible portfolio mixes. Each point represents a different balance between expected return and volatility. The more attractive part of the frontier is the area that offers higher expected return for the same level of risk, or lower risk for the same expected return."
    if "volatility" in q:
        return "Volatility measures how widely returns swing around their average. Higher volatility means larger moves up and down, so the portfolio feels less stable. For many diversified equity portfolios, roughly 10% to 20% annualized volatility is moderate, while 25% and above is fairly aggressive."
    if "value at risk" in q or "var" in q:
        return "Value at Risk estimates a loss threshold under normal market conditions. If the 95% one-day VaR is Rs.500, it means losses should be smaller than that on about 95 out of 100 trading days based on history. It does not show the worst possible loss, only a typical risk threshold."
    if "beta" in q:
        return "Beta tells you how sensitive the portfolio is to benchmark moves. A beta near 1 means it tends to move like the market, above 1 means it usually moves more than the market, and below 1 means it tends to move less. It is a simple gauge of market sensitivity."
    if "alpha" in q:
        return "Alpha estimates how much return the portfolio generated beyond what its market exposure would normally explain. Positive alpha suggests outperformance after adjusting for beta, while negative alpha suggests the strategy did not add value beyond market risk."
    if "correlation" in q or "heatmap" in q:
        return "The correlation heatmap shows how strongly assets move together. Values near 1 mean they often move in sync, values near 0 mean the relationship is weak, and negative values mean they often offset each other. Lower correlation usually improves diversification and can reduce portfolio volatility."
    if "rebalance" in q:
        return "Portfolio rebalancing means resetting weights after market moves change them. It helps control concentration risk, keeps the portfolio aligned with its target risk profile, and can improve discipline, although rebalancing too often may reduce the benefit of letting winners run."
    if "forecast horizon" in q or "forecast" in q or "next trading day" in q or "end of this week" in q or "end of this month" in q or "six months" in q or "end of this year" in q:
        return "Forecast horizons describe expected change over different holding periods rather than specific calendar dates. In this dashboard, the horizons cover the next trading day, roughly one week, one month, six months, and one year, helping you compare short-term momentum with longer-term directional signals."
    return unsupported_message


def _render_ai_assistant() -> None:
    if "assistant_messages" not in st.session_state:
        st.session_state["assistant_messages"] = [
            {
                "role": "assistant",
                "content": "Ask about Sharpe ratio, drawdown, efficient frontier, correlation heatmap, Value at Risk, beta, alpha, or portfolio rebalancing.",
            }
        ]

    st.header("AI Assistant")
    st.caption("Ask portfolio questions in plain language.")
    st.markdown('<div class="assistant-shell">', unsafe_allow_html=True)
    rendered_messages = []
    for message in st.session_state["assistant_messages"]:
        role = "user" if message["role"] == "user" else "assistant"
        rendered_messages.append(f'<div class="assistant-msg {role}">{html.escape(message["content"])}</div>')
    st.markdown(f'<div class="assistant-scroll">{"".join(rendered_messages)}</div>', unsafe_allow_html=True)
    question = st.chat_input(
        "Ask AI — Sharpe ratio, volatility, correlation heatmap, forecast horizons...",
        key="bottom_ai_question",
    )
    if question and question.strip():
        st.session_state["assistant_messages"].append({"role": "user", "content": question.strip()})
        st.session_state["assistant_messages"].append(
            {"role": "assistant", "content": generate_ai_assistant_response(question)}
        )
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Portfolio Research", layout="wide")
    _inject_dashboard_styles()
    st.markdown('<div id="top"></div>', unsafe_allow_html=True)
    st.title("Portfolio Research")
    st.caption("Professional portfolio analytics, benchmark evaluation, forward-looking commentary, and optimization guidance.")

    currency_symbol = "Rs."

    if "prices" not in st.session_state:
        st.session_state["prices"] = None
    if "last_config" not in st.session_state:
        st.session_state["last_config"] = None
    if "last_yahoo_tickers" not in st.session_state:
        st.session_state["last_yahoo_tickers"] = None
    if "last_yahoo_to_display" not in st.session_state:
        st.session_state["last_yahoo_to_display"] = None
    if "last_benchmark_yahoo" not in st.session_state:
        st.session_state["last_benchmark_yahoo"] = None
    if "last_benchmark_label" not in st.session_state:
        st.session_state["last_benchmark_label"] = None
    if "last_resolved_assets" not in st.session_state:
        st.session_state["last_resolved_assets"] = None

    with st.sidebar:
        st.subheader("Portfolio Inputs")
        raw_tickers = st.text_area(
            "Tickers (comma or newline separated)",
            value="RELIANCE, TCS, INFY, HDFCBANK",
            height=90,
            help=(
                "Examples (India): RELIANCE, TCS, INFY ('.NS' is added automatically when Market=India). "
                "Commodities: gold.c, silver.c. Examples (US): AAPL, MSFT, SPY."
            ),
        )
        market = st.selectbox("Market", options=["India (NSE)", "US (NYSE/NASDAQ)"], index=0)
        tickers = _parse_tickers(raw_tickers)
        yahoo_tickers, yahoo_to_display = _to_yahoo_tickers(tickers, market)

        today = dt.date.today()
        default_start = today - dt.timedelta(days=365 * 2)
        start = st.date_input("Start date", value=default_start)
        end = st.date_input("End date", value=today)

        initial_investment = st.number_input(
            f"Initial investment ({currency_symbol})",
            min_value=100.0,
            value=10_000.0,
            step=100.0,
        )

        st.subheader("Advanced Options")
        advanced = st.toggle("Enable advanced options", value=False)
        use_adj_close = st.toggle("Use Adjusted Close", value=True)
        rebalance = st.selectbox(
            "Rebalancing frequency" if advanced else "Rebalancing strategy",
            options=["Buy & Hold", "Monthly Rebalance", "Quarterly Rebalance"],
            index=1 if advanced else 0,
        )

        benchmark_raw = "^NSEI"
        risk_free_rate_pct = 6.5
        if advanced:
            benchmark_raw = st.text_input("Benchmark Index", value="^NSEI", help="Default: NIFTY 50 (^NSEI)")
            rf_choice = st.selectbox(
                "Risk-free rate source",
                options=["Custom", "India Govt Bond 10Y (6.5%)", "India Govt Bond 5Y (6.2%)"],
                index=1,
            )
            if rf_choice == "Custom":
                risk_free_rate_pct = st.number_input("Risk-free rate (%)", min_value=0.0, value=6.5, step=0.1)
            elif "10Y" in rf_choice:
                risk_free_rate_pct = 6.5
            else:
                risk_free_rate_pct = 6.2

        weights: dict[str, float] | None = None
        investment_validation_error: str | None = None
        effective_initial_investment = float(initial_investment)
        if tickers:
            st.caption("Enter the rupee amount invested in each asset. The app will convert these amounts into portfolio weights automatically.")
            investment_inputs: dict[str, float] = {}
            default_investment = float(initial_investment) / len(yahoo_tickers) if yahoo_tickers else float(initial_investment)
            for yt in yahoo_tickers:
                display = yahoo_to_display.get(yt, yt)
                investment_inputs[display] = st.number_input(
                    f"Investment in {display} ({currency_symbol})",
                    min_value=0.0,
                    value=float(default_investment),
                    step=100.0,
                    format="%.2f",
                    key=f"investment_{display}",
                )
            try:
                total_asset_investment = float(sum(investment_inputs.values()))
                if abs(total_asset_investment - float(initial_investment)) > 0.01:
                    investment_validation_error = (
                        f"The sum of asset investments ({currency_symbol}{total_asset_investment:.0f}) does not match "
                        f"the declared portfolio value ({currency_symbol}{float(initial_investment):.0f}). Please correct the inputs."
                    )
                    st.error(investment_validation_error)
                else:
                    weights = _normalize_weights(investment_inputs)
                    effective_initial_investment = total_asset_investment
            except ValueError as e:
                st.error(str(e))
                weights = None

        run = st.button("Run Analysis", type="primary", use_container_width=True)

    if len(tickers) == 0:
        st.error("Please enter at least one ticker.")
        return
    if start >= end:
        st.error("Start date must be before end date.")
        return
    if tickers and weights is None:
        st.error("Please correct the asset investment inputs before running the analysis.")
        return

    benchmark_yahoo, benchmark_display = _to_yahoo_benchmark(benchmark_raw, market)
    benchmark_label = "NIFTY50" if benchmark_yahoo == "^NSEI" else benchmark_display
    config = PortfolioConfig(
        tickers=tickers,
        start=start,
        end=end,
        initial_investment=effective_initial_investment,
        rebalance=rebalance,
        use_adj_close=bool(use_adj_close),
        weights=weights,
    )

    if run:
        with st.spinner("Downloading price data..."):
            all_symbols = list(dict.fromkeys([*yahoo_tickers, benchmark_yahoo]))
            prices = _download_prices(tuple(all_symbols), config.start, config.end)
        if prices.empty:
            st.error(_missing_price_data_message())
            st.session_state["prices"] = None
            return
        failed_assets = _find_failed_assets(
            prices=prices,
            tickers=yahoo_tickers,
            yahoo_to_display=yahoo_to_display,
            use_adj_close=config.use_adj_close,
        )
        expected_assets = len(tickers)
        actual_assets = expected_assets - len(failed_assets)
        if actual_assets != expected_assets:
            st.warning("Some assets could not be resolved or did not return price data. Please verify the symbols.")
        if failed_assets:
            st.warning(
                "The following assets could not be resolved or do not have price data available:\n"
                + "\n".join(f"- {asset}" for asset in failed_assets)
            )
            st.session_state["prices"] = None
            return
        st.session_state["prices"] = prices
        st.session_state["last_config"] = config
        st.session_state["last_yahoo_tickers"] = yahoo_tickers
        st.session_state["last_yahoo_to_display"] = yahoo_to_display
        st.session_state["last_benchmark_yahoo"] = benchmark_yahoo
        st.session_state["last_benchmark_label"] = benchmark_label
        st.session_state["last_resolved_assets"] = _build_resolved_assets_table(tickers, yahoo_tickers)

    prices = st.session_state.get("prices")
    last_config: PortfolioConfig | None = st.session_state.get("last_config")
    last_yahoo_tickers: list[str] | None = st.session_state.get("last_yahoo_tickers")
    last_yahoo_to_display: dict[str, str] | None = st.session_state.get("last_yahoo_to_display")
    last_benchmark_yahoo: str | None = st.session_state.get("last_benchmark_yahoo")
    last_benchmark_label: str | None = st.session_state.get("last_benchmark_label")
    if prices is None or last_config is None or last_yahoo_tickers is None or last_benchmark_yahoo is None:
        st.info("Set your inputs in the sidebar, then click **Run Analysis** to download data.")
        return

    try:
        close = _select_close(prices, last_yahoo_tickers, last_config.use_adj_close)
        close = close.rename(columns=last_yahoo_to_display)
        normalized_close, portfolio_value = _compute_portfolio(close, last_config)
        bench_close = _select_close(prices, [last_benchmark_yahoo], last_config.use_adj_close)
        bench_close = bench_close.rename(columns={last_benchmark_yahoo: last_benchmark_label})
        bench_series = bench_close[last_benchmark_label].sort_index().ffill().dropna()
        chart_close = pd.concat([close, bench_series.rename(last_benchmark_label)], axis=1).sort_index().ffill().dropna(how="all")
        norm_all_for_chart = chart_close / chart_close.iloc[0]
        bench_norm = (bench_series / bench_series.iloc[0]).rename(last_benchmark_label)
    except Exception as e:
        st.error(str(e))
        return

    pv = portfolio_value.dropna()
    portfolio_value_latest = float(pv.iloc[-1]) if not pv.empty else float(last_config.initial_investment)
    total_return_pct = 100.0 * (portfolio_value_latest / float(last_config.initial_investment) - 1.0)
    total_gain_loss = portfolio_value_latest - float(last_config.initial_investment)

    bench_value = (bench_norm * float(last_config.initial_investment)).rename(last_benchmark_label)
    perf_cum = float(pv.iloc[-1] / pv.iloc[0] - 1.0) if len(pv) >= 2 else 0.0
    perf_ann = _annualized_return_from_series(pv)
    daily_ret = pv.pct_change().dropna()
    vol_ann = float(daily_ret.std()) * np.sqrt(252) if not daily_ret.empty else 0.0
    mdd = _max_drawdown(pv)
    ra = _risk_adjusted_metrics(pv, bench_value, risk_free_rate_pct=risk_free_rate_pct)
    var_95 = _historical_var(pv, confidence=0.95)

    bench_daily = bench_value.pct_change().dropna()
    bench_cum = float(bench_value.iloc[-1] / bench_value.iloc[0] - 1.0) if len(bench_value) >= 2 else 0.0
    bench_ann = _annualized_return_from_series(bench_value)
    bench_vol = float(bench_daily.std()) * np.sqrt(252) if not bench_daily.empty else 0.0
    bench_ra = _risk_adjusted_metrics(bench_value, bench_value, risk_free_rate_pct=risk_free_rate_pct)
    bench_mdd = _max_drawdown(bench_value)

    if last_config.weights is None:
        alloc = pd.Series({c: 1.0 / len(close.columns) for c in close.columns})
    else:
        alloc = pd.Series(last_config.weights).reindex(close.columns).fillna(0.0)
        alloc = alloc / alloc.sum() if alloc.sum() > 0 else alloc

    alloc_df = alloc.reset_index()
    alloc_df.columns = ["Asset", "Weight"]
    asset_returns = _daily_returns_from_prices(close)
    asset_mu = asset_returns.mean() * 252.0
    asset_vol = asset_returns.std() * np.sqrt(252.0)
    asset_perf = pd.DataFrame(
        {
            "Asset": close.columns,
            "Weight": [alloc.get(a, 0.0) for a in close.columns],
            "Return (%)": [asset_mu.get(a, 0.0) * 100.0 for a in close.columns],
            "Volatility (%)": [asset_vol.get(a, 0.0) * 100.0 for a in close.columns],
        }
    )

    class_series = pd.Series(dtype=float)
    for a in close.columns:
        cls = _asset_class(a)
        class_series[cls] = class_series.get(cls, 0.0) + alloc.get(a, 0.0)
    if class_series.sum() > 0:
        class_series = class_series / class_series.sum()
    class_df = (100.0 * class_series).reset_index()
    class_df.columns = ["Asset Class", "Weight (%)"]

    rf_annual = float(risk_free_rate_pct) / 100.0
    frontier_df, w_max_sharpe, w_min_vol = _efficient_frontier(asset_returns, rf_annual=rf_annual)
    corr = asset_returns.corr()

    show_benchmark = st.toggle(f"Show Benchmark ({last_benchmark_label})", value=True)
    norm_for_chart = norm_all_for_chart.copy()
    if not show_benchmark:
        norm_for_chart = norm_for_chart.drop(columns=[last_benchmark_label], errors="ignore")

    norm_df = norm_for_chart.rename_axis("Date").reset_index()
    norm_long = norm_df.melt(id_vars="Date", var_name="Ticker", value_name="Normalized")
    fig_norm = px.line(norm_long, x="Date", y="Normalized", color="Ticker", template="plotly_white")
    fig_norm.update_layout(legend_title_text="Ticker", yaxis_title="Normalized price")

    pv_label = f"Portfolio Value ({currency_symbol})"
    pv_df = portfolio_value.rename(pv_label).rename_axis("Date").reset_index()
    fig_pv = px.line(pv_df, x="Date", y=pv_label, template="plotly_white")
    fig_pv.update_layout(yaxis_title=pv_label)

    rr_6m = _rolling_return(portfolio_value, window_days=126).rename("6M Rolling Return")
    rr_1y = _rolling_return(portfolio_value, window_days=252).rename("1Y Rolling Return")
    rr_df = pd.concat([rr_6m, rr_1y], axis=1).dropna(how="all")
    dd = _drawdown_series(portfolio_value)

    fig_alloc = px.pie(alloc_df, names="Asset", values="Weight", template="plotly_white", hole=0.4)
    fig_alloc.update_traces(textinfo="percent+label")

    snapshot_metrics = {
        "portfolio_value": portfolio_value_latest,
        "total_return_pct": total_return_pct,
        "annualized_return_pct": 100.0 * perf_ann,
        "gain_loss": total_gain_loss,
    }
    performance_metrics = {
        "portfolio_cum_return_pct": 100.0 * perf_cum,
        "portfolio_annualized_return_pct": 100.0 * perf_ann,
        "benchmark_cum_return_pct": 100.0 * bench_cum,
        "relative_cum_return_pct": 100.0 * (perf_cum - bench_cum),
    }
    risk_metrics = {
        "volatility_pct": 100.0 * vol_ann,
        "max_drawdown_pct": 100.0 * mdd,
        "var_95": var_95,
        "portfolio_value": portfolio_value_latest,
        "sharpe_ratio": ra["Sharpe Ratio"],
        "sortino_ratio": ra["Sortino Ratio"],
        "beta": ra["Beta vs Benchmark"],
        "alpha_pct": ra["Alpha vs Benchmark (%)"],
        "portfolio_cum_return_pct": 100.0 * perf_cum,
        "benchmark_cum_return_pct": 100.0 * bench_cum,
    }
    structure_insight = generate_structure_insight(alloc, class_series)

    run_forecast = st.button("Run Forecast", key="run_forecast_report")
    forecast_results = get_market_forecast_results(
        yahoo_tickers=last_yahoo_tickers,
        yahoo_to_display=last_yahoo_to_display,
        benchmark_symbol=last_benchmark_yahoo,
        force_refresh=run_forecast,
    )
    portfolio_forecast: dict[str, float] | None = None
    if forecast_results:
        portfolio_forecast = {
            "next_day_return_pct": 0.0,
            "week_return_pct": 0.0,
            "month_return_pct": 0.0,
            "six_month_return_pct": 0.0,
            "year_return_pct": 0.0,
            "prob_positive_pct": 0.0,
        }
        for item in forecast_results:
            asset = item["display_name"]
            weight = float(alloc.get(asset, 0.0))
            base_long_horizon = item["stats"]["20d"]["exp_return"]
            portfolio_forecast["next_day_return_pct"] += weight * item["stats"]["1d"]["exp_return"] * 100.0
            portfolio_forecast["week_return_pct"] += weight * item["stats"]["5d"]["exp_return"] * 100.0
            portfolio_forecast["month_return_pct"] += weight * _scaled_horizon_return(base_long_horizon, 21) * 100.0
            portfolio_forecast["six_month_return_pct"] += weight * _scaled_horizon_return(base_long_horizon, 126) * 100.0
            portfolio_forecast["year_return_pct"] += weight * _scaled_horizon_return(base_long_horizon, 252) * 100.0
            portfolio_forecast["prob_positive_pct"] += weight * item["stats"]["1d"]["prob_positive"] * 100.0

    suggestions, suggestion_df = generate_portfolio_improvement_suggestions(alloc, w_max_sharpe, asset_returns, rf_annual)
    allocation_enhancement_df, removed_assets = build_allocation_enhancement_table(alloc, w_max_sharpe)
    summary_text = generate_portfolio_summary(
        performance_metrics,
        risk_metrics,
        last_benchmark_label,
        structure_insight,
        forecast_metrics=portfolio_forecast,
    )
    risk_score, risk_band, risk_score_explanation = calculate_portfolio_risk_score(
        volatility_pct=100.0 * vol_ann,
        max_drawdown_pct=100.0 * mdd,
        sharpe_ratio=ra["Sharpe Ratio"],
        alloc=alloc,
        corr=corr,
    )

    st.markdown(
        f"""
        <div class="portfolio-summary-card">
            <h3>Portfolio Summary</h3>
            <p>{html.escape(summary_text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.header("Portfolio Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Portfolio Value", _format_currency(portfolio_value_latest, currency_symbol))
    s2.metric("Total Return", _format_pct(total_return_pct))
    s3.metric("Annualized Return", _format_pct(100.0 * perf_ann))
    s4.metric("Total Gain/Loss", _format_currency(total_gain_loss, currency_symbol))
    st.plotly_chart(fig_norm, use_container_width=True)
    with st.expander("Insight"):
        st.write(generate_snapshot_insight(snapshot_metrics, currency_symbol))

    st.divider()
    st.header("Future Outlook")
    st.caption("Forecasts use the existing model pipeline and are shown as horizon-based expectations over the next session, week, month, six months, and year rather than calendar-date predictions.")
    if portfolio_forecast is not None:
        pf1, pf2, pf3, pf4, pf5 = st.columns(5)
        pf1.metric("Portfolio Next Trading Day", _format_pct(portfolio_forecast["next_day_return_pct"]))
        pf2.metric("Portfolio End of This Week", _format_pct(portfolio_forecast["week_return_pct"]))
        pf3.metric("Portfolio End of This Month", _format_pct(portfolio_forecast["month_return_pct"]))
        pf4.metric("Portfolio End of Six Months", _format_pct(portfolio_forecast["six_month_return_pct"]))
        pf5.metric("Portfolio End of This Year", _format_pct(portfolio_forecast["year_return_pct"]))
        st.metric("Portfolio Positive Return Probability", _format_pct(portfolio_forecast["prob_positive_pct"]))
        st.caption("Next Trading Day = expected move next session. End of This Week = about 5 trading days. End of This Month = about 21 trading days. End of Six Months = about 126 trading days. End of This Year = about 252 trading days.")
        with st.expander("Portfolio Outlook Insight"):
            st.write(generate_forecast_explanation(portfolio_forecast, subject="portfolio"))

        forecast_labels = [item["display_name"] for item in forecast_results]
        selected_label = st.selectbox("Forecast asset", forecast_labels, key="forecast_asset")
        selected_forecast = next(item for item in forecast_results if item["display_name"] == selected_label)
        stats = selected_forecast["stats"]
        one = stats["1d"]
        wk = stats["5d"]
        base_long_horizon = stats["20d"]["exp_return"]
        month = _scaled_horizon_return(base_long_horizon, 21)
        six_month = _scaled_horizon_return(base_long_horizon, 126)
        yr = _scaled_horizon_return(base_long_horizon, 252)

        f1, f2, f3, f4, f5 = st.columns(5)
        f1.metric("Next Trading Day", _format_pct(one["exp_return"] * 100.0))
        f2.metric("End of This Week", _format_pct(wk["exp_return"] * 100.0))
        f3.metric("End of This Month", _format_pct(month * 100.0))
        f4.metric("End of Six Months", _format_pct(six_month * 100.0))
        f5.metric("End of This Year", _format_pct(yr * 100.0))
        st.metric("Probability of Positive Return", _format_pct(one["prob_positive"] * 100.0))

        forecast_table = pd.DataFrame(
            {
                "Asset": [item["display_name"] for item in forecast_results],
                "Next Trading Day (%)": [item["stats"]["1d"]["exp_return"] * 100.0 for item in forecast_results],
                "End of This Week (%)": [item["stats"]["5d"]["exp_return"] * 100.0 for item in forecast_results],
                "End of This Month (%)": [_scaled_horizon_return(item["stats"]["20d"]["exp_return"], 21) * 100.0 for item in forecast_results],
                "End of Six Months (%)": [_scaled_horizon_return(item["stats"]["20d"]["exp_return"], 126) * 100.0 for item in forecast_results],
                "End of This Year (%)": [_scaled_horizon_return(item["stats"]["20d"]["exp_return"], 252) * 100.0 for item in forecast_results],
                "Prob. Positive (%)": [item["stats"]["1d"]["prob_positive"] * 100.0 for item in forecast_results],
            }
        )
        st.dataframe(_index_from_one(forecast_table), use_container_width=True)

        with st.expander("Insight"):
            st.write(
                generate_forecast_explanation(
                    {
                        "next_day_return_pct": one["exp_return"] * 100.0,
                        "prob_positive_pct": one["prob_positive"] * 100.0,
                        "week_return_pct": wk["exp_return"] * 100.0,
                        "month_return_pct": month * 100.0,
                        "six_month_return_pct": six_month * 100.0,
                        "year_return_pct": yr * 100.0,
                    },
                    subject=selected_label,
                )
            )
    else:
        st.info("Click **Run Forecast** to train the existing models and populate the outlook section.")
    if run or run_forecast:
        st.markdown(
            """
            <script>
            const element = document.getElementById("top");
            if(element){
                element.scrollIntoView({behavior: "smooth"});
            }
            </script>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.header("Portfolio Growth")
    st.plotly_chart(fig_pv, use_container_width=True)
    if not rr_df.empty:
        rr_df_plot = (100.0 * rr_df).rename_axis("Date").reset_index()
        rr_long = rr_df_plot.melt(id_vars="Date", var_name="Window", value_name="Return (%)")
        fig_rr = px.line(rr_long, x="Date", y="Return (%)", color="Window", template="plotly_white")
        st.plotly_chart(fig_rr, use_container_width=True)
    if not dd.empty:
        dd_df = (100.0 * dd.rename("Drawdown (%)")).rename_axis("Date").reset_index()
        fig_dd = px.area(dd_df, x="Date", y="Drawdown (%)", template="plotly_white")
        fig_dd.update_yaxes(ticksuffix="%")
        st.plotly_chart(fig_dd, use_container_width=True)
    with st.expander("Insight"):
        st.write(generate_growth_insight(portfolio_value, currency_symbol))

    st.divider()
    st.header("Performance Analysis")
    p1, p2, p3 = st.columns(3)
    p1.metric("Cumulative Return", _format_pct(100.0 * perf_cum))
    p2.metric("Annualized Return", _format_pct(100.0 * perf_ann))
    p3.metric("Benchmark Comparison", _format_pct(100.0 * (perf_cum - bench_cum)))
    comp_df = pd.DataFrame(
        {
            "Metric": ["Cumulative Return", "Annualized Return", "Volatility", "Sharpe Ratio", "Max Drawdown"],
            "Portfolio": [
                f"{100.0 * perf_cum:.2f}%",
                f"{100.0 * perf_ann:.2f}%",
                f"{100.0 * vol_ann:.2f}%",
                f"{ra['Sharpe Ratio']:.3f}",
                f"{100.0 * mdd:.2f}%",
            ],
            "Benchmark": [
                f"{100.0 * bench_cum:.2f}%",
                f"{100.0 * bench_ann:.2f}%",
                f"{100.0 * bench_vol:.2f}%",
                f"{bench_ra['Sharpe Ratio']:.3f}",
                f"{100.0 * bench_mdd:.2f}%",
            ],
        }
    )
    st.dataframe(_index_from_one(comp_df), use_container_width=True)
    with st.expander("Insight"):
        st.write(generate_performance_insight(performance_metrics, last_benchmark_label))

    st.divider()
    st.header("Portfolio Risk Score")
    rs1, rs2 = st.columns([1, 3])
    rs1.metric("Risk Score", f"{risk_score:.1f} / 10")
    rs2.metric("Risk Band", risk_band)
    st.write(risk_score_explanation)

    st.divider()
    st.header("Risk Analysis")
    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Volatility", _format_pct(100.0 * vol_ann))
    r2.metric("Maximum Drawdown", _format_pct(100.0 * mdd))
    r3.metric("Value at Risk", _format_currency(var_95, currency_symbol))
    r4.metric("Sharpe Ratio", f"{ra['Sharpe Ratio']:.3f}")
    r5.metric("Sortino Ratio", f"{ra['Sortino Ratio']:.3f}")
    beta_col, alpha_col = st.columns(2)
    beta_col.metric("Beta vs Benchmark", f"{ra['Beta vs Benchmark']:.3f}")
    alpha_col.metric("Alpha vs Benchmark", _format_pct(ra["Alpha vs Benchmark (%)"]))
    with st.expander("Insight"):
        st.write(generate_risk_explanation(risk_metrics, currency_symbol, last_benchmark_label))

    st.divider()
    st.header("Portfolio Structure")
    st.plotly_chart(fig_alloc, use_container_width=True)
    st.subheader("Asset Performance")
    st.dataframe(_index_from_one(asset_perf), use_container_width=True)
    st.subheader("Asset Class Summary")
    st.dataframe(_index_from_one(class_df), use_container_width=True)
    with st.expander("Insight"):
        st.write(structure_insight)

    st.divider()
    st.header("Advanced Analytics")
    if not corr.empty:
        fig_corr = px.imshow(
            corr,
            x=corr.columns,
            y=corr.columns,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            aspect="auto",
            template="plotly_white",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.subheader("Correlation Insight")
        st.write(generate_heatmap_insight(corr))
    if not frontier_df.empty:
        fig_frontier = px.scatter(
            frontier_df,
            x="vol",
            y="ret",
            color="sharpe",
            color_continuous_scale="Viridis",
            template="plotly_white",
            labels={"vol": "Risk (Volatility)", "ret": "Expected Return"},
        )
        fig_frontier.update_traces(marker=dict(size=5), opacity=0.55, selector=dict(mode="markers"))
        fig_frontier.update_layout(margin=dict(l=30, r=30, t=40, b=30))
        fig_frontier.add_scatter(
            x=[frontier_df.loc[frontier_df["sharpe"].idxmax(), "vol"]],
            y=[frontier_df.loc[frontier_df["sharpe"].idxmax(), "ret"]],
            mode="markers",
            marker=dict(color="#ff4d6d", size=14, line=dict(color="white", width=1.5)),
            name="Max Sharpe",
        )
        fig_frontier.add_scatter(
            x=[frontier_df.loc[frontier_df["vol"].idxmin(), "vol"]],
            y=[frontier_df.loc[frontier_df["vol"].idxmin(), "ret"]],
            mode="markers",
            marker=dict(color="orange", size=10, line=dict(color="white", width=1)),
            name="Min Vol",
        )
        st.plotly_chart(fig_frontier, use_container_width=True)
    st.subheader("Optimal Portfolio Allocation")
    if not w_max_sharpe.empty:
        opt_df = (w_max_sharpe / w_max_sharpe.sum()).reset_index()
        opt_df.columns = ["Asset", "Optimal Weight (%)"]
        opt_df["Optimal Weight (%)"] = opt_df["Optimal Weight (%)"] * 100.0
        st.dataframe(_index_from_one(opt_df), use_container_width=True)
    with st.expander("Insight"):
        st.write(generate_advanced_insight(corr, frontier_df, w_max_sharpe))

    st.divider()
    st.header("Portfolio Improvement Suggestions")
    if suggestions:
        for idx, suggestion in enumerate(suggestions, start=1):
            st.markdown(f"**Suggested Adjustment {idx}.** {suggestion}")
    elif w_max_sharpe.empty:
        st.info("Optimization suggestions will appear when the efficient frontier finds a meaningfully better allocation than the current mix.")
    else:
        st.caption("The optimizer does not suggest a large enough reallocation to produce a separate narrative adjustment, but the full allocation comparison is shown below.")

    if not allocation_enhancement_df.empty:
        styled_allocation_df = allocation_enhancement_df.style.format(
            {
                "Current Allocation": "{:.1f}%",
                "Suggested Allocation": "{:.1f}%",
            }
        ).apply(
            lambda row: [
                "font-weight: 700; color: #ff4d6d;" if row["Suggested Allocation"] <= 0.0001 and col == "Suggested Allocation" else ""
                for col in allocation_enhancement_df.columns
            ],
            axis=1,
        )
        st.dataframe(styled_allocation_df, use_container_width=True)
        if removed_assets:
            st.subheader("Removed Assets (0% Allocation)")
            st.write(", ".join(removed_assets))
            st.caption(
                "These assets were excluded from the optimized portfolio because they either reduced the portfolio's Sharpe ratio, increased volatility, or were highly correlated with stronger assets."
            )

    st.divider()
    _render_ai_assistant()


if __name__ == "__main__":
    main()
