# Streamlit Portfolio Performance (Yahoo Finance)

Small Streamlit web app that:
- Accepts stock tickers (comma/newline separated)
- Fetches price data from Yahoo Finance via `yfinance`
- Plots normalized ticker performance and a portfolio equity curve

## Setup

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -r requirements.txt
```

## Run

```bash
python -m streamlit run app.py
```

## Notes

- Uses **Adjusted Close** by default (you can toggle it off).
- Default market is **India (NSE)**. You can enter `RELIANCE, TCS, INFY` and the app will fetch `RELIANCE.NS, TCS.NS, INFY.NS` internally.
- Portfolio modes:
  - **Daily**: fixed weights each day (daily rebalancing)
  - **Buy & Hold**: buy shares on day 1 and let weights drift

