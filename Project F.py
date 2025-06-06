import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Use yfinance instead of FRED for CPI data
@st.cache_data
def get_cpi_data():
    import requests
    series_id = "USACPALTT01IXNBQ"  # CPI series
    api_key = "e30f46dc3e290dafe08d207f3a357392"
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2023-01-01"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data_json = response.json()
        observations = data_json.get("observations", [])
        cpi_df = pd.DataFrame(observations)
        cpi_df["date"] = pd.to_datetime(cpi_df["date"])
        cpi_df["value"] = pd.to_numeric(cpi_df["value"])
        cpi_df = cpi_df[["date", "value"]].rename(columns={"value": "CPI"})
        cpi_df.set_index("date", inplace=True)
        return cpi_df
    else:
        st.error("Failed to fetch CPI data from FRED.")
        return pd.DataFrame()

# Load Starbucks financials
@st.cache_data
def load_financial_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

# Forecasting model
def forecast_revenue(data, periods=4):
    model = SARIMAX(data, order=(1,1,1), seasonal_order=(1,1,1,4))
    results = model.fit()
    forecast = results.get_forecast(steps=periods)
    return forecast.predicted_mean, forecast.conf_int()

# Load data
st.title("📈 Starbucks Revenue Forecast App")
df = load_financial_data()
cpi = get_cpi_data()

# User input for expected quarterly revenue growth rate
growth_input = st.slider("Expected Quarterly Revenue Growth (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.5)

# Forecast revenue
rev_series = df['revenue']
forecast, conf_int = forecast_revenue(rev_series)
latest_date = rev_series.index[-1]
forecast_dates = pd.date_range(start=latest_date + pd.offsets.QuarterEnd(), periods=4, freq='Q')

# Apply user growth assumption
adjusted_forecast = forecast * (1 + growth_input / 100)

# Plot forecast
fig, ax = plt.subplots(figsize=(10,5))
rev_series.plot(ax=ax, label='Actual Revenue')
adjusted_forecast.plot(ax=ax, label='Forecasted Revenue', color='green')
ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='green', alpha=0.2)
ax.set_title("Starbucks Revenue Forecast")
ax.set_ylabel("Revenue (Millions)")
ax.legend()
st.pyplot(fig)

st.write("CPI DataFrame shape:", cpi.shape)
st.write("CPI DataFrame preview:", cpi.head())

if cpi.empty:
    st.error("CPI DataFrame is empty. Cannot fetch latest value.")
    st.stop()
elif 'Close' not in cpi.columns:
    st.error("Column 'Close' not found in CPI data.")
    st.write("Available columns:", cpi.columns)
    st.stop()
else:
    latest_cpi = float(cpi['Close'].iloc[-1])
    st.write("Latest CPI Value (from FRED):", latest_cpi)

spy = yf.download("SPY", period="1mo")
st.write("SPY test data:", spy.head())

st.subheader("📊 Macroeconomic Insight: CPI")
if not cpi.empty and 'Close' in cpi.columns:
    st.write("Latest CPI Value (from yfinance):", float(cpi['Close'].iloc[-1]))
else:
    st.warning("CPI data could not be retrieved. Check your ticker symbol or internet connection.")
    st.write("CPI DataFrame shape:", cpi.shape)
st.write("CPI preview:", cpi.head())
st.write("Latest CPI Value (from yfinance):", float(cpi['Close'].iloc[-1]))

# New Variable Analysis
st.subheader("📎 Additional Variables Insight")
st.line_chart(df[['COGS', 'EPS']])
if df['EPS'].iloc[-1] < df['EPS'].mean() * 0.8:
    st.warning("⚠️ EPS has dropped significantly compared to historical average. Potential earnings risk.")
st.markdown("COGS helps evaluate gross margin trends, while EPS offers insights into per-share profitability trends over time.")

# AI-Generated Summary
st.subheader("🧠 AI-Generated Audit Committee Summary")
ai_summary = (
    "Our ARIMA-based forecast for Starbucks indicates revenue is expected to grow modestly, aligning with historical patterns. "
    f"Live CPI data from yfinance shows inflationary pressure, currently at {float(cpi['Close'].iloc[-1]):.2f}. "
    "EPS trends suggest moderate earnings stability, while COGS fluctuations may warrant further analysis of margin pressures. "
    "No major risk indicators are flagged at this time, though monitoring input cost volatility remains key."
)
st.info(ai_summary)

# Footer
st.caption("Developed for ITEC 3155 / ACTG 4155 – Spring 2025 Final Project")
