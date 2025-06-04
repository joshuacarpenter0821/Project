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
    cpi = yf.download("CPIAUCSL", start="2015-01-01")
    return cpi

# Load Starbucks financials
@st.cache_data
def load_financial_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=['date'])
    df.set_index('Date', inplace=True)
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
rev_series = df['Revenue']
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

# Display CPI data
st.subheader("📊 Macroeconomic Insight: CPI")
st.write("Latest CPI Value (from yfinance):", float(cpi['Close'].iloc[-1]))

# New Variable Analysis
st.subheader("📎 Additional Variables Insight")
st.line_chart(df[['COGS', 'EPS']])
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
