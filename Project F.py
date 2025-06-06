import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Use FRED for CPI data
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

# Forecasting with ARIMAX (revenue with CPI as exogenous)
def forecast_revenue_arimax(data, exog, periods=4):
    model = SARIMAX(data, exog=exog[:-periods], order=(1,1,1), seasonal_order=(1,1,1,4))
    results = model.fit()
    forecast = results.get_forecast(steps=periods, exog=exog[-periods:])
    return forecast.predicted_mean, forecast.conf_int()

# Load data
st.title("📈 Starbucks Revenue Forecast App")
df = load_financial_data()
cpi = get_cpi_data()

# User input for expected quarterly revenue growth rate
growth_input = st.slider("Expected Quarterly Revenue Growth (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.5)

# Resample CPI to quarterly frequency and align with revenue
cpi_quarterly = cpi.resample('Q').mean()
combined = pd.merge(df[['revenue']], cpi_quarterly, left_index=True, right_index=True, how='inner')

# Ensure enough data
if len(combined) < 12:
    st.error("Not enough data after merging CPI and revenue. Extend the data range if possible.")
    st.stop()

rev_series = combined['revenue']
cpi_exog = combined[['CPI']]

# Forecast using ARIMAX
forecast, conf_int = forecast_revenue_arimax(rev_series, cpi_exog)
latest_date = rev_series.index[-1]
forecast_dates = pd.date_range(start=latest_date + pd.offsets.QuarterEnd(), periods=4, freq='Q')

# Apply user growth assumption
adjusted_forecast = forecast * (1 + growth_input / 100)
adjusted_forecast.index = forecast_dates
conf_int.index = forecast_dates

# Plot forecast
fig, ax = plt.subplots(figsize=(10, 5))
rev_series.plot(ax=ax, label='Actual Revenue')
adjusted_forecast.plot(ax=ax, label='Forecasted Revenue (ARIMAX)', color='green')
ax.fill_between(forecast_dates, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='green', alpha=0.2)
ax.set_title("Starbucks Revenue Forecast with CPI (ARIMAX)")
ax.set_ylabel("Revenue (Millions)")
ax.legend()
st.pyplot(fig)

# CPI Insight
st.subheader("📊 Macroeconomic Insight: CPI")
if cpi.empty:
    st.error("CPI data could not be retrieved.")
    st.stop()
else:
    latest_cpi = float(cpi['CPI'].iloc[-1])
    avg_cpi = cpi['CPI'].mean()
    st.write("Latest CPI Value (from FRED):", latest_cpi)
    st.line_chart(cpi)

    if latest_cpi > avg_cpi * 1.05:
        st.warning("📈 CPI is significantly above average. Inflation may pressure input costs.")
    elif latest_cpi < avg_cpi * 0.95:
        st.success("📉 CPI is below average, suggesting lower inflationary pressures.")
    else:
        st.info("CPI is near its recent average. Inflation appears stable.")

# New Variable Analysis
st.subheader("📎 Additional Variables Insight")
st.line_chart(df[['COGS', 'EPS']])
if df['EPS'].iloc[-1] < df['EPS'].mean() * 0.8:
    st.warning("⚠️ EPS has dropped significantly compared to historical average. Potential earnings risk.")
st.markdown("COGS helps evaluate gross margin trends, while EPS offers insights into per-share profitability trends over time.")

# AI-Generated Summary
st.subheader("🧠 AI-Generated Audit Committee Summary")
ai_summary = (
    "Using an ARIMAX model that incorporates CPI data, Starbucks’ revenue forecast reflects macroeconomic conditions. "
    f"Current CPI is {latest_cpi:.2f}, suggesting {'rising' if latest_cpi > avg_cpi else 'moderate or easing'} inflation pressure. "
    "EPS stability and margin insights from COGS provide additional context. Monitoring inflation trends is advised."
)
st.info(ai_summary)

# Footer
st.caption("Developed for ITEC 3155 / ACTG 4155 – Spring 2025 Final Project")

