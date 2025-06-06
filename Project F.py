import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_financial_data_with_cpi():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def forecast_revenue_arimax(data, exog, periods=4):
    # Align indices and drop missing values to ensure matching indices
    combined = pd.concat([data, exog], axis=1).dropna()
    
    data_aligned = combined.iloc[:, 0]   # revenue (endog)
    exog_aligned = combined.iloc[:, 1:]  # CPI or others (exog)
    
    # Training data slices
    data_train = data_aligned.iloc[:-periods]
    exog_train = exog_aligned.iloc[:-periods]
    
    # Forecast exogenous variables for the forecast horizon
    exog_forecast = exog_aligned.iloc[-periods:]
    
    # Check indices alignment
    assert data_train.index.equals(exog_train.index), "Indices for endog and exog are not aligned after slicing!"
    
    model = SARIMAX(data_train, exog=exog_train, order=(1,1,1), seasonal_order=(1,1,1,4))
    results = model.fit()
    
    forecast = results.get_forecast(steps=periods, exog=exog_forecast)
    return forecast.predicted_mean, forecast.conf_int()

# Streamlit app starts here
st.title("📈 Starbucks Revenue Forecast App")

df = load_financial_data_with_cpi()

if 'CPI' not in df.columns:
    st.error("CPI column not found in the data file.")
    st.stop()

growth_input = st.slider("Expected Quarterly Revenue Growth (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.5)

rev_series = df['revenue']
cpi_exog = df[['CPI']]

if len(df) < 12:
    st.error("Not enough data in the CSV file. Please provide more historical data.")
    st.stop()

forecast, conf_int = forecast_revenue_arimax(rev_series, cpi_exog)
latest_date = rev_series.index[-1]
forecast_dates = pd.date_range(start=latest_date + pd.offsets.QuarterEnd(), periods=4, freq='Q')

adjusted_forecast = forecast * (1 + growth_input / 100)
adjusted_forecast.index = forecast_dates
conf_int_scaled = conf_int * (1 + growth_input / 100)
conf_int_scaled.index = forecast_dates

fig, ax = plt.subplots(figsize=(10, 5))
rev_series.plot(ax=ax, label='Actual Revenue')
adjusted_forecast.plot(ax=ax, label='Forecasted Revenue (ARIMAX)', color='green')
ax.fill_between(forecast_dates, conf_int_scaled.iloc[:, 0], conf_int_scaled.iloc[:, 1], color='green', alpha=0.2)
ax.set_title("Starbucks Revenue Forecast with CPI (ARIMAX)")
ax.set_ylabel("Revenue (Millions)")
ax.legend()
st.pyplot(fig)

st.subheader("📊 Macroeconomic Insight: CPI")
latest_cpi = df['CPI'].iloc[-1]
avg_cpi = df['CPI'].mean()
st.write("Latest CPI Value:", latest_cpi)
st.line_chart(df['CPI'])

if latest_cpi > avg_cpi * 1.05:
    st.warning("📈 CPI is significantly above average. Inflation may pressure input costs.")
elif latest_cpi < avg_cpi * 0.95:
    st.success("📉 CPI is below average, suggesting lower inflationary pressures.")
else:
    st.info("CPI is near its recent average. Inflation appears stable.")

st.subheader("📎 Additional Variables Insight")
st.line_chart(df[['COGS', 'EPS']])
if df['EPS'].iloc[-1] < df['EPS'].mean() * 0.8:
    st.warning("⚠️ EPS has dropped significantly compared to historical average. Potential earnings risk.")
st.markdown("COGS helps evaluate gross margin trends, while EPS offers insights into per-share profitability trends over time.")

st.subheader("🧠 AI-Generated Audit Committee Summary")
ai_summary = (
    "Using an ARIMAX model that incorporates CPI data, Starbucks’ revenue forecast reflects macroeconomic conditions. "
    f"Current CPI is {latest_cpi:.2f}, suggesting {'rising' if latest_cpi > avg_cpi else 'moderate or easing'} inflation pressure. "
    "EPS stability and margin insights from COGS provide additional context. Monitoring inflation trends is advised."
)
st.info(ai_summary)

st.caption("Developed for ITEC 3155 / ACTG 4155 – Spring 2025 Final Project")




