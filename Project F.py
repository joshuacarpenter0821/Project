import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

@st.cache_data
def load_financial_data_with_cpi():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def forecast_revenue_arimax(endog, exog, forecast_periods=4):
    # Fit ARIMA(1,1,1) with exog
    model = ARIMA(endog, order=(1,1,1), exog=exog)
    results = model.fit()

    # For forecasting, exog must contain the future exog values
    exog_forecast = exog.tail(forecast_periods)
    forecast = results.get_forecast(steps=forecast_periods, exog=exog_forecast)
    return forecast.predicted_mean, forecast.conf_int()

# Streamlit UI
st.title("📈 Starbucks Revenue Forecast App (Simplified ARIMAX)")

df = load_financial_data_with_cpi()

# Check required columns
required_cols = ['revenue', 'CPI']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    st.error(f"Missing required columns in data: {missing_cols}")
    st.stop()

# Optional GDP growth if available
exog_vars = ['CPI']
if 'GDP_Growth' in df.columns:
    exog_vars.append('GDP_Growth')

exog = df[exog_vars]
endog = df['revenue']

# User input for expected quarterly revenue growth adjustment
growth_input = st.slider("Expected Quarterly Revenue Growth (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.5)

if len(df) < 12:
    st.error("Not enough data in the CSV file. Please provide more historical data.")
    st.stop()

# Forecast
forecast_periods = 4
forecast, conf_int = forecast_revenue_arimax(endog, exog, forecast_periods=forecast_periods)

# Create future dates starting after last date
latest_date = endog.index[-1]
forecast_dates = pd.date_range(start=latest_date + pd.offsets.QuarterEnd(), periods=forecast_periods, freq='Q')

# Adjust forecast by growth input
adjusted_forecast = forecast * (1 + growth_input / 100)
adjusted_forecast.index = forecast_dates

conf_int_scaled = conf_int * (1 + growth_input / 100)
conf_int_scaled.index = forecast_dates

# Plot actual vs forecast
fig, ax = plt.subplots(figsize=(10, 5))
endog.plot(ax=ax, label='Actual Revenue')
adjusted_forecast.plot(ax=ax, label='Forecasted Revenue (ARIMAX)', color='green')
ax.fill_between(forecast_dates, conf_int_scaled.iloc[:, 0], conf_int_scaled.iloc[:, 1], color='green', alpha=0.2)
ax.set_title("Starbucks Revenue Forecast with CPI (ARIMAX, Simplified)")
ax.set_ylabel("Revenue (Millions)")
ax.legend()
st.pyplot(fig)

# CPI insights
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

# Additional variables insight
st.subheader("📎 Additional Variables Insight")
if 'COGS' in df.columns and 'EPS' in df.columns:
    st.line_chart(df[['COGS', 'EPS']])
    if df['EPS'].iloc[-1] < df['EPS'].mean() * 0.8:
        st.warning("⚠️ EPS has dropped significantly compared to historical average. Potential earnings risk.")
    st.markdown("COGS helps evaluate gross margin trends, while EPS offers insights into per-share profitability trends over time.")

# AI summary
st.subheader("🧠 AI-Generated Audit Committee Summary")
ai_summary = (
    "Using a simplified ARIMAX model incorporating CPI data (and GDP growth if available), Starbucks’ revenue forecast "
    f"reflects current macroeconomic conditions. Current CPI is {latest_cpi:.2f}, indicating "
    f"{'rising' if latest_cpi > avg_cpi else 'stable or easing'} inflation pressure. Additional financial metrics like EPS and COGS "
    "offer complementary insights. Monitoring inflation trends is advised."
)
st.info(ai_summary)

# Footer
st.caption("Developed for ITEC 3155 / ACTG 4155 – Spring 2025 Final Project")



