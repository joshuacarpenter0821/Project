import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

st.title("📈 Starbucks Revenue Forecast (Simplified ARIMA Model)")

@st.cache_data
def load_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)
    df = df[["revenue", "CPI"]].dropna()
    return df

df = load_data()

if len(df) < 12:
    st.error("Not enough data to fit the model.")
    st.stop()

growth_input = st.slider("Expected Quarterly Revenue Growth (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.5)

endog = df["revenue"]
exog = df[["CPI"]]

# Fit ARIMA model
model = ARIMA(endog, order=(1, 1, 1), exog=exog)
results = model.fit()

# Forecast the next 4 quarters
last_exog = exog.iloc[-1].values[0]
future_exog = pd.DataFrame({"CPI": [last_exog] * 4})
forecast = results.get_forecast(steps=4, exog=future_exog)
pred = forecast.predicted_mean * (1 + growth_input / 100)
conf_int = forecast.conf_int() * (1 + growth_input / 100)

# Set index for forecast
latest_date = df.index[-1]
forecast_dates = pd.date_range(start=latest_date + pd.offsets.QuarterEnd(), periods=4, freq="Q")
pred.index = forecast_dates
conf_int.index = forecast_dates

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
df["revenue"].plot(ax=ax, label="Actual Revenue")
pred.plot(ax=ax, label="Forecasted Revenue", color="green")
ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color="green", alpha=0.2)
ax.set_title("Starbucks Revenue Forecast with ARIMA + CPI")
ax.set_ylabel("Revenue (Millions)")
ax.legend()
st.pyplot(fig)

st.subheader("Model Summary")
st.text(results.summary())






