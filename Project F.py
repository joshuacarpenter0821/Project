import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.title("📈 Starbucks Revenue Forecast (SARIMAX with CPI)")

@st.cache_data
def load_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"])
    df.set_index("date", inplace=True)
    df = df[["revenue", "CPI"]].dropna()
    return df

df = load_data()

if len(df) < 16:  # seasonal model needs enough data
    st.error("Not enough data to fit the SARIMAX model (need at least 16 records).")
    st.stop()

growth_input = st.slider("Expected Quarterly Revenue Growth (%)", min_value=-10.0, max_value=10.0, value=2.0, step=0.5)

endog = df["revenue"]
exog = df[["CPI"]]

periods = 4  # forecast next 4 quarters

# Prepare future exog: repeat last CPI value for simplicity
last_cpi = exog.iloc[-1].values[0]
future_exog = pd.DataFrame({"CPI": [last_cpi] * periods})

def forecast_revenue_arimax(endog, exog, periods, future_exog):
    # Fit SARIMAX on full data
    model = SARIMAX(endog, exog=exog, order=(1,1,1), seasonal_order=(1,1,1,4))
    results = model.fit(disp=False)
    # Forecast future periods
    forecast_obj = results.get_forecast(steps=periods, exog=future_exog)
    return results, forecast_obj.predicted_mean, forecast_obj.conf_int()

results, forecast_mean, conf_int = forecast_revenue_arimax(endog, exog, periods, future_exog)

# Apply user growth input (simple multiplier)
forecast_mean_adj = forecast_mean * (1 + growth_input / 100)
conf_int_adj = conf_int * (1 + growth_input / 100)

# Create forecast dates index
forecast_dates = pd.date_range(start=df.index[-1] + pd.offsets.QuarterEnd(), periods=periods, freq="Q")
forecast_mean_adj.index = forecast_dates
conf_int_adj.index = forecast_dates

# Plot actual + forecast
fig, ax = plt.subplots(figsize=(10, 5))
df["revenue"].plot(ax=ax, label="Actual Revenue")
forecast_mean_adj.plot(ax=ax, label="Forecasted Revenue", color="green")
ax.fill_between(conf_int_adj.index, conf_int_adj.iloc[:, 0], conf_int_adj.iloc[:, 1], color="green", alpha=0.2)
ax.set_title("Starbucks Revenue Forecast with SARIMAX + CPI")
ax.set_ylabel("Revenue (Millions)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

st.subheader("SARIMAX Model Summary")
st.text(results.summary())
