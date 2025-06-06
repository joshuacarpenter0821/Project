import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load your data
df = pd.read_csv("starbucks_data.csv", parse_dates=["date"])

# Set 'date' as index and sort it (important!)
df.set_index("date", inplace=True)
df.sort_index(inplace=True)

# Select endogenous and exogenous variables
endog = df["revenue"]
exog = df[["CPI"]]

# Ensure endog and exog indices align exactly
common_index = endog.index.intersection(exog.index)
endog_aligned = endog.loc[common_index]
exog_aligned = exog.loc[common_index]

def forecast_revenue_arimax(endog, exog, periods, future_exog):
    model = SARIMAX(
        endog,
        exog=exog,
        order=(1,1,1),
        seasonal_order=(1,1,1,4),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    # Forecast next 'periods' with provided future exog values
    forecast_obj = results.get_forecast(steps=periods, exog=future_exog)
    forecast = forecast_obj.predicted_mean
    conf_int = forecast_obj.conf_int()
    
    return results, forecast, conf_int

# Number of quarters to forecast
periods = 4

# Prepare future exog DataFrame (same columns as exog_aligned)
# Here we just repeat the last known CPI value for simplicity
last_cpi = exog_aligned["CPI"].iloc[-1]
future_cpi = pd.DataFrame(
    {"CPI": [last_cpi]*periods},
    index=pd.date_range(start=exog_aligned.index[-1] + pd.offsets.QuarterEnd(), periods=periods, freq='Q')
)

# Call forecast function
results, forecast, conf_int = forecast_revenue_arimax(endog_aligned, exog_aligned, periods, future_cpi)

print("Forecasted Revenue:")
print(forecast)

print("\nConfidence Intervals:")
print(conf_int)

