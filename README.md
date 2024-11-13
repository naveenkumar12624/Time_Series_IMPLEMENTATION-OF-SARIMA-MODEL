### Date: 
### Developed by : Naveen Kumar S
### Register No : 212221240033

# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL

### AIM:
To implement SARIMA model using python.

### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
   
### PROGRAM:
```py
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Load your dataset
data = pd.read_csv('C:/Users/lenovo/Downloads/archive (2)/NVIDIA/NvidiaStockPrice.csv')  # Replace with the correct path to your dataset

# Ensure 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Select the 'Close' price for time series forecasting
time_series = data['Close']

# Plot the time series data
plt.figure(figsize=(10, 6))
plt.plot(time_series)
plt.title('Stock Close Price Time Series Plot')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.show()

# Function to test stationarity
def test_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    for key, value in result[4].items():
        print(f'Critical Value {key}: {value}')
    if result[1] < 0.05:
        print("The time series is stationary.")
    else:
        print("The time series is not stationary.")

test_stationarity(time_series)

# Differencing the series to make it stationary if necessary
time_series_diff = time_series.diff().dropna()
print("\nAfter Differencing:")
test_stationarity(time_series_diff)

# Set initial SARIMA parameters (p, d, q, P, D, Q, m)
p, d, q = 1, 1, 1
P, D, Q, m = 1, 1, 1, 12  # m = 12 for monthly seasonality if applicable

# Build the SARIMA model
model = SARIMAX(time_series, order=(p, d, q), seasonal_order=(P, D, Q, m), enforce_stationarity=False, enforce_invertibility=False)
sarima_fit = model.fit(disp=False)

# Print the summary of the SARIMA model
print(sarima_fit.summary())

# Forecast for the next 12 periods (adjust forecast period as needed)
forecast_steps = 12  # Adjust forecast period as needed
forecast = sarima_fit.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Plot the forecast
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Historical Data')
plt.plot(forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_ci.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.title('SARIMA Forecast of Stock Close Prices')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.show()

# Calculate and print Mean Absolute Error (MAE) for model performance
from sklearn.metrics import mean_absolute_error
test_data = time_series[-forecast_steps:]  # The last 'forecast_steps' data points for comparison
pred_data = forecast.predicted_mean[:len(test_data)]  # Corresponding predicted data points
mae = mean_absolute_error(test_data, pred_data)
print('Mean Absolute Error:', mae)

```

### OUTPUT:
![image](https://github.com/user-attachments/assets/1e18259b-1248-4aad-b4a7-755a7716f414)

![image](https://github.com/user-attachments/assets/545c33c7-3f3f-424e-bed3-53f0546ec9ca)

![image](https://github.com/user-attachments/assets/4bc5141c-47a7-4157-9af9-6c2d92066fdc)

### RESULT:
Thus the program run successfully based on the SARIMA model.
