import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Read the data
ds1 = xr.open_dataset(r'C:\Users\Mosta\Desktop\OO ANALYSIS AND DESIGN\Final report\Final Report 202253085011+AKIB\Data\data_stream-oper_stepType-instant.nc')
ds2 = xr.open_dataset(r'C:\Users\Mosta\Desktop\OO ANALYSIS AND DESIGN\Final report\Final Report 202253085011+AKIB\Data\data_stream-oper_stepType-accum.nc')

# 2. Extract data for Beijing (39.906217N, 116.3912757E)
latitude = 39.906217
longitude = 116.3912757

# Find the closest latitude and longitude indices
lat_idx = np.abs(ds1.latitude - latitude).argmin()
lon_idx = np.abs(ds1.longitude - longitude).argmin()

# Extract time series data for the closest grid point
time = ds1.valid_time.values
t2m = ds1.t2m[:, lat_idx, lon_idx].values - 273.15  # Temperature (convert from Kelvin to Celsius)
u10 = ds1.u10[:, lat_idx, lon_idx].values  # 10m U wind component (m/s)
v10 = ds1.v10[:, lat_idx, lon_idx].values  # 10m V wind component (m/s)
tp = ds2.tp[:, lat_idx, lon_idx].values  # Total precipitation (m)

# 3. Calculate wind speed
wind_speed = np.sqrt(u10**2 + v10**2)

# 4. Combine the extracted data into a DataFrame
data = pd.DataFrame({
    "time": pd.to_datetime(time),
    "temperature": t2m,
    "wind_speed": wind_speed,
    "precipitation": tp
})

# 5. Create lagged features
def create_lagged_features(df, lag=3):
    features = []
    for col in ['temperature', 'wind_speed']:
        for i in range(1, lag+1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)
        features.extend([f'{col}_lag_{i}' for i in range(1, lag+1)])
    return df.dropna(), features

data, lagged_features = create_lagged_features(data)

# 6. Create features and target
X = data[lagged_features].values
y = data["precipitation"].values

# 7. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False  # Maintain time sequence
)

# Standardize the features using StandardScaler
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Standardize the target variable
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).ravel()

# 8. Train the SVR model
model = SVR(kernel="rbf", C=100, epsilon=0.1, gamma='scale')
model.fit(X_train_scaled, y_train_scaled)

# 9. Make predictions on the test data
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# 10. Evaluate results
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# 11. Visualizations

# 11.1 Time series plot
daily_data = data.set_index('time').resample('D').agg({
    'temperature': 'mean',
    'precipitation': 'sum'
})
daily_data = daily_data.loc['2021-01-01':'2022-12-31']

fig, ax1 = plt.subplots(figsize=(12, 6))
ax2 = ax1.twinx()
ax1.plot(daily_data.index, daily_data['temperature'], 'r-', label='Temperature')
ax2.bar(daily_data.index, daily_data['precipitation'], alpha=0.3, label='Precipitation')
ax1.set_xlabel('Date')
ax1.set_ylabel('Temperature (°C)', color='r')
ax2.set_ylabel('Precipitation (m)', color='b')
plt.title('Daily Average Temperature and Total Precipitation in Beijing (2021-2022)')
fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
plt.tight_layout()
plt.show()

# 11.2 Heatmap
specific_time = np.datetime64('2022-07-01T12:00')

# Use the nearest method to find the closest timestamp in 'valid_time'
temp_data = ds1.t2m.sel(valid_time=specific_time, method='nearest') - 273.15  # Convert from Kelvin to Celsius

plt.figure(figsize=(10, 8))
temp_data.plot()
plt.title(f'Temperature Distribution on {specific_time}')
plt.tight_layout()
plt.show()

# 11.3 Scatter plot
plt.figure(figsize=(8, 8))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Precipitation (m)')
plt.ylabel('Predicted Precipitation (m)')
plt.title('Actual vs Predicted Precipitation')
plt.tight_layout()
plt.show()

# 11.4 Residual plot
residuals = y_test - y_pred
test_times = data['time'].iloc[-len(y_test):]  # Get the actual timestamps for the test set

plt.figure(figsize=(12, 6))
plt.scatter(test_times, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 11.5 Feature importance bar chart
feature_importance = np.abs(model.dual_coef_[0]) @ model.support_vectors_
feature_names = lagged_features
sorted_idx = np.argsort(feature_importance)
plt.figure(figsize=(10, 6))
plt.barh(range(len(feature_names)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_names)), [feature_names[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Precipitation Prediction")
plt.tight_layout()
plt.show()
