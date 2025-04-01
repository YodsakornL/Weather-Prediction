import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
import datetime
import requests
import matplotlib.pyplot as plt


# For train modle(Predict and compare with previous data)
weather = pd.read_excel(r"C:\Users\ASUS\OneDrive\Desktop\AI project\filled.xlsx", index_col="date")
null_pct = weather.apply(pd.isnull).sum()/weather.shape[0]
valid_columns = weather.columns[null_pct < .05]
weather = weather[valid_columns].copy()
weather.columns = weather.columns.str.lower()
weather.index = pd.to_datetime(weather.index)
weather["target"] = weather.shift(-1)["tmax"]
weather = weather.ffill()

# we using Ridge regression
rr = Ridge(alpha=.1)
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]

def backtest(weather, model, predictors, start=365, step=90):
    all_predictions = []
    
    for i in range(start, weather.shape[0], step):
        train = weather.iloc[:i,:]
        test = weather.iloc[i:(i+step),:]
        
        model.fit(train[predictors], train["target"])
        
        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        combined = pd.concat([test["target"], preds], axis=1)
        combined.columns = ["Actual (°C)", "Prediction (°C)"]
        combined["Diff"] = (combined["Prediction (°C)"] - combined["Actual (°C)"]).abs()
        
        all_predictions.append(combined)
    return pd.concat(all_predictions)

predictions = backtest(weather, rr, predictors)

def pct_diff(old, new):
    return (new - old) / old
def compute_rolling(weather, horizon, col):
    label = f"rolling_{horizon}_{col}"
    weather[label] = weather[col].rolling(horizon).mean()
    weather[f"{label}_pct"] = pct_diff(weather[label], weather[col])
    return weather
    
rolling_horizons = [3, 14]
for horizon in rolling_horizons:
    for col in ["tmax", "tmin", "prcp"]:
        weather = compute_rolling(weather, horizon, col)

def expand_mean(df):
    return df.expanding(1).mean()

for col in ["tmax", "tmin", "prcp"]:
    weather[f"month_avg_{col}"] = weather[col].groupby(weather.index.month, group_keys=False).apply(expand_mean)
    weather[f"day_avg_{col}"] = weather[col].groupby(weather.index.day_of_year, group_keys=False).apply(expand_mean)

weather = weather.iloc[14:,:]
weather = weather.fillna(0)
predictors = weather.columns[~weather.columns.isin(["target", "name", "station"])]
predictions = backtest(weather, rr, predictors)
# Result
print(predictions)

#Plot backtest
plt.figure(figsize=(10, 6))
plt.plot(predictions.index, predictions["Actual (°C)"], label="Actual", color='blue', linestyle='-', marker='o')
plt.plot(predictions.index, predictions["Prediction (°C)"], label="Predicted", color='red', linestyle='-', marker='x')
plt.title("Actual vs Predicted Temperatures (Backtest)", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend(loc="upper left")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()


# Find the maximum Diff value
max_diff = predictions["Diff"].max()
max_diff_dates = predictions[predictions["Diff"] == max_diff].index
print("\nMaximum Difference in backtest Predictions:")
print("Date\t\tMax Diff (°C)")
print("-" * 30)

for date in max_diff_dates:
    print(f"{date.strftime('%Y/%m/%d')}\t{max_diff:.2f}")

# Calculate MSE and MAE
print("\nMSE and MAE:")
print("Type of Error\t\tValue")
print("-" * 30)
mse = mean_squared_error(predictions["Actual (°C)"], predictions["Prediction (°C)"])
mae = mean_absolute_error(predictions["Actual (°C)"], predictions["Prediction (°C)"])
print(f"Mean Squared Error:\t{mse:.4f}")
print(f"Mean Absolute Error:\t{mae:.4f}")

# Percentage of error that more than +-2
num_err = count_over2 = (predictions["Diff"] > 2).sum()
percent_error = (count_over2 / len(predictions)) * 100
print("\nError Diff more than 2:")
print("     Error\t\t Value")
print("-" * 35)
print(f"Number of errors:\t{count_over2}")
print(f"Percent of error:\t{percent_error:.2f}%")


# Predict real-time weather
weather.index = pd.to_datetime(weather.index, format='%Y-%m-%d')
weather['day_of_year'] = weather.index.dayofyear
features = ['tavg', 'tmin', 'tmax', 'prcp', 'wspd', 'pres', 'rolling_3_tmax',
            'rolling_3_tmax_pct', 'rolling_3_tmin', 'rolling_3_tmin_pct',
            'rolling_3_prcp', 'rolling_3_prcp_pct', 'rolling_14_tmax',
            'rolling_14_tmax_pct', 'rolling_14_tmin', 'rolling_14_tmin_pct',
            'rolling_14_prcp', 'rolling_14_prcp_pct', 'month_avg_tmax',
            'day_avg_tmax', 'month_avg_tmin', 'day_avg_tmin', 'month_avg_prcp',
            'day_avg_prcp']


predicted_temperatures = []
dates = []

today = datetime.datetime.today()
for i in range(6):
    future_date = today + datetime.timedelta(days=i)
    day_of_year = future_date.timetuple().tm_yday
    future_features = weather.loc[weather['day_of_year'] == day_of_year, features].iloc[0]
    prediction = rr.predict(future_features.to_frame().T)[0]
    predicted_temperatures.append(prediction)
    dates.append(future_date.strftime('%Y/%m/%d'))

# Results of real-time predictions
print("\nPrediction Temp for 6 Days:")
print("Date\t\tPrediction (°C)")
print("-" * 30)
for date, temp in zip(dates, predicted_temperatures):
    print(f"{date}\t{temp:.2f}°C")
    

# We use API from website to be our actual weather value
CITY = "Bangkok"
API_KEY = "23faa9452bf37ea6a8c833688231fcfd"
forecast_url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units=metric"
forecast_response = requests.get(forecast_url)

if forecast_response.status_code == 200:
    forecast_data = forecast_response.json()

    print("\nAPI Temp for 6 days:")
    daily_tmax = {}

    for item in forecast_data['list']:
        date = datetime.datetime.fromtimestamp(item['dt']).strftime('%Y/%m/%d')
        temp_max = item['main']['temp_max']
        
        if date not in daily_tmax or temp_max > daily_tmax[date]:
            daily_tmax[date] = temp_max

    today = datetime.datetime.today()
    count = 0

    print("Date\t\tAPI (°C)")
    print("-" * 25)

    for i in range(6):
        future_date = (today + datetime.timedelta(days=i)).strftime('%Y/%m/%d')
        tmax = daily_tmax.get(future_date, "N/A") 
        print(f"{future_date}\t{tmax}")
        count += 1
        if count == 6:
            break

else:
    print(f"Error fetching data: {forecast_response.status_code}")


# Then we compare it toghter
print("\nComparison between Predicted Temp and API Temp:")
print("Date\t\tMy Prediction (°C)\tAPI Temp (°C)\tDiff (°C)")
print("-" * 65)

for date, pred_temp in zip(dates, predicted_temperatures):
    formatted_date = datetime.datetime.strptime(date, "%Y/%m/%d").strftime("%Y/%m/%d")
    api_temp = daily_tmax.get(formatted_date, "N/A")
    if isinstance(pred_temp, float) and isinstance(api_temp, float):
        diff = round(pred_temp - api_temp, 2)
        print(f"{date}\t{pred_temp:.2f}\t\t\t{api_temp:.2f}\t\t{diff:+.2f}")
    else:
        print(f"{date}\t{pred_temp:.2f}\t\t\t{api_temp}\t\tN/A")


# Compare it in Bar chart
api_temps = [daily_tmax.get(date, None) for date in dates]
plt.figure(figsize=(10, 6))
plt.bar(dates, predicted_temperatures, width=0.4, label="My Prediction", color='red', align='center')
plt.bar(dates, api_temps, width=0.4, label="API Temp", color='blue', align='edge')
plt.title("Comparison of Predicted Temperatures and API Data", fontsize=14)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Temperature (°C)", fontsize=12)
plt.legend(loc="upper left")
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
