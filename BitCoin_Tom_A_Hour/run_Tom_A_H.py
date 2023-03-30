import requests
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Lấy dữ liệu từ API
url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=max'
response = requests.get(url)
data = response.json()
prices = data['prices']

# Tạo DataFrame từ dữ liệu
df = pd.DataFrame(prices, columns=['timestamp', 'price'])
df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
df.drop('timestamp', axis=1, inplace=True)
df.set_index('date', inplace=True)

# Chuẩn hóa dữ liệu giá
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Tạo dữ liệu đầu vào và đầu ra cho mô hình
x_train = []
y_train = []
for i in range(60, len(df)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Tạo mô hình neural network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=1, batch_size=1)

# Lưu lại model
joblib.dump(model, 'bitcoin_prediction_model.pkl')

# Dự đoán giá cho 1 tiếng đồng hồ sau
last_60_minutes = df.iloc[-60:].values
last_60_minutes_scaled = scaler.transform(last_60_minutes)
X_test_minute = []
X_test_minute.append(last_60_minutes_scaled)
X_test_minute = np.array(X_test_minute)
X_test_minute = np.reshape(X_test_minute, (X_test_minute.shape[0], X_test_minute.shape[1], 1))
predicted_price_minute = model.predict(X_test_minute)
predicted_price_minute = scaler.inverse_transform(predicted_price_minute)
one_hour_later = datetime.now() + timedelta(hours=1)
print(f'Giá Bitcoin dự đoán cho 1 tiếng đồng hồ sau ({one_hour_later.strftime("%Y-%m-%d %H:%M:%S")}): {predicted_price_minute[0][0]}')

# Dự đoán giá cho ngày mai
last_60_days = df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test_day = []
X_test_day.append(last_60_days_scaled)
X_test_day = np.array(X_test_day)
X_test_day = np.reshape(X_test_day, (X_test_day.shape[0], X_test_day.shape[1], 1))
predicted_price_day = model.predict(X_test_day)
predicted_price_day = scaler.inverse_transform(predicted_price_day)
tomorrow = datetime.now().date() + timedelta(days=1)
print(f'Giá Bitcoin dự đoán cho ngày {tomorrow}: {predicted_price_day[0][0]}')
