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
last_60_days = df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)

# Lấy giá mới nhất và tính toán lại đầu vào cho mô hình
now = datetime.now()
new_price_response = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_market_cap=false&include_24hr_vol=false&include')
last_60_minutes = df['price'][-60:].values
last_60_minutes_scaled = scaler.transform(last_60_minutes.reshape(-1, 1))
X_test = []
X_test.append(last_60_minutes_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_price = model.predict(X_test)
predicted_price = scaler.inverse_transform(predicted_price)
hour_later = datetime.now() + timedelta(hours=1)
print(f'Giá Bitcoin dự đoán cho 1 tiếng đồng hồ sau ({hour_later.strftime("%H:%M:%S %d/%m/%Y")}): {predicted_price[0][0]}')