import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Lấy dữ liệu giá Bitcoin trong 3650 ngày gần nhất
url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=3650"
response = requests.get(url)
data = response.json()
prices = data["prices"]

# Chuyển đổi dữ liệu sang DataFrame và chỉ lấy giá
df = pd.DataFrame(prices, columns=["timestamp", "price"])
df = df.drop(columns=["timestamp"])

# Chuẩn hóa dữ liệu giá
scaler = StandardScaler()
X = scaler.fit_transform(df.values)

# Tạo mô hình Linear Regression và huấn luyện trên dữ liệu lịch sử
model_file = "bitcoin_price_prediction_model.pkl"
try:
    model = joblib.load(model_file)
    print("Loaded model from file")
except:
    model = LinearRegression()
    model.fit(X[:-1], X[1:])
    joblib.dump(model, model_file)
    print(f"Saved model to {model_file}")

# Dự đoán giá Bitcoin cho ngày mai
new_price = model.predict(X[-1].reshape(1, -1))
new_price = scaler.inverse_transform(new_price.reshape(-1, 1))
print(f"Predicted Bitcoin price for tomorrow: {new_price[0][0]:.2f} USD")
