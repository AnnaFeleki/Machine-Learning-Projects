"""
LSTM Stock Price Forecasting using OHLCV Data

This script trains an LSTM model to predict future stock prices based on historical Close prices.
"""
import yfinance as yf
import numpy as np
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load CSV (replace with your own)


# Download data
ticker = "AAPL"  # You can change this to MSFT, TSLA, etc.
df = yf.download(ticker, start="2015-01-01", end="2024-12-31")
df = df[['Close']]


# Normalize prices
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Create sequences
def create_sequences(data, window_size=60):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train/test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Predict and inverse transform
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test_true, predicted))
mae = mean_absolute_error(y_test_true, predicted)
directional_acc = np.mean(np.sign(np.diff(y_test_true.flatten())) == np.sign(np.diff(predicted.flatten())))

# Plot
trace1 = go.Scatter(x=df.index[-len(y_test_true):], y=y_test_true.flatten(), name='Actual Price')
trace2 = go.Scatter(x=df.index[-len(predicted):], y=predicted.flatten(), name='Predicted Price')
layout = go.Layout(title=f'Stock Price Forecast (RMSE: {rmse:.2f}, Directional Acc: {directional_acc:.2%})',
                   xaxis_title='Date', yaxis_title='Price')
fig = go.Figure(data=[trace1, trace2], layout=layout)
fig.show()
