# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset from sklearn
boston = load_boston()
X = boston.data
y = boston.target
feature_names = boston.feature_names

# Convert to DataFrame for exploration (optional)
df = pd.DataFrame(X, columns=feature_names)
df['medv'] = y

print(df.head())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate
print(f"Training MSE: {mean_squared_error(y_train, y_pred_train):.2f}")
print(f"Testing MSE: {mean_squared_error(y_test, y_pred_test):.2f}")
print(f"Training R²: {r2_score(y_train, y_pred_train):.2f}")
print(f"Testing R²: {r2_score(y_test, y_pred_test):.2f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted House Prices')
plt.show()
