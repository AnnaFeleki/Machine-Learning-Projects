# 🏠 Boston Housing Price Prediction

This project uses **Linear Regression** to predict housing prices in Boston using the `load_boston` dataset from `sklearn.datasets`. The goal is to demonstrate a full regression pipeline including preprocessing, training, evaluation, and visualization.

---

## 📊 Dataset

The [Boston Housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_boston.html) contains information collected by the U.S. Census Service concerning housing in the area of Boston, Massachusetts. It includes 506 rows with 13 numerical features such as:

- CRIM: Crime rate by town
- RM: Average number of rooms per dwelling
- LSTAT: % lower status of the population
- ... and more

> ⚠️ **Note**: The Boston Housing dataset is deprecated in newer versions of `scikit-learn` due to ethical concerns. It is used here only for educational purposes.

---

## 🧠 Model

- **Model**: Linear Regression
- **Library**: `sklearn.linear_model.LinearRegression`
- **Preprocessing**: StandardScaler normalization of features

---

## 🔧 Pipeline

1. Load dataset from `sklearn.datasets.load_boston`
2. Convert to `pandas.DataFrame` for exploration
3. Preprocess: standardize features using `StandardScaler`
4. Train/test split (80/20)
5. Train Linear Regression model
6. Predict and evaluate performance using:
   - Mean Squared Error (MSE)
   - R² Score (Coefficient of Determination)
7. Visualize predicted vs. actual house prices

---

## 📈 Results

The model's performance is evaluated with both MSE and R² on training and test sets. The result plot shows how close the predicted values are to the actual prices.


---

## 📦 Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib


