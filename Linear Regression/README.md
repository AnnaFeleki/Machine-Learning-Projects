# 🏠 Boston Housing Price Prediction using Linear Regression

This project applies a **Linear Regression** model to the **Boston Housing Dataset** to predict median house values based on various socioeconomic and housing-related features.

---

## 📊 Dataset

- **Source:** [Selva86 GitHub Datasets](https://github.com/selva86/datasets/blob/master/BostonHousing.csv)
- **Target Variable:** `medv` (Median value of owner-occupied homes)
- **Features Include:**
  - Crime rate, proportion of non-retail business acres, nitric oxide concentration, average rooms, etc.

---

## 🔧 Tools & Libraries

- Python 3
- `pandas`, `numpy`, `scikit-learn`
- `matplotlib` for visualization

---

## 🚀 How it Works

1. **Load & Explore Data**  
   Read the dataset from a URL and explore its structure.

2. **Preprocessing**  
   - Separate features (`X`) and target (`y`)
   - Train/test split (80/20)
   - Feature scaling using `StandardScaler`

3. **Model Training**  
   Fit a `LinearRegression` model on the training data.

4. **Evaluation Metrics**
   - Mean Squared Error (MSE)
   - R² Score

5. **Visualization**  
   A scatter plot comparing actual vs. predicted house prices.

---
