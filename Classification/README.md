# 🚢 Titanic Survival Prediction using Random Forest

This project demonstrates how to use a **Random Forest Classifier** to predict survival outcomes on the Titanic dataset. It covers preprocessing, model training, evaluation, and visualization of feature importances using Python's `scikit-learn` and `matplotlib`.

---

## 📁 Dataset

- **Source**: [OpenML Titanic Dataset](https://www.openml.org/d/40945)
- **Features** include: `age`, `fare`, `sex`, `embarked`, `pclass`, etc.
- **Target**: `survived` (0 = No, 1 = Yes)

---

## 🧠 Model

- **Algorithm**: Random Forest Classifier
- **Preprocessing Steps**:
  - Encode categorical features (`sex`, `embarked`)
  - Fill missing values with median or mode
  - Drop irrelevant or sparse columns
  - Scale features using `StandardScaler`

---

