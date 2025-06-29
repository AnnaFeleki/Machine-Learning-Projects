# 🧹 Breast Cancer Dataset Cleaning & Classification Pipeline

This project demonstrates a complete pipeline for cleaning, preprocessing, and modeling a real-world clinical dataset — the Breast Cancer Wisconsin dataset. It showcases essential data science steps from EDA to exporting a clean dataset, and is ideal for Upwork clients or employers looking to assess your applied ML skills.

---

## 📊 Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Features:** 30 numeric features related to cell nucleus measurements (mean radius, texture, etc.)
- **Target:** Binary classification – `malignant` vs `benign`

---

## 🧠 Project Highlights

### ✔️ Data Cleaning & Preparation
- Loaded structured dataset into `pandas`
- Handled missing values using `SimpleImputer`
- Identified numeric and categorical features

### ✔️ Preprocessing Pipelines
- **Numeric features:** Imputed with median, standardized using `StandardScaler`
- **Categorical features (if any):** One-hot encoded and imputed with mode
- Combined using `ColumnTransformer`

### ✔️ Machine Learning Model
- Random Forest Classifier wrapped in a `Pipeline`
- Model trained/tested using `train_test_split`
- Evaluated with classification metrics: **precision, recall, F1-score**

### ✔️ Data Export
- Final cleaned and processed dataset exported as `cleaned_breast_cancer_dataset.csv`

---

## 📁 File Structure
```
├── main.py                     # Full pipeline implementation
├── cleaned_breast_cancer_dataset.csv  # Output dataset
├── README.md
```

---

## 🛠️ Requirements
Install required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

---

## 📈 Example Output (Metrics)
```
              precision    recall  f1-score   support

       benign       0.97      0.99      0.98        71
    malignant       0.98      0.96      0.97        43

    accuracy                           0.98       114
```
