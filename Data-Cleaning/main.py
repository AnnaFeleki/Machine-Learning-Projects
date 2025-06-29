import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Load breast cancer dataset from sklearn
def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.feature_names, data.target_names

# Load and preview data
df, feature_names, target_names = load_data()
print("\nDataset shape:", df.shape)
print("\nTarget classes:", target_names)

# EDA: check class distribution
sns.countplot(x='target', data=df)
plt.title("Breast Cancer Diagnosis Distribution")
plt.xticks([0, 1], target_names)
plt.show()

# Define features and target
X = df.drop(columns='target')
y = df['target']

# Identify numeric and categorical features
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# Pipelines
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

clf_pipeline = Pipeline([
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit pipeline
clf_pipeline.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf_pipeline.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Export preprocessed dataset (numeric only for this example)
X_processed = pd.DataFrame(preprocessor.fit_transform(X))
X_processed['target'] = y.values
X_processed.to_csv("cleaned_breast_cancer_dataset.csv", index=False)
