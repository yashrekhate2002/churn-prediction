# =============================
# Customer Churn Prediction
# (NO TensorFlow Version)
# =============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")

# -----------------------------
# 2. Data Cleaning
# -----------------------------
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df.drop('customerID', axis=1, inplace=True)

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

# -----------------------------
# 3. Preprocessing
# -----------------------------
cat_cols = X.select_dtypes(include='object').columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

# -----------------------------
# 4. Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply preprocessing
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# -----------------------------
# 5. Handle Imbalanced Data (SMOTE)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_p, y_train)

# -----------------------------
# 6. Models
# -----------------------------

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_bal, y_train_bal)
y_pred_lr = lr.predict(X_test_p)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_bal, y_train_bal)
y_pred_dt = dt.predict(X_test_p)

# Neural Network (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32),
                    activation='relu',
                    max_iter=500,
                    random_state=42)
mlp.fit(X_train_bal, y_train_bal)
y_pred_mlp = mlp.predict(X_test_p)

# -----------------------------
# 7. Evaluation
# -----------------------------
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))
print("Decision Tree:\n", classification_report(y_test, y_pred_dt))
print("Neural Network (MLP):\n", classification_report(y_test, y_pred_mlp))

# -----------------------------
# 8. Confusion Matrices
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.heatmap(confusion_matrix(y_test, y_pred_lr),
            annot=True, fmt='d', ax=axes[0]).set_title("Logistic Regression")

sns.heatmap(confusion_matrix(y_test, y_pred_dt),
            annot=True, fmt='d', ax=axes[1]).set_title("Decision Tree")

sns.heatmap(confusion_matrix(y_test, y_pred_mlp),
            annot=True, fmt='d', ax=axes[2]).set_title("Neural Network (MLP)")

plt.show()
