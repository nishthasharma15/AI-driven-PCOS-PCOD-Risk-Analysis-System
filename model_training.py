import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import joblib

# ================================
# STEP 1: Load Cleaned Dataset
# ================================

df = pd.read_csv("PCOS_cleaned.csv")
print("Dataset Shape:", df.shape)

# ================================
# STEP 2: Split Features & Target
# ================================

X = df.drop("PCOS (Y/N)", axis=1)
y = df["PCOS (Y/N)"]

# ================================
# STEP 3: Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# ================================
# STEP 4: Feature Scaling (Logistic Regression)
# ================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================
# STEP 5: Baseline Model – Logistic Regression
# ================================

log_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

print("\n--- Logistic Regression Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred_log))
print("Precision:", precision_score(y_test, y_pred_log))
print("Recall   :", recall_score(y_test, y_pred_log))
print("F1 Score :", f1_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))

# ================================
# STEP 6: Final Model – Random Forest
# ================================

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print("Accuracy :", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall   :", recall_score(y_test, y_pred_rf))
print("F1 Score :", f1_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

# ================================
# STEP 7: Feature Importance (Explainable AI)
# ================================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# ================================
# STEP 8: Probability Prediction
# ================================

y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# ================================
# STEP 9: Threshold-Based Risk Scoring
# ================================

def risk_category(prob):
    if prob < 0.30:
        return "Low Risk"
    elif prob < 0.60:
        return "Moderate Risk"
    else:
        return "High Risk"

risk_labels = [risk_category(p) for p in y_prob_rf]

# Show sample predictions
print("\nSample Risk Predictions:")
for i in range(5):
    print(f"Probability: {y_prob_rf[i]:.2f} → {risk_labels[i]}")

# ================================
# STEP 10: Custom Threshold Evaluation
# ================================

custom_threshold = 0.40  # healthcare-friendly threshold
y_pred_custom = (y_prob_rf >= custom_threshold).astype(int)

print("\n--- Random Forest with Custom Threshold (0.40) ---")
print("Precision:", precision_score(y_test, y_pred_custom))
print("Recall   :", recall_score(y_test, y_pred_custom))
print("F1 Score :", f1_score(y_test, y_pred_custom))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_custom))

# ================================
# STEP 11: Save Model & Scaler
# ================================

joblib.dump(rf_model, "pcos_rf_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nModel saved as pcos_rf_model.pkl")
print("Scaler saved as scaler.pkl")
