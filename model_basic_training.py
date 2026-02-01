import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ================================
# STEP 1: Load Cleaned Dataset
# ================================

df = pd.read_csv("PCOS_cleaned.csv")

# ================================
# STEP 2: Select BASIC FEATURES (No Labs / Ultrasound)
# ================================

basic_features = [
    "Age (yrs)",
    "BMI",
    "Cycle(R/I)",
    "Cycle length(days)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Pimples(Y/N)",
    "Fast food (Y/N)",
    "Reg.Exercise(Y/N)"
]

X = df[basic_features]
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

# ================================
# STEP 4: Train BASIC Model
# ================================

basic_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

basic_model.fit(X_train, y_train)

# ================================
# STEP 5: Evaluation
# ================================

y_pred = basic_model.predict(X_test)

print("\n--- BASIC MODEL RESULTS (No Lab Data) ---")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# ================================
# STEP 6: Save BASIC Model
# ================================

joblib.dump(basic_model, "pcos_basic_model.pkl")

print("\nBasic model saved as pcos_basic_model.pkl")
