import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ================================
# Load data and model
# ================================

df = pd.read_csv("PCOS_cleaned.csv")
model = joblib.load("pcos_rf_model.pkl")

X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"]

# Predictions
y_pred = model.predict(X)

# ================================
# Compute metrics
# ================================

acc = accuracy_score(y, y_pred)
prec = precision_score(y, y_pred)
rec = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

# ================================
# Confusion Matrix
# ================================

cm = confusion_matrix(y, y_pred)

# ================================
# Plot: Combined Poster Figure
# ================================

plt.figure(figsize=(12, 6), dpi=300)

# -------- Confusion Matrix --------
plt.subplot(1, 2, 1)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
plt.colorbar()

classes = ["Non-PCOS", "PCOS"]
plt.xticks([0, 1], classes)
plt.yticks([0, 1], classes)

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, cm[i, j],
            ha="center", va="center",
            color="black",
            fontsize=12,
            fontweight="bold"
        )

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# -------- Metrics Table --------
plt.subplot(1, 2, 2)
plt.axis("off")

table_data = [
    ["Accuracy", f"{acc*100:.2f}%"],
    ["Precision", f"{prec:.2f}"],
    ["Recall", f"{rec:.2f}"],
    ["F1-Score", f"{f1:.2f}"]
]

table = plt.table(
    cellText=table_data,
    colLabels=["Metric", "Value"],
    cellLoc="center",
    loc="center"
)

table.scale(1, 2)
table.auto_set_font_size(False)
table.set_fontsize(12)

for key, cell in table.get_celld().items():
    cell.set_linewidth(1)
    cell.set_edgecolor("black")

plt.title("Model Performance Metrics", fontsize=14, fontweight="bold")

# ================================
# Save figure
# ================================

plt.tight_layout()
plt.savefig("pcos_evaluation_poster.png", bbox_inches="tight")
plt.show()

print("Poster-ready evaluation visuals generated successfully.")
