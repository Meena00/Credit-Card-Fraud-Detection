import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# -----------------------------
# Load data
# -----------------------------
credit_df = pd.read_csv("creditcard.csv")
os.makedirs("visuals", exist_ok=True)

# -----------------------------
# Features and target
# -----------------------------
X = credit_df.drop(columns=["Class"])
y = credit_df["Class"]

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -----------------------------
# Model training
# -----------------------------
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=8,
    min_samples_split=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# -----------------------------
# Probability predictions
# -----------------------------
y_prob = model.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_test, y_prob)
print(f"ROC-AUC Score: {auc_score:.4f}")

# -----------------------------
# Cost-sensitive threshold analysis
# -----------------------------
fraud_loss_cost = 500
false_alarm_cost = 10

thresholds = np.arange(0.05, 0.96, 0.01)
results = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    total_cost = (fn * fraud_loss_cost) + (fp * false_alarm_cost)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0

    results.append({
        "Threshold": threshold,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
        "True_Negatives": tn,
        "Precision": precision,
        "Recall": recall,
        "False_Positive_Rate": false_positive_rate,
        "Total_Cost": total_cost
    })

results_df = pd.DataFrame(results)

# -----------------------------
# Best threshold
# -----------------------------
best_row = results_df.loc[results_df["Total_Cost"].idxmin()]
best_threshold = best_row["Threshold"]
best_cost = best_row["Total_Cost"]

print("\nBest threshold based on total cost:")
print(best_row)

# -----------------------------
# Final evaluation at best threshold
# -----------------------------
y_best_pred = (y_prob >= best_threshold).astype(int)

print("\nClassification report at optimal threshold:")
print(classification_report(y_test, y_best_pred, digits=4))

# -----------------------------
# Graph 3: Cost optimization curve
# -----------------------------
plt.figure(figsize=(9, 6))
plt.plot(results_df["Threshold"], results_df["Total_Cost"], linewidth=2)
plt.scatter(
    best_threshold,
    best_cost,
    s=90,
    label=f"Optimal Threshold = {best_threshold:.2f}"
)

plt.annotate(
    f"Minimum Cost = {int(best_cost)}",
    xy=(best_threshold, best_cost),
    xytext=(best_threshold + 0.04, best_cost * 1.08),
    arrowprops=dict(arrowstyle="->")
)

plt.title("Cost-Sensitive Threshold Optimization for Fraud Detection")
plt.xlabel("Classification Threshold")
plt.ylabel("Total Estimated Cost")
plt.legend()
plt.tight_layout()
plt.savefig("visuals/cost-sensitive-threshold-optimization.png", dpi=300)
plt.show()

# -----------------------------
# Save results table
# -----------------------------
results_df.to_csv("visuals/threshold-cost-results.csv", index=False)