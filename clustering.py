import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# -----------------------------
# Load data
# -----------------------------
cc = pd.read_csv("creditcard.csv")
os.makedirs("visuals", exist_ok=True)

# -----------------------------
# Feature selection
# -----------------------------
features = ["V1", "V2", "V3", "V4", "Amount"]
X = cc[features].copy()

# Log-transform Amount to reduce extreme skew
X["Amount"] = np.log1p(X["Amount"])

# -----------------------------
# Scale features
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# K-Means clustering
# -----------------------------
kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
cc["Cluster"] = kmeans.fit_predict(X_scaled)

# -----------------------------
# PCA projection for visualization
# -----------------------------
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

cc["PC1"] = X_pca[:, 0]
cc["PC2"] = X_pca[:, 1]

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())

# -----------------------------
# Cluster summary
# -----------------------------
cluster_summary = (
    cc.groupby("Cluster")
      .agg(
          Total_Transactions=("Class", "count"),
          Fraud_Cases=("Class", "sum"),
          Fraud_Rate=("Class", "mean"),
          Avg_Amount=("Amount", "mean")
      )
      .reset_index()
)

cluster_summary["Fraud_Rate"] = cluster_summary["Fraud_Rate"] * 100

# Rank clusters by fraud rate and assign business labels
cluster_summary = cluster_summary.sort_values("Fraud_Rate").reset_index(drop=True)
segment_names = ["Low-risk", "Moderate-risk", "High-risk"]

cluster_summary["Segment"] = segment_names

cluster_to_segment = dict(
    zip(cluster_summary["Cluster"], cluster_summary["Segment"])
)

cc["Segment"] = cc["Cluster"].map(cluster_to_segment)

print("\nCluster summary:")
print(cluster_summary[["Cluster", "Segment", "Total_Transactions", "Fraud_Cases", "Fraud_Rate", "Avg_Amount"]])

# -----------------------------
# Graph 1: Behavior segmentation
# -----------------------------
plt.figure(figsize=(9, 6))

segments_in_order = ["Low-risk", "Moderate-risk", "High-risk"]

for segment in segments_in_order:
    subset = cc[cc["Segment"] == segment]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        alpha=0.30,
        label=segment
    )

# Overlay fraud transactions
fraud_points = cc[cc["Class"] == 1]
plt.scatter(
    fraud_points["PC1"],
    fraud_points["PC2"],
    marker="x",
    alpha=0.85,
    label="Fraud transactions"
)

plt.title("Transaction Behavior Segmentation (Unsupervised Learning)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.tight_layout()
plt.savefig("visuals/behavior-segmentation.png", dpi=300)
plt.show()

# -----------------------------
# Graph 2: Fraud concentration by segment
# -----------------------------
plot_df = cluster_summary.copy()

plt.figure(figsize=(8, 5))
bars = plt.bar(plot_df["Segment"], plot_df["Fraud_Rate"])

plt.title("Fraud Concentration Across Transaction Behavior Segments")
plt.xlabel("Transaction Segment")
plt.ylabel("Fraud Rate (%)")

for bar, rate in zip(bars, plot_df["Fraud_Rate"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{rate:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig("visuals/fraud-concentration-by-segment.png", dpi=300)
plt.show()

# -----------------------------
# Save summary table
# -----------------------------
cluster_summary.to_csv("visuals/cluster-summary.csv", index=False)