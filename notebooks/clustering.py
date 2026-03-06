from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

cc = pd.read_csv('creditcard.csv')
X = cc[['V1', 'V2', 'Amount']]
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42)
cc['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(7,5))
plt.scatter(cc['V1'], cc['V2'], c=cc['Cluster'], cmap='viridis', alpha=0.5)
plt.title("K-Means Clustering of Transactions (k=3)")
plt.xlabel("Feature V1")
plt.ylabel("Feature V2")
plt.show()