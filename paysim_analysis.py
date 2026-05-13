import pandas as pd
import matplotlib.pyplot as plt

paysim = pd.read_csv('paysim.csv')

plt.figure(figsize=(8,5))
plt.scatter(paysim['step'], paysim['amount'], alpha=0.3, s=5, color='royalblue')
plt.title("Transaction Amounts Over Time (PaySim)")
plt.xlabel("Step (1 Step = 1 Hour)")
plt.ylabel("Transaction Amount")
plt.show()
