import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# CREDIT CARD FRAUD DATASET
# ---------------------------

# Load dataset
credit_df = pd.read_csv("creditcard.csv")
credit_snapshot = credit_df.head(10)
print("Credit Card Dataset Snapshot:\n", credit_snapshot)
# Basic info
print("Credit Card Dataset shape:", credit_df.shape)
print("\nColumns:", credit_df.columns.tolist())
print("\nMissing values per column:\n", credit_df.isnull().sum())

# Summary statistics
print("\nSummary statistics:\n", credit_df.describe())

# Class balance
print("\nFraud vs Non-Fraud counts:\n", credit_df['Class'].value_counts())

# Correlation with fraud
credit_corr = credit_df.corr(numeric_only=True)['Class'].sort_values(ascending=False)
print("\nTop correlated features with fraud:\n", credit_corr.head(10))

# Scatter plot: Amount vs Time (sample 5000 for speed)
plt.figure(figsize=(10,6))
sns.scatterplot(
    data=credit_df.sample(5000),
    x='Time', y='Amount', hue='Class', alpha=0.6,
    palette={0:'blue',1:'red'}
)
plt.title('Credit Card Transactions: Amount vs Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Transaction Amount ($)')
plt.legend(title='Class', labels=['Non-Fraud','Fraud'])
plt.show()
