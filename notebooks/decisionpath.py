import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Load dataset
data = pd.read_csv("creditcard.csv")

# Select features shown in lecture (simplified PCA components + Amount)
X = data[['V1', 'V2', 'V3', 'Amount']]
y = data['Class']  # 1 = Fraud, 0 = Legit

# Split data 70/30 for training/testing (random_state for reproducibility)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train a small tree to illustrate supervised learning structure
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Display accuracy (for reference)
print("Training Accuracy:", round(tree.score(X_train, y_train), 4))
print("Testing Accuracy:", round(tree.score(X_test, y_test), 4))

# Plot simple decision tree
plt.figure(figsize=(10,6))
plot_tree(tree,
          filled=True,
          rounded=True,
          feature_names=['V1', 'V2', 'V3', 'Amount'],
          class_names=['Legit', 'Fraud'])
plt.title("Decision Tree – Credit Card Fraud Detection")
plt.show()
#easiest way to transfer this between computers is to use git, if you have 2fa enabled you will need to use a personal access token instead of your password. to avoid that