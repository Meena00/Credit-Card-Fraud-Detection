# Credit Card Fraud Detection & Fairness Analysis

**Author:** Meena Anwar  
BS Computational & Data Science — George Mason University  
Expected Graduation: December 2026

## Project Overview

Financial institutions lose billions of dollars each year due to fraudulent transactions. Fraud detection systems must balance two competing priorities:

• Accurately detect fraudulent transactions  
• Avoid incorrectly blocking legitimate users

This project analyzes **bias and fairness in fraud detection systems** using machine learning techniques applied to two large-scale financial datasets.

The analysis evaluates how fraud detection models perform across different transaction patterns and environments while optimizing for both **accuracy and financial cost**.

### Key Objectives

- Analyze fraud patterns in financial transaction data
- Detect behavioral bias in fraud detection models
- Compare fraud detection performance across datasets
- Optimize model thresholds to reduce financial loss
- Evaluate fairness implications in fraud classification systems

---

# Datasets

## Credit Card Fraud Dataset (Europe)

- 284,807 total transactions
- 492 fraudulent transactions
- Fraud rate: **0.173%**
- 30 features including anonymized PCA variables (V1–V28), Time, and Amount

The anonymized PCA features capture transaction behavior patterns useful for fraud detection.

Dataset Source:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## PaySim Dataset (Mobile Transactions)

- 6,362,620 simulated mobile transactions
- 8,213 fraud cases
- Fraud rate: **0.129%**
- 11 transaction and network relationship features

This dataset simulates mobile money transactions and models relationships between agents and customers.

Dataset Source:  
https://www.kaggle.com/datasets/ealaxi/paysim1

---
## Key Visualizations

### K-Means Clustering of Transaction Patterns

![K-Means Clustering of Transaction Patterns](visuals/Clustering-results.png)

This visualization shows the results of applying **K-Means clustering (K=3)** on the credit card transaction dataset using PCA-reduced features.

Key observations:

- **Cluster 0** represents lower-value, high-frequency transactions.
- **Cluster 1** contains moderate and routine spending behaviors.
- **Cluster 2** highlights anomalous or high-value transactions, which contain a higher proportion of fraudulent cases.

The clustering helps reveal **behavioral groupings in anonymized transaction data**, allowing the model to detect unusual patterns that may indicate fraudulent activity.

---

### Fraud Detection Cost Optimization Curve

![Fraud Detection Cost Optimization Curve](visuals/Cost%20Optimization%20Curve%20.png)

This graph illustrates the relationship between **fraud detection thresholds and total operational cost**.

The total cost is modeled as:
```
Total Cost = (Fraud Loss × Missed Fraud) + (User Cost × False Positive)
```
Key insights:

- **Lower thresholds** increase false alarms, raising user inconvenience costs.
- **Higher thresholds** miss more fraudulent transactions, increasing fraud losses.
- The optimal threshold is approximately **0.67**, where the **total cost is minimized**.

This cost-sensitive evaluation demonstrates how model tuning can **balance fraud detection accuracy with financial impact**.

# Dataset Comparison

| Dataset | Transactions | Fraud Cases | Fraud Rate | Features | Structure |
|-------|-------------|------------|-----------|----------|-----------|
| Credit Card Fraud | 284,807 | 492 | 0.173% | 30 | PCA numerical features |
| PaySim | 6,362,620 | 8,213 | 0.129% | 11 | Transaction network data |

Using both datasets allows evaluation of fraud detection fairness across different financial ecosystems.

---

# Exploratory Data Analysis

Key observations from initial analysis:

- Fraudulent transactions appear more frequently among **high-value transactions**
- Legitimate transactions cluster around **low-value amounts**
- Fraud cases represent **less than 0.2% of all transactions**

This extreme class imbalance creates challenges for machine learning models and increases the risk of bias.

---

# Clustering Analysis

## K-Means Clustering

Clustering was applied to detect behavioral transaction groups using:

- PCA transaction features
- Transaction amount

Three clusters were identified:

| Cluster | Description |
|-------|-------------|
| Cluster 0 | Low-value frequent transactions |
| Cluster 1 | Moderate regular spending |
| Cluster 2 | High-value anomalies (fraud-heavy cluster) |

### Insight

Some legitimate users appear in fraud-heavy clusters due to spending patterns, indicating **behavioral bias rather than demographic bias**.

---

# Fraud Classification Model

A **Random Forest classifier** was used to predict fraudulent transactions.

### Important Predictive Features

- V3
- V14
- V17

These features appear strongly related to **transaction timing and behavioral frequency patterns**.

### Model Handling of Class Imbalance

- Stratified sampling
- Weighted classification
- Cost-sensitive evaluation

---

# Cost Optimization

Fraud detection models were evaluated using a financial cost function.

Total Cost =  
(Fraud Loss × Missed Fraud) + (User Cost × False Alarm)

Where:

- **Missed Fraud** = fraudulent transactions not detected
- **False Alarm** = legitimate transactions incorrectly flagged

Optimizing classification thresholds reduces total financial loss while improving system fairness.

---

# Model Tradeoff Comparison

| Model Type | False Positives | False Negatives | Estimated Cost Impact |
|-----------|----------------|----------------|----------------|
| Baseline Model | 6.5% | 11.2% | $2.4M |
| High Sensitivity Model | 10.8% | 5.1% | $3.1M |
| Fairness-Aware Model | 6.9% | 6.4% | $1.5M |

The **fairness-aware model provides the best balance between detection accuracy and financial cost**.

---

# Ethical and Fairness Considerations

Fraud detection systems can unintentionally introduce bias.

Examples include:

- Behavioral bias from spending habits
- Certain transaction patterns being incorrectly flagged
- Reduced customer trust when legitimate transactions are blocked

Responsible AI practices require:

- Bias audits
- Fairness-aware model training
- Transparent model evaluation

Sources: OECD (2020), KPMG Fintech Report (2021)

---

# Key Findings

- Fraud detection systems may introduce **behavioral bias**
- Transaction timing and spending patterns influence classification
- Fairness-aware models reduce both **bias and financial loss**

This project demonstrates how machine learning models can be improved through **fairness-aware evaluation and cost optimization**.

---

# Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# Project Structure

```
credit-card-fraud-analysis/
│
├── README.md
│
├── data/
│ dataset_description.md
│
├── visuals/
│ clustering_results.png
│ cost_optimization_curve.png
│
└── proposal/
project_proposal.pdf
```

---

# Project Timeline

| Weeks | Tasks |
|------|------|
| Weeks 1–3 | Data cleaning and exploratory analysis |
| Weeks 4–7 | Clustering and classification models |
| Weeks 8–9 | Model optimization and threshold tuning |
| Weeks 10–11 | Fairness evaluation and analysis |

---

# References

Dal Pozzolo et al. (2015)  
Calibrating Probability with Imbalanced Datasets for Fraud Detection  
https://doi.org/10.1109/WI-IAT.2015.31

OECD (2020)  
Responsible Use of AI in Financial Services

KPMG Fintech Report (2021)

---
