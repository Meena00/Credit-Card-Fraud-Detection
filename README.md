# Credit Card Fraud Detection and Transaction Behavior Analysis

This project analyzes credit card and mobile payment transaction data to understand fraud patterns, transaction behavior, and risk-based segmentation. The goal was to practice data cleaning, exploratory analysis, unsupervised learning, and model evaluation using real-world fraud detection datasets.

The project focuses on identifying how fraudulent transactions differ from normal transactions, how transaction behavior can be grouped into risk segments, and how model thresholds can be adjusted when the cost of missing fraud is higher than the cost of false alarms.

## Skills Demonstrated

- Python data analysis with Pandas and NumPy
- Data cleaning and feature preparation
- Transaction behavior comparison using `groupby()`, `value_counts()`, and summary statistics
- K-Means clustering for unsupervised segmentation
- PCA for visualizing high-dimensional transaction behavior
- Random Forest classification for fraud prediction
- Cost-sensitive threshold evaluation
- Matplotlib visualizations for fraud patterns and model interpretation
- Working with highly imbalanced fraud datasets

## Datasets

This project uses two fraud-related datasets from Kaggle.

The first dataset is the European credit card fraud dataset, which contains anonymized transaction features, transaction amount, and a fraud label. It was used for clustering, behavior segmentation, and fraud classification.

The second dataset is the PaySim mobile money transaction dataset, which simulates financial transactions and includes transaction amount, time step, and fraud-related activity. It was used to analyze transaction amounts over time and identify spikes in transaction behavior.

The datasets are not included in this repository because they come from Kaggle.

## Key Findings

### Transaction Behavior Segmentation

K-Means clustering was used to group credit card transactions into three behavior segments: low-risk, moderate-risk, and high-risk. The clustering used selected transaction features along with a log-transformed transaction amount to reduce the effect of extreme transaction values.

PCA was then used to project the scaled features into two components for visualization. The segmentation showed that fraud transactions were not evenly distributed across the transaction space. Instead, many fraud cases appeared near the moderate-risk and high-risk regions, suggesting that clustering can help identify groups where fraud is more concentrated.

The PCA plot also showed that low-risk transactions covered a wider area, while moderate-risk and high-risk segments were more concentrated. This helped make the transaction groups easier to interpret visually.

### Fraud Concentration by Segment

The fraud concentration chart compared fraud rates across the three transaction behavior segments.

| Segment | Fraud Rate |
|---|---:|
| Low-risk | 0.08% |
| Moderate-risk | 0.16% |
| High-risk | 0.26% |

The high-risk segment had the highest fraud concentration at about **0.26%**, compared to **0.08%** in the low-risk segment. This means the high-risk segment had more than **three times** the fraud rate of the low-risk group.

Although the overall fraud rate remained low, this difference is useful because fraud detection datasets are highly imbalanced. Even small percentage increases can be meaningful when the goal is to identify which transaction groups deserve closer review.

### Transaction Amounts Over Time

The PaySim transaction timeline showed that transaction amounts were not evenly distributed across time. Most transaction amounts stayed relatively low, but there were clear spikes around the middle of the timeline, especially near steps 280 to 310.

Some transaction amounts reached extremely high values, close to the upper range of the chart. These spikes suggest that time-based transaction monitoring can be useful for identifying unusual periods of activity.

This finding is useful because fraud detection is not only about classifying individual transactions. Time-based patterns can also help identify periods where unusually large or concentrated transaction activity may need further review.

### Cost-Sensitive Fraud Classification

A Random Forest classifier was used to predict fraudulent transactions from the credit card dataset. Because fraud cases are rare, the model used class balancing to account for the imbalance between normal and fraudulent transactions.

The analysis also tested multiple classification thresholds instead of relying only on the default threshold. This was important because missing a fraudulent transaction can be more costly than incorrectly flagging a normal transaction.

The threshold analysis estimated total cost using a higher cost for false negatives and a lower cost for false positives. This made the evaluation more realistic for fraud detection because the best model threshold should consider business impact, not only accuracy.

## Analysis Summary

The project showed that fraud detection benefits from both exploratory analysis and model-based evaluation.

Clustering helped separate transactions into behavior-based risk segments, and the high-risk segment had the largest fraud concentration. PCA made these segments easier to visualize and showed where fraud transactions appeared within the transaction space. The PaySim analysis added a time-based perspective by showing spikes in transaction amounts across the transaction timeline.

The supervised model added another layer by testing fraud classification with a Random Forest model and evaluating different decision thresholds. This showed how fraud detection can be treated as a cost-sensitive problem where reducing missed fraud cases may matter more than maximizing overall accuracy.

## Relevance

This project demonstrates skills in data preparation, exploratory analysis, unsupervised learning, classification, visualization, and model evaluation. It connects data science coursework with a practical fraud detection problem where class imbalance, transaction behavior, and decision thresholds all affect the final analysis.

The project also shows how raw transaction records can be turned into useful outputs, including risk segments, fraud concentration summaries, transaction amount trends, and model evaluation results.

## Dataset Access

The CSV files are not included in this repository because they come from Kaggle. To reproduce the analysis, download the datasets from Kaggle and place the files in the project folder before running the scripts.

Datasets used:

- European Credit Card Fraud Dataset  
  File used: `creditcard.csv`

- PaySim Mobile Money Transaction Dataset  
  File used: PaySim transaction CSV file

Source: Kaggle
