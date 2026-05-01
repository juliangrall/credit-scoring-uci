# Credit Scoring — Default Prediction (UCI Credit Card)

Comparative study of supervised learning models for predicting credit card default, using the UCI Default of Credit Card Clients dataset (30,000 customers, 23 features).

## Context
Project for the Statistical Learning course — M2 ISIFAR (Applied Mathematics for Finance), Université Paris-Cité, 2025.

## Data
- **Source**: UCI Machine Learning Repository — [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients+dataset)
- 30,000 observations, 23 explanatory variables (demographics, payment history, bill amounts, payment amounts)
- Target: default payment next month (binary, 22% positive class)

## Methodology

### Preprocessing
- Missing values check and removal of inconsistent codes (EDUCATION, MARRIAGE)
- Conversion of categorical variables to factors
- Train/test split (70/30, stratified)

### Models compared
1. **Logistic Regression** (full + AIC backward selection)
2. **LASSO Logistic Regression** (5-fold CV via tidymodels)
3. **Linear Discriminant Analysis** (LDA)
4. **Decision Tree** (with cross-validation pruning)
5. **Random Forest** (500 trees, default mtry)
6. **SVM** (RBF kernel)

### Evaluation
- AUC, Accuracy, Sensitivity, Specificity
- ROC curves comparison
- Variable importance (Random Forest)
- K-means clustering on numerical features for unsupervised risk segmentation

## Results

| Model | AUC | Accuracy | Sensitivity | Specificity |
|---|---|---|---|---|
| Logistic Regression (AIC) | 0.713 | 0.813 | 0.969 | 0.250 |
| LASSO | 0.713 | 0.814 | 0.970 | 0.252 |
| LDA | 0.710 | 0.815 | 0.966 | 0.270 |
| Decision Tree (pruned) | 0.733 | 0.820 | 0.959 | 0.321 |
| **Random Forest** | **0.761** | 0.818 | 0.942 | **0.373** |
| SVM (RBF) | 0.703 | 0.820 | 0.964 | 0.290 |

**Random Forest** offers the best AUC and best balance between sensitivity and specificity, making it the most reliable model for this imbalanced classification task.

## Key insights
- Recent payment behavior (PAY_0, PAY_2) dominates feature importance — far more predictive than demographics or bill amounts.
- Class imbalance (22% defaulters) keeps recall low across all models, even after sampling adjustments.
- K-means clustering reveals natural risk segments (low-risk ~15%, mid-risk ~18%, high-risk ~29%).

## Stack
R (tidyverse, caret, glmnet, randomForest, e1071, tree, MASS, ROCR, pROC, tidymodels)

## Files
- `credit_scoring.Rmd` — full R Markdown report
- `credit_scoring.pdf` — knitted PDF
- `UCI_Credit_Card.csv` — dataset (download from UCI link above)
