# Credit Card Fraud Detection using One-Class SVM

## Overview
This repository contains the implementation of an anomaly detection model designed to identify fraudulent credit card transactions. Developed as an Artificial Intelligence graduation project, this solution leverages a One-Class Support Vector Machine (SVM) to handle highly imbalanced financial data.

## Dataset
The project utilizes the `creditcardfraud` dataset via Kaggle[cite: 1]. 
* **Total Transactions:** 284,807[cite: 1]
* **Fraud Ratio:** 0.1727% (492 fraudulent transactions)[cite: 1]

*Note: The dataset is not included in this repository due to size constraints. The notebook includes automated data fetching via `kagglehub`[cite: 1].*

## Methodology
Because fraudulent transactions are extremely rare, traditional supervised classification models often struggle. This project frames the problem as an anomaly detection task:
1. **Preprocessing:** Standardized the `Time` and `Amount` features (PCA features V1-V28 were pre-transformed)[cite: 1]. 
2. **Data Isolation:** Filtered the training dataset to exclusively include normal transactions (255,883 samples)[cite: 1]. 
3. **Model Training:** Trained a One-Class SVM to learn the boundaries of "normal" behavior, flagging deviations as potential fraud[cite: 1]. 

## Hyperparameter Tuning
Extensive hyperparameter tuning was conducted to balance the trade-off between precision and recall[cite: 1]. The optimal parameters were found to be:
* `nu` = 0.01[cite: 1]
* `gamma` = 0.01[cite: 1]

## Results
The optimized One-Class SVM achieved the following performance metrics on the test set[cite: 1]:
* **Accuracy:** 99.02%[cite: 1]
* **Recall:** 83.67% (Successfully identifying the vast majority of fraudulent transactions)[cite: 1]
* **F1 Score:** 22.65%[cite: 1]
* **Precision:** 13.10%[cite: 1]

## Project Structure
```text
├── notebooks/
│   └── One_Class_SVM_for_credit_card_fraud_detection.ipynb
├── requirements.txt
├── .gitignore
└── README.md
