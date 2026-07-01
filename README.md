# Fraud Detection on Financial Transactions with BigQuery ML

Detecting fraudulent financial transactions using **supervised and unsupervised machine learning**, built entirely in **Google BigQuery ML**  machine learning written in SQL.

Based on the Google Cloud lab, using the public **PaySim** synthetic dataset. Completed with a score of **100/100**.

---

## The Problem

Financial fraud is a **rare-event problem**: fraudulent transactions make up less than **1%** of all activity. This creates a classic *imbalanced classification* challenge a naive model that always predicts "not fraud" would score 99.9% accuracy while being completely useless. The goal is to build models that actually catch the rare fraudulent cases.

---

## The Dataset

- **Source:** PaySim synthetic dataset (Kaggle)
- **Size:** ~6.3 million transactions
- **Target column:** `isFraud` (1 = fraud, 0 = genuine)
- **Fraud rate:** ~0.1% (highly imbalanced)

Only two transaction types contain fraud: **TRANSFER** and **CASH_OUT** reflecting a real fraud pattern where money is first transferred out of a compromised account, then cashed out.

---

## Pipeline

Load data → Explore → Feature engineering → Unsupervised model → Supervised models → Evaluate → Predict

1. **Load** 6.3M+ transactions into BigQuery
2. **Explore** the data and confirm the class imbalance
3. **Feature engineering** + undersampling to expose the fraud signal
4. **Unsupervised model** (k-means) for anomaly detection
5. **Supervised models** (logistic regression + boosted tree)
6. **Evaluate** and select the champion model
7. **Predict** fraud on held-out test data

---

## The Code

The full commented SQL pipeline — every query explained step by step is in
**[`fraud_detection_pipeline.sql`](fraud_detection_pipeline.sql)**.

---

## Key Concepts Learned

| Concept | Meaning |
|---|---|
| **Imbalanced data** | Fraud is < 1% of transactions; accuracy alone is misleading |
| **Undersampling** | Reducing the majority class to expose the rare pattern |
| **Feature engineering** | Creating signals like `amountError` and zero-balance flags |
| **Supervised learning** | Learning from labelled fraud examples |
| **Unsupervised learning** | Finding anomalies without labels (k-means) |
| **Precision vs Recall** | The metrics that actually matter for rare-event detection |
| **Champion model** | Selecting the best-performing model for scoring |

---

## Tools Used

Google Cloud Platform · BigQuery · BigQuery ML · Cloud Shell · SQL

---

## Credits

Based on Google Cloud Skills Boost lab. Dataset: PaySim (Kaggle).
