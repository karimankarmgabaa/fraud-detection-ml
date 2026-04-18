# 💳 Fraud Detection System

## 📌 Overview
This project detects fraudulent credit card transactions using machine learning.

## ⚙️ Features
- Data preprocessing & scaling
- Logistic Regression model
- ROC Curve & AUC evaluation
- Threshold tuning
- Cost-based optimization

## 📊 Results
- AUC: ~0.97
- Optimized threshold using cost function
- Improved fraud detection performance

## 🧠 Key Insight
Lowering threshold increases recall but reduces precision.  
Optimal threshold selected based on business cost.

## 🚀 How to Use

```python
import joblib

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
