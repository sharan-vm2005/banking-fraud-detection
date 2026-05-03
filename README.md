# Banking Fraud Detection System

## 📌 Overview
This project detects fraudulent transactions using a hybrid approach:
- Machine Learning (Random Forest)
- Rule-Based System

## ⚙️ Features
- SMOTE for handling class imbalance
- Threshold tuning for better performance
- Hybrid fraud detection (ML + rules)

## 📊 Model Performance
- Accuracy: 96%
- Fraud Recall: 77%
- Precision: 22%

## 📁 Project Structure
- train_model.py → Train model
- run_pipeline.py → Run detection pipeline
- models/ → Saved model files
- data/ → Dataset

## 🚀 How to Run
```bash
python train_model.py
python run_pipeline.py
