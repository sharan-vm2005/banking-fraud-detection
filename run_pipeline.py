import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load data
df = pd.read_csv("data/raw/credit_card_fraud_10k.csv")

# 2. Preprocessing
df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)

# 3. Load model FIRST
with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# 4. Align columns


model_features = pd.read_csv("models/feature_columns.csv")
model_features = model_features.iloc[:, 0].tolist()

df = df.reindex(columns=model_features, fill_value=0)



# 5. Split
X = df
y = pd.read_csv("data/raw/credit_card_fraud_10k.csv")['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Predict (ML)
y_probs = model.predict_proba(X_test)[:, 1]
ml_pred = (y_probs > 0.5).astype(int)

def apply_rules(df):
    # Create Series with SAME index as input
    rule_flag = pd.Series(0, index=df.index)

    # Rule 1: High transaction amount
    if 'amount' in df.columns:
        rule_flag = rule_flag | (df['amount'] > 50000)

    return rule_flag.astype(int)

# Rule-based prediction
rule_pred = apply_rules(X_test)

# Final prediction (HYBRID)


final_pred = (ml_pred | rule_pred.values).astype(int)

# 7. Evaluate (IMPORTANT: use final_pred)
print("Confusion Matrix:")
print(confusion_matrix(y_test, final_pred))

print("\nClassification Report:")
print(classification_report(y_test, final_pred))

print("\n✅ Hybrid Pipeline executed successfully!")