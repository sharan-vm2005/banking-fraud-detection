import pandas as pd
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

# 6. Predict
y_pred = model.predict(X_test)

# 7. Evaluate
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n✅ Pipeline executed successfully!")