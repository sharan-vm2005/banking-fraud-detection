import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load data
df = pd.read_csv("data/raw/credit_card_fraud_10k.csv")

# Preprocessing
df = pd.get_dummies(df, columns=['merchant_category'], drop_first=True)

# Features & target
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']

# Save feature columns
feature_columns = X.columns
pd.Series(feature_columns).to_csv("models/feature_columns.csv", index=False)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42
)

model.fit(X_resampled, y_resampled)

# Save model
with open("models/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")