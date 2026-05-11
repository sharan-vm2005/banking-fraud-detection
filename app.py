import streamlit as st
import pandas as pd
import pickle
import mysql.connector

# -----------------------------------
# Load trained model
# -----------------------------------
with open("models/fraud_model.pkl", "rb") as f:
    model = pickle.load(f)

# -----------------------------------
# Load feature columns
# -----------------------------------
feature_columns = pd.read_csv(
    "models/feature_columns.csv"
)

feature_columns = feature_columns.iloc[:, 0].tolist()

# -----------------------------------
# Streamlit Title
# -----------------------------------
st.title("💳 Banking Fraud Detection System")

# ===================================
# CSV UPLOAD SECTION
# ===================================

st.header("Bulk Transaction Detection")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read uploaded CSV
    bulk_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Transactions")
    st.dataframe(bulk_df)

    # -----------------------------------
    # Align columns
    # -----------------------------------
    input_df = bulk_df.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # -----------------------------------
    # Predict probabilities
    # -----------------------------------
    y_probs = model.predict_proba(input_df)[:, 1]

    # ML prediction
    ml_pred = (y_probs > 0.5).astype(int)

    # -----------------------------------
    # Rule-based fraud detection
    # -----------------------------------
    rule_pred = (
            bulk_df['amount'] > 50000
    ).astype(int)

    # -----------------------------------
    # Final hybrid prediction
    # -----------------------------------
    final_pred = (
            ml_pred | rule_pred
    ).astype(int)

    # -----------------------------------
    # Convert labels
    # -----------------------------------
    bulk_df['prediction'] = final_pred

    bulk_df['prediction'] = bulk_df[
        'prediction'
    ].map({
        0: 'Safe',
        1: 'Fraud'
    })

    # -----------------------------------
    # Dashboard Metrics
    # -----------------------------------
    total_transactions = len(bulk_df)

    fraud_count = (
            bulk_df['prediction'] == 'Fraud'
    ).sum()

    safe_count = (
            bulk_df['prediction'] == 'Safe'
    ).sum()

    fraud_percentage = (
                               fraud_count / total_transactions
                       ) * 100

    st.header("📊 Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Transactions",
        total_transactions
    )

    col2.metric(
        "Fraud Transactions",
        fraud_count
    )

    col3.metric(
        "Safe Transactions",
        safe_count
    )

    col4.metric(
        "Fraud %",
        f"{fraud_percentage:.2f}%"
    )

    # -----------------------------------
    # Prediction Results
    # -----------------------------------
    st.subheader("Prediction Results")

    st.dataframe(bulk_df)

    # -----------------------------------
    # Fraud Insights
    # -----------------------------------
    st.header("📈 Fraud Insights")

    # Pie Chart
    pie_data = bulk_df['prediction'].value_counts()

    st.subheader("Fraud vs Safe Transactions")

    st.pyplot(
        pie_data.plot.pie(
            autopct='%1.1f%%',
            figsize=(5, 5)
        ).get_figure()
    )

    # -----------------------------------
    # Fraud by Hour
    # -----------------------------------
    st.subheader("🚨 Fraud Transactions by Hour")

    fraud_only = bulk_df[
        bulk_df['prediction'] == 'Fraud'
        ]

    if len(fraud_only) > 0:
        fraud_hour = fraud_only.groupby(
            'transaction_hour'
        ).size()

        st.bar_chart(fraud_hour)

    # -----------------------------------
    # High Risk Transactions
    # -----------------------------------
    st.subheader("⚠️ High Risk Transactions")

    high_risk = bulk_df[
        bulk_df['prediction'] == 'Fraud'
        ]

    st.dataframe(high_risk)

    # -----------------------------------
    # Save to MySQL
    # -----------------------------------
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="fraud_detection"
    )

    cursor = conn.cursor()

    for _, row in bulk_df.iterrows():
        query = """
        INSERT INTO transactions (
            amount,
            transaction_hour,
            location_mismatch,
            device_trust_score,
            velocity_last_24h,
            cardholder_age,
            prediction
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """

        values = (
            float(row['amount']),
            int(row['transaction_hour']),
            int(row['location_mismatch']),
            float(row['device_trust_score']),
            int(row['velocity_last_24h']),
            int(row['cardholder_age']),
            row['prediction']
        )

        cursor.execute(query, values)

    conn.commit()

    cursor.close()
    conn.close()

    st.success(
        "✅ Transactions stored in MySQL database"
    )

# ===================================
# SINGLE TRANSACTION SECTION
# ===================================

st.header("🧾 Single Transaction Detection")

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0
)

transaction_hour = st.slider(
    "Transaction Hour",
    0,
    23
)

location_mismatch = st.selectbox(
    "Location Mismatch",
    [0, 1]
)

device_trust_score = st.slider(
    "Device Trust Score",
    0.0,
    1.0
)

velocity_last_24h = st.number_input(
    "Transactions in Last 24 Hours",
    min_value=0
)

cardholder_age = st.number_input(
    "Cardholder Age",
    min_value=18
)

# -----------------------------------
# Predict Button
# -----------------------------------
if st.button("Check Fraud"):

    input_data = pd.DataFrame({
        'amount': [amount],
        'transaction_hour': [transaction_hour],
        'location_mismatch': [location_mismatch],
        'device_trust_score': [device_trust_score],
        'velocity_last_24h': [velocity_last_24h],
        'cardholder_age': [cardholder_age]
    })

    # Align columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # ML prediction
    prediction = model.predict(input_data)[0]

    # Rule-based logic
    rule_flag = 0

    if amount > 50000:
        rule_flag = 1

    # Final prediction
    final_prediction = prediction | rule_flag

    # Result
    if final_prediction == 1:
        st.error("⚠️ Fraud Detected")
    else:
        st.success("✅ Safe Transaction")