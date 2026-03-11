import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import datetime
from scipy.special import expit  

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")


@st.cache_resource
def load_resources():
    bundle = joblib.load("model_bundle.pkl")
    model = bundle["model"]
    explainer = shap.TreeExplainer(model)
    return bundle, explainer

bundle, explainer = load_resources()
model = bundle["model"]
expected_columns = bundle["expected_columns"]
encoders = bundle["encoders"]


def safe_label_encode(col, value):
    classes = encoders[col].classes_
    has_unknown = "Unknown" in classes
    if value in classes:
        return value
    return "Unknown" if has_unknown else classes[0]

# =========================
# Sidebar Controls
# =========================
st.sidebar.title("⚙️ Fraud Detection Controls")
threshold = st.sidebar.slider("Fraud Probability Threshold (%)", 0, 100, 50) / 100
input_method = st.sidebar.radio("Select Input Method", ["Manual Input", "CSV / Excel Upload"])

# =========================
# Form Choices from Encoders
# =========================
merchant_choices = sorted(encoders["merchant"].classes_)
job_choices = sorted(encoders["job"].classes_)
state_choices = sorted(encoders["state"].classes_)
city_choices = sorted(encoders["city"].classes_)
category_choices = [
    'grocery_pos', 'shopping_pos', 'gas_transport', 'home', 'kids_pets',
    'shopping_net', 'misc_pos', 'entertainment', 'grocery_net',
    'misc_net', 'personal_care', 'travel', 'health_fitness', 'food_dining'
]
gender_choices = ['M', 'F']

# =========================
# Preprocessing Function
# =========================
def preprocess_data(df):
    df = df.copy()
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'], errors='coerce')
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')

    df['age'] = df['dob'].apply(lambda x: pd.Timestamp.now().year - x.year if pd.notnull(x) else 0)
    df['trans_year'] = df['trans_date_trans_time'].dt.year
    df['trans_month'] = df['trans_date_trans_time'].dt.month
    df['trans_date'] = df['trans_date_trans_time'].dt.day
    df['trans_day'] = df['trans_date_trans_time'].dt.day_name()
    df['trans_hr'] = df['trans_date_trans_time'].dt.hour
    df['year_month'] = df['trans_year'].astype(str) + '-' + df['trans_month'].astype(str)

    for col in encoders.keys():
        df[col] = df[col].apply(lambda x: safe_label_encode(col, x))
        df[col] = encoders[col].transform(df[col])

    df = pd.get_dummies(df, columns=['category', 'gender', 'trans_day', 'trans_month', 'trans_year'], drop_first=True)
    df.drop(columns=['dob', 'trans_date_trans_time'], inplace=True, errors='ignore')

    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[expected_columns]
    return df

# =========================
# Prediction Function
# =========================
def predict(df):
    processed_df = preprocess_data(df)
    proba = model.predict_proba(processed_df)[:, 1]
    results = ["Fraud 🚨" if p >= threshold else "Legit ✅" for p in proba]
    return proba, results, processed_df

# =========================
# Custom Waterfall (SHAP style + probability axis)
# =========================
def custom_waterfall(shap_values_row, feature_names, base_value, max_display=15):
    shap_series = pd.Series(shap_values_row, index=feature_names)
    shap_series = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index)
    shap_series = shap_series.iloc[:max_display]

    colors = ['red' if val > 0 else 'blue' for val in shap_series]

    log_odds = [base_value]
    for val in shap_series:
        log_odds.append(log_odds[-1] + val)

    prob_ticks = [expit(lo) * 100 for lo in log_odds]

    fig, ax = plt.subplots(figsize=(8, 6))

    current_lo = base_value
    for i, (feat, val) in enumerate(zip(shap_series.index, shap_series.values)):
        ax.barh(i, val, left=current_lo, color=colors[i])
        current_lo += val

    ax.set_yticks(range(len(shap_series)))
    ax.set_yticklabels(shap_series.index)
    ax.set_xlabel("Fraud Probability (%)")

    # Set probability-based xtick labels
    xticks = ax.get_xticks()
    ax.set_xticklabels([f"{expit(t) * 100:.1f}%" for t in xticks])

    ax.set_title("SHAP Waterfall Plot (Probability Axis)")
    plt.gca().invert_yaxis()
    return fig

# =========================
# Manual Input Form
# =========================
if input_method == "Manual Input":
    st.title("💳 Credit Card Fraud Detection")
    st.subheader("Enter Transaction Details Manually")

    with st.form(key="manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Transaction Amount ($)", min_value=0.0, step=0.01)
            merchant = st.selectbox("Merchant", merchant_choices)
            category = st.selectbox("Category", category_choices)
            job = st.selectbox("Job", job_choices)
            dob = st.date_input("Date of Birth", min_value=datetime.date(1920, 1, 1), max_value=datetime.date.today())
        with col2:
            gender = st.selectbox("Gender", gender_choices)
            city = st.selectbox("City", city_choices)
            state = st.selectbox("State", state_choices)
            trans_date = st.date_input("Transaction Date", min_value=datetime.date(2000, 1, 1), max_value=datetime.date.today())
            trans_time = st.time_input("Transaction Time", datetime.datetime.now().time())

        submit_button = st.form_submit_button("🔍 Predict")

    if submit_button:
        input_df = pd.DataFrame([{
            "amt": amount, "merchant": merchant, "category": category,
            "gender": gender, "job": job, "state": state, "city": city,
            "dob": dob, "trans_date_trans_time": pd.to_datetime(f"{trans_date} {trans_time}")
        }])
        
        proba, results, processed_df = predict(input_df)
        st.metric(label="Prediction Result", value=results[0], delta=f"Fraud Probability: {proba[0]:.2%}")

        # --- SHAP EXPLANATION ---
        st.write("---")
        with st.expander("💡 See Prediction Explanation"):
            shap_values = explainer.shap_values(processed_df)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            base_val = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value

            # Force Plot
            st.subheader("Force Plot")
            fig_force = shap.force_plot(
                base_val,
                shap_values[0],
                processed_df.iloc[0],
                matplotlib=True,
                show=False
            )
            buf = plt.gcf()
            st.pyplot(buf, bbox_inches='tight', use_container_width=True)
            plt.close()

            # Waterfall (SHAP style + probability axis)
            st.subheader("Waterfall Plot (Probability Axis)")
            fig_waterfall = custom_waterfall(shap_values[0], processed_df.columns, base_val, max_display=15)
            st.pyplot(fig_waterfall, bbox_inches='tight', use_container_width=True)
            plt.close(fig_waterfall)

# =========================
# CSV / Excel Upload
# =========================
else:
    st.title("📂 Batch Fraud Detection")
    st.subheader("Upload a CSV or Excel file for batch processing.")
    uploaded_file = st.file_uploader("Drag & drop your file here ⬇️", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())

        if st.button("🔍 Predict for All Transactions"):
            with st.spinner("Processing all transactions..."):
                proba, results, _ = predict(data)
                data["Fraud_Probability"] = proba
                data["Prediction"] = results
                st.success("Batch prediction complete!")
                st.dataframe(data)

                csv = data.to_csv(index=False).encode('utf-8')
                st.download_button("💾 Download Predictions", csv, "predictions.csv", "text/csv")
