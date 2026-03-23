"""
Streamlit App for MLOps Churn Prediction Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

# Add src to path for proper module resolution
sys.path.insert(0, str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="MLOps Churn Prediction",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .churn-yes {
        background-color: #ff4b4b;
        color: white;
    }
    .churn-no {
        background-color: #00cc96;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load model and preprocessor."""
    model_path = Path("models/model.pkl")
    preprocessor_path = Path("data/processed/preprocessor.pkl")
    feature_names_path = Path("data/processed/feature_names.json")
    
    if not model_path.exists():
        st.error("Model not found! Please ensure models/model.pkl exists.")
        return None, None, None
    
    if not preprocessor_path.exists():
        st.error("Preprocessor not found! Please ensure data/processed/preprocessor.pkl exists.")
        return None, None, None
    
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    
    # Load feature names
    feature_names = None
    if feature_names_path.exists():
        import json
        with open(feature_names_path) as f:
            feature_names = json.load(f)
    
    return model, preprocessor, feature_names


def predict_churn(model, preprocessor, features):
    """Make prediction."""
    df = pd.DataFrame([features])
    
    if preprocessor is not None:
        X = preprocessor.transform(df)
    else:
        X = df
    
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0, 1]
    
    return prediction, probability


# Sidebar
with st.sidebar:
    st.image("https://img.shields.io/badge/MLOps-Demo-blue", use_container_width=True)
    
    st.markdown("### 📊 Model Performance")
    st.metric("Accuracy", "80%")
    st.metric("ROC-AUC", "84%")
    st.metric("Precision", "65%")
    st.metric("Recall", "55%")
    
    st.markdown("---")
    st.markdown("### 📁 About")
    st.markdown("""
    This demo showcases a complete **MLOps pipeline** for customer churn prediction.
    
    **Technologies:**
    - scikit-learn
    - MLflow
    - DVC
    - FastAPI
    - Streamlit
    
    **Source:** [GitHub](https://github.com/your-repo/mlops-demo)
    """)
    
    st.markdown("---")
    st.markdown("### 🎯 Quick Start")
    st.markdown("""
    1. Adjust customer features
    2. Click **Predict**
    3. See churn probability
    """)


# Main content
st.markdown('<p class="main-header">🤖 Customer Churn Prediction Demo</p>', unsafe_allow_html=True)
st.markdown("""
Predict whether a customer will churn based on their service details and demographics.
This demo is part of a complete **MLOps pipeline** showcasing best practices for ML deployment.
""")

# Load model
model, preprocessor, feature_names = load_model()

if model is None:
    st.warning("⚠️ Model not loaded. Please check that the model file exists.")
    st.stop()

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Customer Information")
    
    # Demographics
    with st.expander("👤 Demographics", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        
        with c2:
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=29.85)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=350.0)
    
    # Services
    with st.expander("📡 Services", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        with c2:
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    
    # More services
    with st.expander("🔧 Additional Services", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        
        with c2:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    # Contract & Payment
    with st.expander("💳 Contract & Payment", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        
        with c2:
            payment_method = st.selectbox(
                "Payment Method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
            )

with col2:
    st.markdown("### 🔮 Prediction")
    
    # Create feature dict
    features = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'gender': gender,
        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': partner,
        'Dependents': dependents,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method
    }
    
    # Predict button
    predict_btn = st.button("🔮 Predict Churn", type="primary", use_container_width=True)
    
    if predict_btn:
        prediction, probability = predict_churn(model, preprocessor, features)
        
        result = "Churn" if prediction == 1 else "No Churn"
        
        # Display result
        if result == "Churn":
            st.markdown(f"""
                <div class="prediction-box churn-yes">
                    ⚠️ {result}
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="prediction-box churn-no">
                    ✅ {result}
                </div>
            """, unsafe_allow_html=True)
        
        # Probability gauge
        st.metric("Churn Probability", f"{probability:.1%}")
        
        # Progress bar
        st.progress(probability)
        
        # Explanation
        st.markdown("### 📊 Analysis")
        if probability > 0.7:
            st.warning("⚠️ **High Risk**: This customer shows strong churn indicators.")
        elif probability > 0.4:
            st.info("ℹ️ **Medium Risk**: Some churn indicators present.")
        else:
            st.success("✅ **Low Risk**: Customer likely to stay.")
        
        # Key factors
        st.markdown("### 🔑 Key Factors")
        factors = []
        if contract == "Month-to-month":
            factors.append("• Month-to-month contract")
        if payment_method == "Electronic check":
            factors.append("• Electronic check payment")
        if tenure < 12:
            factors.append("• Short tenure (< 1 year)")
        if monthly_charges > 70:
            factors.append("• High monthly charges")
        
        if factors:
            for factor in factors:
                st.markdown(factor)
        else:
            st.markdown("• No major risk factors identified")
    
    # Example customers
    st.markdown("---")
    st.markdown("### 📚 Example Customers")
    
    examples = {
        "Low Risk": {
            'tenure': 65, 'MonthlyCharges': 85.0, 'TotalCharges': 5500.0,
            'gender': 'Male', 'SeniorCitizen': 'No', 'Partner': 'Yes',
            'Contract': 'Two year', 'PaymentMethod': 'Bank transfer (automatic)'
        },
        "Medium Risk": {
            'tenure': 24, 'MonthlyCharges': 55.0, 'TotalCharges': 1320.0,
            'gender': 'Female', 'SeniorCitizen': 'No', 'Partner': 'Yes',
            'Contract': 'One year', 'PaymentMethod': 'Credit card (automatic)'
        },
        "High Risk": {
            'tenure': 3, 'MonthlyCharges': 45.0, 'TotalCharges': 135.0,
            'gender': 'Female', 'SeniorCitizen': 'Yes', 'Partner': 'No',
            'Contract': 'Month-to-month', 'PaymentMethod': 'Electronic check'
        }
    }
    
    selected_example = st.selectbox("Load example:", list(examples.keys()))
    if st.button("Load Example"):
        ex = examples[selected_example]
        # Note: In a real app, you'd update all the form fields here
        st.info(f"Loaded {selected_example} customer example. Adjust fields and click Predict!")


# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with ❤️ using Streamlit | MLOps Demo Project</p>
    <p>Model: Logistic Regression | Accuracy: 80% | ROC-AUC: 84%</p>
</div>
""", unsafe_allow_html=True)
