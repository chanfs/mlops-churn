"""
Gradio Demo for MLOps Churn Prediction

Interactive UI for testing the churn prediction model.
"""

import gradio as gr
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.insert(0, '/app')


class ChurnPredictor:
    """Simple wrapper for model inference."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_model()
    
    def load_model(self):
        """Load model and preprocessor."""
        model_path = Path("/app/models/model.pkl")
        preprocessor_path = Path("/app/data/processed/preprocessor.pkl")
        
        if model_path.exists():
            self.model = joblib.load(model_path)
        
        if preprocessor_path.exists():
            self.preprocessor = joblib.load(preprocessor_path)
    
    def predict(
        self,
        tenure: float,
        MonthlyCharges: float,
        TotalCharges: float,
        gender: str,
        SeniorCitizen: int,
        Partner: str,
        Dependents: str,
        PhoneService: str,
        InternetService: str,
        Contract: str,
        PaperlessBilling: str,
        PaymentMethod: str
    ) -> tuple:
        """Make prediction."""
        if self.model is None:
            return "Model not loaded", 0.0
        
        # Create feature dict
        features = {
            'tenure': tenure,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'PhoneService': PhoneService,
            'InternetService': InternetService,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Preprocess
        if self.preprocessor is not None:
            X = self.preprocessor.transform(df)
        else:
            X = df
        
        # Predict
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        result = "Churn" if prediction == 1 else "No Churn"
        confidence = probability if prediction == 1 else (1 - probability)
        
        return result, float(probability), float(confidence)


# Initialize predictor
predictor = ChurnPredictor()


def predict_churn(
    tenure,
    MonthlyCharges,
    TotalCharges,
    gender,
    SeniorCitizen,
    Partner,
    Dependents,
    PhoneService,
    MultipleLines,
    InternetService,
    OnlineSecurity,
    OnlineBackup,
    DeviceProtection,
    TechSupport,
    StreamingTV,
    StreamingMovies,
    Contract,
    PaperlessBilling,
    PaymentMethod
):
    """Main prediction function for Gradio."""
    result, probability, confidence = predictor.predict(
        tenure=tenure,
        MonthlyCharges=MonthlyCharges,
        TotalCharges=TotalCharges,
        gender=gender,
        SeniorCitizen=SeniorCitizen,
        Partner=Partner,
        Dependents=Dependents,
        PhoneService=PhoneService,
        InternetService=InternetService,
        Contract=Contract,
        PaperlessBilling=PaperlessBilling,
        PaymentMethod=PaymentMethod
    )
    
    # Create explanation
    if result == "Churn":
        explanation = f"⚠️ This customer is likely to churn with {probability:.1%} probability."
    else:
        explanation = f"✅ This customer is likely to stay with {confidence:.1%} confidence."
    
    return result, probability, explanation


# Create Gradio Interface
with gr.Blocks(title="Churn Prediction Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Customer Churn Prediction Demo
    
    Predict whether a customer will churn based on their service details.
    
    **Model Performance:**
    - Accuracy: 80%
    - ROC-AUC: 84%
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Customer Information")
            
            tenure = gr.Slider(0, 72, value=12, step=1, label="Tenure (months)")
            MonthlyCharges = gr.Number(value=29.85, label="Monthly Charges ($)")
            TotalCharges = gr.Number(value=350.0, label="Total Charges ($)")
            
            gr.Markdown("### Demographics")
            gender = gr.Radio(["Female", "Male"], value="Female", label="Gender")
            SeniorCitizen = gr.Radio(["No", "Yes"], value="No", label="Senior Citizen")
            Partner = gr.Radio(["Yes", "No"], value="Yes", label="Has Partner")
            Dependents = gr.Radio(["Yes", "No"], value="No", label="Has Dependents")
            
            gr.Markdown("### Services")
            PhoneService = gr.Radio(["Yes", "No"], value="Yes", label="Phone Service")
            MultipleLines = gr.Dropdown(
                ["No phone service", "No", "Yes"],
                value="No phone service",
                label="Multiple Lines"
            )
            InternetService = gr.Dropdown(
                ["DSL", "Fiber optic", "No"],
                value="DSL",
                label="Internet Service"
            )
            OnlineSecurity = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="No",
                label="Online Security"
            )
            OnlineBackup = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="Yes",
                label="Online Backup"
            )
            DeviceProtection = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="No",
                label="Device Protection"
            )
            TechSupport = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="No",
                label="Tech Support"
            )
            StreamingTV = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="Yes",
                label="Streaming TV"
            )
            StreamingMovies = gr.Dropdown(
                ["Yes", "No", "No internet service"],
                value="Yes",
                label="Streaming Movies"
            )
            
            gr.Markdown("### Contract & Payment")
            Contract = gr.Dropdown(
                ["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
                label="Contract Type"
            )
            PaperlessBilling = gr.Radio(["Yes", "No"], value="Yes", label="Paperless Billing")
            PaymentMethod = gr.Dropdown(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                value="Electronic check",
                label="Payment Method"
            )
            
            predict_btn = gr.Button("🔮 Predict Churn", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            gr.Markdown("### Prediction Result")
            
            result_label = gr.Label(label="Prediction")
            probability_bar = gr.Slider(0, 1, value=0, interactive=False, label="Churn Probability")
            explanation = gr.Textbox(label="Explanation", lines=3)
            
            gr.Markdown("""
            ### Model Info
            - **Algorithm:** Logistic Regression
            - **Features:** 46 (after preprocessing)
            - **Training Data:** 7,043 customers
            """)
    
    # Connect button
    predict_btn.click(
        fn=predict_churn,
        inputs=[
            tenure, MonthlyCharges, TotalCharges,
            gender, SeniorCitizen, Partner, Dependents,
            PhoneService, MultipleLines, InternetService,
            OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies,
            Contract, PaperlessBilling, PaymentMethod
        ],
        outputs=[result_label, probability_bar, explanation]
    )
    
    # Examples
    gr.Markdown("### Example Customers")
    
    examples = gr.Examples(
        examples=[
            [12, 29.85, 350.0, "Female", "No", "Yes", "No", "Yes", "No phone service", "DSL", "No", "Yes", "No", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check"],
            [65, 85.0, 5500.0, "Male", "No", "Yes", "Yes", "Yes", "Yes", "Fiber optic", "Yes", "Yes", "Yes", "Yes", "Yes", "Yes", "Two year", "No", "Bank transfer (automatic)"],
            [3, 45.0, 135.0, "Female", "Yes", "No", "No", "Yes", "No", "Fiber optic", "No", "No", "No", "No", "No", "No", "Month-to-month", "Yes", "Electronic check"],
        ],
        inputs=[
            tenure, MonthlyCharges, TotalCharges,
            gender, SeniorCitizen, Partner, Dependents,
            PhoneService, MultipleLines, InternetService,
            OnlineSecurity, OnlineBackup, DeviceProtection,
            TechSupport, StreamingTV, StreamingMovies,
            Contract, PaperlessBilling, PaymentMethod
        ],
        label="Click to load example"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
