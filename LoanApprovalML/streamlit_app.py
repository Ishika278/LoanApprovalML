"""
Loan Approval Prediction System - Streamlit Web Application
This script provides a web interface for loan approval predictions using Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="🏦",
    layout="centered"
)

class LoanApprovalPredictor:
    """Loan Approval Prediction System"""
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, 'model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoders = joblib.load(encoders_path)
            return True
        return False
    
    def preprocess_input(self, features):
        """Preprocess user input for prediction"""
        df = pd.DataFrame([features])
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
        
        for col in categorical_cols:
            df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        return df
    
    def predict(self, features):
        """Make prediction"""
        # Preprocess input
        X = self.preprocess_input(features)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        # Decode prediction
        result = self.label_encoders['Loan_Status'].inverse_transform([prediction])[0]
        
        return result, probability


def main():
    """Main function"""
    # Initialize predictor
    predictor = LoanApprovalPredictor()
    
    # Check if model is loaded
    if predictor.model is None:
        st.error("Model not found! Please run train_model.py first.")
        return
    
    # Header
    st.title("🏦 Loan Approval Prediction System")
    st.markdown("---")
    
    # Sidebar with information
    st.sidebar.title("ℹ️ About")
    st.sidebar.info(
        "This ML-powered system predicts whether a loan application "
        "will be approved or rejected based on various applicant features."
    )
    st.sidebar.markdown("### Features Used:")
    st.sidebar.markdown("""
    - Gender
    - Marital Status
    - Number of Dependents
    - Education
    - Employment Status
    - Applicant Income
    - Co-applicant Income
    - Loan Amount
    - Loan Term
    - Credit History
    - Property Area
    """)
    
    # Main form
    st.header("📝 Loan Application Form")
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
        married = st.radio("Married", ["Yes", "No"], horizontal=True)
        dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"])
        education = st.radio("Education", ["Graduate", "Not Graduate"], horizontal=True)
        self_employed = st.radio("Self Employed", ["Yes", "No"], horizontal=True)
    
    with col2:
        applicant_income = st.number_input("Applicant Income ($)", min_value=0, value=5000, step=100)
        coapplicant_income = st.number_input("Co-applicant Income ($)", min_value=0, value=0, step=100)
        loan_amount = st.number_input("Loan Amount ($ thousands)", min_value=0, value=150, step=5)
        loan_term = st.selectbox("Loan Term (months)", [360, 180, 120, 84], index=0)
        credit_history = st.radio("Credit History", ["Good (1)", "Bad (0)"], horizontal=True)
    
    property_area = st.selectbox("Property Area", ["Urban", "Rural", "Semiurban"])
    
    # Convert credit_history
    credit_value = 1 if credit_history == "Good (1)" else 0
    
    # Create feature dictionary
    features = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_value,
        'Property_Area': property_area
    }
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔮 Predict Loan Approval", type="primary", use_container_width=True):
        with st.spinner("Processing prediction..."):
            # Make prediction
            result, probability = predictor.predict(features)
            
            # Display results
            st.markdown("---")
            st.header("📊 Prediction Result")
            
            if result == 'Y':
                st.success("🎉 **LOAN APPROVED!** 🎉")
                st.metric("Confidence", f"{probability[1]*100:.2f}%")
                
                # Success factors
                st.subheader("✅ Factors Contributing to Approval:")
                factors = []
                if credit_value == 1:
                    factors.append("✓ Good credit history")
                if applicant_income > 5000:
                    factors.append("✓ Strong income level")
                if coapplicant_income > 0:
                    factors.append("✓ Additional income from co-applicant")
                if loan_amount < 200:
                    factors.append("✓ Reasonable loan amount")
                if education == "Graduate":
                    factors.append("✓ Graduate education")
                if married == "Yes":
                    factors.append("✓ Stable marital status")
                    
                for factor in factors:
                    st.write(factor)
            else:
                st.error("❌ **LOAN REJECTED** ❌")
                st.metric("Confidence", f"{probability[0]*100:.2f}%")
                
                # Improvement suggestions
                st.subheader("💡 Suggestions for Improvement:")
                suggestions = []
                if credit_value == 0:
                    suggestions.append("• Improve your credit history")
                if applicant_income < 5000:
                    suggestions.append("• Increase your income")
                if loan_amount > 200:
                    suggestions.append("• Reduce the loan amount")
                if coapplicant_income == 0:
                    suggestions.append("• Consider adding a co-applicant")
                    
                for suggestion in suggestions:
                    st.write(suggestion)
            
            # Show applicant summary
            st.markdown("---")
            st.subheader("📋 Application Summary")
            
            summary_data = {
                "Field": ["Gender", "Married", "Dependents", "Education", "Self Employed",
                         "Applicant Income", "Co-applicant Income", "Loan Amount", 
                         "Loan Term", "Credit History", "Property Area"],
                "Value": [gender, married, dependents, education, self_employed,
                         f"${applicant_income:,}", f"${coapplicant_income:,}", 
                         f"${loan_amount:,}", f"{loan_term} months", 
                         "Good" if credit_value == 1 else "Bad", property_area]
            }
            st.table(pd.DataFrame(summary_data))


if __name__ == "__main__":
    main()
