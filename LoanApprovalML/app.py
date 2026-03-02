"""
Loan Approval Prediction System - CLI Application
This script provides a command-line interface for loan approval predictions.
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys

class LoanApprovalPredictor:
    """Loan Approval Prediction System"""
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.scaler = None
        self.label_encoders = None
        self.feature_names = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
            'Loan_Amount_Term', 'Credit_History', 'Property_Area'
        ]
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        model_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(model_dir, 'model.pkl')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        
        if not os.path.exists(model_path):
            print("Error: Model not found! Please run train_model.py first.")
            sys.exit(1)
        
        print("Loading trained model...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.label_encoders = joblib.load(encoders_path)
        print("Model loaded successfully!\n")
    
    def get_user_input(self):
        """Get loan application details from user"""
        print("\n" + "=" * 60)
        print("LOAN APPLICATION FORM")
        print("=" * 60)
        
        # Gender
        print("\nGender:")
        print("  1. Male")
        print("  2. Female")
        gender_choice = input("Enter choice (1-2): ")
        gender = 'Male' if gender_choice == '1' else 'Female'
        
        # Married
        print("\nMarried:")
        print("  1. Yes")
        print("  2. No")
        married_choice = input("Enter choice (1-2): ")
        married = 'Yes' if married_choice == '1' else 'No'
        
        # Dependents
        print("\nNumber of Dependents:")
        print("  1. 0")
        print("  2. 1")
        print("  3. 2")
        print("  4. 3+")
        dep_choice = input("Enter choice (1-4): ")
        dependents = ['0', '1', '2', '3+'][int(dep_choice) - 1]
        
        # Education
        print("\nEducation:")
        print("  1. Graduate")
        print("  2. Not Graduate")
        edu_choice = input("Enter choice (1-2): ")
        education = 'Graduate' if edu_choice == '1' else 'Not Graduate'
        
        # Self Employed
        print("\nSelf Employed:")
        print("  1. Yes")
        print("  2. No")
        emp_choice = input("Enter choice (1-2): ")
        self_employed = 'Yes' if emp_choice == '1' else 'No'
        
        # Income
        print("\nApplicant Income (monthly):")
        applicant_income = float(input("Enter amount: $"))
        
        print("\nCo-applicant Income (monthly):")
        coapplicant_income = float(input("Enter amount (0 if none): $"))
        
        # Loan Details
        print("\nLoan Amount (in thousands):")
        loan_amount = float(input("Enter amount: $"))
        
        print("\nLoan Term:")
        print("  1. 360 months (30 years)")
        print("  2. 180 months (15 years)")
        print("  3. 120 months (10 years)")
        print("  4. 84 months (7 years)")
        term_choice = input("Enter choice (1-4): ")
        loan_term = [360, 180, 120, 84][int(term_choice) - 1]
        
        # Credit History
        print("\nCredit History:")
        print("  1. Good (1)")
        print("  2. Bad (0)")
        credit_choice = input("Enter choice (1-2): ")
        credit_history = 1 if credit_choice == '1' else 0
        
        # Property Area
        print("\nProperty Area:")
        print("  1. Urban")
        print("  2. Rural")
        print("  3. Semi-Urban")
        prop_choice = input("Enter choice (1-3): ")
        property_area = ['Urban', 'Rural', 'Semiurban'][int(prop_choice) - 1]
        
        # Create feature array
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
            'Credit_History': credit_history,
            'Property_Area': property_area
        }
        
        return features
    
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
    
    def display_result(self, features, result, probability):
        """Display prediction result"""
        print("\n" + "=" * 60)
        print("LOAN APPLICATION RESULT")
        print("=" * 60)
        
        print("\nApplicant Details:")
        print(f"  Gender: {features['Gender']}")
        print(f"  Married: {features['Married']}")
        print(f"  Dependents: {features['Dependents']}")
        print(f"  Education: {features['Education']}")
        print(f"  Self Employed: {features['Self_Employed']}")
        print(f"  Applicant Income: ${features['ApplicantIncome']:,.2f}")
        print(f"  Co-applicant Income: ${features['CoapplicantIncome']:,.2f}")
        print(f"  Loan Amount: ${features['LoanAmount']:,.2f}")
        print(f"  Loan Term: {features['Loan_Amount_Term']} months")
        print(f"  Credit History: {'Good' if features['Credit_History'] == 1 else 'Bad'}")
        print(f"  Property Area: {features['Property_Area']}")
        
        print("\n" + "-" * 60)
        
        if result == 'Y':
            print("\n🎉 RESULT: LOAN APPROVED! 🎉")
            print(f"\nConfidence: {probability[1]*100:.2f}%")
        else:
            print("\n❌ RESULT: LOAN REJECTED ❌")
            print(f"\nConfidence: {probability[0]*100:.2f}%")
        
        print("\n" + "-" * 60)
        
        # Display factors that influenced the decision
        print("\nKey Factors:")
        if features['Credit_History'] == 1:
            print("  ✓ Good credit history")
        else:
            print("  ✗ Poor credit history")
        
        if features['ApplicantIncome'] > 5000:
            print("  ✓ Good income level")
        
        if features['CoapplicantIncome'] > 0:
            print("  ✓ Additional income from co-applicant")
        
        if features['LoanAmount'] < 200:
            print("  ✓ Reasonable loan amount")
        
        if features['Education'] == 'Graduate':
            print("  ✓ Graduate education")
    
    def run(self):
        """Run the loan approval prediction system"""
        while True:
            print("\n" + "=" * 60)
            print("LOAN APPROVAL PREDICTION SYSTEM")
            print("=" * 60)
            print("\n1. Apply for Loan")
            print("2. Exit")
            
            choice = input("\nEnter your choice: ")
            
            if choice == '1':
                features = self.get_user_input()
                result, probability = self.predict(features)
                self.display_result(features, result, probability)
            elif choice == '2':
                print("\nThank you for using Loan Approval Prediction System!")
                break
            else:
                print("\nInvalid choice! Please try again.")

def main():
    """Main function"""
    predictor = LoanApprovalPredictor()
    predictor.run()

if __name__ == "__main__":
    main()
