"""
Loan Approval Prediction System - Model Training Script
This script trains a machine learning model to predict loan approval.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# Generate sample loan data
def generate_sample_data():
    """Generate sample loan approval dataset"""
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Married': np.random.choice(['Yes', 'No'], n_samples),
        'Dependents': np.random.choice(['0', '1', '2', '3+'], n_samples),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], n_samples),
        'Self_Employed': np.random.choice(['Yes', 'No'], n_samples),
        'ApplicantIncome': np.random.randint(1500, 50000, n_samples),
        'CoapplicantIncome': np.random.randint(0, 20000, n_samples),
        'LoanAmount': np.random.randint(30, 500, n_samples),
        'Loan_Amount_Term': np.random.choice([360, 180, 120, 84], n_samples),
        'Credit_History': np.random.choice([1, 0], n_samples, p=[0.8, 0.2]),
        'Property_Area': np.random.choice(['Urban', 'Rural', 'Semiurban'], n_samples),
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Generate target variable based on logic
    def determine_approval(row):
        score = 0
        if row['Credit_History'] == 1:
            score += 2
        if row['ApplicantIncome'] > 5000:
            score += 1
        if row['ApplicantIncome'] > 10000:
            score += 1
        if row['CoapplicantIncome'] > 2000:
            score += 1
        if row['LoanAmount'] < 200:
            score += 1
        if row['Education'] == 'Graduate':
            score += 1
        if row['Married'] == 'Yes':
            score += 1
        
        return 'Y' if score >= 5 else 'N'
    
    df['Loan_Status'] = df.apply(determine_approval, axis=1)
    return df

def preprocess_data(df):
    """Preprocess the loan data"""
    # Make a copy
    df = df.copy()
    
    # Drop Loan_ID if exists
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)
    
    # Handle missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    df['Loan_Status'] = le_target.fit_transform(df['Loan_Status'])
    label_encoders['Loan_Status'] = le_target
    
    return df, label_encoders

def train_model():
    """Train the loan approval prediction model"""
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION SYSTEM - MODEL TRAINING")
    print("=" * 60)
    
    # Generate sample data
    print("\n[1] Generating sample loan data...")
    df = generate_sample_data()
    print(f"    Generated {len(df)} samples")
    print(f"    Features: {list(df.columns)}")
    
    # Preprocess data
    print("\n[2] Preprocessing data...")
    df_processed, label_encoders = preprocess_data(df)
    
    # Separate features and target
    X = df_processed.drop('Loan_Status', axis=1)
    y = df_processed['Loan_Status']
    
    # Split data
    print("\n[3] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples: {len(X_test)}")
    
    # Scale features
    print("\n[4] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("\n[5] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    print("\n[6] Evaluating model...")
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"    Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Rejected', 'Approved']))
    
    print("\n    Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    {cm}")
    
    # Feature importance
    print("\n[7] Feature Importance:")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Save model and preprocessing objects
    print("\n[8] Saving model and preprocessing objects...")
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    joblib.dump(model, os.path.join(model_dir, 'model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    joblib.dump(label_encoders, os.path.join(model_dir, 'label_encoders.pkl'))
    
    print(f"    Model saved to: {os.path.join(model_dir, 'model.pkl')}")
    print(f"    Scaler saved to: {os.path.join(model_dir, 'scaler.pkl')}")
    print(f"    Encoders saved to: {os.path.join(model_dir, 'label_encoders.pkl')}")
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return model, scaler, label_encoders

if __name__ == "__main__":
    train_model()
