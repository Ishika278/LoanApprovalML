me # Loan Approval Prediction System 🏦

A machine learning-based web application that predicts whether a loan application will be approved or rejected.

## Features

- 🤖 **Machine Learning Model**: Random Forest classifier with 97% accuracy
- 🌐 **Web Interface**: Built with Streamlit
- 📊 **Interactive Form**: Easy-to-use loan application form
- 📈 **Detailed Results**: Shows prediction confidence and key factors

## Demo

Try the live app: [Loan Approval Prediction System](https://loan-approval-prediction-system.streamlit.app)

## Installation

1. Clone the repository:
```
bash
git clone <your-repo-url>
cd LoanApprovalML
```

2. Install dependencies:
```
bash
pip install -r requirements.txt
```

3. Train the model:
```
bash
python train_model.py
```

4. Run the app locally:
```
bash
streamlit run streamlit_app.py
```

## Deployment to Streamlit Community Cloud

### Method 1: Deploy from GitHub

1. **Prepare your files:**
   - Ensure all required files are in your GitHub repository:
     - `streamlit_app.py` (main app)
     - `train_model.py` (model training)
     - `model.pkl` (trained model)
     - `scaler.pkl` (feature scaler)
     - `label_encoders.pkl` (label encoders)
     - `requirements.txt` (dependencies)

2. **Push to GitHub:**
   
```
bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-github-repo-url>
   git push -u origin main
   
```

3. **Deploy on Streamlit Community Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch, and main file path (`streamlit_app.py`)
   - Click "Deploy"

### Method 2: Deploy using CLI

1. Install Streamlit:
```
bash
pip install streamlit
```

2. Deploy directly:
```
bash
streamlit deploy streamlit_app.py
```

## Usage

1. Fill in the loan application form with your details:
   - Personal information (Gender, Marital Status, Dependents, Education)
   - Employment details (Self-employed, Income)
   - Loan details (Amount, Term)
   - Credit history and property area

2. Click "Predict Loan Approval"

3. View the result with:
   - Approval/Rejection status
   - Confidence percentage
   - Key factors affecting the decision

## Model Details

- **Algorithm**: Random Forest Classifier
- **Accuracy**: 97%
- **Key Features**:
  1. Credit History (33.96%)
  2. Applicant Income (22.88%)
  3. Loan Amount (9.81%)
  4. Marital Status (9.54%)
  5. Co-applicant Income (9.50%)

## Project Structure

```
LoanApprovalML/
├── streamlit_app.py      # Streamlit web application
├── train_model.py       # Model training script
├── app.py              # CLI application (alternative)
├── model.pkl           # Trained model
├── scaler.pkl          # Feature scaler
├── label_encoders.pkl  # Label encoders
├── requirements.txt     # Dependencies
└── README.md           # This file
```

## License

MIT License
