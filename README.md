# 🏦 Banking Customer Prediction System

Complete ML-powered web application for predicting credit default risk, churn risk, customer segmentation, and profitability.

---

## 📋 Project Overview

**4 Trained Models:**
1. **Logistic Regression** → Credit Default Risk (Yes/No)
2. **Random Forest** → Customer Churn Risk (At Risk/Low Risk)
3. **K-Means Clustering** → Customer Segmentation (4 segments)
4. **Linear Regression** → Customer Lifetime Value (Profitability)

**User Input:** 20 customer fields
**Derived Features:** 10 calculated automatically
**Output:** 4 predictions + risk levels + recommendations

---

## 🚀 Quick Start

### 1. Extract & Setup

\`\`\`bash
# Unzip the banking-predictor.zip
unzip banking-predictor.zip
cd banking-predictor

# Install dependencies
pip install -r requirements.txt
\`\`\`

### 2. Add Trained Models

**Critical Step:** Copy the 11 pkl files into the \`models/\` folder

### 3. Run the Application

\`\`\`bash
python app.py
\`\`\`

### 4. Open in Browser

Navigate to: **http://127.0.0.1:5000**

---

## 📊 How It Works

### User Input (20 Fields)

Flask automatically calculates 10 derived features, then runs all 4 models simultaneously to produce:
- Credit Default Risk (0-100%)
- Customer Churn Risk (0-100%)
- Customer Segment (0-3)
- Customer Lifetime Value (₹)

---

## 🛠️ Technical Details

### Preprocessing

All categorical features use **One-Hot Encoding** (same as training):

\`\`\`python
pd.get_dummies(X, columns=categorical_features, drop_first=True)
\`\`\`

### Scaling

- **Model 1 (Default)** ✅ StandardScaler
- **Model 2 (Churn)** ❌ No scaling (Random Forest)
- **Model 3 (Segmentation)** ✅ StandardScaler
- **Model 4 (Profitability)** ✅ StandardScaler

---

## 📁 File Structure

\`\`\`
banking-predictor/
├── app.py                      # Flask backend
├── banking_prediction_form.html # Frontend form
├── requirements.txt            # Dependencies
├── README.md                   # This file
└── models/                     # Pkl files folder
\`\`\`

---

## 🔧 Troubleshooting

**FileNotFoundError:** Copy all 11 pkl files to models/ folder
**ModuleNotFoundError:** Run \`pip install -r requirements.txt\`
**Port in use:** Change port in app.py to 5001 or kill existing process

---

**Built with:** Flask, scikit-learn, pandas, numpy
**Date:** September 2026
