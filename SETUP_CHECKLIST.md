# ✅ SETUP CHECKLIST

Complete this checklist to get the banking prediction app running:

---

## 📦 FILES READY (GENERATED)

✅ `banking_prediction_form.html` - Interactive prediction form with 20 fields
✅ `app.py` - Flask backend with all 4 models loaded
✅ `requirements.txt` - Python dependencies
✅ `README.md` - Full documentation

---

## 🔴 CRITICAL: PKL FILES (YOU MUST PROVIDE)

These 11 pkl files must be copied into the `models/` folder:

**FROM YOUR TRAINING NOTEBOOKS:**

1. ✗ `logistic_default_model.pkl` - Model 1 (Logistic Regression)
2. ✗ `random_forest_churn_model.pkl` - Model 2 (Random Forest)
3. ✗ `kmeans_segmentation_model.pkl` - Model 3 (K-Means)
4. ✗ `linear_regression_profit_model.pkl` - Model 4 (Linear Regression)
5. ✗ `scaler_default.pkl` - Scaler for Model 1
6. ✗ `scaler_clustering.pkl` - Scaler for Model 3
7. ✗ `scaler_profit.pkl` - Scaler for Model 4
8. ✗ `feature_names_default.pkl` - Feature names Model 1
9. ✗ `feature_names_churn.pkl` - Feature names Model 2
10. ✗ `feature_names_clustering.pkl` - Feature names Model 3
11. ✗ `feature_names_profit.pkl` - Feature names Model 4
12. ✗ `segment_names.pkl` - Cluster name mapping

---

## 🚀 DEPLOYMENT STEPS

### Step 1: Get PKL Files
```
Run your 4 training notebooks (AADM_Assignment.ipynb)
Each saves pkl files automatically to models/ folder
Download all 11 pkl files
```

### Step 2: Organize Project Structure
```
banking-predictor/
├── app.py
├── banking_prediction_form.html
├── requirements.txt
├── README.md
└── models/
    ├── logistic_default_model.pkl
    ├── random_forest_churn_model.pkl
    ├── kmeans_segmentation_model.pkl
    ├── linear_regression_profit_model.pkl
    ├── scaler_default.pkl
    ├── scaler_clustering.pkl
    ├── scaler_profit.pkl
    ├── feature_names_default.pkl
    ├── feature_names_churn.pkl
    ├── feature_names_clustering.pkl
    ├── feature_names_profit.pkl
    └── segment_names.pkl
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Flask App
```bash
python app.py
```

### Step 5: Open in Browser
```
http://127.0.0.1:5000
```

### Step 6: Test with Sample Data
Fill form with test customer data:
- Name: John Doe
- Age: 35
- Gender: Male
- Annual Income: ₹500,000
- Credit Score: 750
- Tenure: 36 months
- Other fields: Choose realistic values

Click "Generate Predictions" → Should see 4 results

---

## 🔍 WHAT FLASK DOES

**On Form Submission:**

1. **Receives:** 20 customer input fields
2. **Calculates:** 10 derived features automatically
3. **One-Hot Encodes:** Categorical features (same as training)
4. **Scales:** Numeric features (except Model 2)
5. **Runs 4 Models:** All simultaneously
6. **Returns:** JSON with 4 predictions

**Example Output:**
```json
{
  "customer_name": "John Doe",
  "model_1_default_risk": {
    "prediction": "No",
    "probability": 0.23,
    "risk_percentage": 23.0,
    "risk_level": "Low Risk"
  },
  "model_2_churn_risk": {
    "prediction": "Low Risk",
    "probability": 0.18,
    "risk_percentage": 18.0,
    "risk_level": "Low Risk"
  },
  "model_3_segmentation": {
    "cluster": 1,
    "segment": "High-Value Premium"
  },
  "model_4_profitability": {
    "clv_predicted": 1500000.0,
    "clv_formatted": "₹1,500,000",
    "profit_level": "Medium Value"
  },
  "timestamp": "2026-09-27 10:30:45"
}
```

---

## 🛑 COMMON ERRORS & FIXES

**Error:** FileNotFoundError: models/logistic_default_model.pkl
**Fix:** Copy all 11 pkl files to models/ folder

**Error:** ModuleNotFoundError: No module named 'flask'
**Fix:** Run `pip install -r requirements.txt`

**Error:** Address already in use
**Fix:** Change port in app.py: `app.run(port=5001)`

**Error:** Predictions are NaN
**Fix:** Ensure pkl files are from SAME training session (not mixed)

---

## 📋 FORM FIELDS (20 Total)

### Personal (6)
- Name
- Age
- Gender
- Occupation
- State
- City

### Financial (5)
- Annual_Income_INR
- Credit_Score
- Credit_Limit_INR
- Outstanding_Loan_INR
- Customer_Tenure_Months

### Products (5)
- Primary_Account_Type
- Loan_Product
- Has_Credit_Card (Y/N)
- Has_Investment_Product (Y/N)
- Has_Insurance (Y/N)

### Behavior (4)
- Mobile_App_Logins_Month
- Internet_Banking_Usage
- Customer_Satisfaction_Score
- Complaint_Count

---

## ✨ DERIVED FEATURES (Calculated Auto)

Flask calculates these 10 features from the 20 inputs:

1. Credit_Utilization_Ratio = Outstanding_Loan / Credit_Limit
2. Debt_to_Income_Ratio = Outstanding_Loan / Annual_Income
3. Account_Balance_Before_INR = Annual_Income / 12
4. Average_Monthly_Spend_INR = DTI × Annual_Income / 12
5. Transactions_Last_30_Days = App_Logins
6. Days_Since_Last_Transaction = 30 / (App_Logins + 1)
7. Transaction_Velocity_Score = App_Logins / 3 (capped at 10)
8. Digital_Engagement_Score = (App_Logins + Internet_Banking_Score) / 2
9. Fraud_Risk_Score = 10 - Satisfaction_Score
10. Customer_Lifetime_Value_INR = Annual_Income × (Tenure / 12) × 0.5

---

## 🎯 NEXT STEPS (AFTER SETUP)

1. **Test Locally:** Run app.py and test with multiple customer profiles
2. **Validate Results:** Verify predictions make business sense
3. **Fine-tune Thresholds:** Adjust risk level cutoffs if needed
4. **Deploy to Cloud:** 
   - Heroku (easiest for beginners)
   - AWS Elastic Beanstalk
   - Google Cloud Platform
   - Azure App Service

5. **Create Reports Page:** Add a results display page (optional)
6. **Database Integration:** Store predictions in database (optional)
7. **Authentication:** Add login/security (for production)

---

## 📞 QUESTIONS?

- **Models not loading?** → Verify all 11 pkl files in models/ folder
- **Port error?** → Change port number in app.py
- **Wrong predictions?** → Ensure pkl files from same training session
- **Form not submitting?** → Check browser console for JS errors

---

**Status:** READY FOR DEPLOYMENT ✅
**Date:** September 27, 2026
