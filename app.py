from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='.', static_folder='.')

# ============================================================================
# LOAD ALL 4 MODELS + SCALERS + FEATURE NAMES
# ============================================================================

models_path = 'models/'

try:
    # Load models
    model_1_default = pickle.load(open(os.path.join(models_path, 'logistic_default_model.pkl'), 'rb'))
    model_2_churn = pickle.load(open(os.path.join(models_path, 'random_forest_churn_model.pkl'), 'rb'))
    model_3_clustering = pickle.load(open(os.path.join(models_path, 'kmeans_segmentation_model.pkl'), 'rb'))
    model_4_profit = pickle.load(open(os.path.join(models_path, 'linear_regression_profit_model.pkl'), 'rb'))

    # Load scalers
    scaler_default = pickle.load(open(os.path.join(models_path, 'scaler_default.pkl'), 'rb'))
    scaler_clustering = pickle.load(open(os.path.join(models_path, 'scaler_clustering.pkl'), 'rb'))
    scaler_profit = pickle.load(open(os.path.join(models_path, 'scaler_profit.pkl'), 'rb'))

    # Load feature names
    feature_names_default = pickle.load(open(os.path.join(models_path, 'feature_names_default.pkl'), 'rb'))
    feature_names_churn = pickle.load(open(os.path.join(models_path, 'feature_names_churn.pkl'), 'rb'))
    feature_names_clustering = pickle.load(open(os.path.join(models_path, 'feature_names_clustering.pkl'), 'rb'))
    feature_names_profit = pickle.load(open(os.path.join(models_path, 'feature_names_profit.pkl'), 'rb'))

    # Load segment names
    segment_names = pickle.load(open(os.path.join(models_path, 'segment_names.pkl'), 'rb'))

    print("✓ All models loaded successfully!")

except FileNotFoundError as e:
    print(f"❌ ERROR: {e}")
    print("Make sure all pkl files are in the 'models/' folder")

# ============================================================================
# CATEGORICAL FEATURES (MUST MATCH TRAINING)
# ============================================================================

categorical_features = [
    'Customer_Segment', 'Gender', 'Occupation', 'Primary_Account_Type',
    'Loan_Product', 'Transaction_Type', 'Transaction_Channel',
    'Credit_Risk_Category', 'Internet_Banking_Usage', 'Transaction_Time_Risk'
]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_derived_features(data):
    """
    Calculate derived features from 20 input fields.
    Input: dict with 20 fields
    Output: dict with 20 + 10 derived fields
    """
    
    # Extract inputs
    annual_income = float(data.get('annual_income', 0))
    credit_limit = float(data.get('credit_limit', 0))
    outstanding_loan = float(data.get('outstanding_loan', 0))
    app_logins = float(data.get('app_logins', 0))
    satisfaction_score = float(data.get('satisfaction_score', 0))
    internet_banking = data.get('internet_banking', 'Low')
    
    # Create derived features
    derived = data.copy()
    
    # 1. Credit_Utilization_Ratio
    if credit_limit > 0:
        derived['Credit_Utilization_Ratio'] = outstanding_loan / credit_limit
    else:
        derived['Credit_Utilization_Ratio'] = 0
    
    # 2. Debt_to_Income_Ratio
    if annual_income > 0:
        derived['Debt_to_Income_Ratio'] = outstanding_loan / annual_income
    else:
        derived['Debt_to_Income_Ratio'] = 0
    
    # 3. Account_Balance_Before_INR (proxy: monthly income)
    derived['Account_Balance_Before_INR'] = annual_income / 12
    
    # 4. Average_Monthly_Spend_INR
    if annual_income > 0:
        derived['Average_Monthly_Spend_INR'] = (derived['Debt_to_Income_Ratio'] * annual_income) / 12
    else:
        derived['Average_Monthly_Spend_INR'] = 0
    
    # 5. Transactions_Last_30_Days (proxy: app logins)
    derived['Transactions_Last_30_Days'] = app_logins
    
    # 6. Days_Since_Last_Transaction (proxy: derived from logins)
    derived['Days_Since_Last_Transaction'] = 30 / (app_logins + 1)
    
    # 7. Transaction_Velocity_Score (app logins / 30 scaled to 0-10)
    derived['Transaction_Velocity_Score'] = min(app_logins / 3, 10)
    
    # 8. Digital_Engagement_Score
    # Map internet banking to numeric
    internet_banking_score = {'High': 10, 'Medium': 6, 'Low': 3, 'Never': 0}.get(internet_banking, 0)
    derived['Digital_Engagement_Score'] = (app_logins / 3 + internet_banking_score) / 2
    
    # 9. Fraud_Risk_Score (inverse of satisfaction)
    derived['Fraud_Risk_Score'] = 10 - satisfaction_score
    
    # 10. Customer_Lifetime_Value_INR
    tenure = float(data.get('tenure', 1))
    derived['Customer_Lifetime_Value_INR'] = annual_income * (tenure / 12) * 0.5
    
    return derived


def prepare_features_for_model(data, categorical_cols, feature_names):
    """
    One-hot encode categorical features and align with training feature order.
    
    data: dict with all 30 features (20 input + 10 derived)
    categorical_cols: list of categorical column names
    feature_names: list of feature names from training (from pkl)
    
    Returns: DataFrame aligned to training feature order
    """
    
    # Create DataFrame
    df = pd.DataFrame([data])
    
    # One-hot encode categorical features (same as training)
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Align to training feature order:
    # 1. Create empty DataFrame with all training feature names
    # 2. Fill in columns that exist in encoded data
    # 3. Fill missing columns with 0
    
    result = pd.DataFrame(0, index=[0], columns=feature_names)
    
    for col in df_encoded.columns:
        if col in feature_names:
            result[col] = df_encoded[col].values
    
    return result[feature_names]  # Ensure correct order


# ============================================================================
# ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the prediction form"""
    return render_template('banking_prediction_form.html')


@app.route('/results')
def results():
    """Serve the results page"""
    return render_template('results.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Receive form data, run all 4 models, return predictions.
    """
    
    try:
        # Parse form data
        data = request.get_json()
        
        # Convert string values to appropriate types
        data_processed = {
            'name': data.get('name', ''),
            'age': float(data.get('age', 0)),
            'gender': data.get('gender', ''),
            'occupation': data.get('occupation', ''),
            'state': data.get('state', ''),
            'city': data.get('city', ''),
            'annual_income': float(data.get('annual_income', 0)),
            'credit_score': float(data.get('credit_score', 0)),
            'credit_limit': float(data.get('credit_limit', 0)),
            'outstanding_loan': float(data.get('outstanding_loan', 0)),
            'tenure': float(data.get('tenure', 0)),
            'account_type': data.get('account_type', ''),
            'loan_product': data.get('loan_product', ''),
            'has_credit_card': data.get('has_credit_card', 'No'),
            'has_investment': data.get('has_investment', 'No'),
            'has_insurance': data.get('has_insurance', 'No'),
            'app_logins': float(data.get('app_logins', 0)),
            'internet_banking': data.get('internet_banking', 'Low'),
            'satisfaction_score': float(data.get('satisfaction_score', 0)),
            'complaint_count': float(data.get('complaint_count', 0)),
            # Rename for model compatibility
            'Customer_Segment': 'Mass Market',  # Default
            'Gender': data.get('gender', ''),
            'Occupation': data.get('occupation', ''),
            'Primary_Account_Type': data.get('account_type', ''),
            'Loan_Product': data.get('loan_product', ''),
            'Transaction_Type': 'Debit',  # Default
            'Transaction_Channel': 'Online',  # Default
            'Credit_Risk_Category': 'Medium',  # Default
            'Internet_Banking_Usage': data.get('internet_banking', 'Low'),
            'Transaction_Time_Risk': 'Low',  # Default
            'Age': float(data.get('age', 0)),
            'Annual_Income_INR': float(data.get('annual_income', 0)),
            'Credit_Score': float(data.get('credit_score', 0)),
            'Customer_Tenure_Months': float(data.get('tenure', 0)),
            'Credit_Limit_INR': float(data.get('credit_limit', 0)),
            'Outstanding_Loan_INR': float(data.get('outstanding_loan', 0)),
            'Mobile_App_Logins_Month': float(data.get('app_logins', 0)),
            'Customer_Satisfaction_Score': float(data.get('satisfaction_score', 0)),
            'Complaint_Count': float(data.get('complaint_count', 0)),
        }
        
        # Calculate derived features
        data_with_derived = calculate_derived_features(data_processed)
        
        # ====================================================================
        # MODEL 1: CREDIT DEFAULT RISK (Logistic Regression)
        # ====================================================================
        
        try:
            X_default = prepare_features_for_model(
                data_with_derived,
                categorical_features,
                feature_names_default
            )
            
            # Scale features
            X_default_scaled = scaler_default.transform(X_default)
            
            # Predict
            default_prob = model_1_default.predict_proba(X_default_scaled)[0, 1]
            default_pred = "Yes" if default_prob > 0.5 else "No"
            
            model_1_result = {
                'prediction': default_pred,
                'probability': float(default_prob),
                'risk_percentage': float(default_prob * 100),
                'risk_level': 'High Risk' if default_prob > 0.6 else 'Medium Risk' if default_prob > 0.4 else 'Low Risk'
            }
        except Exception as e:
            model_1_result = {'error': str(e)}
        
        # ====================================================================
        # MODEL 2: CUSTOMER CHURN RISK (Random Forest)
        # ====================================================================
        
        try:
            X_churn = prepare_features_for_model(
                data_with_derived,
                categorical_features,
                feature_names_churn
            )
            
            # NO scaling for Random Forest
            churn_prob = model_2_churn.predict_proba(X_churn)[0, 1]
            churn_pred = "At Risk" if churn_prob > 0.5 else "Low Risk"
            
            model_2_result = {
                'prediction': churn_pred,
                'probability': float(churn_prob),
                'risk_percentage': float(churn_prob * 100),
                'risk_level': 'High Risk' if churn_prob > 0.6 else 'Medium Risk' if churn_prob > 0.4 else 'Low Risk'
            }
        except Exception as e:
            model_2_result = {'error': str(e)}
        
        # ====================================================================
        # MODEL 3: CUSTOMER SEGMENTATION (K-Means)
        # ====================================================================
        
        try:
            # K-Means uses specific features
            clustering_features = [
                'Annual_Income_INR', 'Credit_Score', 'Customer_Tenure_Months',
                'Average_Monthly_Spend_INR', 'Digital_Engagement_Score',
                'Credit_Utilization_Ratio', 'Debt_to_Income_Ratio',
                'Customer_Lifetime_Value_INR', 'Net_Profit_INR',
                'Default_Probability', 'Churn_Probability'
            ]
            
            # Create DataFrame with clustering features
            df_clustering = pd.DataFrame([data_with_derived])
            
            # Handle missing numeric features
            for feat in clustering_features:
                if feat not in df_clustering.columns:
                    df_clustering[feat] = 0
            
            X_clustering = df_clustering[clustering_features]
            
            # Scale
            X_clustering_scaled = scaler_clustering.transform(X_clustering)
            
            # Predict cluster
            cluster = model_3_clustering.predict(X_clustering_scaled)[0]
            segment_name = segment_names.get(cluster, f"Segment {cluster}")
            
            model_3_result = {
                'cluster': int(cluster),
                'segment': segment_name
            }
        except Exception as e:
            model_3_result = {'error': str(e)}
        
        # ====================================================================
        # MODEL 4: PROFITABILITY PREDICTION (Linear Regression)
        # ====================================================================
        
        try:
            X_profit = prepare_features_for_model(
                data_with_derived,
                categorical_features,
                feature_names_profit
            )
            
            # Scale features
            X_profit_scaled = scaler_profit.transform(X_profit)
            
            # Predict
            clv_predicted = model_4_profit.predict(X_profit_scaled)[0]
            
            model_4_result = {
                'clv_predicted': float(clv_predicted),
                'clv_formatted': f"₹{clv_predicted:,.0f}",
                'profit_level': 'High Value' if clv_predicted > 2000000 else 'Medium Value' if clv_predicted > 1000000 else 'Standard'
            }
        except Exception as e:
            model_4_result = {'error': str(e)}
        
        # ====================================================================
        # COMBINE ALL RESULTS
        # ====================================================================
        
        response = {
            'status': 'success',
            'customer_name': data.get('name', 'N/A'),
            'model_1_default_risk': model_1_result,
            'model_2_churn_risk': model_2_result,
            'model_3_segmentation': model_3_result,
            'model_4_profitability': model_4_result,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
