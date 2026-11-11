"""
FinGuard - Real-time Prediction and Explainability Engine
Loads saved preprocessing pipeline and trained Gradient Boosting model.
Performs data validation, feature transformation, inference, and evidence-based explanation.
"""

import os
import sys
import datetime
import numpy as np
import pandas as pd
import joblib

# Ensure preprocessing is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.data_preprocessing import DataPreprocessor


class FraudPredictor:
    """Singleton service to load models once and execute fast real-time inference."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(FraudPredictor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_path='models/trained_model.pkl', pipeline_path='models/preprocessing_pipeline.pkl'):
        if self._initialized:
            return

        print(f"[Predictor] Initializing model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found at: {pipeline_path}")

        model_payload = joblib.load(model_path)
        self.model = model_payload['model']
        self.model_name = model_payload['model_name']
        self.threshold = float(model_payload['threshold'])
        self.feature_names = model_payload['feature_names']
        self.top_features = model_payload.get('top_features', [])
        self.test_metrics = model_payload.get('test_metrics', {})

        self.preprocessor_service = DataPreprocessor.load_pipeline(pipeline_path)
        self._initialized = True
        print(f"[Predictor] Successfully loaded {self.model_name} with optimal threshold {self.threshold:.4f}")

    def validate_inputs(self, data):
        """Validate web form inputs and return sanitized dictionary and error list."""
        errors = []
        sanitized = {}

        # 1. Transaction Amount
        try:
            amt = float(data.get('transaction_amount', -1))
            if amt < 0:
                errors.append("Transaction Amount must be a positive number.")
            sanitized['transaction_amount'] = amt
        except (ValueError, TypeError):
            errors.append("Invalid Transaction Amount format.")

        # 2. Account Balance
        try:
            bal = float(data.get('account_balance', -1))
            if bal < 0:
                errors.append("Account Balance must be a positive number.")
            sanitized['account_balance'] = bal
        except (ValueError, TypeError):
            errors.append("Invalid Account Balance format.")

        # 3. Credit Score
        try:
            score = int(data.get('credit_score', 0))
            if score < 300 or score > 900:
                errors.append("Credit Score must be between 300 and 900.")
            sanitized['credit_score'] = score
        except (ValueError, TypeError):
            errors.append("Invalid Credit Score format.")

        # 4. EMI Amount
        try:
            emi = float(data.get('emi_amount', 0.0))
            if emi < 0:
                errors.append("EMI Amount cannot be negative.")
            sanitized['emi_amount'] = emi
        except (ValueError, TypeError):
            errors.append("Invalid EMI Amount format.")

        # 5. Has Loan
        try:
            has_loan = int(data.get('has_loan', 0))
            if has_loan not in (0, 1):
                has_loan = 1 if str(data.get('has_loan')).lower() in ('yes', 'true', '1') else 0
            sanitized['has_loan'] = has_loan
        except Exception:
            sanitized['has_loan'] = 0

        # Categorical strings
        sanitized['account_type'] = str(data.get('account_type', 'Savings')).strip()
        sanitized['transaction_type'] = str(data.get('transaction_type', 'UPI')).strip()
        sanitized['transaction_direction'] = str(data.get('transaction_direction', 'Debit')).strip()
        sanitized['merchant_category'] = str(data.get('merchant_category', 'Food & Dining')).strip()
        sanitized['state'] = str(data.get('state', 'Maharashtra')).strip()
        sanitized['loan_type'] = str(data.get('loan_type', 'No Loan')).strip()
        sanitized['transaction_status'] = str(data.get('transaction_status', 'Success')).strip()
        sanitized['channel'] = str(data.get('channel', 'Mobile_App')).strip()
        sanitized['kyc_status'] = str(data.get('kyc_status', 'Verified')).strip()

        # Date & Time
        sanitized['transaction_date'] = str(data.get('transaction_date', datetime.date.today().isoformat())).strip()
        sanitized['transaction_time'] = str(data.get('transaction_time', '12:00')).strip()

        if 'transaction_hour' in data and str(data['transaction_hour']).isdigit():
            sanitized['transaction_hour'] = int(data['transaction_hour'])
        else:
            try:
                sanitized['transaction_hour'] = int(sanitized['transaction_time'].split(':')[0])
            except Exception:
                sanitized['transaction_hour'] = 12

        return sanitized, errors

    def generate_explanation(self, sanitized_data, engineered_row, fraud_prob, is_fraud):
        """
        Produce a clear, evidence-based explanation grounded in the transaction's
        concrete values and mathematical model attributes.
        """
        reasons = []

        amt = sanitized_data['transaction_amount']
        bal = sanitized_data['account_balance']
        ratio = engineered_row.get('amount_to_balance_ratio', amt / (bal + 1.0))
        hour = sanitized_data['transaction_hour']
        channel = sanitized_data['channel']
        tx_type = sanitized_data['transaction_type']
        kyc = sanitized_data['kyc_status']

        if is_fraud:
            # High amount flag
            if amt > 100000:
                reasons.append(
                    f"Substantial transaction value (INR {amt:,.2f}) significantly exceeds standard daily retail velocity thresholds."
                )
            elif amt > 50000:
                reasons.append(
                    f"Elevated transaction amount (INR {amt:,.2f}) falls into the 95th percentile of transaction records."
                )

            # High balance depletion ratio
            if ratio > 0.8:
                reasons.append(
                    f"High liquidity depletion ratio ({ratio:.1f}x): requested transaction amount represents an aggressive fraction of total account balance (INR {bal:,.2f})."
                )

            # High-risk transfer channels or payment methods
            if tx_type in ['RTGS', 'Cheque'] and amt > 50000:
                reasons.append(
                    f"Transaction executed via {tx_type}, an instrument statistically exhibiting more than double the baseline fraud incidence."
                )

            # Off-hours execution
            if hour in [0, 1, 2, 3, 4, 5]:
                reasons.append(
                    f"Unusual temporal footprint: transaction initiated during nocturnal off-peak hours ({hour:02d}:00 hrs)."
                )

            # KYC status risk
            if kyc in ['Pending', 'Expired']:
                reasons.append(
                    f"Account compliance vulnerability: user KYC status is flagged as '{kyc}', elevating overall transaction risk profile."
                )

            if not reasons:
                reasons.append(
                    f"Multi-feature combination across {tx_type} transaction type, {sanitized_data['merchant_category']} category, and account profile pushed the model's estimated risk score beyond the decision threshold."
                )
        else:
            if amt <= 25000:
                reasons.append(
                    f"Transaction amount (INR {amt:,.2f}) aligns closely with typical retail transactions."
                )
            if ratio <= 0.3:
                reasons.append(
                    f"Conservative balance exposure: amount represents only {ratio*100:.1f}% of current available funds."
                )
            if kyc == 'Verified':
                reasons.append("Account KYC is fully verified and in active standing.")
            if hour >= 8 and hour <= 22:
                reasons.append(f"Standard operational daylight hours ({hour:02d}:00 hrs).")

            if not reasons:
                reasons.append("Transaction parameters are well within normal behavioral bounds for this account category.")

        return reasons

    def predict(self, raw_data):
        """Validate input, preprocess, run model inference, and return prediction payload."""
        sanitized, errors = self.validate_inputs(raw_data)
        if errors:
            return {
                'success': False,
                'errors': errors
            }

        # Preprocess single transaction
        X_trans, engineered_row = self.preprocessor_service.prepare_single_transaction(sanitized)

        # Run model inference
        proba = float(self.model.predict_proba(X_trans)[0, 1])
        is_fraud = int(proba >= self.threshold)

        explanation_factors = self.generate_explanation(sanitized, engineered_row, proba, is_fraud)

        recommendation = (
            "POTENTIAL FRAUD ALERT: Flag for senior transaction risk team review. Verify beneficiary details and obtain two-factor biometric or voice authorization prior to fund settlement."
            if is_fraud else
            "STANDARD CLEARANCE: Transaction conforms to expected banking parameters. Safe for immediate automated settlement."
        )

        return {
            'success': True,
            'prediction': is_fraud,
            'status_label': 'POTENTIAL FRAUD DETECTED' if is_fraud else 'LEGITIMATE TRANSACTION',
            'is_fraud': bool(is_fraud),
            'fraud_probability': round(proba, 4),
            'fraud_probability_pct': round(proba * 100, 2),
            'threshold_used': round(self.threshold, 4),
            'model_name': self.model_name,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'recommendation': recommendation,
            'explanation_factors': explanation_factors,
            'disclaimer': "This risk assessment is generated by a supervised machine learning model (FinGuard) based on historical statistical patterns and does not constitute a legal or contractual determination of fraud."
        }


# Quick sanity test
if __name__ == '__main__':
    predictor = FraudPredictor()
    test_sample = {
        'transaction_amount': 250000.0,
        'account_balance': 300000.0,
        'account_type': 'Savings',
        'transaction_type': 'RTGS',
        'transaction_direction': 'Debit',
        'merchant_category': 'Investment',
        'state': 'Maharashtra',
        'credit_score': 620,
        'has_loan': 1,
        'loan_type': 'Personal',
        'emi_amount': 5000.0,
        'transaction_status': 'Pending',
        'channel': 'Mobile_App',
        'kyc_status': 'Pending',
        'transaction_date': '2024-05-10',
        'transaction_time': '02:30'
    }
    result = predictor.predict(test_sample)
    print("Test Prediction Output:")
    print(result)
