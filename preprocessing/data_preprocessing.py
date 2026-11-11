"""
FinGuard - Data Cleaning and Preprocessing Pipeline
Handles automated inspection, cleaning, feature engineering,
chronological splitting, and leakage-free transformer fitting.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import joblib


class DataPreprocessor:
    """
    Automated data preprocessing, feature engineering, and pipeline persistence
    for financial transaction fraud detection.
    """

    CATEGORICAL_COLS = [
        'account_type',
        'transaction_type',
        'transaction_direction',
        'merchant_category',
        'state',
        'loan_type',
        'transaction_status',
        'channel',
        'kyc_status'
    ]

    NUMERICAL_COLS = [
        'transaction_amount',
        'account_balance',
        'credit_score',
        'has_loan',
        'emi_amount',
        'transaction_hour',
        'transaction_day',
        'transaction_month',
        'day_of_week',
        'is_weekend',
        'hour_sin',
        'hour_cos',
        'log_amount',
        'log_balance',
        'amount_to_balance_ratio',
        'balance_after_txn'
    ]

    def __init__(self):
        self.preprocessor = None
        self.feature_names = []
        self.report = {}

    def load_data(self, csv_path):
        """Load transactions CSV from disk."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file not found at: {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    def inspect_and_clean(self, df):
        """
        Inspect dataset, handle missing values intelligently, remove duplicates,
        and perform data type conversions without data loss.
        """
        original_shape = df.shape
        missing_before = df.isnull().sum().to_dict()
        duplicates_count = int(df.duplicated().sum())

        # Create working copy to preserve original dataframe
        cleaned = df.copy()

        # Remove duplicate records if present
        if duplicates_count > 0:
            cleaned = cleaned.drop_duplicates().reset_index(drop=True)

        # Missing value handling:
        # If has_loan = 0, replace missing loan_type with 'No Loan'
        # If has_loan = 1 and loan_type is missing, replace with 'Unknown'
        no_loan_mask = (cleaned['has_loan'] == 0) & (cleaned['loan_type'].isna())
        cleaned.loc[no_loan_mask, 'loan_type'] = 'No Loan'

        has_loan_missing_mask = (cleaned['has_loan'] == 1) & (cleaned['loan_type'].isna())
        cleaned.loc[has_loan_missing_mask, 'loan_type'] = 'Unknown'

        # Any remaining missing values in loan_type or others
        if cleaned['loan_type'].isna().any():
            cleaned['loan_type'] = cleaned['loan_type'].fillna('No Loan')

        # Safety fill for any numeric columns if missing (none expected, but for robustness)
        for col in ['transaction_amount', 'account_balance', 'credit_score', 'emi_amount']:
            if col in cleaned.columns and cleaned[col].isna().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

        for col in self.CATEGORICAL_COLS:
            if col in cleaned.columns and cleaned[col].isna().any():
                cleaned[col] = cleaned[col].fillna('Unknown')

        # Datetime conversions
        cleaned['transaction_date'] = pd.to_datetime(cleaned['transaction_date'])

        # Process transaction_time
        if 'transaction_hour' not in cleaned.columns or cleaned['transaction_hour'].isna().any():
            time_parsed = pd.to_datetime(cleaned['transaction_time'].astype(str), format='%H:%M', errors='coerce')
            cleaned['transaction_hour'] = time_parsed.dt.hour.fillna(12).astype(int)
        else:
            cleaned['transaction_hour'] = cleaned['transaction_hour'].astype(int)

        cleaning_log = [
            f"Verified {original_shape[0]:,} rows and {original_shape[1]} columns",
            f"Handled {missing_before.get('loan_type', 0):,} missing loan_type entries (No Loan for non-borrowers, Unknown for borrowers)",
            f"Processed duplicate check: found {duplicates_count} duplicate rows",
            "Parsed transaction_date to ISO datetime and synchronized transaction_hour"
        ]

        return cleaned, {
            'original_rows': int(original_shape[0]),
            'original_cols': int(original_shape[1]),
            'missing_before': {k: int(v) for k, v in missing_before.items() if v > 0},
            'duplicates_removed': duplicates_count,
            'cleaning_log': cleaning_log
        }

    def engineer_features(self, df):
        """
        Derive behavioral, temporal, and financial ratio features.
        Works identically for bulk dataframe and single real-time transaction.
        """
        data = df.copy()

        # Date-based features
        if not pd.api.types.is_datetime64_any_dtype(data['transaction_date']):
            data['transaction_date'] = pd.to_datetime(data['transaction_date'])

        data['transaction_day'] = data['transaction_date'].dt.day
        data['transaction_month'] = data['transaction_date'].dt.month
        data['transaction_year'] = data['transaction_date'].dt.year
        data['day_of_week'] = data['transaction_date'].dt.dayofweek
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(int)

        # Cyclic hour features
        hour = data['transaction_hour'].astype(float)
        data['hour_sin'] = np.sin(2.0 * np.pi * hour / 24.0)
        data['hour_cos'] = np.cos(2.0 * np.pi * hour / 24.0)

        # Financial ratio & log features
        amt = np.maximum(0.0, data['transaction_amount'].astype(float))
        bal = np.maximum(0.0, data['account_balance'].astype(float))

        data['log_amount'] = np.log1p(amt)
        data['log_balance'] = np.log1p(bal)
        data['amount_to_balance_ratio'] = amt / (bal + 1.0)

        # Net balance post-transaction
        is_debit = (data['transaction_direction'].astype(str).str.lower() == 'debit')
        data['balance_after_txn'] = np.where(is_debit, bal - amt, bal + amt)

        return data

    def build_pipeline(self):
        """Construct scikit-learn ColumnTransformer for leakage-free feature scaling and encoding."""
        num_transformer = RobustScaler()
        cat_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.NUMERICAL_COLS),
                ('cat', cat_transformer, self.CATEGORICAL_COLS)
            ]
        )
        return preprocessor

    def split_chronological(self, data):
        """
        Split dataset chronologically:
        - Train: 2019-01-01 to 2021-12-31 (60%, ~330,000 txns)
        - Validation: 2022-01-01 to 2022-12-31 (20%, ~110,000 txns)
        - Test: 2023-01-01 to 2024-01-01 (20%, ~110,000 txns)
        """
        train_mask = data['transaction_year'] <= 2021
        val_mask = data['transaction_year'] == 2022
        test_mask = data['transaction_year'] >= 2023

        feature_cols = self.NUMERICAL_COLS + self.CATEGORICAL_COLS

        X_train = data.loc[train_mask, feature_cols].copy()
        y_train = data.loc[train_mask, 'is_fraud'].astype(int).values

        X_val = data.loc[val_mask, feature_cols].copy()
        y_val = data.loc[val_mask, 'is_fraud'].astype(int).values

        X_test = data.loc[test_mask, feature_cols].copy()
        y_test = data.loc[test_mask, 'is_fraud'].astype(int).values

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def fit_and_transform(self, X_train, X_val, X_test):
        """Fit ColumnTransformer ONLY on X_train, then transform X_train, X_val, X_test."""
        self.preprocessor = self.build_pipeline()
        X_train_trans = self.preprocessor.fit_transform(X_train)
        X_val_trans = self.preprocessor.transform(X_val)
        X_test_trans = self.preprocessor.transform(X_test)

        # Retrieve one-hot encoded feature names
        cat_encoder = self.preprocessor.named_transformers_['cat']
        encoded_cat_names = list(cat_encoder.get_feature_names_out(self.CATEGORICAL_COLS))
        self.feature_names = self.NUMERICAL_COLS + encoded_cat_names

        return X_train_trans, X_val_trans, X_test_trans

    def prepare_single_transaction(self, raw_dict):
        """
        Prepares a single transaction dict from the web form for model prediction.
        Applies cleaning, feature engineering, and the fitted ColumnTransformer.
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor pipeline is not fitted or loaded.")

        row = dict(raw_dict)

        # Loan type defaults
        has_loan = int(row.get('has_loan', 0))
        loan_type = str(row.get('loan_type', '')).strip()
        if has_loan == 0 or not loan_type or loan_type.lower() == 'none':
            loan_type = 'No Loan'
        elif has_loan == 1 and (loan_type == '' or loan_type.lower() == 'unknown'):
            loan_type = 'Unknown'
        row['has_loan'] = has_loan
        row['loan_type'] = loan_type

        # Numeric casts
        for col in ['transaction_amount', 'account_balance', 'credit_score', 'emi_amount']:
            row[col] = float(row.get(col, 0.0))

        # Time and date
        date_str = str(row.get('transaction_date', '2024-01-01'))
        time_str = str(row.get('transaction_time', '12:00'))

        # Extract hour from time if not provided
        if 'transaction_hour' in row and row['transaction_hour'] not in ('', None):
            row['transaction_hour'] = int(row['transaction_hour'])
        else:
            try:
                row['transaction_hour'] = int(time_str.split(':')[0])
            except Exception:
                row['transaction_hour'] = 12

        # Convert to single-row DataFrame
        df_single = pd.DataFrame([row])
        df_single['transaction_date'] = pd.to_datetime(date_str)

        # Run feature engineering
        engineered = self.engineer_features(df_single)
        feature_cols = self.NUMERICAL_COLS + self.CATEGORICAL_COLS
        X_df = engineered[feature_cols]

        # Transform using fitted preprocessor
        X_trans = self.preprocessor.transform(X_df)
        return X_trans, engineered.iloc[0].to_dict()

    def create_sqlite_database(self, cleaned_df, db_path='dataset/finguard.db'):
        """Store cleaned transactions in an indexed SQLite database for rapid API responses."""
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        conn = sqlite3.connect(db_path)

        # Write to sqlite table
        # Store transaction_date as string formatted YYYY-MM-DD
        save_df = cleaned_df.copy()
        if pd.api.types.is_datetime64_any_dtype(save_df['transaction_date']):
            save_df['transaction_date'] = save_df['transaction_date'].dt.strftime('%Y-%m-%d')

        save_df.to_sql('transactions', conn, if_exists='replace', index=False)

        # Create indexes for fast lookup and pagination
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions (transaction_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cust_id ON transactions (customer_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_is_fraud ON transactions (is_fraud);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions (transaction_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_type ON transactions (transaction_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_channel ON transactions (channel);")
        conn.commit()
        conn.close()
        print(f"[DB] SQLite database created successfully at {db_path} with indexes.")

    def save_artifacts(self, pipeline_path='models/preprocessing_pipeline.pkl', report_path='reports/preprocessing_report.json'):
        """Save the fitted preprocessor pipeline and preprocessing report."""
        os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)

        joblib.dump({
            'pipeline': self.preprocessor,
            'feature_names': self.feature_names,
            'numerical_cols': self.NUMERICAL_COLS,
            'categorical_cols': self.CATEGORICAL_COLS
        }, pipeline_path)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=4)

        print(f"[Artifacts] Preprocessor saved to {pipeline_path}")
        print(f"[Artifacts] Preprocessing report saved to {report_path}")

    @classmethod
    def load_pipeline(cls, pipeline_path='models/preprocessing_pipeline.pkl'):
        """Load saved preprocessing pipeline."""
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(f"Pipeline file not found at: {pipeline_path}")
        data = joblib.load(pipeline_path)
        instance = cls()
        instance.preprocessor = data['pipeline']
        instance.feature_names = data['feature_names']
        instance.NUMERICAL_COLS = data['numerical_cols']
        instance.CATEGORICAL_COLS = data['categorical_cols']
        return instance


def run_preprocessing(csv_path='dataset/indian_banking_transactions.csv', create_db=True):
    """Orchestrate the complete preprocessing phase."""
    print("=" * 60)
    print("FINGUARD - EXECUTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)

    processor = DataPreprocessor()
    print(f"[1/6] Loading raw dataset from: {csv_path} ...")
    raw_df = processor.load_data(csv_path)

    print("[2/6] Inspecting and cleaning dataset ...")
    cleaned_df, clean_info = processor.inspect_and_clean(raw_df)

    print("[3/6] Engineering behavioral and temporal features ...")
    engineered_df = processor.engineer_features(cleaned_df)

    if create_db:
        print("[4/6] Initializing high-speed SQLite transaction store ...")
        processor.create_sqlite_database(cleaned_df)

    print("[5/6] Splitting data chronologically (Train: 2019-2021, Val: 2022, Test: 2023-2024) ...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_chronological(engineered_df)

    print(f"      Train set : {X_train.shape[0]:,} rows (Fraud: {int(y_train.sum()):,} / {y_train.mean()*100:.2f}%)")
    print(f"      Val set   : {X_val.shape[0]:,} rows (Fraud: {int(y_val.sum()):,} / {y_val.mean()*100:.2f}%)")
    print(f"      Test set  : {X_test.shape[0]:,} rows (Fraud: {int(y_test.sum()):,} / {y_test.mean()*100:.2f}%)")

    print("[6/6] Fitting ColumnTransformer exclusively on training data (Zero Leakage) ...")
    X_train_trans, X_val_trans, X_test_trans = processor.fit_and_transform(X_train, X_val, X_test)
    print(f"      Transformed feature space: {X_train_trans.shape[1]} features")

    # Build report
    processor.report = {
        'original_dimensions': [clean_info['original_rows'], clean_info['original_cols']],
        'missing_values_before': clean_info['missing_before'],
        'duplicates_removed': clean_info['duplicates_removed'],
        'cleaning_operations': clean_info['cleaning_log'],
        'final_dimensions': [int(engineered_df.shape[0]), int(engineered_df.shape[1])],
        'split_strategy': 'Chronological (Train: 2019-2021, Val: 2022, Test: 2023-2024)',
        'split_metrics': {
            'train_samples': int(X_train.shape[0]),
            'train_frauds': int(y_train.sum()),
            'train_fraud_rate_pct': round(float(y_train.mean() * 100), 3),
            'val_samples': int(X_val.shape[0]),
            'val_frauds': int(y_val.sum()),
            'val_fraud_rate_pct': round(float(y_val.mean() * 100), 3),
            'test_samples': int(X_test.shape[0]),
            'test_frauds': int(y_test.sum()),
            'test_fraud_rate_pct': round(float(y_test.mean() * 100), 3),
        },
        'numerical_features': processor.NUMERICAL_COLS,
        'categorical_features': processor.CATEGORICAL_COLS,
        'total_transformed_features': int(X_train_trans.shape[1]),
        'feature_names': processor.feature_names
    }

    processor.save_artifacts()
    print("=" * 60)
    print("PREPROCESSING PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    return processor, (X_train_trans, y_train), (X_val_trans, y_val), (X_test_trans, y_test)


if __name__ == '__main__':
    run_preprocessing()
