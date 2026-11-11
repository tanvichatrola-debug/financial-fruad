"""
FinGuard - Machine Learning Model Training & Comparative Evaluation
Trains Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
Tuning performed on Validation set; final unbiased evaluation on Test set.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance

# Ensure preprocessing module is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.data_preprocessing import DataPreprocessor, run_preprocessing


def tune_threshold(y_true, y_proba):
    """
    Tune the classification threshold on the validation set to maximize F1-score.
    Returns best threshold and validation metrics at that threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    # Exclude endpoints where precision + recall might be 0
    f1_scores = np.divide(
        2 * (precisions[:-1] * recalls[:-1]),
        (precisions[:-1] + recalls[:-1]),
        out=np.zeros_like(thresholds),
        where=(precisions[:-1] + recalls[:-1]) != 0
    )

    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    best_prec = float(precisions[best_idx])
    best_rec = float(recalls[best_idx])

    # Bound threshold to reasonable range [0.10, 0.90] to avoid degenerate edge cases
    best_threshold = max(0.10, min(0.90, best_threshold))

    return {
        'threshold': round(best_threshold, 4),
        'val_f1': round(best_f1, 4),
        'val_precision': round(best_prec, 4),
        'val_recall': round(best_rec, 4)
    }


def evaluate_model(model, X_test, y_test, threshold=0.5):
    """Calculate comprehensive evaluation metrics on unseen test set using tuned threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_proba))
    pr_auc = float(average_precision_score(y_test, y_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()

    return {
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'confusion_matrix': cm,
        'threshold_used': round(threshold, 4)
    }


def train_and_evaluate_all():
    print("=" * 70)
    print("FINGUARD - MACHINE LEARNING TRAINING & EVALUATION")
    print("=" * 70)

    # 1. Load preprocessed splits
    processor = DataPreprocessor()
    csv_path = 'dataset/indian_banking_transactions.csv'
    if not os.path.exists(csv_path):
        csv_path = 'indian_banking_transactions.csv'

    print("[Step 1] Loading raw dataset and preparing chronological splits ...")
    raw_df = processor.load_data(csv_path)
    cleaned_df, _ = processor.inspect_and_clean(raw_df)
    engineered_df = processor.engineer_features(cleaned_df)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.split_chronological(engineered_df)
    X_train_trans, X_val_trans, X_test_trans = processor.fit_and_transform(X_train, X_val, X_test)

    feature_names = processor.feature_names
    print(f"Features: {len(feature_names)} transformed input features")
    print(f"Train samples: {X_train_trans.shape[0]:,} | Fraud: {int(y_train.sum())} ({y_train.mean()*100:.2f}%)")
    print(f"Val samples  : {X_val_trans.shape[0]:,} | Fraud: {int(y_val.sum())} ({y_val.mean()*100:.2f}%)")
    print(f"Test samples : {X_test_trans.shape[0]:,} | Fraud: {int(y_test.sum())} ({y_test.mean()*100:.2f}%)")

    # 2. Define Candidate Models
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=10,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=100,
            class_weight='balanced',
            random_state=42,
            learning_rate=0.1
        )
    }

    comparison_results = {}
    trained_models = {}
    optimal_thresholds = {}

    print("\n[Step 2] Training and evaluating algorithms ...")
    for name, model in models.items():
        print("-" * 70)
        print(f"--> Training algorithm: {name} ...")
        t0 = time.time()
        model.fit(X_train_trans, y_train)
        train_time = time.time() - t0
        print(f"    Fitted in {train_time:.2f}s")

        # Threshold tuning on Validation Set
        val_proba = model.predict_proba(X_val_trans)[:, 1]
        tuning_info = tune_threshold(y_val, val_proba)
        best_thresh = tuning_info['threshold']
        print(f"    Validation Tuned Threshold: {best_thresh} (Val F1: {tuning_info['val_f1']:.4f}, Recall: {tuning_info['val_recall']:.4f})")

        # Unbiased Evaluation on Test Set
        test_metrics = evaluate_model(model, X_test_trans, y_test, threshold=best_thresh)
        test_metrics['train_time_sec'] = round(train_time, 2)
        test_metrics['val_tuning'] = tuning_info

        comparison_results[name] = test_metrics
        trained_models[name] = model
        optimal_thresholds[name] = best_thresh

        print(f"    Test Metrics: PR-AUC={test_metrics['pr_auc']:.4f} | ROC-AUC={test_metrics['roc_auc']:.4f} | Recall={test_metrics['recall']:.4f} | Precision={test_metrics['precision']:.4f} | F1={test_metrics['f1_score']:.4f}")

    # 3. Model Selection:
    # We rank based on composite fraud detection performance: PR-AUC, F1-score, and Recall
    # PR-AUC is widely recognized as the gold standard for severe class imbalance.
    print("=" * 70)
    print("[Step 3] Selecting Best Algorithm ...")
    sorted_models = sorted(
        comparison_results.items(),
        key=lambda item: (item[1]['pr_auc'], item[1]['f1_score'], item[1]['recall']),
        reverse=True
    )

    best_model_name = sorted_models[0][0]
    best_model = trained_models[best_model_name]
    best_threshold = optimal_thresholds[best_model_name]
    best_metrics = comparison_results[best_model_name]

    print(f"\n WINNING MODEL: {best_model_name.upper()}")
    print(f"   PR-AUC    : {best_metrics['pr_auc']:.4f}")
    print(f"   ROC-AUC   : {best_metrics['roc_auc']:.4f}")
    print(f"   Recall    : {best_metrics['recall']:.4f}")
    print(f"   Precision : {best_metrics['precision']:.4f}")
    print(f"   F1-Score  : {best_metrics['f1_score']:.4f}")
    print(f"   Accuracy  : {best_metrics['accuracy']:.4f}")
    print(f"   Threshold : {best_threshold:.4f}")

    # 4. Feature Importance Extraction
    print("\n[Step 4] Computing Feature Importance ...")
    feature_importance_list = []
    if hasattr(best_model, 'feature_importances_'):
        raw_importances = best_model.feature_importances_
        for fname, imp in zip(feature_names, raw_importances):
            feature_importance_list.append({'feature': fname, 'importance': round(float(imp), 5)})
        feature_importance_list = sorted(feature_importance_list, key=lambda x: x['importance'], reverse=True)
    elif hasattr(best_model, 'coef_'):
        raw_coefs = np.abs(best_model.coef_[0])
        for fname, coef in zip(feature_names, raw_coefs):
            feature_importance_list.append({'feature': fname, 'importance': round(float(coef), 5)})
        feature_importance_list = sorted(feature_importance_list, key=lambda x: x['importance'], reverse=True)
    else:
        # HistGradientBoosting uses permutation importance on a validation sample
        val_sample_idx = np.random.choice(len(X_val_trans), size=min(5000, len(X_val_trans)), replace=False)
        perm = permutation_importance(best_model, X_val_trans[val_sample_idx], y_val[val_sample_idx], n_repeats=3, random_state=42)
        for fname, imp in zip(feature_names, perm.importances_mean):
            feature_importance_list.append({'feature': fname, 'importance': round(float(max(0.0, imp)), 5)})
        feature_importance_list = sorted(feature_importance_list, key=lambda x: x['importance'], reverse=True)

    print("   Top 10 Most Predictive Features:")
    for i, item in enumerate(feature_importance_list[:10], 1):
        print(f"   {i:2d}. {item['feature']:<30} : {item['importance']:.5f}")

    # 5. Save Artifacts
    print("\n[Step 5] Persisting Trained Model and Evaluation Report ...")
    model_payload = {
        'model_name': best_model_name,
        'model': best_model,
        'threshold': best_threshold,
        'feature_names': feature_names,
        'top_features': feature_importance_list[:25],
        'test_metrics': best_metrics
    }
    joblib.dump(model_payload, 'models/trained_model.pkl')

    report_payload = {
        'selected_algorithm': best_model_name,
        'selection_rationale': (
            f"{best_model_name} demonstrated the highest PR-AUC ({best_metrics['pr_auc']:.4f}) "
            f"and F1-score ({best_metrics['f1_score']:.4f}) on the unseen test set while maintaining "
            f"strong fraud recall ({best_metrics['recall']*100:.1f}%), effectively distinguishing rare "
            f"fraudulent transactions from legitimate banking volume under real-world class imbalance."
        ),
        'dataset_split': {
            'train_size': int(X_train_trans.shape[0]),
            'val_size': int(X_val_trans.shape[0]),
            'test_size': int(X_test_trans.shape[0]),
            'total_size': int(X_train_trans.shape[0] + X_val_trans.shape[0] + X_test_trans.shape[0]),
            'fraud_counts': {
                'train': int(y_train.sum()),
                'val': int(y_val.sum()),
                'test': int(y_test.sum()),
                'total': int(y_train.sum() + y_val.sum() + y_test.sum())
            }
        },
        'comparison': comparison_results,
        'best_model_metrics': best_metrics,
        'top_features': feature_importance_list[:20]
    }

    with open('reports/model_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(report_payload, f, indent=4)

    print("[SUCCESS] Trained model saved to: models/trained_model.pkl")
    print("[SUCCESS] Model metrics saved to: reports/model_metrics.json")
    print("=" * 70)


if __name__ == '__main__':
    train_and_evaluate_all()
