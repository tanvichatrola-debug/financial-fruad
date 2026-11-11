"""
FinGuard - Comprehensive Automated Application Test Suite
Tests all pages, API endpoints, error handlers, search, filters, and predictions.
"""

import unittest
import json
from app import app


class FinGuardTestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def test_01_page_routes(self):
        """Verify all HTML pages render successfully (200 OK)."""
        print("\n[Test 1] Testing HTML Page Routes ...")
        pages = ['/dashboard', '/transactions', '/fraud-detection', '/analytics', '/model-performance']
        for page in pages:
            res = self.client.get(page)
            self.assertEqual(res.status_code, 200, f"Page {page} returned status {res.status_code}")
            print(f"   [OK] {page} -> HTTP 200 OK")

        # Test root redirect
        res_root = self.client.get('/')
        self.assertEqual(res_root.status_code, 302)
        print("   [OK] / -> HTTP 302 Redirect to /dashboard")

    def test_02_api_dashboard(self):
        """Verify /api/dashboard returns KPIs, recent txns, and chart data."""
        print("\n[Test 2] Testing /api/dashboard ...")
        res = self.client.get('/api/dashboard')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        kpis = data['kpis']
        self.assertEqual(kpis['total_transactions'], 550000)
        self.assertEqual(kpis['fraudulent_transactions'], 4873)
        self.assertEqual(len(data['recent_transactions']), 10)
        self.assertIn('fraud_vs_legit', data['charts'])
        print(f"   [OK] KPIs: Total={kpis['total_transactions']:,}, Fraud={kpis['fraudulent_transactions']:,}, Model={kpis['selected_model']}")

    def test_03_api_transactions_pagination(self):
        """Verify /api/transactions pagination, search, and filtering."""
        print("\n[Test 3] Testing /api/transactions pagination & filters ...")
        res = self.client.get('/api/transactions?page=1&per_page=15')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data['page'], 1)
        self.assertEqual(data['per_page'], 15)
        self.assertEqual(len(data['transactions']), 15)
        self.assertEqual(data['total_records'], 550000)
        print(f"   [OK] Default pagination: 15 records returned out of {data['total_records']:,} total")

        # Search test
        res_search = self.client.get('/api/transactions?search=TXN000000001')
        data_search = json.loads(res_search.data)
        self.assertTrue(data_search['total_records'] >= 1)
        self.assertEqual(data_search['transactions'][0]['transaction_id'], 'TXN000000001')
        print("   [OK] Search query for 'TXN000000001' returned matching transaction record")

        # Fraud filter test
        res_fraud = self.client.get('/api/transactions?fraud_status=fraud&per_page=10')
        data_fraud = json.loads(res_fraud.data)
        self.assertEqual(data_fraud['total_records'], 4873)
        for tx in data_fraud['transactions']:
            self.assertEqual(tx['is_fraud'], 1)
        print("   [OK] Fraud filter returned strictly fraudulent transactions (Total 4,873)")

        # Instrument filter test
        res_type = self.client.get('/api/transactions?transaction_type=RTGS&per_page=5')
        data_type = json.loads(res_type.data)
        for tx in data_type['transactions']:
            self.assertEqual(tx['transaction_type'], 'RTGS')
        print("   [OK] Transaction type filter (RTGS) verified")

    def test_04_api_transaction_detail(self):
        """Verify /api/transactions/<txn_id> returns complete transaction."""
        print("\n[Test 4] Testing /api/transactions/<txn_id> ...")
        res = self.client.get('/api/transactions/TXN000000001')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        tx = data['transaction']
        self.assertEqual(tx['transaction_id'], 'TXN000000001')
        self.assertIn('customer_id', tx)
        self.assertIn('account_balance', tx)
        print(f"   [OK] Found TXN000000001: Customer={tx['customer_id']}, Amount=INR {tx['transaction_amount']}, Status={tx['transaction_status']}")

    def test_05_api_analytics(self):
        """Verify /api/analytics returns multi-dimensional distributions."""
        print("\n[Test 5] Testing /api/analytics ...")
        res = self.client.get('/api/analytics')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        a = data['analytics']
        self.assertIn('by_type', a)
        self.assertIn('by_channel', a)
        self.assertIn('by_category', a)
        self.assertIn('by_state', a)
        self.assertIn('by_hour', a)
        self.assertIn('by_amount_range', a)
        print(f"   [OK] Analytics loaded: {len(a['by_type'])} types, {len(a['by_channel'])} channels, {len(a['by_hour'])} hours")

    def test_06_api_model_performance(self):
        """Verify /api/model-performance returns evaluation benchmarks and feature importances."""
        print("\n[Test 6] Testing /api/model-performance ...")
        res = self.client.get('/api/model-performance')
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data['success'])
        m = data['metrics']
        self.assertEqual(m['selected_algorithm'], 'Gradient Boosting')
        self.assertIn('Random Forest', m['comparison'])
        self.assertIn('Logistic Regression', m['comparison'])
        self.assertIn('Decision Tree', m['comparison'])
        self.assertTrue(len(m['top_features']) >= 10)
        print(f"   [OK] Model Performance verified: Selected={m['selected_algorithm']}, Compared={list(m['comparison'].keys())}")

    def test_07_api_prediction_engine(self):
        """Verify /api/predict correctly evaluates transaction risk with explanations."""
        print("\n[Test 7] Testing /api/predict inference and explainability ...")

        # Legitimate transaction test
        legit_payload = {
            'transaction_amount': 1500.0,
            'account_balance': 95000.0,
            'account_type': 'Savings',
            'transaction_type': 'UPI',
            'transaction_direction': 'Debit',
            'merchant_category': 'Food & Dining',
            'state': 'Maharashtra',
            'credit_score': 760,
            'has_loan': 0,
            'loan_type': 'No Loan',
            'emi_amount': 0.0,
            'transaction_status': 'Success',
            'channel': 'Mobile_App',
            'kyc_status': 'Verified',
            'transaction_date': '2024-05-15',
            'transaction_time': '14:30'
        }
        res_legit = self.client.post('/api/predict', data=json.dumps(legit_payload), content_type='application/json')
        self.assertEqual(res_legit.status_code, 200)
        data_legit = json.loads(res_legit.data)
        self.assertTrue(data_legit['success'])
        self.assertFalse(data_legit['is_fraud'])
        self.assertEqual(data_legit['status_label'], 'LEGITIMATE TRANSACTION')
        self.assertTrue(len(data_legit['explanation_factors']) > 0)
        print(f"   [OK] Legitimate sample: Probability={data_legit['fraud_probability_pct']}% -> {data_legit['status_label']}")

        # High-risk / fraudulent transaction test (RTGS, high amount, depleted balance, pending KYC)
        fraud_payload = {
            'transaction_amount': 850000.0,
            'account_balance': 35000.0,
            'account_type': 'Current',
            'transaction_type': 'RTGS',
            'transaction_direction': 'Debit',
            'merchant_category': 'Investment',
            'state': 'Maharashtra',
            'credit_score': 580,
            'has_loan': 1,
            'loan_type': 'Personal',
            'emi_amount': 12000.0,
            'transaction_status': 'Pending',
            'channel': 'Web',
            'kyc_status': 'Pending',
            'transaction_date': '2024-05-15',
            'transaction_time': '02:15'
        }
        res_fraud = self.client.post('/api/predict', data=json.dumps(fraud_payload), content_type='application/json')
        self.assertEqual(res_fraud.status_code, 200)
        data_fraud = json.loads(res_fraud.data)
        self.assertTrue(data_fraud['success'])
        self.assertTrue(data_fraud['is_fraud'])
        self.assertEqual(data_fraud['status_label'], 'POTENTIAL FRAUD DETECTED')
        self.assertTrue(len(data_fraud['explanation_factors']) > 0)
        print(f"   [OK] High-risk sample: Probability={data_fraud['fraud_probability_pct']}% -> {data_fraud['status_label']}")
        print(f"     Explanation factors: {data_fraud['explanation_factors'][:2]}")

        # Input validation test (negative amount)
        bad_payload = {'transaction_amount': -500.0}
        res_bad = self.client.post('/api/predict', data=json.dumps(bad_payload), content_type='application/json')
        self.assertEqual(res_bad.status_code, 400)
        data_bad = json.loads(res_bad.data)
        self.assertFalse(data_bad['success'])
        print(f"   [OK] Validation test: correctly caught bad inputs -> {data_bad['errors']}")


if __name__ == '__main__':
    unittest.main()
