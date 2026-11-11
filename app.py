"""
FinGuard - Financial Fraud Detection Web Application
Main Flask application entry point.
"""

import os
from flask import Flask, render_template, redirect, url_for
from backend.routes import api_bp
from backend.prediction import FraudPredictor

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['JSON_SORT_KEYS'] = False

# Register API Blueprint
app.register_blueprint(api_bp)

# Page Routes
@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html', active_page='dashboard')

@app.route('/transactions')
def transactions():
    return render_template('transactions.html', active_page='transactions')

@app.route('/fraud-detection')
def fraud_detection():
    return render_template('fraud_detection.html', active_page='fraud_detection')

@app.route('/analytics')
def analytics():
    return render_template('analytics.html', active_page='analytics')

@app.route('/model-performance')
def model_performance():
    return render_template('model_performance.html', active_page='model_performance')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('dashboard.html', active_page='dashboard'), 404

# Preload ML model on startup to ensure instant prediction response
with app.app_context():
    try:
        FraudPredictor()
        print("[FinGuard] Preloaded Machine Learning Model & Pipeline successfully.")
    except Exception as e:
        print(f"[FinGuard WARNING] Failed to preload model: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"==================================================")
    print(f" FINGUARD - Financial Fraud Detection System")
    print(f" Server running at: http://127.0.0.1:{port}")
    print(f"==================================================")
    app.run(host='127.0.0.1', port=port, debug=False)
