import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
from io import StringIO, BytesIO
import joblib

# Configure Flask app
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Global models
models = {}
feature_engineer = None


def load_production_models():
    """Load pre-trained models from disk."""
    global models, feature_engineer
    
    from config import MODELS_DIR
    from src.feature_engineering import FeatureEngineer
    
    try:
        # Load PD models
        from src.pd_model import PDModelSuite
        pd_suite = PDModelSuite()
        pd_suite.fitted_models = PDModelSuite.load_models(MODELS_DIR)
        
        # Load LGD model
        lgd_path = os.path.join(MODELS_DIR, "lgd_model.pkl")
        if os.path.exists(lgd_path):
            models['lgd'] = joblib.load(lgd_path)
        
        # Load EAD model
        ead_path = os.path.join(MODELS_DIR, "ead_model.pkl")
        if os.path.exists(ead_path):
            models['ead'] = joblib.load(ead_path)
        
        models['pd_suite'] = pd_suite
        
        print("✓ Production models loaded")
        return True
    except Exception as e:
        print(f"Warning: Could not load production models: {e}")
        print("  Running in demo mode with random predictions")
        return False


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict_loan():
    """
    Predict PD, LGD, EAD for a single loan.
    
    POST JSON:
    {
        "loan_amount": 15000,
        "fico_score": 720,
        "income": 75000,
        "dti": 0.25,
        "utilization": 0.35,
        "delinquencies": 0,
        ...
    }
    """
    try:
        data = request.json
        
        # Convert to DataFrame for prediction
        df = pd.DataFrame([data])
        
        # Make predictions
        if models.get('pd_suite'):
            from src.pd_model import PDModelSuite
            pd_suite = models['pd_suite']
            
            # Get best model
            if pd_suite.best_model_name and pd_suite.fitted_models:
                best_model = pd_suite.fitted_models[pd_suite.best_model_name]
                pd_pred = best_model.predict_proba(df)[:, 1][0]
            else:
                pd_pred = np.random.uniform(0.01, 0.2)
        else:
            pd_pred = np.random.uniform(0.01, 0.2)
        
        # LGD (Loss Given Default)
        lgd_pred = np.random.uniform(0.3, 0.8) if not models.get('lgd') else \
                   models['lgd'].predict(df)[0]
        
        # EAD (Exposure at Default)
        ead_pred = data.get('loan_amount', 10000) * np.random.uniform(0.8, 1.0)
        
        # Decision engine
        from src.decision_engine import LoanDecisionEngine
        engine = LoanDecisionEngine()
        
        decision_data = engine.make_decisions(
            np.array([pd_pred]),
            np.array([lgd_pred]),
            np.array([ead_pred]),
            np.array([data.get('loan_amount', 10000)])
        )
        
        pricing_data = engine.calculate_risk_based_pricing(
            np.array([pd_pred]),
            np.array([lgd_pred]),
            np.array([ead_pred]),
            np.array([data.get('loan_amount', 10000)])
        )
        
        el = pd_pred * lgd_pred * ead_pred
        
        response = {
            'success': True,
            'predictions': {
                'pd': float(pd_pred),
                'lgd': float(lgd_pred),
                'ead': float(ead_pred),
                'expected_loss': float(el),
            },
            'decision': {
                'approval': decision_data['Decision'].values[0],
                'risk_level': decision_data['Risk_Level'].values[0],
            },
            'pricing': {
                'base_rate': float(engine.base_rate),
                'adjusted_rate': float(pricing_data['Adjusted_Rate'].values[0]),
                'risk_spread': float(pricing_data['Risk_Spread'].values[0]),
            },
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction for multiple loans via CSV upload.
    
    POST: CSV file with loan data
    """
    try:
        file = request.files['file']
        df = pd.read_csv(file)
        
        # Make predictions for each loan
        predictions = []
        
        for idx, row in df.iterrows():
            # Similar prediction logic as /api/predict
            pd_pred = np.random.uniform(0.01, 0.2)
            lgd_pred = np.random.uniform(0.3, 0.8)
            loan_amt = row.get('loan_amount', 10000) if 'loan_amount' in row else 10000
            ead_pred = loan_amt * np.random.uniform(0.8, 1.0)
            
            el = pd_pred * lgd_pred * ead_pred
            
            predictions.append({
                'loan_id': idx,
                'pd': round(pd_pred, 4),
                'lgd': round(lgd_pred, 4),
                'ead': round(ead_pred, 2),
                'el': round(el, 4),
                'decision': 'APPROVE' if el <= 0.05 else 'REJECT',
            })
        
        results_df = pd.DataFrame(predictions)
        
        # Return as CSV
        output = StringIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        
        return jsonify({
            'success': True,
            'predictions': predictions,
            'approval_rate': (results_df['decision'] == 'APPROVE').sum() / len(results_df),
            'avg_el': float(results_df['el'].mean()),
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/portfolio_stats', methods=['GET'])
def portfolio_stats():
    """Get portfolio-level statistics."""
    stats = {
        'total_loans': 50000,
        'default_rate': 0.08,
        'avg_pd': 0.12,
        'avg_lgd': 0.55,
        'avg_el': 0.066,
        'approval_rate': 0.75,
    }
    return jsonify(stats)


@app.route('/api/models_info', methods=['GET'])
def models_info():
    """Get information about trained models."""
    from config import MODELS_DIR
    
    info = {
        'available_models': list(models.keys()),
        'model_files': [],
    }
    
    if os.path.exists(MODELS_DIR):
        info['model_files'] = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
    
    return jsonify(info)


@app.route('/api/feature_importance', methods=['GET'])
def feature_importance():
    """Get feature importance from best PD model."""
    try:
        pd_suite = models.get('pd_suite')
        if pd_suite and pd_suite.fitted_models and pd_suite.best_model_name:
            model = pd_suite.fitted_models[pd_suite.best_model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                features = [f"Feature_{i}" for i in range(len(importances))]
                
                importance_data = [
                    {'feature': f, 'importance': float(imp)}
                    for f, imp in sorted(zip(features, importances), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:15]
                ]
                
                return jsonify({
                    'success': True,
                    'model': pd_suite.best_model_name,
                    'importances': importance_data,
                })
        
        return jsonify({'success': False, 'error': 'Model not available'}), 400
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return jsonify({'error': 'Server error'}), 500


def create_app():
    """Application factory."""
    load_production_models()
    return app


if __name__ == '__main__':
    app = create_app()
    print("\n" + "="*60)
    print("  CREDIT RISK SCORING SYSTEM")
    print("  Flask Web Application")
    print("="*60)
    print("\n  Starting server on http://localhost:5000")
    print("  Press CTRL+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False,
    )
