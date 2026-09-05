# api.py
# Credit Card Fraud Detection API
# Author: Shreyam
# Description: Flask API for real-time fraud prediction

import os
import sys
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, send_file
from flask_cors import CORS
import joblib
import logging
from datetime import datetime
from functools import wraps
import traceback

# --------------------------------------------------
# 0. CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
MODEL_PATH = "fraud_model_random_forest.pkl"
SCALER_PATH = "scaler.pkl"
ALLOWED_ORIGINS = ["*"]  # Configure as needed

# --------------------------------------------------
# 1. MODEL LOADING WITH ERROR HANDLING
# --------------------------------------------------

def load_model_and_scaler():
    """Load the trained model and scaler with error handling"""
    try:
        # Try to load the best model (Random Forest first)
        model_files = [
            "fraud_model_random_forest.pkl",
            "fraud_model_xgboost.pkl", 
            "fraud_model_logistic_regression.pkl",
            "fraud_model.pkl"
        ]
        
        model = None
        for model_file in model_files:
            if os.path.exists(model_file):
                logger.info(f"Loading model from {model_file}...")
                model = joblib.load(model_file)
                logger.info(f"✅ Model loaded successfully from {model_file}")
                break
        
        if model is None:
            logger.error("❌ No model file found!")
            raise FileNotFoundError("No trained model found. Please run fraud_detection.py first.")
        
        # Load scaler
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            logger.info("✅ Scaler loaded successfully")
        else:
            logger.error("❌ Scaler file not found!")
            raise FileNotFoundError("Scaler file not found. Please run fraud_detection.py first.")
        
        return model, scaler
    
    except Exception as e:
        logger.error(f"Error loading model/scaler: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

# Load models
model, scaler = load_model_and_scaler()

# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def validate_features(data):
    """Validate input features"""
    if not isinstance(data, list):
        return False, "Features must be a list"
    
    if len(data) != 31:
        return False, f"Expected 31 features, got {len(data)}"
    
    # Check for invalid values
    for i, val in enumerate(data):
        if not isinstance(val, (int, float)):
            return False, f"Feature {i} must be numeric"
        if np.isnan(val) or np.isinf(val):
            return False, f"Feature {i} contains invalid value (NaN or Inf)"
    
    return True, "Valid"

def format_prediction_response(prediction, probability, transaction_id=None):
    """Format the prediction response"""
    return {
        "status": "success",
        "prediction": {
            "transaction_id": transaction_id,
            "fraud": int(prediction),
            "fraud_status": "Fraud" if prediction == 1 else "Legitimate",
            "probability": float(probability),
            "risk_level": "High" if probability > 0.8 else "Medium" if probability > 0.5 else "Low"
        },
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "type": type(model).__name__,
            "version": "1.0"
        }
    }

# --------------------------------------------------
# 3. AUTHENTICATION (Optional)
# --------------------------------------------------

API_KEY = os.environ.get('API_KEY', 'your-secret-api-key-here')

def require_api_key(f):
    """Decorator to require API key for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if api_key and api_key == API_KEY:
            return f(*args, **kwargs)
        else:
            logger.warning(f"Unauthorized access attempt from {request.remote_addr}")
            return jsonify({
                "status": "error",
                "message": "Invalid or missing API key"
            }), 401
    return decorated_function

# --------------------------------------------------
# 4. ROUTES
# --------------------------------------------------

@app.route('/', methods=['GET'])
def home():
    """Home page with API documentation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Credit Card Fraud Detection API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            h1 { color: #333; }
            .endpoint { background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 10px 0; }
            .method { color: #0066cc; font-weight: bold; }
            code { background: #e8e8e8; padding: 2px 6px; border-radius: 3px; }
            pre { background: #f0f0f0; padding: 10px; border-radius: 5px; overflow-x: auto; }
        </style>
    </head>
    <body>
        <h1>🛡️ Credit Card Fraud Detection API</h1>
        <p>Real-time fraud detection using Machine Learning</p>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict</h3>
            <p>Predict fraud for a single transaction</p>
            <p><strong>Request Body:</strong></p>
            <pre>{
    "features": [list of 31 numerical features]
}</pre>
            <p><strong>Response:</strong></p>
            <pre>{
    "status": "success",
    "prediction": {
        "fraud": 0 or 1,
        "fraud_status": "Legitimate" or "Fraud",
        "probability": 0.0 to 1.0,
        "risk_level": "Low/Medium/High"
    },
    "timestamp": "2024-01-01T12:00:00"
}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">POST</span> /predict/batch</h3>
            <p>Predict fraud for multiple transactions</p>
            <p><strong>Request Body:</strong></p>
            <pre>{
    "transactions": [
        {"id": "txn_001", "features": [list of 31 features]},
        {"id": "txn_002", "features": [list of 31 features]}
    ]
}</pre>
        </div>
        
        <div class="endpoint">
            <h3><span class="method">GET</span> /health</h3>
            <p>Check API health status</p>
        </div>
        
        <p><strong>Note:</strong> For production use, include API key in headers: <code>X-API-Key: your-api-key</code></p>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_type": type(model).__name__
    })

@app.route('/predict', methods=['POST'])
@require_api_key  # Comment out this line to disable API key requirement
def predict():
    """Predict fraud for a single transaction"""
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "status": "error",
                "message": "No JSON data provided"
            }), 400
        
        # Check for features
        if "features" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'features' field"
            }), 400
        
        features = data["features"]
        transaction_id = data.get("transaction_id", f"txn_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        
        # Validate features
        is_valid, message = validate_features(features)
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": message
            }), 400
        
        # Convert to numpy array
        features_np = np.array(features).reshape(1, -1)
        
        # Scale Time (index 0) and Amount (index 1)
        features_np[:, :2] = scaler.transform(features_np[:, :2])
        
        # Make prediction
        prediction = model.predict(features_np)[0]
        probability = model.predict_proba(features_np)[0][1]
        
        # Format response
        response = format_prediction_response(prediction, probability, transaction_id)
        
        # Log prediction
        logger.info(f"Prediction made for transaction {transaction_id}: {prediction} (prob: {probability:.4f})")
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
@require_api_key  # Comment out this line to disable API key requirement
def predict_batch():
    """Predict fraud for multiple transactions"""
    try:
        data = request.get_json()
        
        if not data or "transactions" not in data:
            return jsonify({
                "status": "error",
                "message": "Missing 'transactions' field"
            }), 400
        
        transactions = data["transactions"]
        
        if not isinstance(transactions, list):
            return jsonify({
                "status": "error",
                "message": "'transactions' must be a list"
            }), 400
        
        if len(transactions) > 100:
            return jsonify({
                "status": "error",
                "message": "Batch size exceeds limit (max 100)"
            }), 400
        
        results = []
        errors = []
        
        for i, transaction in enumerate(transactions):
            try:
                features = transaction.get("features")
                transaction_id = transaction.get("id", f"txn_batch_{i+1}")
                
                if not features:
                    errors.append({
                        "index": i,
                        "id": transaction_id,
                        "message": "Missing features"
                    })
                    continue
                
                # Validate features
                is_valid, message = validate_features(features)
                if not is_valid:
                    errors.append({
                        "index": i,
                        "id": transaction_id,
                        "message": message
                    })
                    continue
                
                # Convert to numpy array
                features_np = np.array(features).reshape(1, -1)
                features_np[:, :2] = scaler.transform(features_np[:, :2])
                
                # Make prediction
                prediction = model.predict(features_np)[0]
                probability = model.predict_proba(features_np)[0][1]
                
                results.append({
                    "transaction_id": transaction_id,
                    "fraud": int(prediction),
                    "fraud_status": "Fraud" if prediction == 1 else "Legitimate",
                    "probability": float(probability),
                    "risk_level": "High" if probability > 0.8 else "Medium" if probability > 0.5 else "Low"
                })
                
            except Exception as e:
                errors.append({
                    "index": i,
                    "id": transaction.get("id", f"txn_batch_{i+1}"),
                    "message": str(e)
                })
        
        return jsonify({
            "status": "success",
            "results": results,
            "errors": errors if errors else None,
            "summary": {
                "total": len(transactions),
                "successful": len(results),
                "failed": len(errors),
                "timestamp": datetime.now().isoformat()
            }
        })
    
    except Exception as e:
        logger.error(f"Error in /predict/batch: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "details": str(e)
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        "model_type": type(model).__name__,
        "features_count": 31,
        "scaler_type": type(scaler).__name__,
        "model_parameters": str(model.get_params()) if hasattr(model, 'get_params') else "N/A"
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "status": "error",
        "message": "Internal server error"
    }), 500

# --------------------------------------------------
# 5. MAIN EXECUTION
# --------------------------------------------------

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Fraud Detection API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-api-key', action='store_true', help='Disable API key requirement')
    
    args = parser.parse_args()
    
    # Disable API key if requested
    if args.no_api_key:
        # Replace the decorator with a no-op version
        def require_api_key(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                return f(*args, **kwargs)
            return decorated_function
        
        # Reapply the decorator
        # Note: In production, you'd want to handle this differently
        logger.warning("⚠️ API key authentication is DISABLED")
    
    # Print startup info
    print("\n" + "="*60)
    print("🛡️ CREDIT CARD FRAUD DETECTION API")
    print("="*60)
    print(f"📍 Host: {args.host}")
    print(f"📍 Port: {args.port}")
    print(f"🔧 Debug: {args.debug}")
    print(f"🔑 API Key Required: {not args.no_api_key}")
    print(f"🤖 Model: {type(model).__name__}")
    print("="*60)
    print("\n📚 API Documentation: http://localhost:5000/")
    print("📊 Health Check: http://localhost:5000/health")
    print("="*60 + "\n")
    
    # Run the app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
