from flask import render_template, request, jsonify
from app import app
from app.utils import load_models, predict_password_complexity

# Load models
rf_model, lstm_model, tokenizer, max_len = load_models()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_password():
    password = request.form.get('password')
    if not password:
        return jsonify({"error": "No password provided"}), 400
    
    rf_result, lstm_result = predict_password_complexity(password, rf_model, lstm_model, tokenizer, max_len)
    
    return jsonify({
        "password": password,
        "random_forest_prediction": rf_result,
        "lstm_prediction": lstm_result
    })
