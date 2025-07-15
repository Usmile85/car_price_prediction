# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 09:20:54 2023
@author: nilay
Updated July 2025 by ChatGPT
"""

from flask import Flask, render_template, request
import numpy as np
import pickle
import os
from werkzeug.middleware.proxy_fix import ProxyFix

# Initialize Flask app
app = Flask(__name__)

# Fix proxy-related 400 errors on mobile/proxy networks
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)

# Home route
@app.route('/')
def index_page():
    return render_template('index.html')

# Predict route
@app.route('/predict', methods=['POST', 'GET'])
def predict_price():
    try:
        # Get form inputs safely
        kms = float(request.form.get('km', 0))
        age = float(request.form.get('age', 0))
        oprice = float(request.form.get('op', 0))
        fuel = request.form.get('fuel_type', '').strip()
        transmission = request.form.get('transmission', '').strip()

        # Encode fuel type
        if fuel == 'Petrol':
            fuel_encoded = [0.0, 1.0]
        elif fuel == 'Diesel':
            fuel_encoded = [1.0, 0.0]
        else:
            fuel_encoded = [0.0, 0.0]

        # Encode transmission
        transmission_encoded = 0.0 if transmission == 'Automatic' else 1.0

        # Prepare input data
        input_array = np.array([kms, oprice, age, *fuel_encoded, transmission_encoded]).reshape(1, -1)

        # Load model
        model_path = os.path.join('models', 'lr_model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Predict and return
        result = np.round(model.predict(input_array))[0]
        msg = f"Predicted car price is ₹{int(result):,}"
        return render_template('index.html', prediction_value=msg)

    except Exception as e:
        error_msg = f"⚠️ Error: {str(e)}"
        return render_template('index.html', prediction_value=error_msg)

# Run locally or via gunicorn in production
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
