# ==============================================================================
# PROJECT: ADVANCED INDOOR MICROBIAL RISK ASSESSMENT SYSTEM (VERITAS)
# FILE: app.py
# VERSION: 2.1 (Holistic & Synergistic Analysis Engine)
#
# PURPOSE:
# A Flask web application that provides a user interface for a saved VERITAS
# model. This version includes a more advanced analysis engine that evaluates
# all co-occurring risks to provide a holistic assessment.
#
# KEY CHANGES:
# - Added a dynamic "Holistic Assessment" that evaluates the overall state
#   based on the number and severity of detected risks.
# - Expanded the library of synergistic risks to identify more combinations.
# - Updated the UI to present the multi-layered analysis clearly.
# ==============================================================================

import os
import numpy as np #type: ignore
import pandas as pd #type: ignore
import joblib #type: ignore
from flask import Flask, request, render_template #type: ignore

app = Flask(__name__)

# --- CONFIGURATION ---
BUNDLE_PATH = 'veritas_model_bundle.pkl'

# --- LOAD MODEL AND ARTIFACTS ---
try:
    print("--> Loading VERITAS model and artifacts...")
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle.get('model')
    scaler = bundle.get('scaler')
    target_labels = bundle.get('target_labels')
    
    # Model DNA: 32 Features matched with .pkl
    feature_names = [
        'hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos',
        'temperature', 'humidity', 'co2', 'pm2.5', 'pm10', 'tvoc', 'co',
        'occupancy_count', 'motion_detected', 'light_intensity',
        'co2_pm_ratio', 'heat_index', 'dew_point', 'dew_point_spread',
        'temp_humidity_interaction',
        'temperature_24h_mean', 'temperature_1h_lag', 'temperature_rate_of_change',
        'humidity_24h_mean', 'humidity_1h_lag', 'humidity_rate_of_change',
        'co2_24h_mean', 'co2_1h_lag', 'co2_rate_of_change',
        'pm2.5_24h_mean', 'pm2.5_1h_lag', 'pm2.5_rate_of_change'
    ]
    print(f"✅ Model loaded. Expecting {len(feature_names)} features.")
except Exception as e:
    print(f"❌ FATAL ERROR: {e}")
    model = None

# --- PROFESSIONAL RULES & SYNERGY ENGINE ---
PROFESSIONAL_RISK_GUIDE = {
    "Risk_High_PM2.5": {
        "Hazard_Level": "High",
        "Interpretation": "Particulate matter (PM2.5) exceeds safety thresholds.",
        "Remedies": ["Check HEPA filters", "Increase air exchange", "Use air purifiers"]
    },
    "Risk_Poor_Ventilation": {
        "Hazard_Level": "Medium",
        "Interpretation": "High CO2 indicates inadequate fresh air exchange.",
        "Remedies": ["Increase outdoor air intake", "Check air diffusers"]
    },
    "Comfort_Humidity_High": {
        "Hazard_Level": "Low",
        "Interpretation": "Excess humidity increases mold risk.",
        "Remedies": ["Engage dehumidification", "Repair leaks"]
    }
}

SYNERGISTIC_RISK_RULES = {
    "High_Microbial_Contamination_Risk": {
        "Required_Risks": ["Comfort_Humidity_High", "Risk_Poor_Ventilation"],
        "Hazard_Level": "Critical",
        "Interpretation": "Stagnant air + High humidity = Pathogen breeding ground.",
        "Remedies": ["Prioritize immediate ventilation & dehumidification."]
    }
}

# --- FEATURE ENGINEERING ENGINE ---
def prepare_input_features(current_data, feature_names):
    df = pd.DataFrame([current_data])
    
    # Constants for single-point prediction
    df['hour'], df['day_of_week'], df['is_weekend'] = 12, 0, 0
    df['hour_sin'], df['hour_cos'] = np.sin(2 * np.pi * 12 / 24), np.cos(2 * np.pi * 12 / 24)
    df['motion_detected'], df['light_intensity'] = 1, 500
    
    # Calculated Metrics
    t, h, c, p25 = df['temperature'][0], df['humidity'][0], df['co2'][0], df['pm2.5'][0]
    df['co2_pm_ratio'] = c / (p25 + 0.1)
    df['heat_index'] = t + (0.55 * (1 - (h/100)) * (t - 14.5))
    df['dew_point'] = t - ((100 - h) / 5)
    df['dew_point_spread'] = t - df['dew_point']
    df['temp_humidity_interaction'] = t * h
    
    # Lag simulations
    for col in ['temperature', 'humidity', 'co2', 'pm2.5']:
        df[f'{col}_24h_mean'], df[f'{col}_1h_lag'], df[f'{col}_rate_of_change'] = df[col], df[col], 0
        
    return df[feature_names]

# --- ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None: return render_template('index.html', error="Model not loaded.")
    
    try:
        # 1. Inputs
        raw_data = {
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'co2': float(request.form['co2']),
            'pm2.5': float(request.form['pm2.5']),
            'pm10': float(request.form['pm10']),
            'tvoc': float(request.form['tvoc']),
            'co': float(request.form['co']),
            'occupancy_count': int(request.form['occupancy_count'])
        }

        # 2. Match 32 features
        input_df = prepare_input_features(raw_data, feature_names)
        scaled_input = scaler.transform(input_df)
        predictions = model.predict(scaled_input)

        # 3. Apply Rules
        primary_risks, detected_labels = [], []
        for i, label in enumerate(target_labels):
            if predictions[0, i] == 1:
                detected_labels.append(label)
                info = PROFESSIONAL_RISK_GUIDE.get(label, {"Hazard_Level": "Detected", "Interpretation": "Pollutant detected.", "Remedies": []})
                primary_risks.append({"name": label.replace('_', ' '), "level": info["Hazard_Level"], "interpretation": info["Interpretation"], "remedies": info["Remedies"]})

        # 4. Synergy & Holistic Analysis
        synergistic = []
        for name, rule in SYNERGISTIC_RISK_RULES.items():
            if all(risk in detected_labels for risk in rule['Required_Risks']):
                synergistic.append({"name": name.replace('_', ' '), "level": rule["Hazard_Level"], "interpretation": rule["Interpretation"], "remedies": rule["Remedies"]})

        holistic = {"level": "Low", "text": "Stable Environment."}
        if len(detected_labels) >= 3: holistic = {"level": "Severe", "text": "Multiple co-occurring issues detected."}

        # 5. Quantified Advice
        advice = []
        if raw_data['co2'] > 1000: advice.append(f"🌬️ Reduce CO2 by {raw_data['co2'] - 800:.0f} ppm.")
        if raw_data['humidity'] > 65: advice.append(f"💧 Reduce Humidity by {raw_data['humidity'] - 50:.0f}%.")

        return render_template('index.html', results={"primary_risks": primary_risks, "synergistic_risks": synergistic, "holistic_assessment": holistic}, veritas_advice=advice)

    
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    # Ek hi line mein host, port aur debug config
    app.run(host='0.0.0.0', port=5000, debug=True)
