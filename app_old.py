# ==============================================================================
# PROJECT: ADVANCED INDOOR MICROBIAL RISK ASSESSMENT SYSTEM (AIMRAS)
# FILE: app.py
# VERSION: 2.1 (Holistic & Synergistic Analysis Engine)
#
# PURPOSE:
# A Flask web application that provides a user interface for a saved AIMRAS
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
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, render_template

# --- INITIALIZATION ---
app = Flask(__name__)

# --- CONFIGURATION ---
BUNDLE_PATH = 'veritas_model_bundle.pkl'
# --- LOAD MODEL AND ARTIFACTS AT STARTUP ---
try:
    print("--> Loading AIMRAS model and artifacts...")
    if not os.path.exists(BUNDLE_PATH):
        raise FileNotFoundError(f"Model bundle not found at '{BUNDLE_PATH}'")
    
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle['model']
    scaler = bundle['scaler']
    feature_names = bundle['feature_names']
    target_labels = bundle['target_labels']
    print("✅ Model and artifacts loaded successfully.")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model from '{MODEL_DIR}'. Ensure the directory exists and contains the model files.")
    print(f"Error details: {e}")
    model = None

# --- PROFESSIONAL RECOMMENDATION & SYNERGY ENGINE ---
PROFESSIONAL_RISK_GUIDE = {
    "Risk_High_PM2.5": {
        "Hazard_Level": "High",
        "Interpretation": "Particulate matter (PM2.5) exceeds safety thresholds. This is a direct respiratory health hazard.",
        "Remedies": [
            "Verify HVAC system's HEPA filter integrity and check for filter bypass.",
            "Increase air exchange rate in the affected zone.",
            "Deploy localized air purification units if HVAC is insufficient."
        ]
    },
     "Risk_Moderate_PM2.5": {
        "Hazard_Level": "Medium",
        "Interpretation": "Particulate matter (PM2.5) is at a level that could affect sensitive groups. Prolonged exposure is not recommended.",
        "Remedies": [
            "Ensure HVAC filters are clean and functioning correctly.",
            "Consider increasing ventilation during periods of high occupancy."
        ]
    },
    "Risk_Poor_Ventilation": {
        "Hazard_Level": "Medium",
        "Interpretation": "High CO2 indicates inadequate fresh air exchange, leading to reduced cognitive performance and stuffiness.",
        "Remedies": [
            "Adjust HVAC settings to increase the percentage of outdoor air intake.",
            "Check for blocked air diffusers or return vents.",
            "Review zone occupancy against its designed ventilation capacity."
        ]
    },
    "Comfort_Humidity_High": {
        "Hazard_Level": "Low",
        "Interpretation": "Excess humidity reduces thermal comfort and, if persistent, significantly increases the risk of microbial contamination (mold, mildew).",
        "Remedies": [
            "Engage HVAC dehumidification cycle or increase its intensity.",
            "Check for and repair any water intrusion or leaks.",
            "Deploy supplemental dehumidifiers for chronically damp areas."
        ]
    },
    "Comfort_Thermal_Hot": {
        "Hazard_Level": "Low",
        "Interpretation": "The temperature is uncomfortably hot, affecting productivity and well-being.",
        "Remedies": [
            "Adjust thermostat setpoint to a lower temperature.",
            "Increase fan speed for better air circulation.",
            "Review solar gain through windows and consider blinds/film."
        ]
    },
    "Comfort_Thermal_Cold": {
        "Hazard_Level": "Low",
        "Interpretation": "The temperature is uncomfortably cold, affecting productivity and well-being.",
        "Remedies": [
            "Adjust thermostat setpoint to a higher temperature.",
            "Check for drafts from windows or doors and seal them."
        ]
    },
}

SYNERGISTIC_RISK_RULES = {
    "High_Microbial_Contamination_Risk": {
        "Required_Risks": ["Comfort_Humidity_High", "Risk_Poor_Ventilation"],
        "Hazard_Level": "Critical",
        "Interpretation": "A critical microbial hazard exists. The combination of high humidity and stagnant, poorly ventilated air creates an ideal breeding ground for mold, bacteria, and other pathogens.",
        "Remedies": [
            "Prioritize immediate ventilation and dehumidification improvements.",
            "Schedule an inspection for existing microbial contamination.",
            "Implement continuous humidity and CO2 monitoring."
        ]
    },
    "Severe_Respiratory_Hazard": {
        "Required_Risks": ["Risk_High_PM2.5", "Risk_Poor_Ventilation"],
        "Hazard_Level": "Critical",
        "Interpretation": "A severe respiratory hazard exists. High levels of particulate matter are trapped in a poorly ventilated space, leading to pollutant accumulation and increased exposure risk.",
        "Remedies": [
            "Consider reducing occupancy in the zone until resolved.",
            "Combine all recommendations for the primary risks (HEPA filtration and ventilation)."
        ]
    },
    "Oppressive_Atmosphere_Hazard": {
        "Required_Risks": ["Comfort_Humidity_High", "Comfort_Thermal_Hot"],
        "Hazard_Level": "High",
        "Interpretation": "A 'heat index' hazard exists. The combination of high heat and high humidity creates an oppressive atmosphere that can feel significantly hotter than the measured temperature and can lead to heat-related stress and illness.",
        "Remedies": [
            "Aggressively cool and dehumidify the space using all available HVAC resources.",
            "Encourage occupants to stay hydrated."
        ]
    }
}

# --- HELPER FUNCTIONS ---
def prepare_input_features(current_data, feature_names):
    """Creates a DataFrame with engineered features for the XGBoost model."""
    df = pd.DataFrame([current_data])
    df['timestamp'] = pd.to_datetime(pd.Timestamp.now())
    df = df.set_index('timestamp')

    history_df = pd.concat([df] * 24, ignore_index=True)
    history_df.index = pd.date_range(end=df.index[0], periods=24, freq='H')
    
    df_out = history_df.copy()
    df_out['hour'] = df_out.index.hour
    df_out['day_of_week'] = df_out.index.dayofweek
    df_out['is_weekend'] = df_out['day_of_week'].isin([5, 6]).astype(int)
    df_out['hour_sin'] = np.sin(2 * np.pi * df_out['hour'] / 24)
    df_out['hour_cos'] = np.cos(2 * np.pi * df_out['hour'] / 24)
    key_sensors = ['temperature', 'humidity', 'co2', 'pm2.5']
    for col in key_sensors:
        if col in df_out.columns:
            df_out[f'{col}_24h_mean'] = df_out[col].rolling('24H', min_periods=1).mean()
            df_out[f'{col}_1h_lag'] = df_out[col].shift(1)
            df_out[f'{col}_rate_of_change'] = df_out[col].diff()
    df_out['temp_humidity_interaction'] = df_out['temperature'] * df_out['humidity']
    df_out['dew_point'] = df_out['temperature'] - ((100 - df_out['humidity']) / 5)
    df_out['dew_point_spread'] = df_out['temperature'] - df_out['dew_point']
    df_out = df_out.ffill().bfill().fillna(0)
    
    final_features = df_out.tail(1)
    final_features = final_features.reindex(columns=feature_names, fill_value=0)
    
    return final_features


# --- FLASK ROUTES ---
def generate_veritas_advice(inputs):
    """
    PHASE 2: Actionable Recourse Logic
    Calculates quantified reductions for environmental safety.
    """
    advice = []
    
    # 1. CO2 Quantified Recourse
    co2_val = float(inputs.get('co2', 0))
    if co2_val > 1000:
        reduction = co2_val - 800  # Target safety level is 800 ppm
        advice.append(f"🌬️ Action: CO2 level high hai. Isse kam se kam {reduction:.0f} ppm kam karke ventilation sudharein.")

    # 2. Humidity Quantified Recourse
    hum_val = float(inputs.get('humidity', 0))
    if hum_val > 65:
        reduction = hum_val - 50  # Target safety level is 50%
        advice.append(f"💧 Action: Mold risk! Humidity ko {reduction:.0f}% niche laane ke liye dehumidifier ka use karein.")

    # 3. PM2.5 Quantified Recourse
    pm25_val = float(inputs.get('pm2.5', 0))
    if pm25_val > 25:
        reduction = pm25_val - 12  # Target safety level is 12 µg/m³
        advice.append(f"😷 Action: Air quality khrab hai. PM2.5 ko {reduction:.1f} µg/m³ kam karne ke liye air purifier chalaein.")

    return advice if advice else ["✅ Environment ideal conditions mein hai. Koi action required nahi."]
@app.route('/')
def home():
    """Renders the main page with the input form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the form."""
    current_advice = generate_veritas_advice(request.form)
    if model is None:
        return render_template('index.html', error="Model is not loaded. Please check the server logs.")

    try:
        form_data = {
            'temperature': float(request.form['temperature']),
            'humidity': float(request.form['humidity']),
            'co2': float(request.form['co2']),
            'pm2.5': float(request.form['pm2.5']),
            'pm10': float(request.form['pm10']),
            'tvoc': float(request.form['tvoc']),
            'co': float(request.form['co']),
            'occupancy_count': int(request.form['occupancy_count']),
            'motion_detected': 1,
            'light_intensity': 500,
        }

        input_df = prepare_input_features(form_data, feature_names)
        scaled_input = scaler.transform(input_df)
        predictions_array = model.predict(scaled_input)

        primary_risks = []
        detected_risk_labels = []
        for i, label in enumerate(target_labels):
            if predictions_array[0, i] == 1:
                detected_risk_labels.append(label)
                risk_info = PROFESSIONAL_RISK_GUIDE.get(label, {})
                primary_risks.append({
                    "name": label.replace('_', ' '),
                    "level": risk_info.get("Hazard_Level", "N/A"),
                    "interpretation": risk_info.get("Interpretation", "No details available."),
                    "remedies": risk_info.get("Remedies", [])
                })

        synergistic_risks = []
        for name, rule in SYNERGISTIC_RISK_RULES.items():
            if all(risk in detected_risk_labels for risk in rule['Required_Risks']):
                synergistic_risks.append({
                    "name": name.replace('_', ' '),
                    "level": rule.get("Hazard_Level", "N/A"),
                    "interpretation": rule.get("Interpretation", "No details available."),
                    "remedies": rule.get("Remedies", [])
                })
        
        # --- NEW: Holistic Assessment Logic ---
        holistic_assessment = None
        num_risks = len(detected_risk_labels)
        if num_risks >= 3:
            holistic_assessment = {
                "level": "Severe",
                "text": "The environment has multiple co-occurring issues, indicating a complex and unstable indoor climate that requires comprehensive and immediate intervention."
            }
        elif num_risks == 2:
             holistic_assessment = {
                "level": "Moderate",
                "text": "Multiple environmental parameters are outside of optimal ranges. Addressing these issues in tandem is recommended to prevent compounding problems."
            }
        elif num_risks == 1:
             holistic_assessment = {
                "level": "Low",
                "text": "A single environmental parameter is outside the optimal range. Targeted intervention is required."
            }


        results = {
            "primary_risks": primary_risks,
            "synergistic_risks": synergistic_risks,
            "holistic_assessment": holistic_assessment
        }

        return render_template('index.html', 
                               results=results, 
                               veritas_advice=current_advice)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return render_template('index.html', error=f"An error occurred: {e}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
