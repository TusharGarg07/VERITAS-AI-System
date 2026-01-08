# ==============================================================================
# PROJECT: VERITAS (Versatile Environmental Risk Intelligence & Transparency Analysis System)
# FILE: main.py
# ==============================================================================

import os
import numpy as np
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field, validator, EmailStr
from typing import List, Optional, Dict, Any
from config import PROFESSIONAL_RISK_GUIDE, SYNERGISTIC_RISK_RULES

# --- INITIALIZATION ---
app = FastAPI(
    title="VERITAS",
    description="Versatile Environmental Risk Intelligence & Transparency Analysis System",
    version="3.0.0"
)

# Setup templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# --- CONFIGURATION ---
BUNDLE_PATH = os.path.join(BASE_DIR, 'veritas_model_bundle.pkl')
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
ADMIN_EMAIL = "phenominal0525@gmail.com"

# --- GLOBAL VARIABLES ---
model = None
scaler = None
feature_names = None
target_labels = None

# --- PYDANTIC MODELS ---
class SensorData(BaseModel):
    temperature: float = Field(..., description="Temperature in Celsius", ge=-50, le=100)
    humidity: float = Field(..., description="Relative Humidity in %", ge=0, le=100)
    co2: float = Field(..., description="CO2 levels in ppm", ge=0)
    pm2_5: float = Field(..., alias="pm2.5", description="PM2.5 concentration in µg/m³", ge=0)
    pm10: float = Field(..., description="PM10 concentration in µg/m³", ge=0)
    tvoc: float = Field(..., description="Total Volatile Organic Compounds in ppb", ge=0)
    co: float = Field(..., description="Carbon Monoxide in ppm", ge=0)
    occupancy_count: int = Field(..., description="Number of occupants", ge=0)
    motion_detected: int = Field(1, description="Motion detected (0 or 1)", ge=0, le=1)
    light_intensity: float = Field(500, description="Light intensity in Lux", ge=0)
    email: Optional[str] = Field(None, description="User email for reports")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "temperature": 24.5,
                "humidity": 45.0,
                "co2": 600.0,
                "pm2.5": 10.0,
                "pm10": 15.0,
                "tvoc": 100.0,
                "co": 0.5,
                "occupancy_count": 2,
                "motion_detected": 1,
                "light_intensity": 500
            }
        }

class RiskDetail(BaseModel):
    name: str
    level: str
    interpretation: str
    remedies: List[str]

class HolisticAssessment(BaseModel):
    level: str
    text: str
    score: float

class AnalysisResponse(BaseModel):
    primary_risks: List[RiskDetail]
    synergistic_risks: List[RiskDetail]
    holistic_assessment: Optional[HolisticAssessment]
    veritas_advice: List[str]
    health_score: float

# --- LIFECYCLE EVENTS ---
@app.on_event("startup")
async def startup_event():
    global model, scaler, feature_names, target_labels
    print("--> Loading VERITAS model and artifacts...")
    if not os.path.exists(BUNDLE_PATH):
        print(f"❌ WARNING: Model bundle not found at '{BUNDLE_PATH}'. Application will run but predictions will fail.")
        return
    
    try:
        bundle = joblib.load(BUNDLE_PATH)
        # Basic integrity check - support both naming conventions
        model = bundle.get('model')
        scaler = bundle.get('scaler')
        
        # Handle variations in key names
        feature_names = bundle.get('feature_names') or bundle.get('features')
        target_labels = bundle.get('target_labels') or bundle.get('targets')

        # FIX: Patch for sklearn version incompatibility (ClassifierChain unpickling issue)
        # sklearn 1.8.0+ expects 'estimator' attribute, but 1.6.1 pickle has 'base_estimator'
        if model is not None:
            if hasattr(model, 'base_estimator') and not hasattr(model, 'estimator'):
                print("⚠️ Patching model: Setting model.estimator = None to fix sklearn compatibility.")
                model.estimator = None

        if model is None or scaler is None or feature_names is None or target_labels is None:
             # Fallback for feature names if absolutely necessary (based on app.py history)
             if feature_names is None:
                 print("⚠️ Feature names not found in bundle. Using default fallback.")
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
             
             if target_labels is None:
                 raise ValueError(f"Model bundle missing required keys. Found: {list(bundle.keys())}")

        print(f"✅ VERITAS Model and artifacts loaded successfully. ({len(feature_names)} features)")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load model. Details: {e}")

# --- HELPER FUNCTIONS ---
def prepare_input_features(data: SensorData, feature_names: List[str]):
    """Creates a DataFrame with engineered features for the XGBoost model."""
    # Convert Pydantic model to dict, handling alias
    input_data = data.dict(by_alias=True)
    
    # Remove email if present as it's not a model feature
    if 'email' in input_data:
        del input_data['email']
    
    # Ensure keys match what pandas expects (renaming pm2_5 back to pm2.5 is handled by by_alias=True if we use the alias)
    # However, we need to be careful with the keys.
    # The original code expected keys like 'pm2.5'.
    
    df = pd.DataFrame([input_data])
    df['timestamp'] = pd.to_datetime(pd.Timestamp.now())
    df = df.set_index('timestamp')

    # Simulate history (since we only have one data point, we assume steady state for 24h)
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
    
    # Advanced metrics (from v2.1)
    df_out['co2_pm_ratio'] = df_out['co2'] / (df_out['pm2.5'] + 0.1)
    df_out['heat_index'] = df_out['temperature'] + (0.55 * (1 - (df_out['humidity']/100)) * (df_out['temperature'] - 14.5))
    
    df_out['dew_point'] = df_out['temperature'] - ((100 - df_out['humidity']) / 5)
    df_out['dew_point_spread'] = df_out['temperature'] - df_out['dew_point']
    df_out = df_out.ffill().bfill().fillna(0)
    
    final_features = df_out.tail(1)
    # Reindex to ensure all features expected by the model are present
    final_features = final_features.reindex(columns=feature_names, fill_value=0)
    
    # Explicit Type Casting: Ensure all features are float
    final_features = final_features.astype(float)
    
    return final_features

def generate_veritas_advice(inputs: SensorData) -> List[str]:
    """
    Calculates quantified reductions for environmental safety.
    """
    advice = []
    
    # 1. CO2 Quantified Recourse
    if inputs.co2 > 1000:
        reduction = inputs.co2 - 800
        advice.append(f"🌬️ Action: CO2 level high hai. Isse kam se kam {reduction:.0f} ppm kam karke ventilation sudharein.")

    # 2. Humidity Quantified Recourse
    if inputs.humidity > 65:
        reduction = inputs.humidity - 50
        advice.append(f"💧 Action: Mold risk! Humidity ko {reduction:.0f}% niche laane ke liye dehumidifier ka use karein.")

    # 3. PM2.5 Quantified Recourse
    if inputs.pm2_5 > 25:
        reduction = inputs.pm2_5 - 12
        advice.append(f"😷 Action: Air quality khrab hai. PM2.5 ko {reduction:.1f} µg/m³ kam karne ke liye air purifier chalaein.")

    return advice if advice else ["✅ Environment ideal conditions mein hai. Koi action required nahi."]

def calculate_health_score(num_risks: int, synergistic_count: int) -> float:
    base_score = 100 - (num_risks * 25)
    penalty = synergistic_count * 30
    final_score = base_score - penalty
    return max(0.0, min(100.0, float(final_score)))

def send_email_task(to_email: str, subject: str, html_content: str):
    """Sends an email asynchronously."""
    if not EMAIL_USER or not EMAIL_PASS:
        print("⚠️ Email credentials not found. Skipping email.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = to_email
        msg['Subject'] = subject
        msg.attach(MIMEText(html_content, 'html'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print(f"✅ Email sent to {to_email}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_environment(data: SensorData, background_tasks: BackgroundTasks):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Debug: Print received data
        print(f"📥 Received Input: {data}")

        # Prepare and predict
        input_df = prepare_input_features(data, feature_names)
        
        # Debug: Print DataFrame shape and columns
        print(f"📊 Processed DataFrame Shape: {input_df.shape}")
        print(f"📋 Columns: {input_df.columns.tolist()}")

        if scaler:
            try:
                scaled_input = scaler.transform(input_df)
            except Exception as se:
                print(f"❌ Scaler Error: {se}")
                # If scaler fails, maybe input_df doesn't match scaler expectation
                if hasattr(scaler, 'n_features_in_'):
                    print(f"Scaler expects {scaler.n_features_in_} features, but got {input_df.shape[1]}")
                raise se
        else:
            scaled_input = input_df

        predictions_array = model.predict(scaled_input)


        # Process Primary Risks
        primary_risks = []
        detected_risk_labels = []
        for i, label in enumerate(target_labels):
            if predictions_array[0, i] == 1:
                detected_risk_labels.append(label)
                risk_info = PROFESSIONAL_RISK_GUIDE.get(label, {})
                primary_risks.append(RiskDetail(
                    name=label.replace('_', ' '),
                    level=risk_info.get("Hazard_Level", "N/A"),
                    interpretation=risk_info.get("Interpretation", "No details available."),
                    remedies=risk_info.get("Remedies", [])
                ))

        # Process Synergistic Risks
        synergistic_risks = []
        for name, rule in SYNERGISTIC_RISK_RULES.items():
            if all(risk in detected_risk_labels for risk in rule['Required_Risks']):
                synergistic_risks.append(RiskDetail(
                    name=name.replace('_', ' '),
                    level=rule.get("Hazard_Level", "N/A"),
                    interpretation=rule.get("Interpretation", "No details available."),
                    remedies=rule.get("Remedies", [])
                ))

        # Holistic Assessment
        num_risks = len(detected_risk_labels)
        synergistic_count = len(synergistic_risks)
        health_score = calculate_health_score(num_risks, synergistic_count)
        
        holistic_assessment = None
        if num_risks >= 3:
            holistic_assessment = HolisticAssessment(
                level="Severe",
                text="The environment has multiple co-occurring issues, indicating a complex and unstable indoor climate.",
                score=health_score
            )
        elif num_risks == 2:
             holistic_assessment = HolisticAssessment(
                level="Moderate",
                text="Multiple environmental parameters are outside of optimal ranges.",
                score=health_score
            )
        elif num_risks == 1:
             holistic_assessment = HolisticAssessment(
                level="Low",
                text="A single environmental parameter is outside the optimal range.",
                score=health_score
            )
        else:
             holistic_assessment = HolisticAssessment(
                level="Optimal",
                text="Environmental conditions are within optimal ranges.",
                score=health_score
            )

        # Generate Advice
        advice = generate_veritas_advice(data)

        # --- EMAIL ALERT SYSTEM ---
        # Trigger condition: Health Score < 60 OR Risk == 'Severe' (mapped from 'High')
        is_high_risk = health_score < 60 or (holistic_assessment and holistic_assessment.level == "Severe")
        
        if is_high_risk and data.email:
            # 1. Send User Report
            user_html = templates.get_template("user_report.html").render(
                health_score=round(health_score),
                risk_class="high-risk" if health_score < 50 else "moderate-risk",
                holistic_assessment=holistic_assessment.text if holistic_assessment else "N/A",
                advice_list=advice,
                primary_risks=primary_risks,
                year=datetime.now().year
            )
            background_tasks.add_task(
                send_email_task, 
                data.email, 
                "VERITAS Health Report: Critical Action Required", 
                user_html
            )

            # 2. Send Admin Alert
            admin_html = templates.get_template("admin_alert.html").render(
                health_score=round(health_score),
                user_email=data.email,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                primary_risks=primary_risks,
                data=data.dict()
            )
            background_tasks.add_task(
                send_email_task, 
                ADMIN_EMAIL, 
                f"🚨 VERITAS HIGH RISK ALERT - {data.email}", 
                admin_html
            )
            print(f"⚠️ High risk detected ({health_score}). Email tasks queued.")

        return AnalysisResponse(
            primary_risks=primary_risks,
            synergistic_risks=synergistic_risks,
            holistic_assessment=holistic_assessment,
            veritas_advice=advice,
            health_score=health_score
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
