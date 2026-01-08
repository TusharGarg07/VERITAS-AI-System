# VERITAS (Versatile Environmental Risk Intelligence & Transparency Analysis System)

## Overview
VERITAS is a state-of-the-art environmental risk assessment system designed to monitor, analyze, and provide actionable intelligence on indoor air quality. It leverages advanced machine learning models and a holistic synergy engine to detect complex, co-occurring environmental hazards.

This repository contains the modernized architecture of the system, migrated from a legacy Flask application to a high-performance FastAPI backend with a futuristic, responsive frontend.

## Key Features
*   **Real-time Risk Intelligence**: Instant analysis of sensor data (Temperature, Humidity, CO2, PM2.5, etc.).
*   **Synergistic Risk Detection**: Identifies critical combinations of factors (e.g., High Humidity + Poor Ventilation).
*   **Holistic Health Score**: A dynamic 0-100 score reflecting the overall safety of the environment.
*   **AI Agent Insights**: Provides actionable, localized advice (Hinglish support included) for remediation.
*   **Modern Tech Stack**: Built with FastAPI, Pydantic, Tailwind CSS, and Scikit-Learn/XGBoost.

## System Architecture

### Backend (`main.py`)
*   **Framework**: FastAPI (Asynchronous, High Performance).
*   **Validation**: Pydantic models ensure strict data integrity for all sensor inputs.
*   **ML Integration**: Secure loading of `veritas_model_bundle.pkl` with fallback mechanisms.
*   **Logic**:
    *   `prepare_input_features`: Handles complex feature engineering (lag features, interaction terms).
    *   `generate_veritas_advice`: Rule-based engine for actionable recommendations.

### Configuration (`config.py`)
*   Modularized configuration separating business logic from application code.
*   Contains `PROFESSIONAL_RISK_GUIDE` and `SYNERGISTIC_RISK_RULES`.

### Frontend (`templates/index.html`)
*   **Design**: Glassmorphism aesthetic with Neon accents using Tailwind CSS.
*   **Interactivity**: Vanilla JavaScript for lightweight, fast interactions with the API.
*   **Visualization**: SVG-based Health Score gauge and dynamic risk cards.

## Setup Instructions

### Prerequisites
*   Python 3.9+
*   pip

### Installation
1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd veritas-system
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify Model File**:
    Ensure `veritas_model_bundle.pkl` is present in the root directory.

### Running the Application
Start the server using Uvicorn:
```bash
uvicorn main:app --reload
```

The application will be available at:
*   **Dashboard**: `http://localhost:8000`
*   **API Docs**: `http://localhost:8000/docs`

## API Endpoints

### `POST /api/analyze`
Analyzes environmental data and returns a comprehensive risk assessment.

**Request Body (JSON):**
```json
{
  "temperature": 25.0,
  "humidity": 50.0,
  "co2": 800,
  "pm2.5": 10.0,
  "pm10": 15.0,
  "tvoc": 50,
  "co": 0.5,
  "occupancy_count": 2,
  "motion_detected": 1,
  "light_intensity": 500
}
```

**Response:**
*   `primary_risks`: List of detected single-source risks.
*   `synergistic_risks`: List of combined hazard scenarios.
*   `holistic_assessment`: Overall status text and severity level.
*   `health_score`: Numeric score (0-100).
*   `veritas_advice`: List of recommended actions.

## Deployment Guidelines
*   **Production**: Use a process manager like Gunicorn with Uvicorn workers.
    ```bash
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
    ```
*   **Environment**: Ensure the server has read access to `veritas_model_bundle.pkl`.
*   **Security**: Place behind a reverse proxy (Nginx/Apache) with SSL termination.

## Migration Notes 
*   Renamed project identity to VERITAS.
*   Refactored `app_old.py` to `main.py` and `config.py`.
*   Updated `index.html` to use modern Tailwind CSS classes.
*   Preserved all feature engineering logic for model compatibility.
