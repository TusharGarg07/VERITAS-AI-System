import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import BackgroundTasks
import numpy as np

# Add project root to sys.path to allow imports from main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app, send_email_task, SensorData

client = TestClient(app)

# Mock environment variables
os.environ["EMAIL_USER"] = "test@example.com"
os.environ["EMAIL_PASS"] = "password"

def test_send_email_task_success():
    """Test if send_email_task calls smtplib correctly."""
    # Patch the global variables in main module
    with patch("main.EMAIL_USER", "test@example.com"), \
         patch("main.EMAIL_PASS", "password"), \
         patch("smtplib.SMTP") as mock_smtp:
        
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__.return_value = mock_server
        
        send_email_task("user@example.com", "Subject", "<h1>Content</h1>")
        
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_with("test@example.com", "password")
        mock_server.send_message.assert_called_once()

def test_send_email_task_no_credentials():
    """Test if send_email_task handles missing credentials gracefully."""
    with patch("main.EMAIL_USER", None):
        with patch("smtplib.SMTP") as mock_smtp:
            send_email_task("user@example.com", "Subject", "Content")
            mock_smtp.assert_not_called()

def test_api_analyze_triggers_email_high_risk():
    """Test if the analyze endpoint triggers email tasks for high risk inputs."""
    
    high_risk_data = {
        "temperature": 35.0,
        "humidity": 80.0,
        "co2": 2000.0,
        "pm2.5": 100.0,
        "pm10": 150.0,
        "tvoc": 500.0,
        "co": 10.0,
        "occupancy_count": 5,
        "motion_detected": 1,
        "light_intensity": 500,
        "email": "victim@example.com"
    }

    # Patch EMAIL_USER so logic proceeds to add task
    with patch("main.EMAIL_USER", "test@example.com"), \
         patch("main.BackgroundTasks.add_task") as mock_add_task, \
         patch("main.model") as mock_model, \
         patch("main.target_labels", ["Risk_A", "Risk_B", "Risk_C", "Risk_D"]), \
         patch("main.scaler") as mock_scaler:
            
        mock_model.predict.return_value = np.array([[1, 1, 1, 1]]) # Simulate multiple risks
        
        # We need to mock scaler.transform to return something valid
        mock_scaler.transform.return_value = np.array([[1]*32]) # Dummy scaled data
            
        response = client.post("/api/analyze", json=high_risk_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["health_score"] < 60
        
        assert mock_add_task.call_count == 2

def test_api_analyze_no_email_low_risk():
    """Test if the analyze endpoint does NOT trigger email for low risk or missing email."""
    
    low_risk_data = {
        "temperature": 22.0,
        "humidity": 45.0,
        "co2": 400.0,
        "pm2.5": 5.0,
        "pm10": 10.0,
        "tvoc": 50.0,
        "co": 0.0,
        "occupancy_count": 1,
        "motion_detected": 0,
        "light_intensity": 500,
        "email": "safe@example.com"
    }

    with patch("main.EMAIL_USER", "test@example.com"), \
         patch("main.BackgroundTasks.add_task") as mock_add_task, \
         patch("main.model") as mock_model, \
         patch("main.target_labels", ["Risk_A", "Risk_B", "Risk_C", "Risk_D"]), \
         patch("main.scaler") as mock_scaler:
            
        mock_model.predict.return_value = np.array([[0, 0, 0, 0]])
        mock_scaler.transform.return_value = np.array([[0]*32])

        response = client.post("/api/analyze", json=low_risk_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["health_score"] > 80
        
        mock_add_task.assert_not_called()
