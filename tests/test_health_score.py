import pytest
from main import calculate_health_score, SensorData

# Helper to create a default SensorData object
def create_sensor_data(**kwargs):
    defaults = {
        "temperature": 22, "humidity": 50, "co2": 400, "pm2.5": 10,
        "pm10": 15, "tvoc": 50, "co": 0.5, "occupancy_count": 1,
        "motion_detected": 0, "light_intensity": 300, "email": "test@example.com"
    }
    defaults.update(kwargs)
    return SensorData(**defaults)

def test_perfect_conditions():
    """Test score under ideal conditions."""
    data = create_sensor_data()
    score = calculate_health_score(data, num_risks=0, synergistic_count=0)
    assert score == 100.0

def test_minor_risks_score_clamp():
    """Test that score is clamped to 99 if risks are present but no deductions are triggered."""
    data = create_sensor_data()
    score = calculate_health_score(data, num_risks=1, synergistic_count=0)
    assert score == 99.0

def test_high_co2_deductions():
    """Test deductions for high and very high CO2 levels."""
    data_high = create_sensor_data(co2=1500)
    score_high = calculate_health_score(data_high, num_risks=1, synergistic_count=0)
    assert score_high == 80.0  # 100 - 20

    data_severe = create_sensor_data(co2=2500)
    score_severe = calculate_health_score(data_severe, num_risks=1, synergistic_count=0)
    assert score_severe == 50.0  # 100 - 50

def test_high_humidity_deductions():
    """Test deductions for high and very high humidity."""
    data_high = create_sensor_data(humidity=75)
    score_high = calculate_health_score(data_high, num_risks=1, synergistic_count=0)
    assert score_high == 85.0  # 100 - 15

    data_severe = create_sensor_data(humidity=90)
    score_severe = calculate_health_score(data_severe, num_risks=1, synergistic_count=0)
    assert score_severe == 70.0  # 100 - 30

def test_high_co_deduction():
    """Test deduction for high CO levels."""
    data = create_sensor_data(co=10)
    score = calculate_health_score(data, num_risks=1, synergistic_count=0)
    assert score == 60.0  # 100 - 40

def test_high_pm25_deduction():
    """Test deduction for high PM2.5 levels."""
    data = create_sensor_data(**{'pm2.5': 40})
    score = calculate_health_score(data, num_risks=1, synergistic_count=0)
    assert score == 80.0  # 100 - 20

def test_combined_deductions():
    """Test multiple deductions applied correctly."""
    data = create_sensor_data(co2=1500, humidity=90, co=10)
    # Deductions: -20 (CO2), -30 (Humidity), -40 (CO)
    score = calculate_health_score(data, num_risks=3, synergistic_count=1)
    assert score == 10.0  # 100 - 20 - 30 - 40

def test_score_minimum_zero():
    """Test that the score cannot go below 0."""
    data = create_sensor_data(co2=2500, humidity=90, co=10, **{'pm2.5': 40})
    # Deductions: -50 (CO2), -30 (Humidity), -40 (CO), -20 (PM2.5) = -140
    score = calculate_health_score(data, num_risks=4, synergistic_count=2)
    assert score == 0.0
