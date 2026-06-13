import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from veritas.utils.logger import logger

class VeritasSimulator:
    SCENARIOS = {
        "closed_room": {"temp": (24, 28), "co2": (600, 1000), "humidity": (50, 65)},
        "high_occupancy": {"temp": (28, 32), "co2": (1000, 1800), "humidity": (65, 80)},
        "poor_ventilation": {"temp": (30, 38), "co2": (1800, 3500), "humidity": (75, 95)}
    }

    def __init__(self):
        try:
            logger.info("Simulation Engine Initialized")
        except Exception as e:
            print(f"Logger fail-safe: {e}")

    def generate_scenario_data(self, scenario: str, duration: int, interval: int) -> List[Dict[str, Any]]:
        try:
            if scenario not in self.SCENARIOS:
                logger.warning(f"Unknown scenario: {scenario}. Using default.")
                scenario = "closed_room"

            config = self.SCENARIOS[scenario]
            steps = (duration * 60) // interval
            data_points = []
            start_time = datetime.now()

            for i in range(steps):
                timestamp = (start_time + timedelta(seconds=i * interval)).isoformat()
                sensors = {
                    "temperature": float(np.random.uniform(*config["temp"])),
                    "co2": float(np.random.uniform(*config["co2"])),
                    "humidity": float(np.random.uniform(*config["humidity"]))
                }
                data_points.append({"timestamp": timestamp, "sensors": sensors})

            logger.info(f"Generated {len(data_points)} data points for scenario: {scenario}")
            return data_points
        except Exception as e:
            logger.error(f"Simulation generation failed: {str(e)}")
            return []

simulator_engine = VeritasSimulator()
