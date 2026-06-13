from typing import Dict, List
from veritas.utils.logger import logger

class HumanImpactEngine:
    def map_impact(self, sensor_data: Dict[str, float]) -> List[str]:
        try:
            impacts = []
            co2 = sensor_data.get("co2", 0)
            
            if co2 > 5000:
                impacts.append("Risk of toxicity and asphyxiation")
            elif co2 > 2000:
                impacts.append("Headaches, sleepiness, and loss of attention")
            elif co2 > 1000:
                impacts.append("Drowsiness and poor air quality perception")
            else:
                impacts.append("Normal indoor air quality - no significant impact")
                
            return impacts
        except Exception as e:
            logger.error(f"Impact mapping failed: {str(e)}")
            return ["Error determining human impact"]

impact_engine = HumanImpactEngine()
