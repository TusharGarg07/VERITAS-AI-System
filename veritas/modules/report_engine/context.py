from typing import Dict, Any
from veritas.utils.logger import logger

class ContextIntelligence:
    THRESHOLDS = {
        "hospital": {"co2": 600, "temp": 22},
        "home": {"co2": 1000, "temp": 24},
        "industrial": {"co2": 5000, "temp": 35}
    }

    def evaluate_context(self, context_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            logger.info(f"Evaluating context-aware logic for: {context_type}")
            thresholds = self.THRESHOLDS.get(context_type, self.THRESHOLDS["home"])
            
            sensors = data.get("sensors", data.get("results", {}))
            violations = []
            
            if sensors.get("co2", 0) > thresholds["co2"]:
                violations.append(f"CO2 exceeds {context_type} limit ({thresholds['co2']} ppm)")
            
            return {
                "context": context_type,
                "thresholds": thresholds,
                "violations": violations,
                "is_safe": len(violations) == 0
            }
        except Exception as e:
            logger.error(f"Context evaluation failed: {str(e)}")
            return {"error": str(e)}

context_engine = ContextIntelligence()
