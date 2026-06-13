from typing import List, Dict, Any
from veritas.utils.logger import logger
from veritas.schemas.action_response import PriorityLevel, ActionItem

class ActionIntelligenceEngine:
    def __init__(self):
        try:
            logger.info("Action Intelligence Engine Initialized")
        except Exception as e:
            print(f"Logger fail-safe: {e}")

    def generate_actions(self, analysis_data: Dict[str, Any]) -> List[ActionItem]:
        try:
            actions = []
            # Extract sensor data if coming from simulation or analyze
            sensors = analysis_data.get("sensors", analysis_data.get("results", {}))
            
            # Clean up values for humans
            co2 = round(sensors.get("co2", 0), 0)
            temp = round(sensors.get("temperature", 0), 1)
            humidity = round(sensors.get("humidity", 0), 1)
            
            # Decision Logic: CO2
            if co2 > 2500:
                actions.append(ActionItem(
                    action="CRITICAL: Immediate Evacuation",
                    priority=PriorityLevel.CRITICAL,
                    timeframe="Immediate",
                    rationale=f"CO2 levels are life-threatening ({int(co2)} ppm)"
                ))
            elif co2 > 1500:
                actions.append(ActionItem(
                    action="High Priority: Maximum Ventilation",
                    priority=PriorityLevel.CRITICAL,
                    timeframe="Within 5 mins",
                    rationale=f"CO2 levels are hazardous ({int(co2)} ppm)"
                ))
            elif co2 > 1000:
                actions.append(ActionItem(
                    action="Increase Ventilation Rate",
                    priority=PriorityLevel.MODERATE,
                    timeframe="Within 15 mins",
                    rationale=f"CO2 levels are above comfort threshold ({int(co2)} ppm)"
                ))

            # Decision Logic: Temperature
            if temp > 35:
                actions.append(ActionItem(
                    action="CRITICAL: Emergency Cooling",
                    priority=PriorityLevel.CRITICAL,
                    timeframe="Immediate",
                    rationale=f"Temperature is dangerously high ({temp}°C)"
                ))
            elif temp > 30:
                actions.append(ActionItem(
                    action="Activate Cooling System",
                    priority=PriorityLevel.MODERATE,
                    timeframe="Within 10 mins",
                    rationale=f"Temperature exceeds 30°C ({temp}°C)"
                ))

            # Decision Logic: Humidity
            if humidity > 85:
                actions.append(ActionItem(
                    action="Activate Dehumidifier",
                    priority=PriorityLevel.MODERATE,
                    timeframe="Within 20 mins",
                    rationale=f"Humidity is excessively high ({humidity}%)"
                ))
            elif humidity > 70:
                actions.append(ActionItem(
                    action="Increase Air Circulation",
                    priority=PriorityLevel.LOW,
                    timeframe="Within 30 mins",
                    rationale=f"Humidity levels are rising ({humidity}%)"
                ))

            # Synergy Logic: CO2 + Humidity (Poor Ventilation)
            if co2 > 1000 and humidity > 70:
                actions.append(ActionItem(
                    action="RISK: Poor Ventilation Synergy Detected",
                    priority=PriorityLevel.CRITICAL,
                    timeframe="Immediate",
                    rationale=f"Combined high CO2 ({int(co2)} ppm) and Humidity ({humidity}%) indicate stagnant air"
                ))

            if not actions:
                actions.append(ActionItem(
                    action="No immediate action required",
                    priority=PriorityLevel.LOW,
                    timeframe="N/A",
                    rationale="All parameters within safety limits"
                ))

            logger.info(f"Generated {len(actions)} actionable insights")
            return actions
        except Exception as e:
            logger.error(f"Action generation failed: {str(e)}")
            return []

action_engine = ActionIntelligenceEngine()
