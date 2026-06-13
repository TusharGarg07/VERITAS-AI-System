import math

class VeritasFormatter:
    """
    Utility class for rounding and formatting environmental data according to enterprise rules.
    """
    
    @staticmethod
    def format_value(key: str, value: float) -> str:
        """
        Applies rounding rules based on the parameter type.
        """
        try:
            if value is None:
                return "N/A"
            
            if key.lower() == 'co2':
                return f"{int(round(value))} ppm"
            elif key.lower() in ['temperature', 'temp']:
                return f"{round(value, 1)} °C"
            elif key.lower() == 'humidity':
                return f"{round(value, 1)} %"
            elif key.lower() == 'pm2_5':
                return f"{round(value, 1)} µg/m³"
            elif key.lower() == 'tvoc':
                return f"{int(round(value))} ppb"
            elif key.lower() == 'co':
                return f"{round(value, 1)} ppm"
            elif key.lower() == 'illuminance':
                return f"{int(round(value))} lux"
            elif key.lower() == 'noise':
                return f"{round(value, 1)} dB"
            
            return str(round(value, 2))
        except Exception:
            return str(value)

    @staticmethod
    def get_status(key: str, value: float) -> str:
        """
        Determines the risk status for a given parameter and value.
        """
        try:
            if value is None:
                return "UNKNOWN"
                
            key = key.lower()
            
            if key == 'co2':
                if value > 2000: return "CRITICAL"
                if value > 1000: return "HIGH"
                if value > 600: return "MODERATE"
                return "GOOD"
            
            if key == 'temperature':
                if value > 35 or value < 15: return "CRITICAL"
                if value > 30 or value < 18: return "HIGH"
                return "GOOD"
                
            if key == 'humidity':
                if value > 80 or value < 20: return "CRITICAL"
                if value > 70 or value < 30: return "HIGH"
                return "GOOD"
                
            if key == 'pm2_5':
                if value > 55: return "CRITICAL"
                if value > 35: return "HIGH"
                if value > 12: return "MODERATE"
                return "GOOD"

            if key == 'tvoc':
                if value > 3000: return "CRITICAL"
                if value > 1000: return "HIGH"
                if value > 500: return "MODERATE"
                return "GOOD"

            return "NORMAL"
        except Exception:
            return "NORMAL"

    @staticmethod
    def get_risk_level(score: float) -> str:
        """
        Returns risk level based on score (0-100).
        """
        if score >= 80: return "CRITICAL"
        if score >= 60: return "HIGH"
        if score >= 40: return "MODERATE"
        return "SAFE"
