# ==============================================================================
# PROJECT: VERITAS (Versatile Environmental Risk Intelligence & Transparency Analysis System)
# FILE: config.py
# ==============================================================================

PROFESSIONAL_RISK_GUIDE = {
    "Risk_High_PM2.5": {
        "Hazard_Level": "High",
        "Interpretation": "Particulate matter (PM2.5) exceeds safety thresholds. This is a direct respiratory health hazard.",
        "Remedies": [
            "Verify HVAC system's HEPA filter integrity and check for filter bypass.",
            "Increase air exchange rate in the affected zone.",
            "Deploy localized air purification units if HVAC is insufficient."
        ]
    },
    "Risk_Moderate_PM2.5": {
        "Hazard_Level": "Medium",
        "Interpretation": "Particulate matter (PM2.5) is at a level that could affect sensitive groups. Prolonged exposure is not recommended.",
        "Remedies": [
            "Ensure HVAC filters are clean and functioning correctly.",
            "Consider increasing ventilation during periods of high occupancy."
        ]
    },
    "Risk_Poor_Ventilation": {
        "Hazard_Level": "Medium",
        "Interpretation": "High CO2 indicates inadequate fresh air exchange, leading to reduced cognitive performance and stuffiness.",
        "Remedies": [
            "Adjust HVAC settings to increase the percentage of outdoor air intake.",
            "Check for blocked air diffusers or return vents.",
            "Review zone occupancy against its designed ventilation capacity."
        ]
    },
    "Comfort_Humidity_High": {
        "Hazard_Level": "Low",
        "Interpretation": "Excess humidity reduces thermal comfort and, if persistent, significantly increases the risk of microbial contamination (mold, mildew).",
        "Remedies": [
            "Engage HVAC dehumidification cycle or increase its intensity.",
            "Check for and repair any water intrusion or leaks.",
            "Deploy supplemental dehumidifiers for chronically damp areas."
        ]
    },
    "Comfort_Thermal_Hot": {
        "Hazard_Level": "Low",
        "Interpretation": "The temperature is uncomfortably hot, affecting productivity and well-being.",
        "Remedies": [
            "Adjust thermostat setpoint to a lower temperature.",
            "Increase fan speed for better air circulation.",
            "Review solar gain through windows and consider blinds/film."
        ]
    },
    "Comfort_Thermal_Cold": {
        "Hazard_Level": "Low",
        "Interpretation": "The temperature is uncomfortably cold, affecting productivity and well-being.",
        "Remedies": [
            "Adjust thermostat setpoint to a higher temperature.",
            "Check for drafts from windows or doors and seal them."
        ]
    },
}

SYNERGISTIC_RISK_RULES = {
    "High_Microbial_Contamination_Risk": {
        "Required_Risks": ["Comfort_Humidity_High", "Risk_Poor_Ventilation"],
        "Hazard_Level": "Critical",
        "Interpretation": "A critical microbial hazard exists. The combination of high humidity and stagnant, poorly ventilated air creates an ideal breeding ground for mold, bacteria, and other pathogens.",
        "Remedies": [
            "Prioritize immediate ventilation and dehumidification improvements.",
            "Schedule an inspection for existing microbial contamination.",
            "Implement continuous humidity and CO2 monitoring."
        ]
    },
    "Severe_Respiratory_Hazard": {
        "Required_Risks": ["Risk_High_PM2.5", "Risk_Poor_Ventilation"],
        "Hazard_Level": "Critical",
        "Interpretation": "A severe respiratory hazard exists. High levels of particulate matter are trapped in a poorly ventilated space, leading to pollutant accumulation and increased exposure risk.",
        "Remedies": [
            "Consider reducing occupancy in the zone until resolved.",
            "Combine all recommendations for the primary risks (HEPA filtration and ventilation)."
        ]
    },
    "Oppressive_Atmosphere_Hazard": {
        "Required_Risks": ["Comfort_Humidity_High", "Comfort_Thermal_Hot"],
        "Hazard_Level": "High",
        "Interpretation": "A 'heat index' hazard exists. The combination of high heat and high humidity creates an oppressive atmosphere that can feel significantly hotter than the measured temperature and can lead to heat-related stress and illness.",
        "Remedies": [
            "Aggressively cool and dehumidify the space using all available HVAC resources.",
            "Encourage occupants to stay hydrated."
        ]
    }
}
