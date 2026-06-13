from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class SimulationRequest(BaseModel):
    scenario: str = Field(..., description="Scenario name: closed_room, high_occupancy, poor_ventilation")
    duration_minutes: int = Field(default=60, ge=1, le=1440)
    sampling_rate_seconds: int = Field(default=60, ge=1, le=3600)
    initial_conditions: Optional[Dict[str, float]] = Field(default_factory=dict)
