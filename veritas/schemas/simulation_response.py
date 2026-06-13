from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class SimulationDataPoint(BaseModel):
    timestamp: str
    sensors: Dict[str, float]

class SimulationResponse(BaseModel):
    status: str
    scenario: str
    data_points: List[SimulationDataPoint]
    summary: Dict[str, Any]
    error: Optional[str] = None
