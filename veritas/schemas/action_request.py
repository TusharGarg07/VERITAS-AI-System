from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ActionRequest(BaseModel):
    analysis_results: Dict[str, Any] = Field(..., description="Output from Core Analysis or Simulation")
    thresholds: Optional[Dict[str, float]] = Field(default_factory=dict)
    context_id: Optional[str] = None
