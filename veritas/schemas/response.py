from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class AnalysisResponse(BaseModel):
    status: str = Field(..., description="Success or failure status")
    task_id: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
