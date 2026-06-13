from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ReportRequest(BaseModel):
    context_type: str = Field(..., description="Context: hospital, home, industrial")
    analysis_data: Dict[str, Any] = Field(..., description="Data from core/simulation modules")
    actions: Optional[Dict[str, Any]] = Field(default=None, description="Data from action engine")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
