from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class PriorityLevel(str, Enum):
    CRITICAL = "Critical"
    MODERATE = "Moderate"
    LOW = "Low"

class ActionItem(BaseModel):
    action: str
    priority: PriorityLevel
    timeframe: str
    rationale: str

class ActionResponse(BaseModel):
    status: str
    actions: List[ActionItem]
    risk_score: float
    metadata: Dict[str, Any]
    error: Optional[str] = None
