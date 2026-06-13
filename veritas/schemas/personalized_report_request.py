from pydantic import BaseModel, Field, EmailStr
from typing import Dict, Any, Optional

class PersonalizedReportRequest(BaseModel):
    user_name: str = Field(..., min_length=2)
    user_email: EmailStr
    domain: str = Field(..., description="e.g. Healthcare, Residential, Industrial")
    analysis_data: Dict[str, Any]
    context_type: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
