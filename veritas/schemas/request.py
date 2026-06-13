from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any

class AnalysisRequest(BaseModel):
    task_id: str = Field(..., description="Unique task identifier")
    content: str = Field(..., min_length=1, description="Text content to analyze")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace')
        return v
