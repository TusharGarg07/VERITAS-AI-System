from pydantic import BaseModel
from typing import Optional

class PersonalizedReportResponse(BaseModel):
    status: str
    report_id: str
    email_sent: bool
    file_path: str
    error: Optional[str] = None
