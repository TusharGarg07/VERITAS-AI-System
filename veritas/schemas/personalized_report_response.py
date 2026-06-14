from pydantic import BaseModel
from typing import Optional

class PersonalizedReportResponse(BaseModel):
    status: str
    report_id: str
    email_sent: bool
    file_path: str
    download_url: Optional[str] = None
    error: Optional[str] = None
