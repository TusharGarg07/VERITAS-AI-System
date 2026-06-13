from pydantic import BaseModel, Field
from typing import Optional

class ReportResponse(BaseModel):
    status: str
    report_id: str
    download_url: Optional[str] = None
    file_path: Optional[str] = None
    error: Optional[str] = None
