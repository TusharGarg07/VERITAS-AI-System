from fastapi import APIRouter, HTTPException
from veritas.schemas.request import AnalysisRequest
from veritas.schemas.response import AnalysisResponse
from veritas.core.pipeline import pipeline
from veritas.utils.logger import logger

router = APIRouter()

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_data(request: AnalysisRequest):
    try:
        logger.info(f"Received analysis request: {request.task_id}")
        
        result = await pipeline.execute(
            task_id=request.task_id,
            content=request.content,
            params=request.parameters
        )
        
        if "error" in result:
            return AnalysisResponse(
                status="error",
                task_id=request.task_id,
                error=result["error"],
                metadata={"status_code": 500}
            )
            
        return AnalysisResponse(
            status="success",
            task_id=request.task_id,
            results=result,
            metadata={"processed_by": "veritas-api"}
        )

    except Exception as e:
        logger.error(f"API Route error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
