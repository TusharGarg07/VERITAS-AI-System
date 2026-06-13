from fastapi import APIRouter, HTTPException
from veritas.schemas.report_request import ReportRequest
from veritas.schemas.report_response import ReportResponse
from veritas.modules.report_engine.context import context_engine
from veritas.modules.report_engine.impact import impact_engine
from veritas.modules.report_engine.explain import xai_engine
from veritas.modules.report_engine.generator import report_generator
from veritas.utils.logger import logger
import uuid

router = APIRouter(
    prefix="/report",
    tags=["report"]
)

@router.post("/generate")
async def create_report(request: ReportRequest):
    try:
        report_id = str(uuid.uuid4())
        logger.info(f"Generating report for ID: {report_id}")

        # 1. Context Evaluation
        ctx_results = context_engine.evaluate_context(request.context_type, request.analysis_data)
        
        # 2. Human Impact Mapping
        sensors = request.analysis_data.get("sensors", request.analysis_data.get("results", {}))
        impacts = impact_engine.map_impact(sensors)
        
        # 3. Explainable AI
        explanation = xai_engine.explain_risk(request.analysis_data)
        
        # 4. Generate PDF
        report_payload = {
            "context": request.context_type,
            "sensors": sensors,
            "impacts": impacts,
            "explanation": explanation,
            "violations": ctx_results.get("violations", []),
            "risk_score": 0.3 # Mock for now
        }
        
        file_path = report_generator.generate_pdf(report_id, report_payload)

        return ReportResponse(
            status="success",
            report_id=report_id,
            file_path=file_path
        )

    except Exception as e:
        logger.error(f"Report API failure: {str(e)}")
        raise HTTPException(status_code=500, detail="Report generation failed")
