from fastapi import APIRouter, HTTPException
from veritas.schemas.action_request import ActionRequest
from veritas.schemas.action_response import ActionResponse
from veritas.modules.action_engine.actions import action_engine
from veritas.utils.logger import logger

router = APIRouter(
    prefix="/actions",
    tags=["actions"]
)

@router.post("/generate")
async def get_actions(request: ActionRequest):
    try:
        logger.info(f"Action request received for context: {request.context_id}")
        
        actions = action_engine.generate_actions(request.analysis_results)

        if not actions:
            return ActionResponse(
                status="error",
                actions=[],
                risk_score=0.0,
                metadata={},
                error="Failed to generate actions"
            )

        # Calculate risk score based on action priorities
        critical_count = sum(1 for a in actions if a.priority == "Critical")
        moderate_count = sum(1 for a in actions if a.priority == "Moderate")
        
        if critical_count > 0:
            risk_score = min(0.9 + (critical_count * 0.05), 1.0)
        elif moderate_count > 0:
            risk_score = min(0.4 + (moderate_count * 0.1), 0.8)
        else:
            risk_score = 0.1

        return ActionResponse(
            status="success",
            actions=actions,
            risk_score=risk_score,
            metadata={"engine": "Veritas-Action-v1", "critical_triggers": critical_count}
        )

    except Exception as e:
        logger.error(f"Action API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Action Engine Failure")
