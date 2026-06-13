from fastapi import APIRouter, HTTPException
from veritas.schemas.simulation_request import SimulationRequest
from veritas.schemas.simulation_response import SimulationResponse
from veritas.modules.simulation.simulator import simulator_engine
from veritas.utils.logger import logger

router = APIRouter(
    prefix="/simulation",
    tags=["Simulation"]
)

@router.post("/simulate")
async def run_simulation(request: SimulationRequest):
    try:
        logger.info(f"Simulation request received: {request.scenario}")
        
        data = simulator_engine.generate_scenario_data(
            scenario=request.scenario,
            duration=request.duration_minutes,
            interval=request.sampling_rate_seconds
        )

        if not data:
            return SimulationResponse(
                status="error",
                scenario=request.scenario,
                data_points=[],
                summary={},
                error="Failed to generate simulation data"
            )

        summary = {
            "total_points": len(data),
            "duration_minutes": request.duration_minutes,
            "engine": "Veritas-Sim-v1"
        }

        # Format output precisely for frontend (data_points key is mandatory)
        return {
            "status": "success",
            "scenario": request.scenario,
            "data_points": data,
            "summary": summary
        }

    except Exception as e:
        logger.error(f"Simulation API Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Simulation Engine Failure")
