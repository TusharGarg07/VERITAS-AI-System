import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from veritas.config.settings import settings
from veritas.api.routes import analyze, simulation, actions, report, personalized_report
from veritas.utils.logger import logger
from veritas.core.model_loader import model_loader

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version="1.0.0",
        openapi_url=f"{settings.API_V1_STR}/openapi.json"
    )

    # Startup event
    @app.on_event("startup")
    async def startup_event():
        try:
            logger.info("Starting VERITAS AI System...")
            model_loader.load()
        except Exception as e:
            logger.critical(f"System startup failed: {str(e)}")

    # Include routers
    app.include_router(
        analyze.router, 
        prefix=f"{settings.API_V1_STR}",
        tags=["analysis"]
    )

    app.include_router(
        simulation.router,
        prefix=f"{settings.API_V1_STR}",
        tags=["simulation"]
    )

    app.include_router(
        actions.router,
        prefix=f"{settings.API_V1_STR}",
        tags=["actions"]
    )

    app.include_router(
        report.router,
        prefix=f"{settings.API_V1_STR}",
        tags=["report"]
    )

    app.include_router(
        personalized_report.router,
        prefix=f"{settings.API_V1_STR}",
        tags=["personalized-report"]
    )

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model_loaded": model_loader.is_loaded}

    # Mount Static Files
    app.mount("/reports", StaticFiles(directory="reports"), name="reports")
    app.mount("/", StaticFiles(directory="veritas/frontend", html=True), name="frontend")

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False
    )
