import sys
from loguru import logger
from veritas.config.settings import settings

def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.LOG_LEVEL,
    )
    logger.add(
        "logs/veritas.log",
        rotation="10 MB",
        retention="10 days",
        level="INFO",
        compression="zip"
    )

setup_logging()
