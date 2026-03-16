import sys
import os
from loguru import logger


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured loguru logging."""
    logger.remove()

    # Console handler — pretty in dev, JSON in prod
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=log_level,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler for persistent logs
    os.makedirs("storage/logs", exist_ok=True)
    logger.add(
        "storage/logs/app.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        level=log_level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        serialize=False,
    )

    logger.info(f"Logging initialized at level {log_level}")
