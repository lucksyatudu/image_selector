import logging
from core.config import Config

def setup_logging():
    """Configures the logging for the application."""
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

# Initialize the logger for the package
logger = setup_logging()