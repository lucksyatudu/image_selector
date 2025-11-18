import logging
from datetime import datetime
from core.config import Config

def setup_logging():
    """Configures the logging for the application."""
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    log_path = 'runs/run_log_' + datetime.now().strftime('%Y%m%d-%H%M') + '.log'
    logger = logging.getLogger()
     # Prevent adding handlers repeatedly (important in notebooks / multiple imports)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)

    # ---- Formatter ----
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # ---- Console Handler ----
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ---- File Handler ----
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Writing logs to: {log_path}")

    return logger

# Initialize the logger for the package
logger = setup_logging()