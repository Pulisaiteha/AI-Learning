import os
import logging


def setup_logger(name: str = "rag_app"):
    """Configures a custom logger that outputs to both console and a logs/ file."""
    # Build path to root-level logs folder
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "app.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if setup is called multiple times
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Console Handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


# Module-level instance for quick imports
logger = setup_logger()