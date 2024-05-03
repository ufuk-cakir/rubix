import logging
import os
import rubix._version as version
from rubix import config as rubix_config

def get_logger(config = None):
    if config is None:
        config = rubix_config["logger"]
    logger = logging.getLogger("rubix")
    logger.setLevel(getattr(logging, config["log_level"].upper(), "INFO"))
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    # Console Handler
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if config["log_file_path"]:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config["log_file_path"]), exist_ok=True)
        # Configure FileHandler to overwrite the file
        file_handler = logging.FileHandler(config["log_file_path"], mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {config['log_file_path']}")

    return logger