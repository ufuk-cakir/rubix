import logging
import os
import rubix._version as version


# Get Environment Variables
RUBIX_LOG_LEVEL = os.getenv("RUBIX_LOG_LEVEL", None)
RUBIX_LOG_FILEPATH = os.getenv("RUBIX_LOG_PATH", default=None)


if RUBIX_LOG_LEVEL:
    # Set up logging
    logger = logging.getLogger("rubix")

    # Set log level
    logger.setLevel(RUBIX_LOG_LEVEL)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Create a formatter if we are logging to a file
    if RUBIX_LOG_FILEPATH:
        # Create directory if it does not exist
        os.makedirs(RUBIX_LOG_FILEPATH, exist_ok=True)
        file_name = os.path.join(RUBIX_LOG_FILEPATH, "rubix.log")
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {file_name}")
        logger.info(
            "If you want to disable logging to file, set the environment variable RUBIX_LOG_PATH to None"
        )

    # Create a formatter if we are logging to the console

    # Log a message
    logger.info("RUBIX Version: %s", version.version)


else:

    class empty_logger:
        def debug(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

    logger = empty_logger()
