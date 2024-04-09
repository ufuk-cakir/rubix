import pytest
import importlib


# Define the fixture with request parameter to access test parameters
@pytest.fixture
def setup_logger_env(monkeypatch, request):
    log_level = request.param  # Access the parameter passed to the test
    if log_level is not None:
        monkeypatch.setenv("RUBIX_LOG_LEVEL", log_level)
    else:
        monkeypatch.delenv("RUBIX_LOG_LEVEL", raising=False)

    # Import and reload the rubix.logger module inside the fixture
    import rubix.logger  # type: ignore

    importlib.reload(rubix.logger)
    return rubix.logger


# Use pytest.mark.parametrize to pass different values to your fixture
@pytest.mark.parametrize("setup_logger_env", ["DEBUG", None], indirect=True)
def test_logger(setup_logger_env):
    logger = setup_logger_env
    if logger.logger.level == 10:  # DEBUG level
        assert logger.logger.name == "rubix" and logger.logger.level == 10
    else:  # If environment variable is not set, check the default or last set level
        assert logger.logger.name == "empty-logger"
