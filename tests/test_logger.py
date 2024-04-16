import pytest
import os
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


# Fixture to setup and teardown the test directory
@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    test_log_dir = tmp_path / "test_logs"
    monkeypatch.setenv("RUBIX_LOG_PATH", str(test_log_dir))
    yield str(test_log_dir)  # Provide the directory path to the test
    # No explicit teardown needed if using tmp_path, as pytest handles it


def test_log_file_creation(monkeypatch, tmp_path):

    log_dir = tmp_path / "test_logs"
    monkeypatch.setenv("RUBIX_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RUBIX_LOG_PATH", str(log_dir))
    import rubix.logger  # type: ignore

    importlib.reload(rubix.logger)
    logger = rubix.logger.logger
    assert logger.name == "rubix"
    expected_log_file = os.path.join(log_dir, "rubix.log")
    assert os.path.isdir(log_dir), "Log directory was not created"
    assert os.path.exists(expected_log_file), "Log file was not created"
