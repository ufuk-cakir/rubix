from rubix.logger import get_logger


def test_logger_file_path(tmpdir):
    log_file = tmpdir / "test.log"
    config = {
        "log_file_path": str(log_file),
        "log_level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }
    logger = get_logger(config)
    logger.info("test")
    assert log_file.exists()
    with open(log_file, "r") as f:
        assert "test" in f.read()
