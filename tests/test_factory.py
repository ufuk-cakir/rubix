import pytest
from unittest.mock import patch, MagicMock
from rubix.galaxy.input_handler.factory import get_input_handler
import logging


def test_get_input_handler_illustris():
    config = {"simulation": {"name": "IllustrisTNG", "args": {"path": "value1"}}}

    with patch("rubix.galaxy.input_handler.factory.IllustrisHandler") as mock_handler:
        mock_instance = MagicMock()
        mock_handler.return_value = mock_instance

        result = get_input_handler(config)

        # Check if the mock instance is returned
        assert result == mock_instance

        # Ensure the constructor is called with the correct arguments
        call_args = mock_handler.call_args[1]
        assert call_args["path"] == "value1"
        assert isinstance(call_args["logger"], logging.Logger)


def test_get_input_handler_nihao():
    config = {"simulation": {"name": "NIHAO", "args": {"path": "value2"}}}

    with patch("rubix.galaxy.input_handler.factory.NihaoHandler") as mock_handler:
        mock_instance = MagicMock()
        mock_handler.return_value = mock_instance

        result = get_input_handler(config)

        assert result == mock_instance

        call_args = mock_handler.call_args[1]
        assert call_args["path"] == "value2"
        assert isinstance(call_args["logger"], logging.Logger)


def test_get_input_handler_unsupported():
    config = {"simulation": {"name": "UnknownSim", "args": {}}}

    with pytest.raises(ValueError) as excinfo:
        get_input_handler(config)

    assert "not supported" in str(excinfo.value)
