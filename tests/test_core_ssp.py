import pytest
from unittest.mock import patch, MagicMock
from rubix.core.ssp import get_lookup
from rubix import config

ssp_config = config["ssp"]
supported_templates = ssp_config["templates"]
TEMPLATE_NAME = list(supported_templates.keys())[0]
print("supported_templates:", supported_templates)
print("TEMPLATE_NAME:", TEMPLATE_NAME)


def test_get_lookup_with_valid_config():
    config = {
        "ssp": {
            "template": {"name": TEMPLATE_NAME},
            "method": "cubic",
        },
    }
    lookup = get_lookup(config)
    assert callable(lookup)


def test_get_lookup_with_missing_ssp_field():
    config = {}
    with pytest.raises(ValueError) as excinfo:
        get_lookup(config)
    assert "Configuration does not contain 'ssp' field" in str(excinfo.value)


def test_get_lookup_with_missing_template_field():
    config = {"ssp": {}}
    with pytest.raises(ValueError) as excinfo:
        get_lookup(config)
    assert "Configuration does not contain 'template' field" in str(excinfo.value)


def test_get_lookup_with_missing_name_field():
    config = {"ssp": {"template": {}}}
    with pytest.raises(ValueError) as excinfo:
        get_lookup(config)
    assert "Configuration does not contain 'name' field" in str(excinfo.value)


def test_get_lookup_with_missing_method_field():
    config = {
        "ssp": {"template": {"name": TEMPLATE_NAME}},
    }
    lookup = get_lookup(config)
    assert callable(lookup)


# Additional tests from the provided example
@patch("rubix.core.ssp.get_logger")
@patch("rubix.core.ssp.get_ssp")
def test_get_lookup_default_method(mock_get_ssp, mock_get_logger):
    config = {"ssp": {"template": {"name": TEMPLATE_NAME}}}
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_ssp = MagicMock()
    mock_get_ssp.return_value = mock_ssp
    mock_ssp.get_lookup.return_value = "lookup_function"

    result = get_lookup(config)

    mock_get_logger.assert_called_once()
    mock_get_ssp.assert_called_once_with(config)
    mock_logger.debug.assert_called_with(
        "Method not defined, using default method: cubic"
    )
    mock_ssp.get_lookup.assert_called_once_with(method="cubic")
    assert result == "lookup_function"


@patch("rubix.core.ssp.get_logger")
@patch("rubix.core.ssp.get_ssp")
def test_get_lookup_defined_method(mock_get_ssp, mock_get_logger):
    config = {"ssp": {"template": {"name": TEMPLATE_NAME}, "method": "linear"}}
    mock_logger = MagicMock()
    mock_get_logger.return_value = mock_logger
    mock_ssp = MagicMock()
    mock_get_ssp.return_value = mock_ssp
    mock_ssp.get_lookup.return_value = "lookup_function"

    result = get_lookup(config)

    mock_get_logger.assert_called_once()
    mock_get_ssp.assert_called_once_with(config)
    mock_logger.debug.assert_called_with(f"Using method defined in config: linear")
    mock_ssp.get_lookup.assert_called_once_with(method="linear")
    assert result == "lookup_function"


@patch("rubix.core.ssp.get_lookup")
def test_get_lookup_vmap(mock_get_lookup):
    config = {"ssp": {"template": {"name": TEMPLATE_NAME}}}
    mock_lookup = MagicMock()
    mock_get_lookup.return_value = mock_lookup
    mock_vmap = MagicMock(return_value="vmap_lookup")

    with patch("jax.vmap", mock_vmap):
        from rubix.core.ssp import get_lookup_vmap

        result = get_lookup_vmap(config)

    mock_get_lookup.assert_called_once_with(config)
    mock_vmap.assert_called_once_with(mock_lookup, in_axes=(0, 0))
    assert result == "vmap_lookup"


@patch("rubix.core.ssp.get_lookup_vmap")
def test_get_lookup_pmap(mock_get_lookup_vmap):
    config = {"ssp": {"template": {"name": TEMPLATE_NAME}}}
    mock_lookup_vmap = MagicMock()
    mock_get_lookup_vmap.return_value = mock_lookup_vmap
    mock_pmap = MagicMock(return_value="pmap_lookup")

    with patch("jax.pmap", mock_pmap):
        from rubix.core.ssp import get_lookup_pmap

        result = get_lookup_pmap(config)

    mock_get_lookup_vmap.assert_called_once_with(config)
    mock_pmap.assert_called_once_with(mock_lookup_vmap, in_axes=(0, 0))
    assert result == "pmap_lookup"
