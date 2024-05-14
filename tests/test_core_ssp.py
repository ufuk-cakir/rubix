import pytest
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
