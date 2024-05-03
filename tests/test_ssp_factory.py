import pytest
from unittest.mock import patch
from rubix.spectra.ssp.factory import get_ssp_template
from copy import deepcopy


# Fixture to reset the configuration after each test
@pytest.fixture(autouse=True)
def reset_config():
    # Setup: Store original config
    original_config = get_config().copy()

    # This yields control to the test function
    yield

    # Teardown: Reset config to original after each test
    with patch("rubix.spectra.ssp.factory.rubix_config", original_config):
        pass


def get_config():
    from rubix import config

    return deepcopy(config)


def test_get_ssp_template_existing_template():
    config = get_config()
    suppored_templates = config["ssp"]["templates"].copy()

    for template_name in suppored_templates:
        template = get_ssp_template(template_name)
        template_class_name = config["ssp"]["templates"][template_name]["name"]
        assert template.__class__.__name__ == template_class_name


def test_get_ssp_template_non_existing_template():
    template_name = "unknown_template"

    with pytest.raises(ValueError) as excinfo:
        get_ssp_template(template_name)

    assert (
        str(excinfo.value)
        == "SSP template unknown_template not found in the supported configuration file."
    )


def test_get_ssp_template_invalid_format():
    config = get_config()
    config_copy = config.copy()
    suppored_templates = config_copy["ssp"]["templates"]

    # get   the first template
    template_name = list(suppored_templates.keys())[0]
    suppored_templates[template_name]["format"] = "invalid_format"

    config_copy["ssp"]["templates"] = suppored_templates
    with patch("rubix.spectra.ssp.factory.rubix_config", config_copy):
        with pytest.raises(ValueError) as excinfo:
            get_ssp_template(template_name)

        assert (
            str(excinfo.value)
            == "Currently only HDF5 format is supported for SSP templates."
        )


def test_get_ssp_template_error_loading_file():

    config = get_config()
    print("before", config["ssp"])
    suppored_templates = config["ssp"]["templates"]

    # get   the first template
    template_name = list(suppored_templates.keys())[0]
    suppored_templates[template_name]["file_name"] = "invalid_file"

    suppored_templates[template_name]["format"] = "HDF5"
    config["ssp"]["templates"] = suppored_templates
    print(config["ssp"])
    with patch("rubix.spectra.ssp.factory.rubix_config", config):
        with pytest.raises(FileNotFoundError) as excinfo:
            print("template_name", template_name)
            get_ssp_template(template_name)

    assert "No such file or directory" in str(excinfo.value)