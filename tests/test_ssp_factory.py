import pytest
from unittest.mock import patch
from rubix.spectra.ssp.factory import get_ssp_template
from copy import deepcopy
import sys


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


class MockFSPS:
    class StellarPopulation:
        def __init__(self, zcontinuous=0, **kwargs):
            self.zlegend = np.array([0.001, 0.01, 0.1])
            self.log_age = np.array([9.0, 9.1, 9.2])

        def get_spectrum(self, peraa=True, **kwargs):
            return (
                np.array([4000, 4100, 4200]),
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            )


# fake the FSPS package
sys.modules["fsps"] = MockFSPS()


def test_get_ssp_template_existing_template():
    config = get_config()
    supported_templates = config["ssp"]["templates"].copy()

    with patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True):
        for template_name in supported_templates:
            print("template_name", template_name)
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
    supported_templates = config_copy["ssp"]["templates"]

    # get the first template
    template_name = list(supported_templates.keys())[0]
    supported_templates[template_name]["format"] = "invalid_format"

    config_copy["ssp"]["templates"] = supported_templates
    with patch("rubix.spectra.ssp.factory.rubix_config", config_copy):
        with pytest.raises(ValueError) as excinfo:
            get_ssp_template(template_name)

        assert (
            str(excinfo.value)
            == "Currently only HDF5 format and fits files in the format of pyPipe3D format are supported for SSP templates."
        )


def test_get_ssp_template_error_loading_file():

    config = get_config()
    supported_templates = config["ssp"]["templates"]

    # get the first template
    template_name = list(supported_templates.keys())[0]
    supported_templates[template_name]["file_name"] = "invalid_file"

    supported_templates[template_name]["format"] = "HDF5"
    config["ssp"]["templates"] = supported_templates
    with patch("rubix.spectra.ssp.factory.rubix_config", config):
        with pytest.raises(FileNotFoundError) as excinfo:
            print("template_name", template_name)
            get_ssp_template(template_name)
        
    assert "Could not download file" in str(excinfo.value)




def test_get_ssp_template_existing_fsps_template():
    config = get_config()
    config_copy = config.copy()

    supported_templates = config_copy["ssp"]["templates"]

    # get the fsps template
    supported_templates["FSPS"]["source"] = "load_from_file"
    config_copy["ssp"]["templates"] = supported_templates

    with (
        patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True),
        patch("rubix.spectra.ssp.factory.rubix_config", config_copy),
    ):
        template = get_ssp_template("FSPS")
        template_class_name = supported_templates["FSPS"]["name"]
        assert template.__class__.__name__ == template_class_name


def test_get_fsps_template_wrong_source_keyword():
    config = get_config()
    config_copy = config.copy()

    supported_templates = config_copy["ssp"]["templates"]

    # get the fsps template
    supported_templates["FSPS"]["source"] = "wrong_source_keyword"
    config_copy["ssp"]["templates"] = supported_templates

    with (
        patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True),
        patch("rubix.spectra.ssp.factory.rubix_config", config_copy),
    ):
        with pytest.raises(ValueError) as excinfo:
            get_ssp_template("FSPS")
    assert (
        f"The source {supported_templates['FSPS']['source']} of the FSPS SSP template is not supported."
        == str(excinfo.value)
    )

