import pytest
from rubix.core.telescope import get_spaxel_assignment
from unittest.mock import patch


def test_get_spaxel_assignment():
    config = {
        "telescope": {
            "name": "MUSE",
        },
        "galaxy": {"dist_z": 0.5},
        "cosmology": {"name": "PLANCK15"},
    }

    spaxel_assignment = get_spaxel_assignment(config)

    assert callable(spaxel_assignment)
    # Add more test cases as needed

    # Test for unsupported telescope
    config["telescope"]["name"] = "unsupprted"
    with pytest.raises(ValueError):
        get_spaxel_assignment(config)

    # Test for unsupported pixel type
    with patch("rubix.core.telescope.get_telescope") as mock_get_telescope:
        mock_get_telescope.return_value.pixel_type = "unsupported"

        with pytest.raises(ValueError):
            get_spaxel_assignment(config)
