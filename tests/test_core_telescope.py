import pytest
from rubix.core.telescope import (
    get_spaxel_assignment,
    get_telescope,
    get_spatial_bin_edges,
)
from unittest.mock import patch, MagicMock


class MockRubixData:
    def __init__(self, stars, gas):
        self.stars = stars
        self.gas = gas


class MockStarsData:
    def __init__(self, coords):
        self.coords = coords


class MockGasData:
    def __init__(self, coords):
        self.coords = coords


@patch("rubix.core.telescope.TelescopeFactory")
def test_get_telescope(mock_factory):
    config = {"telescope": {"name": "MUSE"}}
    mock_telescope = MagicMock()
    mock_factory.return_value.create_telescope.return_value = mock_telescope

    result = get_telescope(config)

    mock_factory.return_value.create_telescope.assert_called_once_with("MUSE")
    assert result == mock_telescope


def test_get_spaxel_assignment():
    config = {
        "telescope": {
            "name": "MUSE",
        },
        "galaxy": {"dist_z": 0.5},
        "cosmology": {"name": "PLANCK15"},
    }

    with (
        patch("rubix.core.telescope.get_telescope") as mock_get_telescope,
        patch(
            "rubix.core.telescope.get_spatial_bin_edges"
        ) as mock_get_spatial_bin_edges,
        patch(
            "rubix.core.telescope.square_spaxel_assignment"
        ) as mock_square_spaxel_assignment,
    ):

        mock_get_telescope.return_value = MagicMock(pixel_type="square")
        mock_get_spatial_bin_edges.return_value = "spatial_bin_edges"
        mock_square_spaxel_assignment.return_value = "pixel_assignment"

        spaxel_assignment = get_spaxel_assignment(config)

        assert callable(spaxel_assignment)

        # input_data = {"coords": "coords"}
        input_data = MockRubixData(
            MockStarsData(
                coords="coords",
            ),
            MockGasData(
                coords=None,
            ),
        )
        result = spaxel_assignment(input_data)

        assert result.stars.pixel_assignment == "pixel_assignment"
        assert result.stars.spatial_bin_edges == "spatial_bin_edges"
        assert result.stars.coords == "coords"

    # Test for unsupported pixel type
    with patch("rubix.core.telescope.get_telescope") as mock_get_telescope:
        mock_get_telescope.return_value.pixel_type = "unsupported"

        with pytest.raises(ValueError):
            get_spaxel_assignment(config)


@patch("rubix.core.telescope.calculate_spatial_bin_edges")
@patch("rubix.core.telescope.get_cosmology")
@patch("rubix.core.telescope.get_telescope")
def test_get_spatial_bin_edges(
    mock_get_telescope, mock_get_cosmology, mock_calculate_spatial_bin_edges
):
    config = {
        "telescope": {"name": "MUSE"},
        "galaxy": {"dist_z": 0.5},
        "cosmology": {"name": "PLANCK15"},
    }

    mock_telescope = MagicMock(fov=1.0, sbin=10)
    mock_get_telescope.return_value = mock_telescope
    mock_get_cosmology.return_value = "cosmology"
    mock_calculate_spatial_bin_edges.return_value = (
        "spatial_bin_edges",
        "spatial_bin_size",
    )

    result = get_spatial_bin_edges(config)

    mock_get_telescope.assert_called_once_with(config)
    mock_get_cosmology.assert_called_once_with(config)
    mock_calculate_spatial_bin_edges.assert_called_once_with(
        fov=1.0,
        spatial_bins=10,
        dist_z=0.5,
        cosmology="cosmology",
    )
    assert result == "spatial_bin_edges"
