'''import pytest
import h5py
import numpy as np
from rubix.utils import (
    _get_quantity_from_hdf5,
    _get_fields_with_units,
    get_meta_from_hdf5,
    load_stars_data,
)
from unxt import Quantity


def create_mock_hdf5(tmp_path):
    # Create a mock HDF5 file structure
    hdf5_file = tmp_path / "mock_data.h5"
    with h5py.File(hdf5_file, "w") as f:
        group = f.create_group("particles/stars")
        ds = group.create_dataset("coords", data=np.array([[1, 2, 3], [-1, -2, -3]]))
        ds.attrs["unit"] = "kpc"
        ds = group.create_dataset("velocity", data=np.array([[-1, 0, 1], [-1, -2, -3]]))
        ds.attrs["unit"] = "km/s"

        meta_group = f.create_group("meta")
        meta_group.create_dataset("age", data=13.8)
    return hdf5_file


@pytest.fixture
def mock_hdf5_file(tmp_path):
    return create_mock_hdf5(tmp_path)


def test_get_quantity_from_hdf5(mock_hdf5_file):
    with h5py.File(mock_hdf5_file, "r") as f:
        quantity = _get_quantity_from_hdf5(f, "particles/stars/coords")
        assert np.array_equal(
            quantity[0].value, np.array([1, 2, 3])
        ), "Arrays are not equal"


def test_get_wrong_key_from_hdf5(mock_hdf5_file):
    with h5py.File(mock_hdf5_file, "r") as f:
        with pytest.raises(
            KeyError, match="Key particles/stars/coords_wrong not found in HDF5 file"
        ):
            _get_quantity_from_hdf5(f, "particles/stars/coords_wrong")


def test_get_fields_with_units(mock_hdf5_file):
    with h5py.File(mock_hdf5_file, "r") as f:
        fields = _get_fields_with_units(f, "particles/stars")
        assert len(fields) == 2
        assert all(isinstance(field, Quantity) for field in fields.values())


def test_get_meta_from_hdf5(mock_hdf5_file):
    with h5py.File(mock_hdf5_file, "r") as f:
        meta = get_meta_from_hdf5(f)
        assert meta["age"] == 13.8


def test_load_stars_data(mock_hdf5_file):
    # Prepare mock return values that reflect the expected dimensions
    with h5py.File(mock_hdf5_file, "r") as f:
        data, is_centered = load_stars_data(f, np.array([0.5, 0.5, 0.5]), False)
        assert is_centered, "Data should be marked as centered"
        # check if they are centered
        assert np.array_equal(
            data["coords"][0], np.array([0.5, 1.5, 2.5])
        ), "Arrays are not equal"


def test_load_stars_data_centered(mock_hdf5_file):
    # Prepare mock return values that reflect the expected dimensions
    with h5py.File(mock_hdf5_file, "r") as f:
        data, is_centered = load_stars_data(f, np.array([0, 0, 0]), True)
        assert is_centered, "Data should be marked as centered"
        assert np.array_equal(
            data["coords"][0].value, np.array([1, 2, 3])
        ), "Arrays are not equal"
'''