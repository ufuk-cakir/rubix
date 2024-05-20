import pytest
import jax
import jax.numpy as jnp
from unittest.mock import Mock

from rubix.spectra.ifu import resample_spectrum
from rubix.core.data import reshape_array
from rubix.core.telescope import get_telescope
from rubix.core.ifu import (
    get_calculate_spectra,
    get_scale_spectrum_by_mass,
    get_resample_spectrum_vmap,
    get_resample_spectrum_pmap,
    get_doppler_shift_and_resampling,
)

# Mocking logger and necessary functions from rubix and other modules
mock_logger = Mock()
mock_get_logger = Mock(return_value=mock_logger)
mock_get_lookup_pmap = Mock(return_value=lambda metallicity, age: metallicity + age)
mock_get_telescope = Mock(return_value=Mock(wave_seq=jnp.array([4000, 5000, 6000])))
mock_get_ssp = Mock(return_value=Mock(wavelength=jnp.array([4500, 5500, 6500])))
mock_resample_spectrum = Mock(return_value=jnp.array([1, 2, 3]))

# Sample input data
sample_inputs = {
    "metallicity": jnp.array([0.1, 0.2]),
    "age": jnp.array([1, 2]),
    "mass": jnp.array([0.5, 1.0]),
    "velocities": jnp.array([[1, 2, 3], [1, 2, 3]]),
    "spectra": jnp.array([[1, 2, 3], [4, 5, 6]]),
}


# reshape sample inputs

print("Sample_inputs:")
for key in sample_inputs:
    sample_inputs[key] = reshape_array(sample_inputs[key])
    print(f"Key: {key}, shape: {sample_inputs[key].shape}")


# Sample configuration
sample_config = {
    "pipeline": {"name": "calc_ifu"},
    "logger": {
        "log_level": "DEBUG",
        "log_file_path": None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "telescope": {"name": "MUSE"},
    "cosmology": {"name": "PLANCK15"},
    "galaxy": {"dist_z": 0.1},
    "ssp": {
        "template": {"name": "BruzualCharlot2003"},
    },
}


# Sample inputs for testing
initial_spectra = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
initial_wavelengths = jnp.array([[4500.0, 5500.0, 6500.0], [4500.0, 5500.0, 6500.0]])
target_wavelength = jnp.array([4000.0, 5000.0, 6000.0])


def test_resample_spectrum_vmap():
    print("initial_spectra shape", initial_spectra.shape)
    print("initial_wavelengths shape", initial_wavelengths.shape)
    print("target_wavelength shape", target_wavelength.shape)
    resample_spectrum_vmap = get_resample_spectrum_vmap(target_wavelength)
    result_vmap = resample_spectrum_vmap(initial_spectra, initial_wavelengths)

    expected_result = jnp.stack(
        [
            resample_spectrum(
                initial_spectra[0], initial_wavelengths[0], target_wavelength
            ),
            resample_spectrum(
                initial_spectra[1], initial_wavelengths[1], target_wavelength
            ),
        ]
    )
    assert jnp.allclose(result_vmap, expected_result)


def test_resample_spectrum_pmap():
    # For pmap we need to reshape, such that first axis is the device axis
    initial_spectra = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    initial_wavelengths = jnp.array(
        [[4500.0, 5500.0, 6500.0], [4500.0, 5500.0, 6500.0]]
    )
    initial_spectra = reshape_array(initial_spectra)
    initial_wavelengths = reshape_array(initial_wavelengths)
    resample_spectrum_pmap = get_resample_spectrum_pmap(target_wavelength)
    result_pmap = resample_spectrum_pmap(initial_spectra, initial_wavelengths)

    expected_result = jnp.stack(
        [
            resample_spectrum(
                initial_spectra[0, 0], initial_wavelengths[0, 0], target_wavelength
            ),
            resample_spectrum(
                initial_spectra[0, 1], initial_wavelengths[0, 1], target_wavelength
            ),
        ]
    )
    assert jnp.allclose(result_pmap, expected_result)


@pytest.fixture
def setup_environment(monkeypatch):
    monkeypatch.setattr("rubix.core.ifu.get_logger", mock_get_logger)
    monkeypatch.setattr("rubix.core.ifu.get_lookup_pmap", mock_get_lookup_pmap)
    monkeypatch.setattr("rubix.core.ifu.get_telescope", mock_get_telescope)
    monkeypatch.setattr("rubix.core.ifu.get_ssp", mock_get_ssp)
    monkeypatch.setattr("rubix.core.ifu.resample_spectrum", mock_resample_spectrum)


def test_calculate_spectra(setup_environment):
    calculate_spectra = get_calculate_spectra(sample_config)
    result = calculate_spectra(sample_inputs.copy())
    expected_spectra = sample_inputs["metallicity"] + sample_inputs["age"]
    assert jnp.array_equal(result["spectra"], expected_spectra)


def test_scale_spectrum_by_mass(setup_environment):
    scale_spectrum_by_mass = get_scale_spectrum_by_mass(sample_config)
    result = scale_spectrum_by_mass(sample_inputs.copy())
    expected_spectra = sample_inputs["spectra"] * jnp.expand_dims(
        sample_inputs["mass"], axis=-1
    )
    assert jnp.array_equal(result["spectra"], expected_spectra)


def test_doppler_shift_and_resampling(setup_environment):
    doppler_shift_and_resampling = get_doppler_shift_and_resampling(sample_config)
    result = doppler_shift_and_resampling(sample_inputs.copy())

    assert "spectra" in result

    # check if telescope bins are correctly resamplbed

    telescope = get_telescope(sample_config)

    print("result spectra shape:", result["spectra"].shape)
    print("result spectra:", result["spectra"])
    print("Mock resampled spectrum:", mock_resample_spectrum)
    mocked_returned_spec = jnp.array([1, 2, 3])
    expected_result = jnp.array([mocked_returned_spec, mocked_returned_spec])
    assert jnp.all(result["spectra"] == reshape_array(expected_result))

    n_gpus = jax.local_device_count()
    assert result["spectra"].shape == (n_gpus, 2, 3)  # Based on mocked data
