import jax
import jax.numpy as jnp

from rubix.spectra.ifu import resample_spectrum
from rubix.core.data import reshape_array
from rubix.core.ssp import get_ssp
from rubix.core.ifu import (
    get_calculate_spectra,
    get_scale_spectrum_by_mass,
    get_resample_spectrum_vmap,
    get_resample_spectrum_pmap,
    get_doppler_shift_and_resampling,
)

RTOL = 1e-5
ATOL = 1e-6
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


class MockRubixData:
    def __init__(self, stars, gas):
        self.stars = stars
        self.gas = gas


class MockStarsData:
    def __init__(self, velocity, metallicity, mass, age, spectra=None):
        # self.coords = coords
        self.velocity = velocity
        self.metallicity = metallicity
        self.mass = mass
        self.age = age
        # self.pixel_assignment = pixel_assignment
        self.spectra = spectra


class MockGasData:
    def __init__(self, spectra):
        self.spectra = None


# Sample inputs for testing
initial_spectra = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
initial_wavelengths = jnp.array([[4500.0, 5500.0, 6500.0], [4500.0, 5500.0, 6500.0]])
target_wavelength = jnp.array([4000.0, 5000.0, 6000.0])


def _get_sample_inputs(subset=None):
    ssp = get_ssp(sample_config)
    """metallicity = reshape_array(ssp.metallicity)
    age = reshape_array(ssp.age)
    spectra = reshape_array(ssp.flux)"""
    metallicity = ssp.metallicity
    age = ssp.age
    spectra = ssp.flux

    print("Metallicity shape: ", metallicity.shape)
    print("Age shape: ", age.shape)
    print("Spectra shape: ", spectra.shape)
    print(".............")

    import numpy as np

    # Create meshgrid for metallicity and age to cover all combinations
    metallicity_grid, age_grid = np.meshgrid(
        metallicity.flatten(), age.flatten(), indexing="ij"
    )
    metallicity_grid = reshape_array(metallicity_grid.flatten())
    age_grid = reshape_array(age_grid.flatten())
    print("Metallicity grid shape: ", metallicity_grid.shape)
    print("Age grid shape: ", age_grid.shape)

    spectra = spectra.reshape(-1, spectra.shape[-1])
    print("spectra after reshape: ", spectra.shape)
    spectra = reshape_array(spectra)

    print("spectra after reshape_array call: ", spectra.shape)

    # reshape spectra
    num_combinations = metallicity_grid.shape[1]
    spectra_reshaped = spectra.reshape(
        spectra.shape[0], num_combinations, spectra.shape[-1]
    )

    # Create Velocities for each combination

    velocities = jnp.ones((metallicity_grid.shape[0], num_combinations, 3))
    mass = jnp.ones_like(metallicity_grid)

    if subset is not None:
        metallicity_grid = metallicity_grid[:, :subset]
        age_grid = age_grid[:, :subset]
        velocities = velocities[:, :subset]
        mass = mass[:, :subset]
        spectra_reshaped = spectra_reshaped[:, :subset]
    # inputs = dict(
    #    metallicity=metallicity_grid, age=age_grid, velocities=velocities, mass=mass
    # )
    inputs = MockRubixData(
        MockStarsData(
            velocity=velocities,
            metallicity=metallicity_grid,
            mass=mass,
            age=age_grid,
        ),
        MockGasData(spectra=None),
    )
    return inputs, spectra_reshaped


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
    assert not jnp.any(jnp.isnan(result_vmap))


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

    # Check how many GPUs are available, since this defines the shape of the result
    if jax.device_count() > 1:
        expected_result = jnp.array(
            [
                resample_spectrum(
                    initial_spectra[0, 0], initial_wavelengths[0, 0], target_wavelength
                ),
                resample_spectrum(
                    initial_spectra[1, 0], initial_wavelengths[1, 0], target_wavelength
                ),
            ]
        )
        expected_result = reshape_array(expected_result)

    else:
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
    assert not jnp.any(jnp.isnan(result_pmap))


def test_calculate_spectra():
    calculate_spectra = get_calculate_spectra(sample_config)

    inputs, expected_spectra = _get_sample_inputs()
    result = calculate_spectra(inputs)  # type: ignore
    print("Calculataed Spectra shape:", result.stars.spectra.shape)
    print("Expected Spectra shape:", expected_spectra.shape)

    # print the first 5 values of the calculated and expected spectra
    print("Calculated Spectra:", result.stars.spectra[:5])
    print("Expected Spectra:", expected_spectra[:5])

    is_close = jnp.isclose(
        result.stars.spectra, expected_spectra, rtol=RTOL, atol=ATOL
    )  # noqa

    print("N_close:", jnp.sum(is_close))
    print("N_total:", len(is_close.flatten()))

    # check where it is not close to the expected spectra
    not_close_indices = jnp.where(~is_close)
    # Get the difference between the calculated and expected spectra
    diff = jnp.abs(result.stars.spectra - expected_spectra)
    print("Difference of not close index:", diff[not_close_indices])
    print("sum of difference of not close index:", jnp.sum(diff[not_close_indices]))
    assert jnp.isclose(
        result.stars.spectra, expected_spectra, rtol=RTOL, atol=ATOL
    ).all()
    assert not jnp.any(jnp.isnan(result.stars.spectra))


def test_scale_spectrum_by_mass():
    input = MockRubixData(
        MockStarsData(
            velocity=sample_inputs["velocities"],
            metallicity=sample_inputs["metallicity"],
            mass=sample_inputs["mass"],
            age=sample_inputs["age"],
            spectra=sample_inputs["spectra"],
        ),
        MockGasData(spectra=None),
    )
    expected_spectra = input.stars.spectra * jnp.expand_dims(input.stars.mass, axis=-1)
    scale_spectrum_by_mass = get_scale_spectrum_by_mass(sample_config)
    result = scale_spectrum_by_mass(input)

    # Print the values for debugging
    print("Input Mass:", input.stars.mass)
    print("Input Spectra:", input.stars.spectra)
    print("Result Spectra:", result.stars.spectra)
    print("Expected Spectra:", expected_spectra)
    assert jnp.array_equal(result.stars.spectra, expected_spectra)
    assert not jnp.any(jnp.isnan(result.stars.spectra))


def test_doppler_shift_and_resampling():
    doppler_shift_and_resampling = get_doppler_shift_and_resampling(sample_config)
    inputs, expected_spectra = _get_sample_inputs(subset=10)
    inputs.stars.spectra = expected_spectra
    result = doppler_shift_and_resampling(inputs)  # type: ignore

    assert hasattr(result.stars, "spectra")
    assert not jnp.any(jnp.isnan(result.stars.spectra))
