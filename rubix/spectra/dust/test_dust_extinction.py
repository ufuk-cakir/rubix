import pytest
from rubix.core.data import RubixData
from rubix.spectra.dust.dust_extinction import efficient_spaxel_extinction

import jax.numpy as jnp

@pytest.fixture
def mock_config():
    return {
        "logger": None,
        "constants": {
            "MASS_OF_PROTON": 1.6726219e-24,
            "MSUN_TO_GRAMS": 1.989e33,
            "KPC_TO_CM": 3.086e21
        },
        "ssp": {
            "dust": {
                "dust_grain_density": 3.0,
                "extinction_model": "Cardelli89",
                "Rv": 3.1,
                "dust_to_gas_model": "power law slope free",
                "Xco": "MW"
            }
        }
    }

@pytest.fixture
def mock_rubixdata():
    class MockGas:
        def __init__(self):
            self.coords = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            self.pixel_assignment = jnp.array([[0, 1]])
            self.metals = jnp.array([[[0.01, 0.02, 0.03, 0.04, 0.05], [0.06, 0.07, 0.08, 0.09, 0.1]]])
            self.mass = jnp.array([[1.0, 2.0]])

    class MockStars:
        def __init__(self):
            self.coords = jnp.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])
            self.pixel_assignment = jnp.array([[0, 1]])
            self.mass = jnp.array([[1.0, 2.0]])
            self.spectra = jnp.array([[[1.0, 2.0], [3.0, 4.0]]])

    class MockRubixData:
        def __init__(self):
            self.gas = MockGas()
            self.stars = MockStars()

    return MockRubixData()

def test_efficient_spaxel_extinction(mock_config, mock_rubixdata):
    wavelength = jnp.array([5000.0, 6000.0])
    n_spaxels = 2
    spaxel_area = jnp.array([1.0, 1.0])

    result = efficient_spaxel_extinction(mock_config, mock_rubixdata, wavelength, n_spaxels, spaxel_area)

    assert result.shape == (1, 2, 2)
    assert jnp.all(result >= 0)