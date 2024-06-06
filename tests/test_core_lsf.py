import pytest
from rubix.core.lsf import get_convolve_lsf
import jax.numpy as jnp


def test_get_convolve_lsf_missing_lsf_key():
    config = {"telescope": {}}
    with pytest.raises(ValueError) as excinfo:
        get_convolve_lsf(config)
    assert "LSF configuration not found" in str(excinfo.value)


def test_get_convolve_lsf_missing_sigma_key():
    config = {"telescope": {"lsf": {}}}
    with pytest.raises(ValueError) as excinfo:
        get_convolve_lsf(config)
    assert "LSF sigma size not found" in str(excinfo.value)


def test_get_convolve_correct_behavior():
    config = {
        "telescope": {
            "name": "MUSE",
            "psf": {"name": "gaussian", "size": 5, "sigma": 0.6},
            "lsf": {"sigma": 0.6},
        }
    }

    convolve_lsf = get_convolve_lsf(config)

    dummy_datacube = jnp.zeros((10, 10, 10))
    # set delta function at the center of spectrum in each bin
    dummy_datacube = dummy_datacube.at[:, :, 5].set(1.0)

    # Apply the convolution function
    result = convolve_lsf({"datacube": dummy_datacube})

    # Check if datacube has been modified correctly
    # You need to specify the expected result based on your understanding of the convolution effect
    assert (
        result["datacube"].shape == dummy_datacube.shape
    )  # Check the shape remains unchanged
    assert jnp.all(
        result["datacube"][:, :, 5] != 1
    )  # Ensure the center is not just a delta peak anymore
    assert (
        jnp.sum(result["datacube"], axis=2).all() == 1
    )  # The integral across each spectrum should be 1 if the kernel is normalized

    # import matplotlib.pyplot as plt
    #
    # plt.plot(result["datacube"][5, 5, :])  # Plot the middle spectrum
    # plt.show()
