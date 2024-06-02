from rubix.telescope.psf.psf import get_psf_kernel, apply_psf

from typing import Callable, Dict
import jax.numpy as jnp


def get_convolve_psf(config: dict) -> Callable:
    """Get the point spread function (PSF) kernel based on the configuration."""
    if config["telescope"]["psf"]["name"] == "gaussian":
        m, n = config["telescope"]["psf"]["size"], config["telescope"]["psf"]["size"]
        sigma = config["telescope"]["psf"]["sigma"]
        psf_kernel = get_psf_kernel("gaussian", m, n, sigma=sigma)

    else:
        raise ValueError(
            f"Unknown PSF kernel name: {config['telescope']['psf']['name']}"
        )

    # Define the function to convolve the datacube with the PSF kernel
    def convolve_psf(input: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """Convolve the input datacube with the PSF kernel."""
        input["datacube"] = apply_psf(input["datacube"], psf_kernel)
        return input

    return convolve_psf
