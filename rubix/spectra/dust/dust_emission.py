from typing import Tuple

import equinox as eqx
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
from astropy.io import fits
from beartype import beartype as typechecker
from jaxtyping import Array, Float, jaxtyped

from rubix.logger import get_logger

# Values of Umin for each of the Draine + Li (2007) dust emission grids.
umin_vals = jnp.array(
    [
        0.10,
        0.15,
        0.20,
        0.30,
        0.40,
        0.50,
        0.70,
        0.80,
        1.00,
        1.20,
        1.50,
        2.00,
        2.50,
        3.00,
        4.00,
        5.00,
        7.00,
        8.00,
        10.0,
        12.0,
        15.0,
        20.0,
        25.0,
    ]
)

# Values of qpah for each of the Draine + Li (2007) dust emission grids.
qpah_vals = jnp.array(
    [0.10, 0.47, 0.75, 1.12, 1.49, 1.77, 2.37, 2.50, 3.19, 3.90, 4.58]
)


@jaxtyped(typechecker=typechecker)
def load_dust_emission_grids(
    grid_dir: str,
):  # -> Tuple[Array[Float, "n_wavelengths"], Array[Float, "n_wavelengths"]]:
    """Load the dust emission grids from Draine + Li (2007)."""

    _logger = get_logger()

    try:
        # Draine + Li (2007) dust emission grids, stored as a FITS HDUList.
        dust_grid_umin_only = [
            jnp.asarray(
                fits.open(grid_dir + "/dl07_grids_umin_only.fits")[i].data,
                dtype=jnp.float32,
            )
            for i in range(len(qpah_vals) + 1)
        ]

        dust_grid_umin_umax = [
            jnp.asarray(
                fits.open(grid_dir + "/dl07_grids_umin_umax.fits")[i].data,
                dtype=jnp.float32,
            )
            for i in range(len(qpah_vals) + 1)
        ]

    except IOError as e:
        _logger.warning(f"[Dust emission] Error: {e}")
        _logger.warning(
            f"Failed to load dust emission grids, these should be placed in the {grid_dir} directory."
        )
        raise FileNotFoundError(
            f"Could not find the dust emission grids in the {grid_dir} directory."
        )

    return dust_grid_umin_only, dust_grid_umin_umax


def weighted_interp(qpah_fact, higher_slice, lower_slice):
    """Perform weighted interpolation using JAX's vectorized operations."""
    return qpah_fact * higher_slice + (1 - qpah_fact) * lower_slice


@jaxtyped(typechecker=typechecker)
class DustEmission(eqx.Module):
    """
    Dust emission model from Draine + Li (2007). This is a simple
    class that allows access to the dust emission models of Draine + Li
    (2007). Currently very simple, possibly could be sped up in some
    circumstances by pre-interpolating to the input wavs.
    Inspiration taken from the bagpipes implementation of DL07.
    See https://github.com/ACCarnall/bagpipes/blob/master/bagpipes/models/dust_emission_model.py
    for more details.

    Parameters
    ----------

    wavelengths :
        1D array of wavelength values desired for the DL07 models.
    """

    wavelengths: Float[Array, "n_wavelengths"]

    def spectrum(
        self, dust_emission_grid_path: str, qpah: float, umin: float, gamma: float
    ) -> Float[Array, "n_wavelengths"]:
        """Get the 1D spectrum for a given set of model parameters.

        Parameters
        ----------
        dust_emission_grid_path : str
            The path to the dust emission grid files
        qpah : float
            The PAH fraction in the dust model. Must be between 0 and 4.
        umin : float
            The minimum radiation field strength in the dust model. Must be between 0 and 25.
        gamma : float
            The fraction of the maximum radiation field strength in the dust model.
            Must be between 0 and 1.
        """

        qpah_ind = jnp.sum(qpah_vals < qpah)
        umin_ind = jnp.sum(umin_vals < umin)

        qpah_fact = (qpah - qpah_vals[qpah_ind - 1]) / (
            qpah_vals[qpah_ind] - qpah_vals[qpah_ind - 1]
        )

        umin_fact = (umin - umin_vals[umin_ind - 1]) / (
            umin_vals[umin_ind] - umin_vals[umin_ind - 1]
        )

        umin_w = jnp.array([(1 - umin_fact), umin_fact])

        dust_grid_umin_only, dust_grid_umin_umax = load_dust_emission_grids(
            dust_emission_grid_path
        )

        # Get the grids at the relevant indices
        lqpah_only = dust_grid_umin_only[qpah_ind]
        hqpah_only = dust_grid_umin_only[qpah_ind + 1]

        # Use dynamic_slice for better JAX compilation
        # Get first dimension size for dynamic slicing
        rows = lqpah_only.shape[0]

        # Dynamic slice for the lower qpah grid
        lqpah_slice_only = lax.dynamic_slice(lqpah_only, (0, umin_ind), (rows, 2))

        # Dynamic slice for the higher qpah grid
        hqpah_slice_only = lax.dynamic_slice(hqpah_only, (0, umin_ind), (rows, 2))

        # Perform the weighted interpolation
        tqpah_only = weighted_interp(qpah_fact, hqpah_slice_only, lqpah_slice_only)

        # Same approach for the umax grids
        lqpah_umax = dust_grid_umin_umax[qpah_ind]
        hqpah_umax = dust_grid_umin_umax[qpah_ind + 1]

        # Use dynamic slicing for umax grids
        rows_umax = lqpah_umax.shape[0]
        lqpah_slice_umax = lax.dynamic_slice(lqpah_umax, (0, umin_ind), (rows_umax, 2))

        hqpah_slice_umax = lax.dynamic_slice(hqpah_umax, (0, umin_ind), (rows_umax, 2))

        tqpah_umax = weighted_interp(qpah_fact, hqpah_slice_umax, lqpah_slice_umax)

        # Use vectorized operations for weighted sums
        interp_only = jnp.sum(umin_w * tqpah_only, axis=1)
        interp_umax = jnp.sum(umin_w * tqpah_umax, axis=1)

        # Final model combines the two interpolated spectra
        model = gamma * interp_umax + (1 - gamma) * interp_only

        # Create the final interpolated spectrum
        wavelength_grid = dust_grid_umin_only[1][:, 0]

        # Interpolate to the desired wavelength grid
        spectrum = jnp.interp(
            self.wavelengths,
            wavelength_grid,
            model,
            left=0.0,
            right=0.0,
        )

        return spectrum
