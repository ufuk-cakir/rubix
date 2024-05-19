import jax.numpy as jnp
import jax
from rubix import config
from jaxtyping import Float, Array


def convert_luminoisty_to_flux(
    luminosity,
    observation_lum_dist,
    observation_z,
    pixel_size,
    CONSTANTS=config["constants"],
):
    """Convert luminosity to flux in units erg/s/cm^2/Angstrom as observed by the telescope"""
    CONST = CONSTANTS.get("LSOL_TO_ERG") / CONSTANTS.get("MPC_TO_CM") ** 2
    FACTOR = (
        CONST
        / (4 * jnp.pi * observation_lum_dist**2)
        / (1 + observation_z)
        / pixel_size
    )
    spectral_dist = luminosity * FACTOR
    return spectral_dist


def cosmological_doppler_shift(
    z: float, wavelength: Float[Array, " n_bins"]
) -> Float[Array, " n_bins"]:
    """Calculate the cosmological Doppler shift of a wavelength.
    Parameters
    ----------
    z : float
        The redshift.
    wavelength : float
        The wavelength in Angstrom.
    Returns
    -------
    float
        The Doppler shifted wavelength in Angstrom.
    """
    # Calculate the cosmological Doppler shift of a wavelength
    return (1 + z) * wavelength


def calculate_diff(vec, pad_with_zero=True):
    """Calculate the difference between each element in a vector.

    Parameters
    ----------
    vec : array-like
        The input vector.

    pad_with_zero : bool, optional
        Whether to prepend the first element of the vector to the differences.
        Default is True.
    """

    if pad_with_zero:
        differences = jnp.diff(vec, prepend=vec[0])
    else:
        differences = jnp.diff(vec)
    return differences


def _get_velocity_component_single(vec: Float[Array, "3"], direction: str) -> Float:
    # Check that vec is of size 3
    if not vec.size == 3:
        raise ValueError(f"Expected vector of size 3, but got {vec.size}.")

    if direction == "x":
        return vec[0]
    elif direction == "y":
        return vec[1]
    elif direction == "z":
        return vec[2]

    else:
        raise ValueError(
            f"{direction} is not a valid direction. Supported directions are 'x', 'y', or 'z'."
        )


def _get_velocity_component_multiple(
    vecs: Float[Array, "n_particles 3"], direction: str
) -> Float[Array, "n_particles 1"]:
    # Check that vecs has shape (n_particles, 3)
    if vecs.shape[1] != 3:
        raise ValueError(
            f"Expected vectors of shape (n_particles, 3), but got {vecs.shape}."
        )

    if direction == "x":
        return vecs[:, 0]
    elif direction == "y":
        return vecs[:, 1]
    elif direction == "z":
        return vecs[:, 2]
    else:
        raise ValueError(
            f"{direction} is not a valid direction. Supported directions are 'x', 'y', or 'z'."
        )


def get_velocity_component(
    vec: Float[Array, " * 3"], direction: str
) -> Float[Array, " * 1"]:
    if isinstance(vec, jax.Array) and vec.ndim == 2:
        return _get_velocity_component_multiple(vec, direction)
    elif isinstance(vec, jax.Array) and vec.ndim == 1:
        return _get_velocity_component_single(vec, direction)
    else:
        raise ValueError("Got wrong shapes.")


def _velocity_doppler_shift_single(
    wavelength: Float[Array, " n_bins"],
    velocity: Float[Array, "3"],
    direction="y",
    SPEED_OF_LIGHT=config["constants"]["SPEED_OF_LIGHT"],
) -> Float[Array, " n_bins"]:
    """Calculate the Doppler shift of a wavelength due to a velocity.

    Parameters
    ----------
    wavelength : float
        The wavelength in Angstrom.
    velocity : float
        The velocity in km/s.

    Returns
    -------
    float
        The Doppler shifted wavelength in Angstrom.

    """
    velocity = get_velocity_component(velocity, direction)
    # Calculate the Doppler shift of a wavelength due to a velocity
    return wavelength * jnp.exp(velocity / SPEED_OF_LIGHT)


def velocity_doppler_shift(
    wavelength: Float[Array, " n_bins"],
    velocity: Float[Array, " * 3"],
    direction="y",
    SPEED_OF_LIGHT=config["constants"]["SPEED_OF_LIGHT"],
) -> Float[Array, " n_bins"]:
    """Calculate the Doppler shift of a wavelength due to a velocity.

    Parameters
    ----------
    wavelength : float
        The wavelength in Angstrom.
    velocity : float
        The velocity in km/s.

    Returns
    -------
    float
        The Doppler shifted wavelength in Angstrom.

    """
    # Vmap the function to handle multiple velocities with the same wavelength
    return jax.vmap(
        lambda v: _velocity_doppler_shift_single(
            wavelength, v, direction, SPEED_OF_LIGHT
        )
    )(velocity)


def resample_spectrum(
    initial_spectrum: Float[Array, " n_bins_initial"],
    initial_wavelength: Float[Array, " n_bins_initial"],
    target_wavelength: Float[Array, " n_bins_target"],
) -> Float[Array, " n_bins_target"]:
    """Resample a spectrum to the wavelength grid of a telescope.
    Parameters
    ----------
    initial_spectrum : array-like
        The initial spectrum.
    initial_wavelength : array-like
        The initial wavelength grid.
    target_wavelength : array-like
        The target wavelength grid.

    Returns
    -------
    array-like
        The resampled spectrum.
    """
    # Get wavelengths inside the telescope range
    in_range_mask = (initial_wavelength >= jnp.min(target_wavelength)) & (
        initial_wavelength <= jnp.max(target_wavelength)
    )
    intrinsic_wave_diff = calculate_diff(initial_wavelength) * in_range_mask

    # Get total luminsoity within the wavelength range
    total_lum = jnp.sum(initial_spectrum * intrinsic_wave_diff)

    # Interpolate the wavelegnth to the telescope grid
    particle_lum = jnp.interp(target_wavelength, initial_wavelength, initial_spectrum)

    # New total luminosity
    new_total_lum = jnp.sum(particle_lum * calculate_diff(target_wavelength))

    # Factor to conserve flux in the new spectrum
    scale_factor = total_lum / new_total_lum

    lum = particle_lum * scale_factor

    return lum
