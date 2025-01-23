import jax.numpy as jnp
import jax
from rubix.spectra.cue.grid import CueGasLookup
from rubix.logger import get_logger
from rubix.core.telescope import get_telescope
from rubix.cosmology.base import BaseCosmology
from rubix.spectra.ifu import convert_luminoisty_to_flux_gas
from rubix import config as rubix_config


def preprocess_config(config: dict):
    """
    Perform any necessary pre-processing based on the config and return
    pre-processed data that can be passed to the JAX-compiled function.
    """
    # Initialize the logger
    logger = get_logger(config.get("logger", None))

    # Retrieve the telescope configuration
    telescope = get_telescope(config)
    spatial_res = telescope.spatial_res

    # Cosmology calculations
    logger.warning("Assuming Planck Cosmology for converting from luminosity to flux.")
    cosmo = BaseCosmology(0.3089, -1.0, 0.0, 0.6774)
    observation_lum_dist = cosmo.luminosity_distance_to_z(config["galaxy"]["dist_z"])

    # Calculate the factor to convert luminosity to flux
    factor = convert_luminoisty_to_flux_gas(
        observation_lum_dist,
        observation_z=config["galaxy"]["dist_z"],
        pixel_size=spatial_res,
        CONSTANTS=rubix_config["constants"],
    )
    factor = jnp.float32(factor)

    return {
        "config": config,
        "telescope": telescope,
        "observation_lum_dist": observation_lum_dist,
        "factor": factor,
    }


def prepare_theta(config: dict, rubixdata):
    """
    Returns the theta parameters for the Cue model (Li et al. 2024) for the shape of the ionizing spectrum and the ionizing gas properties.
    The theta parameters are calculated for each gas cell with the Illustris data.
    Be aware that we are using default values for the theta parameters for the ionizing spectrum shape from the CUE github repository.
    https://github.com/yi-jia-li/cue

    Parameters:
    rubixdata (RubixData): The RubixData object containing the gas data.

    Returns:
    jnp.ndarray: The theta parameters that are the input for the Cue model.

    Cue states in their code in line.py
    Nebular Line Emission Prediction
    :param theta: nebular parameters of n samples, (n, 12) matrix
    :param gammas, log_L_ratios, log_QH, n_H, log_OH_ratio, log_NO_ratio, log_CO_ratio: 12 input parameters
    """
    logger = get_logger(config.get("logger", None))
    logger.warning(
        "Using default theta parameters for the Cue model (Li et al. 2024) for the shape of the ionizing spectrum. Ionizing gas properties are calculated for each gas cell with the Illustris data."
    )
    alpha_HeII = jnp.full(len(rubixdata.gas.mass), 21.5)
    alpha_OII = jnp.full(len(rubixdata.gas.mass), 14.85)
    alpha_HeI = jnp.full(len(rubixdata.gas.mass), 6.45)
    alpha_HI = jnp.full(len(rubixdata.gas.mass), 3.15)
    log_OII_HeII = jnp.full(len(rubixdata.gas.mass), 4.55)
    log_HeI_OII = jnp.full(len(rubixdata.gas.mass), 0.7)
    log_HI_HeI = jnp.full(len(rubixdata.gas.mass), 0.85)
    # log_QH = rubixdata.gas.electron_abundance
    n_H = jnp.full(len(rubixdata.gas.mass), 10**2.5)  # rubixdata.gas.density
    # n_H = jnp.full(len(rubixdata.gas.mass), 10**2.5)
    OH_ratio = rubixdata.gas.metals[:, 4] / rubixdata.gas.metals[:, 0]
    NO_ratio = rubixdata.gas.metals[:, 3] / rubixdata.gas.metals[:, 4]
    CO_ratio = rubixdata.gas.metals[:, 2] / rubixdata.gas.metals[:, 4]

    log_oh_sol = -3.07
    log_co_sol = -0.37
    log_no_sol = -0.88

    oh_factor = 16 / 1
    co_factor = 12 / 16
    no_factor = 14 / 16

    final_log_oh = jnp.log10(OH_ratio * oh_factor) / log_oh_sol
    final_log_co = jnp.log10(CO_ratio * co_factor) / 10**log_co_sol
    final_log_no = jnp.log10(NO_ratio * no_factor) / 10**log_no_sol
    log_QH = jnp.full(len(rubixdata.gas.mass), 49.58)

    log_OH_ratio = jnp.full(len(rubixdata.gas.mass), -0.85)
    log_NO_ratio = jnp.full(len(rubixdata.gas.mass), -0.134)
    log_CO_ratio = jnp.full(len(rubixdata.gas.mass), -0.134)

    theta = [
        alpha_HeII,
        alpha_OII,
        alpha_HeI,
        alpha_HI,
        log_OII_HeII,
        log_HeI_OII,
        log_HI_HeI,
        log_QH,
        n_H,
        # final_log_oh,
        # final_log_no,
        # final_log_co,
        final_log_oh,
        final_log_no,
        final_log_co,
    ]
    theta = jnp.transpose(jnp.array(theta))
    logger.debug(f"theta: {theta.shape}")
    logger.debug(f"theta: {theta}")
    return theta


def preprocess_cue(config: dict):
    """
    Perform any necessary pre-processing based on the config and return
    pre-processed data that can be passed to the JAX-compiled function.
    """
    pre_config = preprocess_config(config)
    CueClass = CueGasLookup(pre_config)

    # Perform any necessary lookup and pre-processing steps here
    preprocessed_lookup_data = CueClass  # For now, just passing the class itself
    return preprocessed_lookup_data


def get_gas_emission(config: dict):

    if "gas" not in config["data"]["args"]["particle_type"]:
        raise ValueError("No gas particle data in this galaxy")

    if "gas" not in config["data"]["args"]["cube_type"]:
        raise ValueError("Not specified in the config to calculate gas emission cube")

    logger = get_logger()

    # Perform the pre-processing step outside of the returned function
    preprocessed_lookup_data = preprocess_cue(config)

    def gas_emission(rubixdata: object) -> object:
        """Calculate gas emission lines and gas continuum emission."""
        logger.info("Calculation gas emission...")

        theta = prepare_theta(config, rubixdata)
        # Use the pre-processed lookup data here
        logger.debug("Calculating gas emission")
        logger.debug("theta: %s", theta.shape)
        # logger.debug("theta1 %s", theta[1])
        logger.debug("internal_energy: %s", rubixdata.gas.internal_energy.shape)
        logger.debug("electron_abundance: %s", rubixdata.gas.electron_abundance.shape)
        # wave, flux = preprocessed_lookup_data.get_gas_emission_flux(theta, rubixdata.gas.internal_energy, rubixdata.gas.electron_abundance)
        # rubixdata = jax.block_until_ready(rubixdata)
        # rubixdata = jax.device_put(rubixdata)

        """
        # Define the vectorized function
        vectorized_get_gas_emission_flux = jax.vmap(
            preprocessed_lookup_data.get_gas_emission_flux,
            in_axes=(0, 0, 0)  # Vectorize over the first dimension of each input
        )

        # Apply the vectorized function to the inputs
        wave, flux = vectorized_get_gas_emission_flux(
            theta,
            rubixdata.gas.internal_energy,
            rubixdata.gas.electron_abundance
        )

        rubixdata.gas.wave = wave
        rubixdata.gas.spectra = flux
        """
        # Initialize the output arrays
        num_particles = theta.shape[0]
        wave_list = []
        flux_list = []

        # par = [[21.5, 14.85, 6.45, 3.15, 4.55, 0.7, 0.85, 49.58, 10**2.5, -0.85, -0.134, -0.134]]
        # par = jnp.array(par)

        # Loop over the particles
        for i in range(num_particles):
            par = theta[i]
            par = jnp.array(par)
            par = jnp.reshape(par, (1, 12))
            wave, flux = preprocessed_lookup_data.get_gas_emission_flux(
                par,
                rubixdata.gas.internal_energy[i] * 1e-12,
                rubixdata.gas.electron_abundance[i],
            )
            wave_list.append(wave)
            flux_list.append(flux)
            logger.debug("flux: %s", flux)
        # Stack the results
        wave = wave_list[0]
        flux = jnp.stack(flux_list)

        rubixdata.gas.wave = wave
        rubixdata.gas.spectra = flux

        logger.debug("Completed gas emission calculation: %s", rubixdata)
        # logger.debug(
        #    "test core module: temperature: %s", jnp.array(rubixdata.gas.temperature)
        # )
        # logger.debug(
        #    "test core module: continuum: %s", jnp.array(rubixdata.gas.continuum)
        # )
        # logger.debug(
        #    "test core module: emission: %s", jnp.array(rubixdata.gas.emission_spectra)
        # )
        return rubixdata

    return gas_emission
