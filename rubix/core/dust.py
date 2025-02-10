import jax.numpy as jnp 
import jax

from rubix.logger import get_logger
from .data import RubixData
from rubix.spectra.dust.dust_extinction import apply_spaxel_extinction
from .telescope import get_telescope
from rubix.telescope.utils import calculate_spatial_bin_edges
from rubix.core.cosmology import get_cosmology

from typing import Callable
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

@jaxtyped(typechecker=typechecker)
def dust_to_gas_ratio(gas_metallicity: Float[Array, "n_gas"], model: str, Xco: str) -> Float[Array, "n_gas"]:
    """
    Calculate the dust_to_gas ratio following the empirical relations from Remy-Ruyer et al. 2014.
    We use the fitting formula from table 1.

    Parameters
    ----------
    gas_metallicity : Float[Array, "n_gas"]
        The metallicity of the gas cells. Remy-Ruyer et al. 2014 use 12 + log(O/H) as a proxy for metallicity.

    model : str
        The model to use for the gas-to-dust ratio as specified in Table 1 of Remy-Ruyer et al. 2014. Options are: 
        - power law fixed slope
        - power law free slope
        - broken power law fit
    Returns
    -------
    Float[Array, "n_gas"]
        The gas-to-dust ratio for each gas cell. 
    """

    x_sol = 8.69  # solar oxygen abundance from Asplund et al. 2009

    #convert the gas gas_metallicity to 12 + log(O/H)
    # TODO!!!

    if Xco == "MW":
        if model == "power law slope fixed":
            raise NotImplementedError("power law slope fixed not implemented yet.")
        elif model == "power law slope free":
            # power law slope fixed
            # log(D/G) = a + b * log(O/H)
            alpha = 1.62
            a = 2.21
            dust_to_gas_ratio = 1 / 10**(a + alpha * (x_sol - gas_metallicity))
        elif model == "broken power law fit":
            # broken power law fit
            # log(D/G) = a + b * log(O/H) for log(O/H) < 8.4
            # log(D/G) = c + d * log(O/H) for log(O/H) >= 8.4
            a = 2.21
            alpha_h = 1.00
            b = 0.68
            alpha_l = 3.08
            x_transition = 7.96
            dust_to_gas_ratio = 1 / jnp.where(gas_metallicity > x_transition, 10**(a + alpha_h * (x_sol - gas_metallicity)), 10**(b + alpha_l * (x_sol - gas_metallicity)))
    elif Xco == "Z":
        if model == "power law slope fixed":
            raise NotImplementedError("power law slope fixed not implemented yet.")
        elif model == "power law slope free":
            # power law slope fixed
            # log(D/G) = a + b * log(O/H)
            alpha = 2.02
            a = 2.21
            dust_to_gas_ratio = 1 / 10**(a + alpha * (x_sol - gas_metallicity))
        elif model == "broken power law fit":
            # broken power law fit
            # log(D/G) = a + b * log(O/H) for log(O/H) < 8.4
            # log(D/G) = c + d * log(O/H) for log(O/H) >= 8.4
            a = 2.21
            alpha_h = 1.00
            b = 0.96
            alpha_l = 3.10
            x_transition = 8.10
            dust_to_gas_ratio = 1 / jnp.where(gas_metallicity > x_transition, 10**(a + alpha_h * (x_sol - gas_metallicity)), 10**(b + alpha_l * (x_sol - gas_metallicity)))
    
    return dust_to_gas_ratio


@jaxtyped(typechecker=typechecker)
def _apply_spaxel_extinction(config: dict, rubixdata: RubixData, wavelength: Float[Array, "n_wave"], spaxel_area: Float[Array, "..."], spaxel_idx: Array) -> Float[Array, "m n_stars n_wave"]: #RubixData:
    """
    Apply the dust extinction to the spaxel data using vmap.

    Parameters
    ----------
    rubixdata : RubixData
        The RubixData object containing the spaxel data.
    spaxel_idx : int
        The index of the spaxel to apply extinction to.

    Returns
    -------
    RubixData
        The RubixData object with the dust extinction applied to the spaxel data.
    """
    logger = get_logger(config.get("logger", None))
    logger.info("Applying dust extinction to the spaxel data using vmap...")

    #jax.debug.print("spaxel assignment for gas: {rubixdata.gas.pixel_assignment}", rubixdata=rubixdata)
    #jax.debug.print("gas mask for spaxel: {gas_mask}", gas_mask=rubixdata.gas.pixel_assignment == spaxel_idx)

    dust_to_gas_ratio = dust_to_gas_ratio(rubixdata.gas.metallicity, config["ssp"]["dust"]["dust_to_gas_model"], config["ssp"]["dust"]["Xco"])
    dust_mass = rubixdata.gas.mass * dust_to_gas_ratio

    # get the data for spaxel spaxel_idx
    gas_mask = rubixdata.gas.pixel_assignment == spaxel_idx
    star_mask = rubixdata.stars.pixel_assignment == spaxel_idx

    #gas_density = jnp.where(gas_mask, rubixdata.gas.density, 0)
    
    #jax.debug.print("gas densities: {gas_density}", gas_density=gas_density)

    dust_mass = jnp.where(gas_mask, dust_mass, 0)
    dust_z_pos = jnp.where(gas_mask, rubixdata.gas.coords[:,:,2], 0)
    
    star_z_pos = jnp.where(star_mask, rubixdata.stars.coords[:,:,2], 0)
    ssp_template_fluxes = jnp.where(star_mask[:, :, None], rubixdata.stars.spectra, 0)

    #rubixdata.stars.spectra = jnp.where(star_mask[:, :, None], apply_spaxel_extinction(config, gas_density[0], gas_z_pos[0], star_z_pos[0], ssp_template_fluxes[0], wavelength), rubixdata.stars.spectra)

    tmp_spectra = jnp.where(star_mask[:, :, None], apply_spaxel_extinction(config, dust_mass[0], dust_z_pos[0], star_z_pos[0], ssp_template_fluxes[0], wavelength, spaxel_area), 0)

    return tmp_spectra #rubixdata


@jaxtyped(typechecker=typechecker)
def get_extinction(config: dict) -> Callable:
    """
    Get the function to apply the dust extinction to the spaxel data.
    
    Parameters
    ----------
    config : dict
        The configuration dictionary.

    Returns
    -------
    Callable
        The function to apply the dust extinction to the spaxel data.
    """
    logger = get_logger(config.get("logger", None))
    
    # check if dust key exists in config file to ensure we really want to apply dust extinction
    if "dust" not in config["ssp"]:
        raise ValueError("Dust configuration not found in config file.")
    if "extinction_model" not in config["ssp"]["dust"]:
        raise ValueError("Extinction model not found in dust configuration.")

    # Get the telescope wavelength and spaxel number
    telescope = get_telescope(config)
    n_spaxels = int(telescope.sbin**2)
    wavelength = telescope.wave_seq

    galaxy_dist_z = config["galaxy"]["dist_z"]
    cosmology = get_cosmology(config)
    # Calculate the spatial bin edges
    _, spatial_bin_size = calculate_spatial_bin_edges(
        fov=telescope.fov,
        spatial_bins=telescope.sbin,
        dist_z=galaxy_dist_z,
        cosmology=cosmology,
    )

    spaxel_area = spatial_bin_size**2
    #spaxel_area = jnp.asarray(0.01)

    def calculate_extinction(rubixdata: RubixData) -> RubixData:
        """Apply the dust extinction to the spaxel data."""
        logger.info("Applying dust extinction to the spaxel data...")

        spaxel_indices = jnp.arange(n_spaxels)
        #rubixdata = _apply_spaxel_extinction(config, rubixdata, wavelength, 0)

        def _dummy_apply_spaxel_extinction(spaxel_indices):
            return _apply_spaxel_extinction(config, rubixdata, wavelength, spaxel_area, spaxel_indices)
        
        apply_spaxel_extinction_vmap = jax.vmap(_dummy_apply_spaxel_extinction)
        #rubixdata = apply_spaxel_extinction_vmap(config, rubixdata, wavelength, spaxel_indices)
        rubixdata.stars.spectra = jnp.sum(apply_spaxel_extinction_vmap(spaxel_indices), axis=0)
        return rubixdata
    
    return calculate_extinction

