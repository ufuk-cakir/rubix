import jax.numpy as jnp
import jax
from rubix import config
from rubix.logger import get_logger
from .extinction_models import *

from typing import Tuple
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def calculate_spaxel_column_density(
    dust_mass: Float[Array, "n_gas"],
    dust_z_pos: Float[Array, "n_gas"],
    spaxel_area: Float[Array, "..."]
) -> Tuple[Float[Array, "n_gas"], Float[Array, "n_gas"]]:
    """
    Calculate the dust column density for a single spaxel.

        The dust column density is calculated by effectively integrating the gas density along the z-axis.
        This is done by first sorting the gas cells by their z position and then calculating the column density
        of the gas cells as a function of distance.

        Note: This function makes some approximations that should be valid for densly populated gas cells, i.e.
        the gas cells are much smaller than a spaxel size. The behaviour of this function might be improved by 
        rasterizing the gas cells first onto a regular grid.
        TODO: Implement rasterization of gas cells onto a regular grid.

        Parameters
        ----------
        dust_mass : Float[Array, "n_gas"]
            The dust mass of each gas cell.
        dust_z_pos : Float[Array, "n_gas"]
            The z positions of the gas cells.

        Returns
        -------
        Float[Array, "n_gas"]
            The gas column density as a function of distance.
        
        Float[Array, "n_gas"]
            The sorted z positions of the gas cells.
        
        Notes
        -----
        
        .. math::
            \Sigma(z) = \sum_{i=0}^{n} \rho_i \Delta z_i

        where:
        - :math:`\Sigma(z)` is the column density at position :math:`z`
        - :math:`\rho_i` is the gas density of the i-th cell
        - :math:`\Delta z_i` is the difference between consecutive z positions
    """
    sorted_indices = jnp.argsort(dust_z_pos)
    sorted_gas_density = dust_mass[sorted_indices]
    sorted_gas_z_pos = dust_z_pos[sorted_indices]

    # Calculate the differences between consecutive z positions
    dz = jnp.diff(sorted_gas_z_pos, append=sorted_gas_z_pos[-1])
    #jax.debug.print("masses: {masses}", masses=sorted_gas_density)
    #jax.debug.print("max mass: {max_mass}", max_mass=jnp.max(sorted_gas_density))

    #column_density = jnp.cumsum(sorted_gas_density * dz) / spaxel_area
    column_density = jnp.cumsum(sorted_gas_density) / spaxel_area
    #jax.debug.print("max column_density: {max_column_density}", max_column_density=jnp.max(column_density))
    return column_density, sorted_gas_z_pos


@jaxtyped(typechecker=typechecker)
def calculate_extinction(
    dust_column_density: Float[Array, "n_gas"],
    dust_grain_density: float,
    effective_wavelength: float = 5448 # Johnson V band effective wavelength in Angstrom
) -> Float[Array, "n_gas"]:
    """
    Calculate the extinction of the gas cells due to dust.
    
        The extinction is calculated using the dust column density and the dust-to-gas ratio.
        The dust column density is calculated by multiplying the gas column density with the dust-to-gas ratio.
        See e.g. formula A5 and A6 in the appendix of the paper by Ibarra-Medel et al. 2018.
        
        Parameters
        ----------
        dust_column_density : Float[Array, "n_gas"]
            The gas column density of each gas cell.
        dust_grain_density : Float
            The dust grain density.
        effective_wavelength : Float
            The effective wavelength in Angstroem of the light at which we calculate the extinction.
            Default value is 5448 Angstrom for Johnson V as taken from here: https://www.aavso.org/filters

        Returns
        -------
        Float[Array, "n_gas"]
            The extinction of the gas cells due to dust.
        
        Notes
        -----
        Extinction is calculated as:

        .. :math: `A_{\lambda}(z)=\frac{3m_H\pi \Sigma(z)}{0.4 \log(10)\lambda_V \rho_{D}}\times (D/G)`

        where:
        - :math:`A_{\lambda}` is the extinction at the effective wavelength
        - :math:`m_H` is the mass of a hydrogen atom
        - :math:`\Sigma(z)` is the column density of the gas cells
        - :math:`\lambda_V` is the effective wavelength`
        - :math:`\rho_{D}` is the dust grain density
        - :math:`D/G` is the dust-to-gas ratio
    """

    # Constants
    m_H = config["constants"]["MASS_OF_PROTON"]  # mass of a hydrogen atom in grams

    #TODO: check units carefully!
    # dust_grain_density is in g/cm^3
    # gas_column_density is internally in Msun per ???, but should be in g/cm^2
    # coordinates internally are in kpc
    # effective_wavelength is in Angstrom = 10^-10 m
    # dust_to_gas_ratio is dimensionless and ~ 0.01 for MW
    # m_H is in grams

    # equation is wrong, column density should be in number of particles per cm^2 
    #dust_extinction = 3 * m_H * jnp.pi * gas_column_density / (0.4 * jnp.log(10) * effective_wavelength * 1e-8 * dust_grain_density) * dust_to_gas_ratio
    
    # convert the surface density to grams per cm^2
    CONVERT_MASS_PER_AREA = float(config["constants"]["MSUN_TO_GRAMS"]) / float(config["constants"]["KPC_TO_CM"])**2
    effective_wavelength = effective_wavelength * 1e-8  # convert to cm
    # in order to get the right column density we might need to add mass of the gas particles and divide by the area of the telescope pixels...
    dust_extinction = 3 * jnp.pi * dust_column_density * CONVERT_MASS_PER_AREA / (0.4 * jnp.log(10) * effective_wavelength * dust_grain_density)
    #dust_extinction = jnp.array([2.0]*len(gas_column_density)) 

    ###############################################
    #meta_g=gas['GFM_Metallicity'][:]
    #volm=volm/ho**3.0
    #dens=dens*ho**2.0
    #volm=float_((volm/(4.0*np.pi/3.0))**(1./3.0)*(3.08567758e19*100))
    #dens=dens*1e10/(3.08567758e19*100)**3.0*1.9891e30/1.67262178e-27
    #Av_g=meta_g*(3.0*1.67262e-24*np.pi*dens*volm)/(4.0*np.log(10.)*3.0*5494e-8)

    return dust_extinction

@jaxtyped(typechecker=typechecker)
def apply_spaxel_extinction(config: dict, dust_mass: Float[Array, "n_gas"], dust_z_pos: Float[Array, "n_gas"], star_z_pos: Float[Array, "n_star"], ssp_template_fluxes: Float[Array, "n_star n_wave"], wavelength: Float[Array, "n_wave"], spaxel_area: Float[Array, "..."]) -> Float[Array, "n_star n_wave"]:
    """
    Calculate the extinction for each star in the spaxel and apply dust extinction to it's associated SSP.

        The dust column density is calculated by effectively integrating the dust mass along the z-axis and dividing by pixel area.
        This is done by first sorting the gas cells by their z position and then calculating the column density
        of the dust as a function of distance.

        The dust column density is then interpolated to the z positions of the stars.
        The extinction is calculated using the dust column density and the dust-to-gas ratio.
        The extinction is then applied to the SSP fluxes using an Av/Rv dependent extinction model. Default is chosen as Cardelli89.

        Parameters
        ----------
        dust_mass : Float[Array, "n_gas"]
            The gas density of each gas cell.
        dust_z_pos : Float[Array, "n_gas"]
            The z positions of the gas cells.
        star_z_pos : Float[Array, "n_star"]
            The z positions of the stars.
        ssp_template_fluxes : Float[Array, "n_star, n_wave"]
            The SSP template fluxes of each star.
        ssp_wavelength : Float[Array, "n_wave"]
            The wavelength of the SSP template fluxes.

        Returns
        -------

        Float[Array, "n_star, n_wave"]
            The SSP template fluxes after applying the dust extinction.

        Notes
        -----
        The extinction is calculated as described in the function calculate_extinction.
    """
    logger = get_logger(config.get("logger", None))
    
    # might become a function of gas metallicity
    dust_grain_density = config["ssp"]["dust"]["dust_grain_density"]
    
    ext_model = config["ssp"]["dust"]["extinction_model"]
    Rv = config["ssp"]["dust"]["Rv"] # check if this is the right place to put it, at some point we might want to have models that don't need Rv.
    
    logger.info(f"Chosen extinction model: {ext_model} with Rv parameter Rv: {Rv}")
    
    # Dynamically choose the extinction model based on the string name
    if ext_model not in RV_MODELS:
        raise ValueError(f"Extinction model '{ext_model}' is not available. Choose from {RV_MODELS}.")

    # Calculate the gas column density
    dust_column_density, sorted_gas_z_pos = calculate_spaxel_column_density(dust_mass, dust_z_pos, spaxel_area)

    # interpolate the gas column density to the star positions
    dust_column_density_interpolated = jnp.interp(star_z_pos, sorted_gas_z_pos, dust_column_density)
    Av_array = calculate_extinction(dust_column_density_interpolated, dust_grain_density)
    jax.debug.print("Av_array: {Av_array}", Av_array=Av_array)

    ext_model_class = Rv_model_dict[ext_model]
    ext = ext_model_class(Rv=Rv)

    # get the extinguished SSP flux for different amounts of dust
    # Vectorize the extinction calculation using vmap
    extinguish_vmap = jax.vmap(ext.extinguish, in_axes=(None, 0))
    # note, we need to pass wavelength in microns here to the extinction model.
    # in Rubix the wavelength is in Angstroms, so we divide by 1e4 to get microns. 
    extincted_ssp_template_fluxes = ssp_template_fluxes * extinguish_vmap(wavelength/1e4, Av_array)

    return extincted_ssp_template_fluxes