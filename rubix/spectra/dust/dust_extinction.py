import jax.numpy as jnp
from rubix import config

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker


@jaxtyped(typechecker=typechecker)
def calculate_spaxel_column_density(
    gas_density: Float[Array, "n_gas"],
    gas_z_pos: Float[Array, "n_gas"]
) -> Float[Array, "n_gas"]:
    """
    Calculate the dust column density for a single spaxel.

        The gas column density is calculated by effectively integrating the gas density along the z-axis.
        This is done by first sorting the gas cells by their z position and then calculating the column density
        of the gas cells as a function of distance.

        Note: This function makes some approximations that should be valid for densly populated gas cells, i.e.
        the gas cells are much smaller than a spaxel size. The behaviour of this function might be improved by 
        rasterizing the gas cells first onto a regular grid.
        TODO: Implement rasterization of gas cells onto a regular grid.

        Parameters
        ----------
        gas_density : Float[Array, "n_gas"]
            The gas density of each gas cell.
        gas_z_pos : Float[Array, "n_gas"]
            The z positions of the gas cells.

        Returns
        -------
        Float[Array, "n_gas"]
            The gas column density as a function of distance.
        
        Notes
        -----
        
        .. math::
            \Sigma(z) = \sum_{i=0}^{n} \rho_i \Delta z_i

        where:
        - :math:`\Sigma(z)` is the column density at position :math:`z`
        - :math:`\rho_i` is the gas density of the i-th cell
        - :math:`\Delta z_i` is the difference between consecutive z positions
    """
    sorted_indices = jnp.argsort(gas_z_pos)
    sorted_gas_density = gas_density[sorted_indices]
    sorted_gas_z_pos = gas_z_pos[sorted_indices]

    # Calculate the differences between consecutive z positions
    dz = jnp.diff(sorted_gas_z_pos, append=sorted_gas_z_pos[-1])

    column_density = jnp.cumsum(sorted_gas_density * dz)

    return column_density


@jaxtyped(typechecker=typechecker)
def calculate_spaxel_extinction(
    gas_column_density: Float[Array, "n_gas"],
    dust_to_gas_ratio: Float,
    dust_grain_density: Float,
    effective_wavelength: Float
) -> Float[Array, "n_gas"]:
    """
    Calculate the extinction of the gas cells due to dust.
    
        The extinction is calculated using the dust column density and the dust-to-gas ratio.
        The dust column density is calculated by multiplying the gas column density with the dust-to-gas ratio.
        See e.g. formula A5 and A6 in the appendix of the paper by Ibarra-Medel et al. 2018.
        
        Parameters
        ----------
        gas_column_density : Float[Array, "n_gas"]
            The gas column density of each gas cell.
        dust_to_gas_ratio : Float
            The dust-to-gas ratio.
        dust_grain_density : Float
            The dust grain density.
        effective_wavelength : Float
            The effective wavelength of the light at whihc we calculate the extinction.
            Default value is 5448 Angstrom.

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
    m_H = config.m_H

    #TODO: check units carefully!
    dust_extinction = 3 * m_H * jnp.pi * gas_column_density / (0.4 * jnp.log(10) * effective_wavelength * dust_grain_density) * dust_to_gas_ratio

    return dust_extinction
