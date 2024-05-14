from jax import lax, vmap, jit
from jax import numpy as jnp
from .utils import trapz

import equinox as eqx


# TODO: maybe change this to load from the config file?
C_SPEED = 2.99792458e8  # m/s
RHO_CRIT0_KPC3_UNITY_H = 277.536627  # multiply by h**2 in cosmology conversion
MPC = 3.08567758149e24  # Mpc in cm
YEAR = 31556925.2  # year in seconds


class BaseCosmology(eqx.Module):
    """Class to handle cosmological calculations.

    The methods in this class are mainly taken from https://github.com/ArgonneCPAC/dsps/blob/main/dsps/cosmology/flat_wcdm.py.
    Here they are wrapped in a class to be used in JAX.

    Once initialized with the cosmological parameters, the class can be used to calculate various cosmological quantities.

    Parameters
    ----------
    Om0 : float
        The present day matter density.
    w0 : float
        The present day dark energy equation of state.
    wa : float
        The dark energy equation of state parameter.
    h : float
        The Hubble constant.

    Returns
    -------
    Cosmology
        A Cosmology instance.

    Examples
    --------
    >>> # Create Planck15 cosmology
    >>> cosmo = Cosmology(0.3089, -1.0, 0.0, 0.6774)
    >>> # Calculate the angular diameter distance to redshift 0.5
    >>> cosmo.angular_diameter_distance_to_z(0.5)
    """

    Om0: jnp.float32
    w0: jnp.float32
    wa: jnp.float32
    h: jnp.float32

    def __init__(self, Om0, w0, wa, h):
        self.Om0 = jnp.float32(Om0)
        self.w0 = jnp.float32(w0)
        self.wa = jnp.float32(wa)
        self.h = jnp.float32(h)

    @jit
    def scale_factor_to_redshift(self, a):
        z = 1.0 / a - 1.0
        return z

    @jit
    def _rho_de_z(self, z):
        a = 1.0 / (1.0 + z)
        de_z = a ** (-3.0 * (1.0 + self.w0 + self.wa)) * lax.exp(
            -3.0 * self.wa * (1.0 - a)
        )
        return de_z

    @jit
    def _Ez(self, z):
        zp1 = 1.0 + z
        Ode0 = 1.0 - self.Om0
        t = self.Om0 * zp1**3 + Ode0 * self._rho_de_z(z)
        E = jnp.sqrt(t)
        return E

    @jit
    def _integrand_oneOverEz(self, z):
        return 1 / self._Ez(z)

    @jit
    def comoving_distance_to_z(self, redshift):
        z_table = jnp.linspace(0, redshift, 256)
        integrand = self._integrand_oneOverEz(z_table)
        return trapz(z_table, integrand) * C_SPEED * 1e-5 / self.h

    @jit
    def luminosity_distance_to_z(self, redshift):
        return self.comoving_distance_to_z(redshift) * (1 + redshift)

    @jit
    def angular_diameter_distance_to_z(self, redshift):
        return self.comoving_distance_to_z(redshift) / (1 + redshift)

    @jit
    def distance_modulus_to_z(self, redshift):
        d_lum = self.luminosity_distance_to_z(redshift)
        mu = 5.0 * jnp.log10(d_lum * 1e5)
        return mu

    @jit
    def _hubble_time(self, z):
        E0 = self._Ez(z)
        htime = 1e-16 * MPC / YEAR / self.h / E0
        return htime

    @jit
    def lookback_to_z(self, redshift):
        z_table = jnp.linspace(0, redshift, 512)
        integrand = 1 / self._Ez(z_table) / (1 + z_table)
        res = trapz(z_table, integrand)
        th = self._hubble_time(0.0)
        return th * res

    @jit
    def age_at_z0(self):
        z_table = jnp.logspace(0, 3, 512) - 1.0
        integrand = 1 / self._Ez(z_table) / (1 + z_table)
        res = trapz(z_table, integrand)
        th = self._hubble_time(0.0)
        return th * res

    @jit
    def _age_at_z_kern(self, redshift):
        t0 = self.age_at_z0()
        tlook = self.lookback_to_z(redshift)
        return t0 - tlook

    @jit
    def age_at_z(self, redshift):
        fun = self._age_at_z_vmap()
        return fun(jnp.atleast_1d(redshift))

    def _age_at_z_vmap(self):
        return jit(vmap(self._age_at_z_kern))

    @jit
    def angular_scale(self, z) -> jnp.float32:
        """Angular scale in kpc/arcsec at redshift z.

        Parameters
        ----------
        z : float
            Redshift.

        Returns
        -------
        scale : float
            Angular scale in kpc/arcsec at redshift z.
        """
        # Angular scale in kpc/arcsec at redshift z.
        D_A = self.angular_diameter_distance_to_z(z)  # in Mpc
        scale = D_A * (jnp.pi / (180 * 60 * 60)) * 1e3  # in kpc/arcsec
        return scale

    """
    I dont think we need this currently, but keeping it here for reference
    @jit
    def rho_crit(self, redshift):
        rho_crit0 = RHO_CRIT0_KPC3_UNITY_H * self.h * self.h
        rho_crit = rho_crit0 * self._Ez(redshift) ** 2
        return rho_crit
    
    @jit
    def _integrand_oneOverEz1pz(self, z):
        return 1.0 / self._Ez(z) / (1.0 + z)
    
    @jit
    def _Om_at_z(self, z):
        E = self._Ez(z)
        return self.Om0 * (1.0 + z) ** 3 / E / E
    

    
    
    @jit
    def _delta_vir(self, z):
        x = self._Om(z) - 1.0
        Delta = 18 * jnp.pi**2 + 82.0 * x - 39.0 * x**2
        return Delta
    
    @jit
    def virial_dynamical_time(self, redshift):
        delta = self._delta_vir(redshift)
        t_cross = 2**1.5 * self._hubble_time(redshift) * delta**-0.5
        return t_cross

"""

