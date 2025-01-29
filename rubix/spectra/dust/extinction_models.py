import jax.numpy as jnp
import equinox

from typing import ClassVar
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

__all__ = ["BaseExtModel", "BaseExtRvModel", "BaseExtRvAfAModel", "BaseExtGrainModel"]

def test_valid_x_range(wave: Float[Array, "n"], wave_range: Float[Array, "2"], outname: str) -> None:
    """
    Test if the input wavelength is within the valid range of the model.

    Parameters
    ----------
    wave : Float[Array, "n"]
        The input wavelength to test.
        
    wave_range : Float[Array, "2"]
        The valid range of the model.
        
    outname : str
        The name of the model for error message.

    Returns
    -------
    None
    """

    deltacheck = 1e-6  # delta to allow for small numerical issues
    if jnp.logical_or(
        jnp.any(wave <= (wave_range[0] - deltacheck)), jnp.any(wave >= (wave_range[1] + deltacheck))
    ):
        raise ValueError(
            "Input wave outside of range defined for "
            + outname
            + " ["
            + str(wave_range[0])
            + " <= wave <= "
            + str(wave_range[1])
            + ", wave has units 1/micron]"
        )
        
@jaxtyped(typechecker=typechecker)
class BaseExtModel(equinox.Module): 
    """
    Base class for dust extinction models.
    """

    #wave: equinox.AbstractClassVar[Float[Array, "n_wave"]]
    wave_range: equinox.AbstractClassVar[Float[Array, "2"]]
    #wave: Float[Array, "n_wave"] = equinox.field(static=True)
    #wave_range: Float[Array, "2"] = equinox.field(converter=jnp.asarray, static=True)

    def __call__(self, wave: Float[Array, "n_wave"]) -> Float[Array, "n_wave"]:
        """
        Evaluate the dust extinction model at the input wavelength for the given model parameters.
        """

        test_valid_x_range(wave, self.wave_range, self.__class__.__name__)
        
        return self.evaluate(wave)

    def extinguish(self) -> Float[Array, "n_wave"]:
        """
        Abstract function to calculate the dust extinction for a given wavelength as a fraction.

        Parameters
        ----------
        wave : Float[Array, "n_wave"]
            The wavelength to calculate the dust extinction for.
            The wavelength has to be passed as wavenumber in units of [1/microns].

        Returns
        -------
        Float[Array, "n_wave"]
            The fractional extinction as a function of wavenumber.
        """
        pass


@jaxtyped(typechecker=typechecker)
class BaseExtRvModel(BaseExtModel):
    """
    Base class for dust extinction models with Rv parameter.
    """

    Rv: float #equinox.AbstractClassVar[Float]
    Rv_range: equinox.AbstractClassVar[Float[Array, "2"]]

    """
    The Rv parameter (R(V) = A(V)/E(B-V) total-to-selective extinction) of the dust extinction model and its valid range. 
    """

    def __check_init__(self) -> None:
        """
        Check if the Rv parameter of the dust extinction model is within Rv_range.

        Parameters
        ----------
        Rv : Float
            The Rv parameter of the dust extinction model.
        
        Raises
        ------
        ValueError
            If the Rv parameter is outsied of defined range.
        """
        if not (self.Rv_range[0] <= self.Rv <= self.Rv_range[1]):
            raise ValueError(
                "parameter Rv must be between "
                + str(self.Rv_range[0])
                + " and "
                + str(self.Rv_range[1])
            )
        else:
            pass
    
    def extinguish(self, wave: Float[Array, "n_wave"], Av: Float = None, Ebv: Float = None) -> Float[Array, "n_wave"]:
        """
        Calculate the dust extinction for a given wavelength as a fraction.

        Parameters
        ----------
        wave : Float[Array, "n_wave"]
            The wavelength to calculate the dust extinction for.
            The wavelength has to be passed as wavenumber in units of [1/microns].

        Av : Float
            The visual extinction.
            A(V) value of dust column.
            Note: Av or Ebv must be set.
        
        Ebv : Float
            The color excess.
            E(B-V) value of dust column.
            Note: Av or Ebv must be set.

        Returns
        -------
        Float[Array, "n_wave"]
            The fractional extinction as a function of wavenumber.
        """
        # get the extinction curve
        axav = self(wave)

        # check that av or ebv is set
        if (Av is None) and (Ebv is None):
            raise ValueError("neither Av or Ebv passed, one of them is required!")

        # if Av is not set and Ebv set, convert to Av
        if Av is None:
            Av = self.Rv * Ebv

        # return fractional extinction
        return jnp.power(10.0, -0.4 * axav * Av)



@jaxtyped(typechecker=typechecker)
class Cardelli89(BaseExtRvModel):
    """
    Calculate the extinction curve of the Milky Way according to the 
    Cardelli, Clayton, & Mathis (1989) Milky Way R(V) dependent model.

    Parameters
    ----------
    Rv : Float
        R(V) = A(V)/E(B-V) = total-to-selective extinction

    Returns
    -------
    Float[Array, "n_wave"]
        A(x)/A(V) extinction curve [mag]

    Raises
    ------
    InputParameterError
        Input Rv values outside of defined range

    Notes
    -----
    From Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245)

    Example showing CCM89 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from rubix.spectra.dust_extinction import Cardelli89

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(0.5,10.0,0.1) # units of micron

        Rvs = ['2.0','3.0','4.0','5.0','6.0']
        for cur_Rv in Rvs:
            ext_model = Cardelli89(Rv=cur_Rv)
            ax.plot(x,ext_model,label='R(V) = ' + str(cur_Rv))

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.1, 0.12, 0.15, 0.2, 0.3, 0.5, 1.0])
        new_ticks = 1 / axis_xs
        new_ticks_labels = ["%.2f" % z for z in axis_xs]
        tax = ax.twiny()
        tax.set_xlim(ax.get_xlim())
        tax.set_xticks(new_ticks)
        tax.set_xticklabels(new_ticks_labels)
        tax.set_xlabel(r"$\lambda$ [$\mu$m]")

        ax.legend(loc='best')
        plt.show()
    """

    #wave: Float[Array, "n_wave"]
    #Rv: Float
    wave_range: ClassVar[Float[Array, "2"]] = equinox.field(converter=jnp.asarray, static=True, default=jnp.array([0.3, 10.0]))
    Rv_range: ClassVar[Float[Array, "2"]] = equinox.field(converter=jnp.asarray, static=True, default=jnp.array([2.0, 6.0]))

    def evaluate(self, wave: Float[Array, "n_wave"]) -> Float[Array, "n_wave"]:
        """
            Cardelli, Clayton, and Mathis (1989, ApJ, 345, 245) function

            Parameters
            ----------
            wave: float
                expects wave as wavelengths in wavenumbers [1/micron]

            Returns
            -------
            axav: jax numpy array (float)
                A(x)/A(V) extinction curve [mag]
        """

        # setup the a & b coefficient vectors
        a = jnp.zeros(wave.shape)
        b = jnp.zeros(wave.shape)

        # define the ranges
        ir_indxs = jnp.where(jnp.logical_and(0.3 <= wave, wave < 1.1))
        opt_indxs = jnp.where(jnp.logical_and(1.1 <= wave, wave < 3.3))
        nuv_indxs = jnp.where(jnp.logical_and(3.3 <= wave, wave <= 8.0))
        fnuv_indxs = jnp.where(jnp.logical_and(5.9 <= wave, wave <= 8))
        fuv_indxs = jnp.where(jnp.logical_and(8 < wave, wave <= 10))

        # Infrared
        a = a.at[ir_indxs].set(0.574 * wave[ir_indxs] ** 1.61)
        b = b.at[ir_indxs].set(-0.527 * wave[ir_indxs] ** 1.61)

        # NIR/optical
        y = wave[opt_indxs] - 1.82
        a = a.at[opt_indxs].set(1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
        b = b.at[opt_indxs].set(1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4 - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)

        # NUV
        a = a.at[nuv_indxs].set(
            1.752 - 0.316 * wave[nuv_indxs] - 0.104 / ((wave[nuv_indxs] - 4.67) ** 2 + 0.341)
        )
        b = b.at[nuv_indxs].set(
            -3.09 + 1.825 * wave[nuv_indxs] + 1.206 / ((wave[nuv_indxs] - 4.62) ** 2 + 0.263)
        )

        # far-NUV
        y = wave[fnuv_indxs] - 5.9
        a = a.at[fnuv_indxs].set(a[fnuv_indxs] + (-0.04473 * (y**2) - 0.009779 * (y**3)))
        b = b.at[fnuv_indxs].set(b[fnuv_indxs] + (0.2130 * (y**2) + 0.1207 * (y**3)))
        #a = a.at[fnuv_indxs] += -0.04473 * (y**2) - 0.009779 * (y**3)
        #b[fnuv_indxs] += 0.2130 * (y**2) + 0.1207 * (y**3)

        # FUV
        y = wave[fuv_indxs] - 8.0
        a = a.at[fuv_indxs].set(-1.073 - 0.628*y + 0.137*y**2 - 0.070*y**3)
        b = b.at[fuv_indxs].set(13.670 + 4.257*y - 0.420*y**2 + 0.374*y**3)

        # return A(x)/A(V)
        return a + b / self.Rv



#TODO: Implement more jax versions of extinction models from astropy, see https://dust-extinction.readthedocs.io/en/latest/index.html

