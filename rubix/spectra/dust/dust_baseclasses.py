import jax.numpy as jnp
import equinox

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

__all__ = ["BaseExtModel", "BaseExtRvModel"]#, "BaseExtRvAfAModel", "BaseExtGrainModel"]

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
