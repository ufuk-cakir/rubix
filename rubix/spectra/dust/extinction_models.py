import jax.numpy as jnp
import equinox
from scipy.special import comb #whenever there is a jax version of comb, replace this!!! 
#Might come soon according to this github PR: https://github.com/jax-ml/jax/pull/18389

from .dust_baseclasses import BaseExtRvModel
from .dust_baseclasses import test_valid_x_range

from typing import Union, Tuple
from typing import ClassVar
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

__all__ = ["Cardelli89", "Gordon23"] #"O94", "F99", "F04", "VCG04", "GCC09", "M14", "G16", "F19", "D22", "G23"]

wave_range_CCM89 = [0.3, 10.0]
Rv_range_CCM89 = [2.0, 6.0]

wave_range_G23 = [0.0912, 32.0]
Rv_range_G23 = [2.3, 5.6]

@jaxtyped(typechecker=typechecker)
def PowerLaw1d(x: Float[Array, "n_wave"], amplitude: float, x_0: float, alpha: float) -> Float[Array, "n_wave"]:
    """
    Calculate a power law function.
    Function inspired by astropy.modeling.functional_models.PowerLaw1D.

    Parameters
    ----------
    x : Float[Array, "n_wave"]
        Input array.
    amplitude : float
        Amplitude of the power law.
    x_0 : float
        Reference x value.
    alpha : float
        Power law index.

    Returns
    -------
    Float[Array, "n_wave"]
        Output array after applying the power law.
    
    Notes
    -----
    Model formula (with :math:`A` for ``amplitude`` and :math:`\\alpha` for ``alpha``):

        .. math:: f(x) = A (x / x_0) ^ {-\\alpha}
    """
    xx = x / x_0
    return amplitude * xx ** (-alpha)

@jaxtyped(typechecker=typechecker)
def _smoothstep(x: Float[Array, "n_wave"], x_min: float = 0, x_max: float = 1, N: int = 1) -> Float[Array, "n_wave"]: 
    x = jnp.clip((x - x_min) / (x_max - x_min), 0, 1)

    result = 0
    for n in range(0, N + 1):
        result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)

    return result

@jaxtyped(typechecker=typechecker)
def poly_map_domain(oldx: Float[Array, "n"], domain: Tuple[float, float], window: Tuple[float, float]) -> Float[Array, "n"]:
    """
    Map domain into window by shifting and scaling.

    Parameters
    ----------
    oldx : array
          original coordinates
    domain : tuple of length 2
          function domain
    window : tuple of length 2
          range into which to map the domain
    """
    domain = jnp.array(domain)
    window = jnp.array(window)
    if domain.shape != (2,) or window.shape != (2,):
        raise ValueError('Expected "domain" and "window" to be a tuple of size 2.')
    scl = (window[1] - window[0]) / (domain[1] - domain[0])
    off = (window[0] * domain[1] - window[1] * domain[0]) / (domain[1] - domain[0])
    return off + scl * oldx



def Polynomial1d(x: Float[Array, "n"], coeffs: Float[Array, "m"], domain: Tuple[float, float] = (-1., 1.), window: Tuple[float, float] = (-1., 1.)) -> Float[Array, "n"]:
    r"""
    Evaluate a 1D polynomial model defined as 

    .. math::

        P = \sum_{i=0}^{i=n}C_{i} * x^{i}
    
    This function inspired by astropy.modelling.polynomial.Polynomial1D. 

    Parameters
    ----------
    x : ndarray
        Input values.
    coeffs : ndarray
        Coefficients of the polynomial, ordered from the constant term to the highest degree term.
    domain : tuple, optional
        Domain of the input values. Default is (-1, 1).
    window : tuple, optional
        Window to which the domain is mapped. Default is (-1, 1).

    Returns
    -------
    result : ndarray
        Evaluated polynomial values.
    """

    def horner(x: Float[Array, "n"], coeffs: Float[Array, "m"]) -> Float[Array, "n"]:
        """
        Evaluate polynomial using Horner's method.
        """
        if len(coeffs) == 1:
            return coeffs[-1] * jnp.ones_like(x, subok=False)
        c0 = coeffs[-1]
        for i in range(2, len(coeffs) + 1):
            c0 = coeffs[-i] + c0 * x
        return c0

    if domain is not None:
        x = poly_map_domain(x, domain, window)
    return horner(x, coeffs)

@jaxtyped(typechecker=typechecker)
def Drude1d(x: Float[Array, "n"], amplitude: float = 1.0, x_0: float = 1.0, fwhm: float = 1.0):
    r"""
    Evaluate the Drude model function.
    This function is inspired by astropy.modeling.functional_models.Drude1D.

    Model formula:

        .. math:: f(x) = A \\frac{(fwhm/x_0)^2}{((x/x_0 - x_0/x)^2 + (fwhm/x_0)^2}

    Parameters
    ----------
    x : ndarray
        Input values.
    amplitude : float, optional
        Peak value. Default is 1.0.
    x_0 : float, optional
        Position of the peak. Default is 1.0.
    fwhm : float, optional
        Full width at half maximum. Default is 1.0.

    Returns
    -------
    result : ndarray
        Evaluated Drude model values.
    
    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt

        from rubix.spectra.dust.extinction_models import Drude1D

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(7.5 , 12.5 , 0.1)

        dmodel = Drude1D(amplitude=1.0, fwhm=1.0, x_0=10.0)
        ax.plot(x, dmodel(x))

        ax.set_xlabel('x')
        ax.set_ylabel('F(x)')

        plt.show()
    """
    if x_0 == 0:
        raise ValueError("0 is not an allowed value for x_0")
    return (
        amplitude
        * ((fwhm / x_0) ** 2)
        / ((x / x_0 - x_0 / x) ** 2 + (fwhm / x_0) ** 2)
    )

@jaxtyped(typechecker=typechecker)
def _modified_drude(x: Float[Array, "n"], scale: float, x_o: float, gamma_o: float, asym: float) -> Float[Array, "n"]:
    """
    Modified Drude function to have a variable asymmetry.  Drude profiles
    are intrinsically asymmetric with the asymmetry fixed by specific central
    wavelength and width.  This modified Drude introduces an asymmetry
    parameter that allows for variable asymmetry at fixed central wavelength
    and width.

    Parameters
    ----------
    x : ndarray
        input wavelengths

    scale : float
        central amplitude

    x_o : float
        central wavelength

    gamma_o : float
        full-width-half-maximum of profile

    asym : float
        asymmetry where a value of 0 results in a standard Drude profile
    
    Returns
    -------
    y : ndarray
        output profile
    """
    gamma = 2.0 * gamma_o / (1.0 + jnp.exp(asym * (x - x_o)))
    y = scale * ((gamma / x_o) ** 2) / ((x / x_o - x_o / x) ** 2 + (gamma / x_o) ** 2)

    return y


@jaxtyped(typechecker=typechecker)
def FM90(x: Float[Array, "n"], C1: float = 0.10, C2: float = 0.70, C3: float = 3.23, C4: float = 0.41, xo: float = 4.59, gamma: float = 0.95) -> Float[Array, "n"]:
    r"""
    Fitzpatrick & Massa (1990) 6 parameter ultraviolet shape model

    Parameters
    ----------
    x: float
        wavenumber x in units of [1/micron]

    C1: float
       y-intercept of linear term

    C2: float
       slope of liner term

    C3: float
       strength of "2175 A" bump (true amplitude is C3/gamma^2)

    C4: float
       amplitude of FUV rise

    xo: float
       centroid of "2175 A" bump

    gamma: float
       width of "2175 A" bump

    Returns
        -------
        exvebv: np array (float)
            E(x-V)/E(B-V) extinction curve [mag]

    Raises
    ------
        ValueError
           Input x values outside of defined range

    Notes
    -----
    From Fitzpatrick & Massa (1990, ApJS, 72, 163)

    Only applicable at UV wavelengths

    Example showing a FM90 curve with components identified.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from rubix.spectra.dust.extinction_models import FM90

        fig, ax = plt.subplots()

        # generate the curves and plot them
        x = np.arange(3.8,8.6,0.1)/u.micron

        ext_model = FM90(x=x)
        ax.plot(x,ext_model,label='total')

        ext_model = FM90(x=x, C3=0.0, C4=0.0)
        ax.plot(x,ext_model,label='linear term')

        ext_model = FM90(x=x, C1=0.0, C2=0.0, C4=0.0)
        ax.plot(x,ext_model,label='bump term')

        ext_model = FM90(x=x, C1=0.0, C2=0.0, C3=0.0)
        ax.plot(x,ext_model,label='FUV rise term')

        ax.set_xlabel(r'$x$ [$\mu m^{-1}$]')
        ax.set_ylabel(r'$E(\lambda - V)/E(B - V)$')

        # for 2nd x-axis with lambda values
        axis_xs = np.array([0.12, 0.15, 0.2, 0.3])
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

    # Define bounds based on Gordon et al. (2024) results
    bounds = {
        "C1": (-10.0, 5.0),
        "C2": (-0.1, 5.0),
        "C3": (-1.0, 6.0),
        "C4": (-0.5, 1.5),
        "xo": (4.5, 4.9),
        "gamma": (0.6, 1.7)
    }
    
    # Check if parameters are within bounds
    if not (bounds["C1"][0] <= C1 <= bounds["C1"][1]):
        raise ValueError(f"C1 is out of bounds: {C1}")
    if not (bounds["C2"][0] <= C2 <= bounds["C2"][1]):
        raise ValueError(f"C2 is out of bounds: {C2}")
    if not (bounds["C3"][0] <= C3 <= bounds["C3"][1]):
        raise ValueError(f"C3 is out of bounds: {C3}")
    if not (bounds["C4"][0] <= C4 <= bounds["C4"][1]):
        raise ValueError(f"C4 is out of bounds: {C4}")
    if not (bounds["xo"][0] <= xo <= bounds["xo"][1]):
        raise ValueError(f"xo is out of bounds: {xo}")
    if not (bounds["gamma"][0] <= gamma <= bounds["gamma"][1]):
        raise ValueError(f"gamma is out of bounds: {gamma}")

    x_range = [1 / 0.35, 1 / 0.09]
    test_valid_x_range(x, x_range, "FM90")


    # linear term
    exvebv = C1 + C2 * x

    # bump term
    x2 = x**2
    exvebv += C3 * (x2 / ((x2 - xo**2) ** 2 + x2 * (gamma**2)))

    # FUV rise term
    fnuv_indxs = jnp.where(x >= 5.9)
    if len(fnuv_indxs) > 0:
        y = x[fnuv_indxs] - 5.9
        exvebv = exvebv.at[fnuv_indxs].add( C4 * (0.5392 * (y**2) + 0.05644 * (y**3)))

    # return E(x-V)/E(B-V)
    return exvebv









@jaxtyped(typechecker=typechecker)
class Cardelli89(BaseExtRvModel):
    r"""
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
    wave_range: ClassVar[Float[Array, "2"]] = equinox.field(converter=jnp.asarray, static=True, default=jnp.array(wave_range_CCM89))
    Rv_range: ClassVar[Float[Array, "2"]] = equinox.field(converter=jnp.asarray, static=True, default=jnp.array(Rv_range_CCM89))

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
                A(wave)/A(V) extinction curve [mag]
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


@jaxtyped(typechecker=typechecker)
class Gordon23(BaseExtRvModel):
    r"""
    Gordon et al. (2023) Milky Way R(V) dependent model

    Parameters
    ----------
    Rv: float
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
    From Gordon et al. (2023, ApJ, in press)

    Example showing G23 curves for a range of R(V) values.

    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u

        from dust_extinction.parameter_averages import G23

        fig, ax = plt.subplots()

        # generate the curves and plot them
        lam = np.logspace(np.log10(0.0912), np.log10(30.0), num=1000) * u.micron

        Rvs = [2.5, 3.1, 4.0, 4.75, 5.5]
        for cur_Rv in Rvs:
           ext_model = G23(Rv=cur_Rv)
           ax.plot(lam,ext_model(lam),label='R(V) = ' + str(cur_Rv))

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlabel('$\lambda$ [$\mu$m]')
        ax.set_ylabel(r'$A(x)/A(V)$')

        ax.legend(loc='best')
        plt.show()
    """
    
    wave_range: ClassVar[Float[Array, "2"]] = equinox.field(converter=jnp.asarray, static=True, default=jnp.array(wave_range_G23))
    Rv_range: ClassVar[Float[Array, "2"]] = equinox.field(converter=jnp.asarray, static=True, default=jnp.array(Rv_range_G23))

    def evaluate(self, wave: Float[Array, "n_wave"]) -> Float[Array, "n_wave"]:
        """
        Gordon 2023 function (The Astrophysical Journal, Volume 950, Issue 2, id.86, 13 pp.)

        Parameters
        ----------
        wave: float
           expects wave as wavelengths in wavenumbers [1/micron]

        Returns
        -------
        axav: np array (float)
            A(wave)/A(V) extinction curve [mag]

        Raises
        ------
        ValueError
           Input wave values outside of defined range
        """
        # setup the a & b coefficient vectors
        a = jnp.zeros(wave.shape)
        b = jnp.zeros(wave.shape)

        # define the ranges
        ir_indxs = jnp.where(jnp.logical_and(1.0 <= wave, wave < 35.0))
        opt_indxs = jnp.where(jnp.logical_and(0.3 <= wave, wave < 1.1))
        uv_indxs = jnp.where(jnp.logical_and(0.09 <= wave, wave <= 0.3))

        # overlap ranges
        optir_waves = [0.9, 1.1]
        optir_overlap = (wave >= optir_waves[0]) & (wave <= optir_waves[1])
        uvopt_waves = [0.3, 0.33]
        uvopt_overlap = (wave >= uvopt_waves[0]) & (wave <= uvopt_waves[1])

        # NIR/MIR
        # fmt: off
        # (scale, alpha1, alpha2, swave, swidth), sil1, sil2
        ir_a = [0.38526, 1.68467, 0.78791, 4.30578, 4.78338,
                0.06652, 9.8434, 2.21205, -0.24703,
                0.0267 , 19.58294, 17., -0.27]
        # fmt: on

        a = a.at[ir_indxs].set(self.nirmir_intercept(wave[ir_indxs], ir_a))
        b = b.at[ir_indxs].set(PowerLaw1d(x = wave[ir_indxs], amplitude = -1.01251, x_0 = 1.0, alpha = -1.06099))

        # optical
        # fmt: off
        # polynomial coeffs, ISS1, ISS2, ISS3
        opt_a = [-0.35848, 0.7122 , 0.08746, -0.05403, 0.00674,
                 0.03893, 2.288, 0.243,
                 0.02965, 2.054, 0.179,
                 0.01747, 1.587, 0.243]
        opt_b = [0.12354, -2.68335, 2.01901, -0.39299, 0.03355,
                 0.18453, 2.288, 0.243,
                 0.19728, 2.054, 0.179,
                 0.1713 , 1.587, 0.243]
        # fmt: on

        def compound_polynomial_drude_model(x: Float[Array, "n_wave"], params: Float[Array, "m"]) -> Float[Array, "n_wave"]:
            """
            Compound polynomial and Drude model
            """
            # Extract polynomial coefficients and Drude model parameters from opt_a
            poly_coeffs = params[:5]  # First 5 elements for polynomial coefficients
            drude_params = params[5:]  # Remaining elements for Drude model parameters

            # Evaluate the polynomial model
            poly_result = Polynomial1d(x, poly_coeffs)

            # Evaluate the Drude models
            drude_result_1 = Drude1d(x, amplitude=drude_params[0], x_0=drude_params[1], fwhm=drude_params[2])
            drude_result_2 = Drude1d(x, amplitude=drude_params[3], x_0=drude_params[4], fwhm=drude_params[5])
            drude_result_3 = Drude1d(x, amplitude=drude_params[6], x_0=drude_params[7], fwhm=drude_params[8])

            # Combine the results
            return poly_result + drude_result_1 + drude_result_2 + drude_result_3
        

        m20_model_a_result = compound_polynomial_drude_model(1 / wave[opt_indxs], opt_a)
        a = a.at[opt_indxs].set(m20_model_a_result)
        
        m20_model_b_result = compound_polynomial_drude_model(1 / wave[opt_indxs], opt_b)
        b = b.at[opt_indxs].set(m20_model_b_result)

        # overlap between optical/ir
        # weights = (1.0 / optir_waves[1] - x[optir_overlap]) / (
        #     1.0 / optir_waves[1] - 1.0 / optir_waves[0]
        # )
        weights = _smoothstep(
            wave[optir_overlap], x_min=optir_waves[0], x_max=optir_waves[1], N=1
        )

        m20_model_a_result_2 = compound_polynomial_drude_model(1 / wave[optir_overlap], opt_a)
        a = a.at[optir_overlap].set((1.0 - weights) * m20_model_a_result_2)
        a = a.at[optir_overlap].add(weights * self.nirmir_intercept(wave[optir_overlap], ir_a))
        
        m20_model_b_result_2 = compound_polynomial_drude_model(1 / wave[optir_overlap], opt_b)
        b = b.at[optir_overlap].set((1.0 - weights) * m20_model_b_result_2)
        b = b.at[optir_overlap].add(weights * PowerLaw1d(x = wave[optir_overlap], amplitude = -1.01251, x_0 = 1.0, alpha = -1.06099))

        # Ultraviolet
        fm90_model_a = FM90(1 / wave[uv_indxs], 0.81297, 0.2775, 1.06295, 0.11303, 4.60, 0.99)
        a = a.at[uv_indxs].set(fm90_model_a)
        fm90_model_b = FM90(1 / wave[uv_indxs], -2.97868, 1.89808, 3.10334, 0.65484, 4.60, 0.99)
        b = b.at[uv_indxs].set(fm90_model_b)

        # overlap between uv/optical
        # weights = (1.0 / uvopt_waves[1] - x[uvopt_overlap]) / (
        #     1.0 / uvopt_waves[1] - 1.0 / uvopt_waves[0]
        # )
        weights = _smoothstep(
            wave[uvopt_overlap], x_min=uvopt_waves[0], x_max=uvopt_waves[1], N=1
        )
        fm90_model_a_overlap = FM90(1 / wave[uvopt_overlap], 0.81297, 0.2775, 1.06295, 0.11303, 4.60, 0.99)
        a = a.at[uvopt_overlap].set((1.0 - weights) * fm90_model_a_overlap)
        m20_model_a_result_3 = compound_polynomial_drude_model(1 / wave[uvopt_overlap], opt_a)
        a = a.at[uvopt_overlap].add(weights * m20_model_a_result_3)
        fm90_model_b_overlap = FM90(1 / wave[uvopt_overlap], -2.97868, 1.89808, 3.10334, 0.65484, 4.60, 0.99)
        b = b.at[uvopt_overlap].set((1.0 - weights) * fm90_model_b_overlap)
        m20_model_b_result_3 = compound_polynomial_drude_model(1 / wave[uvopt_overlap], opt_b)
        b = b.at[uvopt_overlap].add(weights * m20_model_b_result_3)

        # return A(x)/A(V)
        return a + b * (1 / self.Rv - 1 / 3.1)



    @staticmethod
    def nirmir_intercept(wave, params):
        """
        Functional form for the NIR/MIR intercept term.
        Based on modifying the G21 shape model to have two power laws instead
        of one with a break wavelength.

        Parameters
        ----------
        wave: float
           expects x in wavenumbers [1/micron]
        params: floats
           paramters of function

        Returns
        -------
        axav: np array (float)
            A(x)/A(V) extinction curve [mag]
        """

        # fmt: off
        (scale, alpha, alpha2, swave, swidth,
            sil1_amp, sil1_center, sil1_fwhm, sil1_asym,
            sil2_amp, sil2_center, sil2_fwhm, sil2_asym) = params
        # fmt: on

        # broken powerlaw with a smooth transition
        axav_pow1 = scale * (wave ** (-1.0 * alpha))

        norm_ratio = swave ** (-1.0 * alpha) / swave ** (-1.0 * alpha2)
        axav_pow2 = scale * norm_ratio * (wave ** (-1.0 * alpha2))

        # use smoothstep to smoothly transition between the two powerlaws
        weights = _smoothstep(
            wave, x_min=swave - swidth / 2, x_max=swave + swidth / 2, N=1
        )
        axav = axav_pow1 * (1.0 - weights) + axav_pow2 * weights

        # silicate feature drudes
        axav += _modified_drude(wave, sil1_amp, sil1_center, sil1_fwhm, sil1_asym)
        axav += _modified_drude(wave, sil2_amp, sil2_center, sil2_fwhm, sil2_asym)

        return axav


#TODO: Implement more jax versions of extinction models from astropy, see https://dust-extinction.readthedocs.io/en/latest/index.html

