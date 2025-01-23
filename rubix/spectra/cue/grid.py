import jax.numpy as jnp
import numpy as np
from jax import vmap
import jax
from cue.line import predict as line_predict
from cue.continuum import predict as cont_predict

# from cue.line import predict as line_predict
# from cue.continuum import predict as cont_predict
from rubix.core.telescope import get_telescope
from rubix.spectra.ifu import convert_luminoisty_to_flux_gas
from rubix import config as rubix_config
from rubix.logger import get_logger
from rubix.cosmology.base import BaseCosmology
from jax.experimental import jax2tf
import tensorflow as tf


class CueGasLookup:

    def __init__(self, preprocessed_data):
        self.config = preprocessed_data["config"]
        self.telescope = preprocessed_data["telescope"]
        self.observation_lum_dist = preprocessed_data["observation_lum_dist"]
        self.factor = preprocessed_data["factor"]

    def get_wavelengthrange(self, steps=1000):
        """
        Returns a range of wavelengths for the spectra dependent on the wavelength range of the emission lines.
        The wavelength range is set to 1000 to 10000 Angstrom, because we currently focus on the optical range.
        This can be easily changed in the future, as long as the Cue model is able to predict the emission lines in the desired wavelength range.
        """

        logger = get_logger(self.config.get("logger", None))
        logger.warning("Wavelengthrange is set to 1000 to 10000 Angstrom.")
        steps = int(steps)
        wave_start = 1e3
        wave_end = 1e4
        wavelengthrange = jnp.linspace(wave_start, wave_end, steps)
        return wavelengthrange

    def continuum_tf(self, theta):
        return cont_predict(theta=theta).nn_predict()

    def calculate_continuum(self, theta):
        """
        Calculate the gas continuum using the provided theta parameters.

        My interpretation of the input: these are the parameters from Table 1 in Li et al. 2024
        gammas: power-law slopes (alpha_HeII, alpha_OII, alpha_HeI and alpha_HI)
        log_L_ratios:   flux ratios of the two bluest segments (log F_OII/F_HeII)
                        flux ratios of the second and third segment (log F_HeI/F_OII)
                        fliux ratios of the two reddest segments (log F_HI/F_HeI)
        log_QH: ionization parameter log U (in Illustris ElectronAbundance)
        n_H: hydrogen density (not in log scale!!!) in 1/cm^3 (in Illustris Denisty)
        log_OH_ratio: oxygen abundance (in Illustris GFM_Metals[4]/GFM_Metals[0])
        log_NO_ratio: nitrogen-to-oxygen ratio (in Illustris GFM_Metals[3]/GFM_Metals[4])
        log_CO_ratio: carbon-to-oxygen ratio (in Illustris GFM_Metals[2]/GFM_Metals[4])

        My interpretation of the output:
        first array: wavelengt in Angstrom
        second array: luminosity in erg/s

        Parameters:
        theta (jnp.ndarray): The theta parameters describing the shape of the ionizing spectrum and the ionizing gas properties, 12 values.

        Returns:
        jnp.ndarray: The predicted continuum.
        continuum[0] is the wavelength in Angstrom
        continuum[1] is the luminosity in erg/s for the continuum in each gas cell
        """
        logger = get_logger(self.config.get("logger", None))
        logger.info("Calculating continuum")
        par = theta
        wavelength_cont, continuum = jax2tf.call_tf(self.continuum_tf)(par)

        return wavelength_cont, continuum

    def get_resample_continuum(self, theta):
        """
        Resamples the spectrum of the gas continuum to the new wavelength range using interpolation.
        The new wavelength range is the same for the emission lines.
        We do this step to be able to add the continuum and the emission lines together later.

        Parameters:
        rubixdata (RubixData): The RubixData object containing the gas data.

        original_wavelength (jnp.ndarray): The original wavelength array of continuum.
        continuum (jnp.ndarray): The original spectrum array.
        new_wavelength (jnp.ndarray): The new wavelength array to resample to, which is the same for the emission lines.

        Returns:
        rubixdata.gas.continuum (jnp.ndarray): The resampled wavelength and spectrum array.
        """
        logger = get_logger(self.config.get("logger", None))

        new_wavelength = self.get_wavelengthrange()
        original_wavelength, continuum = self.calculate_continuum(theta)

        # Define the interpolation function
        def interp_fn(continuum_i):
            if original_wavelength.shape != continuum_i.shape:
                raise ValueError(
                    f"Shapes do not match: original_wavelength {original_wavelength.shape}, continuum_i {continuum_i.shape}"
                )
            return jnp.interp(new_wavelength, original_wavelength, continuum_i)

        resampled_continuum = interp_fn(continuum)

        logger.debug(
            f"new_wavelength: {new_wavelength.shape}, resampled_continuum: {resampled_continuum.shape}"
        )
        # logger.debug(
        #    f"new_wavelength: {new_wavelength}, resampled_continuum: {resampled_continuum}"
        # )
        return new_wavelength, resampled_continuum

    def lines_tf(self, theta):
        return line_predict(theta=theta).nn_predict()

    def calculate_lines(self, theta):
        """
        Calculate the lines using the provided theta parameters.

        My interpretation of the input: these are the parameters from Table 1 in Li et al. 2024
        gammas: power-law slopes (alpha_HeII, alpha_OII, alpha_HeI and alpha_HI)
        log_L_ratios:   flux ratios of the two bluest segments (log F_OII/F_HeII)
                        flux ratios of the second and third segment (log F_HeI/F_OII)
                        fliux ratios of the two reddest segments (log F_HI/F_HeI)
        log_QH: ionization parameter log U (in Illustris ElectronAbundance)
        n_H: hydrogen density (not in log scale!!!) in 1/cm^3 (in Illustris Denisty)
        log_OH_ratio: oxygen abundance (in Illustris GFM_Metals[4]/GFM_Metals[0])
        log_NO_ratio: nitrogen-to-oxygen ratio (in Illustris GFM_Metals[3]/GFM_Metals[4])
        log_CO_ratio: carbon-to-oxygen ratio (in Illustris GFM_Metals[2]/GFM_Metals[4])

        My interpretation of the output:
        first array: wavelengt in Angstrom
        second array: luminosity in erg/s

        Parameters:
        theta (jnp.ndarray): The theta parameters describing the shape of the ionizing spectrum and the ionizing gas properties, 12 values.

        Returns:
        jnp.ndarray: The predicted emission lines.
        lines[0] is the wavelength in Angstrom
        lines[1] is the luminosity in erg/s for the emission lines in each gas cell
        """
        logger = get_logger(self.config.get("logger", None))
        logger.warning(
            "Calculating emission lines assumes that we trust the outcome of the Cue model (Li et al. 2024)."
        )
        par = theta
        wavelength_lines, lines = jax2tf.call_tf(self.lines_tf)(par)

        return wavelength_lines, lines

    def illustris_gas_temp(self, internal_energy_u, electron_abundance):
        """
        Calculation the tempeature for each gas cell in the galaxy.

        Returns the temperature of the gas in the galaxy according to the Illustris simulation.
        See https://www.tng-project.org/data/docs/faq/ under Section General point 6 for more details.

        Parameters:
        rubixdata (RubixData): The RubixData object containing the gas data.

        Returns:
        rubixdata (RubixData): The RubixData object with the gas temperature added to rubixdata.gas.temperature.
        """
        logger = get_logger(self.config.get("logger", None))
        logger.info("Calculating gas temperature")

        # Constants
        x_h = 0.76  # hydrogen mass fraction
        m_p = 1.6726219e-24  # proton mass in CGS units (g)
        gamma = 5 / 3  # adiabatic index
        k_b = 1.38064852e-16  # Boltzmann constant in CGS units (erg/K), https://www.physics.rutgers.edu/~abrooks/342/constants.html

        # Mean molecular weight
        mean_molecular_weight = 4.0 / (1 + 3 * x_h + 4 * x_h * electron_abundance) * m_p

        # Temperature calculation
        temperature = (gamma - 1) * internal_energy_u / k_b * mean_molecular_weight

        return temperature

    def dispersionfactor(self, temperature):
        """
        Calculates the thermal broadening of the emission lines.
        We follow the formular of https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/thermalBroad.html
        To get the dispersion, this factor has to be multiplied by the wavelength of the emission line.
        Expected to be around 1 Angstom for temperatures of 10^4 K.

        Parameters:
        rubixdata (RubixData): The RubixData object containing the gas data.

        Returns:
        rubixdata (RubixData): The RubixData object with the gas dispersion factor added to rubixdata.gas.dispersionfactor.
        """
        logger = get_logger(self.config.get("logger", None))
        logger.info("Calculating dispersion factor")
        logger.warning(
            "The dispersion factor for line width currentl only assumes thermal broadening."
        )
        # Constants (https://www.physics.rutgers.edu/~abrooks/342/constants.html)
        k_B = 1.3807 * 10 ** (-16)  # cm2 g s-2 K-1
        c = 2.99792458 * 10**10  # cm s-1
        m_p = 1.6726e-24  # g

        dispersionfactor = jnp.sqrt((8 * k_B * temperature * np.log(2)) / (m_p * c**2))
        # dispersionfactor = jnp.ones(len(rubixdata.gas.mass))*10
        # dispersionfactor = dispersionfactor[:, None]

        # dispersion = dispersionfactor * wavelengths
        logger.debug(f"dispersionfactor: {dispersionfactor.shape}")
        logger.debug(f"dispersionfactor: {dispersionfactor}")
        return (
            dispersionfactor * 1e7
        )  # because otherwise the dispersion factor is orders of magnitudeds too small

    def gaussian(self, x, a, b, c):
        """
        Returns a Gaussian function.

        Parameters:
        x (jnp.ndarray): The wavelength range.
        a (float): The amplitude of the Gaussian function.
        b (float): The peak position of the Gaussian function.
        c (float): The standard deviation of the Gaussian function.

        Returns:
        jnp.ndarray: The Gaussian function.
        """
        return a * jnp.exp(-((x - jnp.array(b)) ** 2) / (2 * c**2))

    def get_emission_lines(self, theta, internal_energy_u, electron_abundance):
        """
        Returns the spectra of the gas in the galaxy according to the Cue lookup table.
        The spectra takes the luminosity and dispersion factor of each gas cell and calculates the Gaussian emission line and adds all up for each gas cell.

        Parameters:
        rubixdata (RubixData): The RubixData object containing the gas data.

        Returns:
        rubixdata (RubixData): The RubixData object with the gas emission spectra added to rubixdata.gas.emission_spectra.
        """
        logger = get_logger(self.config.get("logger", None))
        logger.info("Calculating gas emission lines")
        wavelengths, emission_peaks = self.calculate_lines(theta)
        emission_peaks = jnp.nan_to_num(emission_peaks, posinf=0.0, neginf=0.0, nan=0.0)
        logger.debug(f"wavelengths: {wavelengths}")
        logger.debug(f"emission_peaks: {emission_peaks}")

        wavelengthrange = self.get_wavelengthrange()
        temperature = self.illustris_gas_temp(internal_energy_u, electron_abundance)
        dispersionfactor = self.dispersionfactor(temperature)

        # Define a function to compute the Gaussian for a single set of parameters
        def compute_gaussian(l, wl, fwhm):
            return self.gaussian(wavelengthrange, l, wl, fwhm)

        # Vectorize the compute_gaussian function
        vmap_gaussian = vmap(compute_gaussian, in_axes=(0, 0, 0))

        # Define a function to compute the spectrum for a single particle
        def compute_spectrum(luminosity, fwhm):
            gaussians = vmap_gaussian(luminosity, wavelengths, fwhm)
            return jnp.sum(gaussians, axis=0)

        spectrum = compute_spectrum(emission_peaks, dispersionfactor * wavelengths)

        logger.debug(
            f"wavelengthrange: {wavelengthrange.shape}, spectra_all: {spectrum.shape}"
        )
        # logger.debug(f"wavelengthrange: {wavelengthrange}, spectra_all: {spectra_all}")
        return wavelengthrange, spectrum

    def get_gas_emission(self, theta, internal_energy_u, electron_abundance):
        """ "
        Returns the added spectrum of gas contnuum and emission lines, both from the Cue lookup

        Parameters:
        rubixdata (RubixData): The RubixData object containing the gas data.

        Returns:
        rubixdata (RubixData): The RubixData object with the gas emission added to rubixdata.gas.spectra.
        """
        logger = get_logger(self.config.get("logger", None))
        logger.info("Calculating gas emission (continuum and emission lines combined)")

        # rubixdata = self.get_emission_lines(rubixdata)
        wave_cont, continuum = self.get_resample_continuum(theta)
        wave_lines, emission_lines = self.get_emission_lines(
            theta, internal_energy_u, electron_abundance
        )

        # continuum = rubixdata.gas.continuum
        # emission_lines = rubixdata.gas.emission_spectra

        gas_emission = continuum + emission_lines

        gas_emission_cleaned = jnp.nan_to_num(
            gas_emission, posinf=0.0, neginf=0.0, nan=0.0
        )

        # rubixdata.gas.spectra = gas_emission_cleaned

        logger.debug(
            f"continuum: {continuum.shape}"  # , emission_lines: {emission_lines.shape}"
        )
        logger.debug(f"gas_emission: {gas_emission.shape}")
        logger.debug(f"gas_emission_cleaned: {gas_emission_cleaned.shape}")
        return wave_cont, gas_emission_cleaned

    def get_gas_emission_flux(self, theta, internal_energy_u, electron_abundance):
        logger = get_logger(self.config.get("logger", None))
        """
        Converts the gas emission spectra to flux.
        Because of very small and very large values, we have to multiply the luminosity and factor with a factor.
        Flux in erg/s/cm^2/Angstrom.

        Parameters:
        rubixdata (RubixData): The RubixData object containing the gas data.

        Returns:
        rubixdata (RubixData): The RubixData object with the gas emission flux added to rubixdata.gas.spectra.
        """
        logger = get_logger(self.config.get("logger", None))
        logger.info("Calculating gas emission flux from luminosity")

        wave, emission = self.get_gas_emission(
            theta, internal_energy_u, electron_abundance
        )

        # Convert luminosity to flux using the preprocessed factor
        luminosity = emission * 1e-30
        luminosity = luminosity * 1e-20

        flux = luminosity * self.factor

        logger.debug(f"luminosity: {luminosity.shape}, flux: {flux.shape}")
        logger.debug(f"luminosity: {luminosity}, flux: {flux}")

        return wave, flux
