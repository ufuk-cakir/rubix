import jax.numpy as jnp
import numpy as np
from jax import vmap
from rubix.spectra.cue.cue.src.cue.line import predict as line_predict
from rubix.spectra.cue.cue.src.cue.continuum import predict as cont_predict
from rubix.core.telescope import get_telescope


class CueGasLookup:
    def __init__(self, config):
        self.config = config

    def illustris_gas_temp(self, rubixdata):
        """
        Returns the temperature of the gas in the galaxy according to the Illustris simulation.
        See https://www.tng-project.org/data/docs/faq/ under Section General point 6 for more details.
        """
        # Convert internal energy
        internal_energy_u = rubixdata.gas.internal_energy
        # Electron abundance
        electron_abundance = rubixdata.gas.electron_abundance
        # Constants
        x_h = 0.76  # hydrogen mass fraction
        m_p = 1.6726219e-24  # proton mass in CGS units (g)
        gamma = 5 / 3  # adiabatic index
        k_b = 1.38064852e-16  # Boltzmann constant in CGS units (erg/K), https://www.physics.rutgers.edu/~abrooks/342/constants.html

        # Mean molecular weight
        mean_molecular_weight = 4.0 / (1 + 3 * x_h + 4 * x_h * electron_abundance) * m_p

        # Temperature calculation
        temperature = (gamma - 1) * internal_energy_u / k_b * mean_molecular_weight

        # Assign temperature to rubixdata
        rubixdata.gas.temperature = temperature

        return rubixdata

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
        second array: luminosity in erg/Hz
        """
        self.lines = line_predict(theta=theta).nn_predict()
        return self.lines

    def calculate_continuum(self, theta):
        """Calculate the continuum using the provided theta parameters."""
        self.continuum = cont_predict(theta=theta).nn_predict()
        return self.continuum

    def get_theta(self, rubixdata):
        alpha_HeII = jnp.full(len(rubixdata.gas.mass), 21.5)
        alpha_OII = jnp.full(len(rubixdata.gas.mass), 14.85)
        alpha_HeI = jnp.full(len(rubixdata.gas.mass), 6.45)
        alpha_HI = jnp.full(len(rubixdata.gas.mass), 3.15)
        log_OII_HeII = jnp.full(len(rubixdata.gas.mass), 4.55)
        log_HeI_OII = jnp.full(len(rubixdata.gas.mass), 0.7)
        log_HI_HeI = jnp.full(len(rubixdata.gas.mass), 0.85)
        log_QH = rubixdata.gas.electron_abundance
        n_H = rubixdata.gas.density
        log_OH_ratio = rubixdata.gas.metals[:, 4] / rubixdata.gas.metals[:, 0]
        log_NO_ratio = rubixdata.gas.metals[:, 3] / rubixdata.gas.metals[:, 4]
        log_CO_ratio = rubixdata.gas.metals[:, 2] / rubixdata.gas.metals[:, 4]
        # theta = []
        # for i in range(len(rubixdata.gas.mass)):
        #    theta_i = [alpha_HeII[i], alpha_OII[i], alpha_HeI[i], alpha_HI[i], log_OII_HeII[i], log_HeI_OII[i], log_HI_HeI[i], log_QH[i], n_H[i], log_OH_ratio[i], log_NO_ratio[i], log_CO_ratio[i]]
        #    theta.append(theta_i)
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
            log_OH_ratio,
            log_NO_ratio,
            log_CO_ratio,
        ]
        theta = jnp.transpose(jnp.array(theta))
        return theta

    def get_wavelengthrange(self, steps=10000):
        """
        Returns a range of wavelengths for the spectra dependent on the wavelength range of the emission lines.
        """
        telescope = get_telescope(self.config)
        wave_start = telescope.wave_range[0]
        wave_end = telescope.wave_range[1]
        wavelengthrange = jnp.linspace(wave_start, wave_end, steps)
        return wavelengthrange

    def get_continuum(self, rubixdata):
        """
        Returns the continuum of the gas in the galaxy according to the cue lookup.
        The continuum is calculated using the provided theta parameters.
        continuum[0] is the wavelength in Angstrom
        continuum[1] is the luminosity in erg/Hz
        Stores the continuum in rubixdata.gas.continuum.
        Output: rubixdata
        """
        theta = self.get_theta(rubixdata)
        # continuum = []
        # for i in range(theta.shape[0]):
        #    continuum_i = self.calculate_continuum(theta[i, :].reshape(1, -1))
        #    continuum.append(continuum_i)
        continuum = self.calculate_continuum(theta)
        rubixdata.gas.continuum = continuum
        # rubixdata.gas.continuum[0] is the wavelength
        # rubixdata.gas.continuum[1][i] is the luminosity for the ith particle
        return rubixdata

    def get_resample_continuum(self, rubixdata):
        """
        Resamples the spectrum to the new wavelength range using interpolation.

        Parameters:
        original_wavelength (jnp.ndarray): The original wavelength array.
        continuum (jnp.ndarray): The original spectrum array.
        new_wavelength (jnp.ndarray): The new wavelength array to resample to.

        Returns:
        rubixdata.gas.continuum (jnp.ndarray): The resampled wavelength and spectrum array.
        """
        new_wavelength = self.get_wavelengthrange()
        continuum = rubixdata.gas.continuum[1]
        original_wavelength = rubixdata.gas.continuum[0]
        resampled_continuum = []
        for i in range(len(rubixdata.gas.mass)):
            resampled_continuum_i = jnp.interp(
                new_wavelength, original_wavelength, continuum[i]
            )
            resampled_continuum.append(resampled_continuum_i)
        # resampled_continuum = jnp.interp(new_wavelength, original_wavelength, continuum)
        rubixdata.gas.continuum = [new_wavelength, resampled_continuum]
        return rubixdata

    def get_emission_peaks(self, rubixdata):
        """
        Returns the wavelength and the luminosity (erg/Hz) for 138 emission lines for each gas cell.
        Stores in rubixdata.gas.emission_peaks.
        """
        theta = self.get_theta(rubixdata)
        emission_peaks = self.calculate_lines(theta)
        rubixdata.gas.emission_peaks = emission_peaks

        emission_peaks_modify = rubixdata.gas.emission_peaks[1]
        # Replace inf values with zero
        emission_peaks_cleaned = jnp.nan_to_num(
            emission_peaks_modify, posinf=0.0, neginf=0.0
        )
        # Update the original array
        rubixdata.gas.emission_peaks = [emission_peaks[0], emission_peaks_cleaned]

        return rubixdata

    def dispersionfactor(self, rubixdata):
        """
        Calculates the thermal broadening of the emission lines.
        We follow the formular of https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/thermalBroad.html
        To get the dispersion, this factor has to be multiplied by the wavelength of the emission line.
        Expected to be around 1 Angstom for temperatures of 10^4 K.
        """
        # Constants (https://www.physics.rutgers.edu/~abrooks/342/constants.html)
        k_B = 1.3807 * 10 ** (-16)  # cm2 g s-2 K-1
        c = 2.99792458 * 10**10  # cm s-1
        m_p = 1.6726e-24  # g

        rubixdata = self.get_emission_peaks(rubixdata)
        rubixdata = self.illustris_gas_temp(rubixdata)
        wavelengths = rubixdata.gas.emission_peaks[0]

        dispersionfactor = np.sqrt(
            (8 * k_B * rubixdata.gas.temperature * np.log(2)) / (m_p * c**2)
        )
        dispersionfactor = dispersionfactor[:, None]

        dispersion = dispersionfactor * wavelengths
        rubixdata.gas.dispersionfactor = dispersion
        return rubixdata

    def gaussian(self, x, a, b, c):
        """
        Returns a Gaussian function.
        """
        return a * jnp.exp(-((x - jnp.array(b)) ** 2) / (2 * c**2))

    def get_emission_lines(self, rubixdata):
        """
        Returns the spectra of the gas in the galaxy according to the Cloudy lookup table.
        The spectra takes the flux and dispersion factor of each gas cell and calculates the Gaussian emission line and adds all up for each gas cell.
        Stores the spectra in rubixdata.gas.spectra.
        """
        # get wavelengths of lookup and wavelengthrange of telescope
        rubixdata = self.get_emission_peaks(rubixdata)
        wavelengths = rubixdata.gas.emission_peaks[0]
        wave_start = get_telescope(self.config).wave_range[0]
        wave_end = get_telescope(self.config).wave_range[1]
        wavelengthrange = self.get_wavelengthrange()
        # update rubixdata with temperature, dispersionfactor and luminosity
        rubixdata = self.illustris_gas_temp(rubixdata)
        rubixdata = self.dispersionfactor(rubixdata)

        rubixdata = self.get_emission_peaks(rubixdata)

        spectra_all = []

        # Define a function to compute the Gaussian for a single set of parameters
        def compute_gaussian(l, wl, fwhm):
            return self.gaussian(wavelengthrange, l, wl, fwhm)

        # Vectorize the compute_gaussian function
        vmap_gaussian = vmap(compute_gaussian, in_axes=(0, 0, 0))

        # Define a function to compute the spectrum for a single particle
        def compute_spectrum(luminosity, fwhm):
            gaussians = vmap_gaussian(luminosity, wavelengths, fwhm)
            return jnp.sum(gaussians, axis=0)

        # Vectorize the compute_spectrum function over all particles
        vmap_spectrum = vmap(compute_spectrum, in_axes=(0, 0))

        # Compute the spectra for all particles
        spectra_all = vmap_spectrum(
            rubixdata.gas.emission_peaks[1], rubixdata.gas.dispersionfactor
        )

        # Store the spectra and wavelength range in rubixdata
        rubixdata.gas.spectra = spectra_all
        rubixdata.gas.wavelengthrange = wavelengthrange

        return rubixdata
