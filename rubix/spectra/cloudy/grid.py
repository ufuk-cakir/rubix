import pickle
import jax.numpy as jnp
import numpy as np
from jax import vmap

# redshift 0 to 10
# metals -5 to 0.5
# hden -7 to 4
# temp 2 to 8

# Lines
# O  6 1031.91A
# O  6 1037.62A
# BLND 1035.00A
# H  1 1215.67A
# C  4 1548.19A
# C  4 1550.78A
# BLND 1549.00A
# HE 2 1640.43A
# C  3 1906.68A
# C  3 1908.73A
# MG 2 2795.53A
# MG 2 2802.71A


class CloudyGasLookup:
    @staticmethod
    def load_anything(filepath):
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def __init__(self, datafile_path):
        self.datafile_path = datafile_path
        self.data = self.load_anything(datafile_path)
        self.line_names = self.data["line_name"]
        self.line_emissivity = self.data["line_emissivity"]
        self.redshift_grid = self.data["redshift"]
        self.metallicity_grid = self.data["metals"]
        self.hden_grid = self.data["hden"]
        self.temp_grid = self.data["temp"]

    def get_line_names(self):
        return self.line_names

    def get_wavelength(self, line_name):
        # Split the line name by spaces and take the last part
        wavelength_str = line_name.split()[-1]
        # Remove the 'A' character and convert to float
        wavelength = float(wavelength_str[:-1])
        return wavelength

    def get_all_wavelengths(self):
        line_names = self.get_line_names()
        wavelengths = [self.get_wavelength(line_name) for line_name in line_names]
        return jnp.array(wavelengths)

    def find_nearest(self, grid, value):
        """
        Returns the index of the nearest value in an grid.
        For the Cloudy lookup, we need to find the nearest value in the Cloudy grid.
        """
        return jnp.argmin(jnp.abs(grid - value))

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

    def get_intrinsic_emissivity(self, rubixdata):
        """
        Returns the intrinsic emissivity of the gas in the galaxy according to the Cloudy lookup table.
        Takes the metallicity, hden, and temp of each gas cell and finds the nearest grid point in the Cloudy lookup table.
        Gets the intrinsic emissivity for all lines available in the lookup table.
        Stores the intrinsic emissivity in rubixdata.gas.intr_emissivity (unit of intrinsic emissivity: erg s^-1 cm^-3).
        """
        # Calculate temperature
        rubixdata = self.illustris_gas_temp(rubixdata)

        # Find nearest indices in the lookup grids
        metallicity_indices = vmap(self.find_nearest, in_axes=(None, 0))(
            self.metallicity_grid, jnp.log10(rubixdata.gas.metallicity)
        )
        hden_indices = vmap(self.find_nearest, in_axes=(None, 0))(
            self.hden_grid, jnp.log10(rubixdata.gas.density)
        )
        temp_indices = vmap(self.find_nearest, in_axes=(None, 0))(
            self.temp_grid, jnp.log10(rubixdata.gas.temperature)
        )
        redshift_idx = self.find_nearest(self.redshift_grid, rubixdata.galaxy.redshift)

        # Define a function to get emissivity for a single particle
        def get_emissivity_for_one_particle(metallicity_idx, hden_idx, temp_idx):
            # return jnp.array([line_data[redshift_idx, metallicity_idx, hden_idx, temp_idx] for line_data in self.line_emissivity])
            return vmap(
                lambda line_data: line_data[
                    redshift_idx, metallicity_idx, hden_idx, temp_idx
                ]
            )(jnp.array(self.line_emissivity))

        vmap_get_emissivity = vmap(get_emissivity_for_one_particle, in_axes=(0, 0, 0))

        # Calculate intrinsic emissivity for all particles
        intr_emissivity_all_particles = vmap_get_emissivity(
            metallicity_indices, hden_indices, temp_indices
        )

        # Stack the arrays to form a single large jnp.array
        rubixdata.gas.intr_emissivity = jnp.stack(intr_emissivity_all_particles, axis=0)

        return rubixdata

    def get_luminosity(self, rubixdata):
        """
        Returns the luminosity of the gas particles in the galaxy.
        Takes the emergent emissivity and the mass and volume of each gas cell and calculates the luminosity.
        Stores the luminosity in rubixdata.gas.luminosity.
        """
        rubixdata = self.get_intrinsic_emissivity(rubixdata)
        intr_emissivity = rubixdata.gas.intr_emissivity
        mass = rubixdata.gas.mass
        density = rubixdata.gas.density
        factor = mass / density  # Shape (n,)
        # Reshape factor to be broadcastable with intr_emissivity
        factor = factor[:, None]  # Shape (n, 1)
        # Multiply each value in each row of intr_emissivity by the corresponding factor
        luminosity = intr_emissivity * factor
        rubixdata.gas.luminosity = luminosity
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

        wavelengths = self.get_all_wavelengths()
        rubixdata = self.illustris_gas_temp(rubixdata)

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

    def get_wavelengthrange(self, steps=4000):
        """
        Returns a range of wavelengths for the spectra dependent on the wavelength range of the emission lines.
        """
        wavelengths = self.get_all_wavelengths()
        wavelengthrange = jnp.linspace(
            wavelengths[0] * 0.9, wavelengths[-1] * 1.1, steps
        )
        return wavelengthrange

    def get_spectra(self, rubixdata):
        """
        Returns the spectra of the gas in the galaxy according to the Cloudy lookup table.
        The spectra takes the flux and dispersion factor of each gas cell and calculates the Gaussian emission line and adds all up for each gas cell.
        Stores the spectra in rubixdata.gas.spectra.
        """
        # get wavelengths and wavelengthrange
        wavelengths = self.get_all_wavelengths()
        wavelengthrange = self.get_wavelengthrange()
        # update rubixdata with temperature, dispersionfactor and luminosity
        rubixdata = self.illustris_gas_temp(rubixdata)
        rubixdata = self.dispersionfactor(rubixdata)

        rubixdata = self.get_luminosity(rubixdata)

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
            rubixdata.gas.luminosity, rubixdata.gas.dispersionfactor
        )

        # Store the spectra and wavelength range in rubixdata
        rubixdata.gas.spectra = spectra_all
        rubixdata.gas.wavelengthrange = wavelengthrange

        return rubixdata
