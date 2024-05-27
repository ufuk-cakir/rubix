from jaxtyping import Float, Array
import equinox as eqx
import jax.numpy as jnp
from astropy import units as u
import os, wget
import h5py
import jax.numpy as jnp
from rubix import config as rubix_config
from typing import Dict
from interpax import interp2d
from jax.tree_util import Partial
from dataclasses import dataclass

SSP_UNITS = rubix_config["ssp"]["units"]


@dataclass
class SSPGrid:
    """
    Base class for all SSP
    """

    age: Float[Array, " age_bins"]
    metallicity: Float[Array, " metallicity_bins"]
    wavelength: Float[Array, " wavelength_bins"]
    flux: Float[Array, "metallicity_bins age_bins wavelength_bins"]
    # This does not work with jax.jit, gives error that str is not valid Jax type
    # units: Dict[str, str] = eqx.field(default_factory=dict)

    def __init__(self, age, metallicity, wavelength, flux):
        self.age = jnp.asarray(age)
        self.metallicity = jnp.asarray(metallicity)
        self.wavelength = jnp.asarray(wavelength)
        self.flux = jnp.asarray(flux)
        # self.units = SSP_UNITS

    def get_lookup(self, method="cubic", extrap=0):
        """Returns a 2D interpolation function for the SSP grid.

        The function can be called with metallicity and age as arguments to get the flux at that metallicity and age.

        Parameters
        ----------
        method : str
            The method to use for interpolation. Default is "cubic".
        extrap: float, bool or tuple
            The value to return for points outside the interpolation domain. Default is 0.
            See https://interpax.readthedocs.io/en/latest/_api/interpax.Interpolator2D.html#interpax.Interpolator2D

        Returns
        -------
        Interp2D
            The 2D interpolation function.

        Examples
        --------
        >>> grid = SSPGrid(...)
        >>> lookup = grid.get_lookup()
        >>> metallicity = 0.02
        >>> age = 1e9
        >>> flux = lookup(metallicity, age)
        """

        # Bind the SSP grid to the interpolation function
        interp = Partial(
            interp2d,
            method=method,
            x=self.metallicity,
            y=self.age,
            f=self.flux,
            extrap=extrap,
        )
        interp.__doc__ = (
            "Interpolation function for SSP grid, args: f(metallicity, age)"
        )
        return interp

    @staticmethod
    def convert_units(data, from_units, to_units):
        quantity = u.Quantity(data, from_units)
        return quantity.to(to_units).value
    
    @staticmethod
    def checkout_SSP_template(config: dict, file_location: str):
        """
        Check if the SSP template exists on disk, if not download it 
        from the given URL in the configuration dictionary.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        file_location : str
            Location to save the template file.
        
        Returns
        -------
        file_path : str
            The path to the file.
        """ 

        file_path = os.path.join(file_location, config["file_name"])

        if not os.path.exists(file_path):
            print(f'[SSPModels] File {file_path} not found. Downloading it from {config["url"]}')
            try:
                wget.download(config["url"], file_location) 
            except OSError as err:
                print(f'[SSPModels] OS error: {err}')
            except Exception as ex:
                print(f'[SSPModels] Unexpected {ex=}, {type(ex)=}')
                raise ValueError(f"Could not download file {file_path} from url {config["url"]}.")    
        
        return file_path

    @classmethod
    def from_file(cls, config: dict, file_location: str) -> "SSPGrid":
        """
        Template function to load a SSP grid from a file.

        Parameters
        ----------
        config : dict
            Configuration dictionary.
        
        file_location : str
            Location of the file.

        Returns
        -------
        SSPGrid
            The SSP grid in the correct units.
        """

        # Initialize an empty zero length array for each field
        # in the SSP configuration.
        # Actual loading of templates needs to be implemented in the subclasses.

        ssp_data = {}
        for field_name in config["fields"].items():
            ssp_data[field_name] = jnp.empty(0)

        grid = cls(**ssp_data)
        grid.__class__.__name__ = config["name"]
        return grid

@SSPGrid
class HDF5SSPGrid:
    """
    Class for SSP models stored in HDF5 format.
    Mainly used for custom collection of Bruzual & Charlot 2003 models and MILES models .
    """

    # Do we need this again or is this taken care of by inheriting from SSPGrid?
    age: Float[Array, " age_bins"]
    metallicity: Float[Array, " metallicity_bins"]
    wavelength: Float[Array, " wavelength_bins"]
    flux: Float[Array, "metallicity_bins age_bins wavelength_bins"]
    # This does not work with jax.jit, gives error that str is not valid Jax type
    # units: Dict[str, str] = eqx.field(default_factory=dict)

    def __init__(self, age, metallicity, wavelength, flux):
        super().__init__(age, metallicity, wavelength, flux)

    @classmethod
    def from_file(cls, config: dict, file_location: str) -> "SSPGrid":
        """
        Load a SSP grid from a HDF5 file.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        SSPGrid
            The SSP grid in the correct units.
        """

        if config.get("format", "").lower() != "hdf5":
            raise ValueError("Configured file format is not HDF5.")

        file_path = cls.checkout_SSP_template(config, file_location)
        
        ssp_data = {}
        with h5py.File(file_path, "r") as f:
            for field_name, field_info in config["fields"].items():
                data = f[field_info["name"]][:]  # type: ignore
                data = jnp.power(10, data) if field_info["in_log"] else data  # type: ignore
                data = cls.convert_units(
                    data, field_info["units"], SSP_UNITS[field_name]
                )
                ssp_data[field_name] = data

        grid = cls(**ssp_data)
        grid.__class__.__name__ = config["name"]
        return grid
    
@SSPGrid
class pyPipe3DSSPGrid:
    """
    Class for all SSP models supported by the pyPipe3D project.
    See http://ifs.astroscu.unam.mx/pyPipe3D/templates/ for more information.
    """

    age: Float[Array, " age_bins"]
    metallicity: Float[Array, " metallicity_bins"]
    wavelength: Float[Array, " wavelength_bins"]
    flux: Float[Array, "metallicity_bins age_bins wavelength_bins"]
    # This does not work with jax.jit, gives error that str is not valid Jax type
    # units: Dict[str, str] = eqx.field(default_factory=dict)

    def __init__(self, age, metallicity, wavelength, flux):
        super().__init__(age, metallicity, wavelength, flux)

    @staticmethod
    def get_wavelength_from_header(header, wave_axis=None):
        """
        Generates a wavelength array using `header`, a :class:`astropy.io.fits.header.Header`
        instance, at axis `wave_axis`.

        wavelengths = CRVAL + CDELT*([0, 1, ..., NAXIS] + 1 - CRPIX)

        adapted from https://github.com/reginasar/TNG_MaNGA_mocks/blob/3229dd47b441aef380ef7dbfdf110f39e5c5a77c/sin_ifu_clean.py#L1466

        Parameters
        ----------
        header : :class:`astropy.io.fits.header.Header`
            FITS header with spectral data.

        wave_axis : int, optional
            The axis where the wavelength information is stored in `header`,
            (CRVAL, CDELT, NAXIS, CRPIX).
            Defaults to 1.

        Returns
        -------
        array like
            Wavelengths array.

            wavelengths = CRVAL + CDELT*([0, 1, ..., NAXIS] + 1 - CRPIX)
        """
        if wave_axis is None:
            wave_axis = 1
        h = header
        crval = h[f'CRVAL{wave_axis}']
        cdelt = h[f'CDELT{wave_axis}']
        naxis = h[f'NAXIS{wave_axis}']
        crpix = h[f'CRPIX{wave_axis}']
        if not cdelt:
            cdelt = 1
        return crval + cdelt*(np.arange(naxis) + 1 - crpix)
    
    @staticmethod
    def get_normalization_wavelength(header, wavelength, flux_models, n_models):
        """ 
        Search for the normalization wavelength at the FITS header.
        If the key WAVENORM does not exists in the header, sweeps all the
        models looking for the wavelengths where the flux is closer to 1,
        calculates the median of those wavelengths and returns it.

        TODO: defines a better normalization wavelength if it's not present
        in the header.

        adapted from https://github.com/reginasar/TNG_MaNGA_mocks/blob/3229dd47b441aef380ef7dbfdf110f39e5c5a77c/sin_ifu_clean.py#L1466

        Parameters
        ----------
        header : :class:`astropy.io.fits.header.Header`
            FITS header with spectral data.
    
        wavelength : array like, wavelength of the model SSPs.

        flux_models : array like, flux of the model SSPs.

        n_models : int, number of models in the SSP grid.

        Returns
        -------
        float
            The normalization wavelength.
        """
        try:
            wave_norm = header['WAVENORM']
        except Exception as ex:
            _closer = 1e-6
            probable_wavenorms = np.hstack([wavelength[(np.abs(flux_models[i] - 1) < _closer)]
                                        for i in range(n_models)])
            wave_norm = np.median(probable_wavenorms)
        
            print(f'[SSPModels] {ex}')
            print(f'[SSPModels] setting normalization wavelength to {wave_norm} A')
        return wave_norm

    @staticmethod    
    def get_tZ_models(header, n_models):
        """ 
        Reads the values of age, metallicity and mass-to-light at the
        normalization flux from the SSP models FITS file.

        adapted from https://github.com/reginasar/TNG_MaNGA_mocks/blob/3229dd47b441aef380ef7dbfdf110f39e5c5a77c/sin_ifu_clean.py#L1466

        Parameters
        ----------
        header : :class:`astropy.io.fits.header.Header`
            FITS header with spectral data.
    
        n_models : int, number of models in the SSP grid.

        Returns
        -------
        array like
            Ages, in Gyr, in the sequence as they appear in FITS data.

        array like
            Metallicities in the sequence as they appear in FITS data.

        array like
            Mass-to-light value at the normalization wavelength.
        """
        ages = np.zeros(n_models, dtype='float')
        Zs = ages.copy()
        mtol = ages.copy()
        for i in range(n_models):
            mult = {'Gyr': 1, 'Myr': 1/1000}
            name_read_split = header[f'NAME{i}'].split('_')
            # removes 'spec_ssp_' from the name
            name_read_split = name_read_split[2:]
            _age = name_read_split[0]
            if 'yr' in _age:
                mult = mult[_age[-3:]]  # Gyr or Myr
                _age = _age[:-3]
            else:
                mult = 1  # Gyr
            age = mult*np.float64(_age)
            _Z = name_read_split[1].split('.')[0]
            Z = np.float64(_Z.replace('z', '0.'))
            ages[i] = age
            Zs[i] = Z
            mtol[i] = 1/np.float64(header[f'NORM{i}'])
            
        return jnp.unique(ages), jnp.unique(Zs), mtol

    @classmethod
    def from_file(cls, config: dict, file_location: str) -> "SSPGrid":
        """
        Load a SSP grid from a fits file in pyPipe3D format.

        Parameters
        ----------
        config : dict
            Configuration dictionary.

        Returns
        -------
        SSPGrid
            The SSP grid in the correct units.
        """
        if config.get("format", "").lower() != "fits":
            raise ValueError("Configured file format is not fits.")

        file_path = cls.checkout_SSP_template(config, file_location)

        ssp_data = {}
        with fits.open(file_path) as f:
            _header = f[0].header
            #n_wave = _header['NAXIS1']
            n_models = _header['NAXIS2']

            # pyPIPE3D uses the key WAVENORM to store the normalization wavelength
            # not sure what this is actually used for in the end.
            # Here we enable reading it, but we should make sure we understand what it is used for.
            # normalization_wavelength = get_normalization_wavelength(_header, wavelength, flux_models, n_models)
            ages, metallicities, m2l = cls.get_tZ_models(_header, n_models)
            wavelength = cls.get_wavelength_from_header(_header)

            # read in the flux of the models and multiply by the mass-to-light ratio to get the flux in Lsun/Msun
            # see also eq. A1 here https://arxiv.org/pdf/1811.04856.pdf
            template_flux = f[0].data * m2l[:, None, None]
            # reshape and bring into the correct order of metallcity, age, wavelength
            # to conform with the SSPGrid dataclass
            flux_models = jnp.swapaxes(template_flux, 0, 1)

            for field_name, field_info in config["fields"].items():
                if field_name == 'flux':
                    data = flux_models
                elif field_name == 'wavelength':
                    data = wavelength
                elif field_name == 'age':
                    data = ages
                elif field_name == 'metallicity':
                    data = metallicities
                elif field_name == 'mass_to_light':
                    data = mass_to_light
                else:
                    raise ValueError(f'Field {field_name} not recognized')
                
                data = jnp.power(10, data) if field_info["in_log"] else data  # type: ignore
                data = cls.convert_units(
                    data, field_info["units"], SSP_UNITS[field_name]
                )
                ssp_data[field_name] = data

        grid = cls(**ssp_data)
        grid.__class__.__name__ = config["name"]
        return grid


#TODO: build another class that handles eMILES, sMILES templates that are also used by the GECKOS survey.
# those will also have alpha enhancement and not only metallicity dependence. might need some changes to the 
# interpolation function further down the pipeline... 