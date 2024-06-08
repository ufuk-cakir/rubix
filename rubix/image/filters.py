import speclite.filters as spl
import equinox as eqx
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jaxtyping import Array, Float
from typing import List


class Filter(eqx.Module):
    """
    A class representing a single filter with wavelength and response data.

    Attributes
    ----------
    wavelength : Float[Array, "n_wavelengths"]
        The wavelengths at which the filter response is defined.
    response : Float[Array, "n_wavelengths"]
        The filter response at the corresponding wavelengths.
    name : str
        The name of the filter.
    """

    wavelength: Float[Array, " n_wavelengths"]
    response: Float[Array, " n_wavelengths"]
    name: str

    def __init__(self, wavelength, response, name: str):
        """
        Initialize the Filter with given wavelength, response, and name.

        Parameters
        ----------
        wavelength : array-like
            The wavelengths at which the filter response is defined.
        response : array-like
            The filter response at the corresponding wavelengths.
        name : str
            The name of the filter.
        """
        self.wavelength = jnp.array(wavelength)
        self.response = jnp.array(response)
        self.name = name

    def plot(self, ax=None):
        """
        Plot the filter response.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes on which to plot. If None, the current axes (`plt.gca()`) will be used.
        """
        if ax is None:
            ax = plt.gca()
        ax.plot(self.wavelength, self.response, label=self.name)
        ax.set_xlabel("Wavelength")
        ax.set_ylabel("Response")
        ax.set_title("Filter Responses")
        ax.legend()

    def __call__(self, new_wavelengths):
        """
        Interpolate the filter response at new wavelengths.

        Parameters
        ----------
        new_wavelengths : array-like
            The new wavelengths at which to interpolate the filter response.

        Returns
        -------
        jax.numpy.ndarray
            The interpolated filter response at the new wavelengths.
        """
        new_response = jnp.interp(new_wavelengths, self.wavelength, self.response)
        return new_response

    def __str__(self):
        """
        Return the name of the filter.

        Returns
        -------
        str
            The name of the filter.
        """
        return self.name

    def __repr__(self):
        """
        Return the name of the filter for representation.

        Returns
        -------
        str
            The name of the filter.
        """
        return self.name


class FilterCurves(eqx.Module):
    """
    A class representing a collection of filter curves.

    Attributes
    ----------
    filters : List[Filter]
        The list of filter objects.
    """

    filters: List[Filter]

    def __init__(self, filters):
        """
        Initialize the FilterCurves with a list of filters.

        Parameters
        ----------
        filters : List[Filter]
            The list of filter objects.
        """
        self.filters = filters

    def plot(self):
        """
        Plot all filter responses on the same figure.
        """
        fig, ax = plt.subplots()
        for filter in self.filters:
            filter.plot(ax)
        plt.show()

    def get_images(self, cube, wavelengths):
        """
        Get the images of a cube of spectra through all filters.
        Parameters
        ----------
        cube : jax.numpy.ndarray
            The cube of spectra.
        wavelengths : jax.numpy.ndarray
            The wavelengths of the cube.
        Returns
        -------
        List[jax.numpy.ndarray]
            The list of images through each filter.
        """
        images = {"filter": [], "image": []}
        for filter in self.filters:
            images["filter"].append(filter.name)
            images["image"].append(convolve_filter_with_cube(filter, cube, wavelengths))

        return images

    def __getitem__(self, key):
        """
        Get a filter by index.

        Parameters
        ----------
        key : int
            The index of the filter to retrieve.

        Returns
        -------
        Filter
            The filter at the specified index.
        """
        return self.filters[key]

    def __len__(self):
        """
        Get the number of filters.

        Returns
        -------
        int
            The number of filters.
        """
        return len(self.filters)


def load_filters(filter_name: str):
    """
    Load multiple filter from the speclite library as Filter objects.

    Parameters
    ----------
    filter_name : str
        Name of the filter to load. e.g 'sdss2010-*' loads all SDSS filters.
    Returns
    -------
    FilterCurves
        FilterCurves object containing the Filter objects.
    """
    # Load the filters from the speclite library
    filters = spl.load_filters(filter_name)

    # Create a list of Filter objects
    # Filter object has wavelength and response attributes
    filter_list = []
    for filter in filters:
        filter_list.append(Filter(filter.wavelength, filter.response, filter.name))

    return FilterCurves(filter_list)


def convolve_filter_with_spectrum(
    filter: Filter,
    spectrum_wavelengths: Float[Array, " n_wavelengths"],
    spectrum_flux: Float[Array, " n_wavelengths"],
) -> Float[Array, "1"]:
    """
    Convolves a single filter with a single spectrum.

    Parameters
    ----------
    filter : Filter
        The filter to convolve with the spectrum.
    spectrum_wavelengths : jax.numpy.ndarray
        The wavelengths of the spectrum.
    spectrum_flux : jax.numpy.ndarray
        The flux values of the spectrum.

    Returns
    -------
    jax.numpy.ndarray
    The convolved flux value. Corresponds to the total flux of the spectrum through the filter.

    """
    # Interpolate the filter response to the spectrum wavelengths
    filter_response = filter(spectrum_wavelengths)

    # Perform the convolution (element-wise multiplication and integration)
    convolved_flux = jnp.trapezoid(
        spectrum_flux * filter_response, spectrum_wavelengths
    )

    return convolved_flux


def convolve_filter_with_cube(
    filter: Filter,
    cube: Float[Array, " n_x n_y n_wavelengths"],
    wavelengths: Float[Array, " n_wavelengths"],
) -> Float[Array, " n_x n_y"]:
    """
    Convolves a single filter with a cube of spectra.
    Parameters
    ----------
    filter : Filter
        The filter to convolve with the cube.
    cube : jax.numpy.ndarray
        The cube of spectra.
    wavelengths : jax.numpy.ndarray
        The wavelengths of the cube.
    Returns
    -------
    jax.numpy.ndarray
        The convolved image. Corresponds to the total flux in each pixel through the filter.
    """
    # Interpolate the filter response to the cube wavelengths
    filter_response = filter(wavelengths)
    # Perform the convolution (element-wise multiplication and integration)
    convolved_image = jnp.trapezoid(cube * filter_response, wavelengths, axis=-1)
    return convolved_image
