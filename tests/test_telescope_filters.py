import pytest
from rubix.telescope.filters.filters import (
    Filter,
    FilterCurves,
    convolve_filter_with_spectra,
    load_filters,
)

import jax.numpy as jnp

import matplotlib
import matplotlib.pyplot as plt

# Use the Agg backend for testing to avoid opening a figure window
matplotlib.use("Agg")


def test_filter_initialization():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    assert jnp.all(filt.wavelength == wavelength)
    assert jnp.all(filt.response == response)
    assert filt.name == name


def test_filter_plot():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    # Test with provided axes
    fig, ax = plt.subplots()
    filt.plot(ax)
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == name
    plt.close(fig)

    # Test with no axes provided
    fig = plt.figure()
    filt.plot()  # This should use plt.gca()
    ax = plt.gca()
    assert len(ax.lines) == 1
    assert ax.lines[0].get_label() == name
    plt.close(fig)


def test_filter_call():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    new_wavelengths = jnp.array([450, 550])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    interpolated_response = filt(new_wavelengths)

    assert jnp.allclose(interpolated_response, jnp.array([0.3, 0.7]), atol=1e-2)


def test_filter_curves_initialization():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)
    filt_curves = FilterCurves([filt])

    assert len(filt_curves.filters) == 1
    assert filt_curves.filters[0] == filt


def test_filter_curves_plot():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    name = "Test Filter"

    filt = Filter(wavelength, response, name)
    filt_curves = FilterCurves([filt])

    # filt_curves.plot()


def test_apply_filter_curves():
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([0.1, 0.5, 0.9])
    spectra = jnp.array([1.0, 2.0, 3.0])
    cube = jnp.ones((2, 2, 3))
    name = "Test Filter"

    filt = Filter(wavelength, response, name)
    filt_curves = FilterCurves([filt])

    images = filt_curves.apply_filter_curves(cube, wavelength)

    assert len(images["image"]) == 1
    assert images["filter"][0] == name
    assert images["image"][0].shape == (2, 2)


def test_convolve_filter_with_spectra_single():
    # Define a simple filter and spectrum where the result of convolution is known
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([1.0, 1.0, 1.0])  # Flat response for simplicity
    spectra = jnp.array([2.0, 2.0, 2.0])  # Uniform spectra
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    # The convolved flux should be the sum of the spectra values times the spacing between wavelengths
    # Here, the spacing between each wavelength is 100, so the result should be 2*100 + 2*100 = 400
    convolved_flux = convolve_filter_with_spectra(filt, spectra, wavelength)

    expected_flux = 2.0 * 100 + 2.0 * 100  # Using trapezoidal integration

    assert jnp.isclose(convolved_flux, expected_flux, atol=1e-2)


def test_convolve_filter_with_spectra_cube():
    # Define a simple filter and spectrum cube where the result of convolution is known
    wavelength = jnp.array([400, 500, 600])
    response = jnp.array([1.0, 1.0, 1.0])  # Flat response for simplicity
    cube = jnp.ones((2, 2, 3))  # Uniform cube of spectra, all values are 1
    name = "Test Filter"

    filt = Filter(wavelength, response, name)

    # The convolved image should be the sum of the spectra values times the spacing between wavelengths
    # Here, the spacing between each wavelength is 100, and each spectrum sum is 1*100 + 1*100 = 200
    # Thus, each pixel in the convolved image should be 1*100 + 1*100 = 200
    convolved_image = convolve_filter_with_spectra(filt, cube, wavelength)

    expected_image_value = 1.0 * 100 + 1.0 * 100  # Using trapezoidal integration

    assert convolved_image.shape == (2, 2)
    assert jnp.allclose(convolved_image, expected_image_value, atol=1e-2)


# Load filter tests
def test_load_filters():
    filter_curves = load_filters("sdss2010-*")
    assert isinstance(filter_curves, FilterCurves)
    assert len(filter_curves.filters) > 0
