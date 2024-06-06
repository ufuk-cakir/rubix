import jax.numpy as jnp


def test_bruzual_charlot():
    from rubix.spectra.ssp.templates import BruzualCharlot2003 as ssp

    print(ssp)
    assert ssp.__class__.__name__ == "Bruzual & Charlot (2003)"
    assert isinstance(ssp.wavelength, jnp.ndarray)
    assert isinstance(ssp.flux, jnp.ndarray)
    assert isinstance(ssp.age, jnp.ndarray)
    assert isinstance(ssp.metallicity, jnp.ndarray)
#
#def test_pyPipe3D():
#    from rubix.spectra.ssp.templates import MaStar_CB19_SLOG_1_5 as ssp
#
#    print(ssp)
#    assert ssp.__class__.__name__ == "Mastar Charlot & Bruzual (2019)"
#    assert isinstance(ssp.wavelength, jnp.ndarray)
#    assert isinstance(ssp.flux, jnp.ndarray)
#    assert isinstance(ssp.age, jnp.ndarray)
#    assert isinstance(ssp.metallicity, jnp.ndarray)