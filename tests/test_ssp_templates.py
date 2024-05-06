import jax.numpy as jnp


def test_bruzual_charlot():
    from rubix.spectra.ssp.templates import BruzualCharlot2003 as ssp

    print(ssp)
    assert ssp.__class__.__name__ == "Bruzual & Charlot (2003)"
    assert isinstance(ssp.wavelength, jnp.ndarray)
    assert isinstance(ssp.flux, jnp.ndarray)
    assert isinstance(ssp.age, jnp.ndarray)
    assert isinstance(ssp.metallicity, jnp.ndarray)
