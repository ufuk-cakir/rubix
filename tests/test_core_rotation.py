import pytest
from rubix.core.rotation import get_galaxy_rotation


def test_rotation_info_not_provided():
    config = {"galaxy": {}}
    with pytest.raises(
        ValueError, match="Rotation information not provided in galaxy config"
    ):
        get_galaxy_rotation(config)


def test_alpha_not_provided():
    config = {"galaxy": {"rotation": {"beta": 0, "gamma": 0}}}
    with pytest.raises(ValueError, match="alpha not provided in rotation information"):
        get_galaxy_rotation(config)


def test_beta_not_provided():
    config = {"galaxy": {"rotation": {"alpha": 0, "gamma": 0}}}
    with pytest.raises(ValueError, match="beta not provided in rotation information"):
        get_galaxy_rotation(config)


def test_gamma_not_provided():
    config = {"galaxy": {"rotation": {"alpha": 0, "beta": 0}}}
    with pytest.raises(ValueError, match="gamma not provided in rotation information"):
        get_galaxy_rotation(config)
