import pytest

from rubix.core.noise import get_apply_noise


def test_no_noise_in_config():
    config = {"telescope": {}}
    with pytest.raises(
        ValueError, match="Noise information not provided in telescope config"
    ):
        get_apply_noise(config)


def test_no_signal_to_noise_in_noise_config():
    config = {"telescope": {"noise": {}}}
    with pytest.raises(
        ValueError, match="Signal to noise information not provided in noise config"
    ):
        get_apply_noise(config)


def test_no_noise_distribution_in_noise_config():
    config = {"telescope": {"noise": {"signal_to_noise": 10}}}
    with pytest.raises(ValueError) as e:
        get_apply_noise(config)
