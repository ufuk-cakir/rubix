import pytest
import jax.numpy as jnp
from rubix.pipeline import transformer as transformer
from rubix.core.pipeline import RubixPipeline
from rubix.spectra.ssp.grid import SSPGrid
from rubix.telescope.base import BaseTelescope


# Dummy data functions
def dummy_get_rubix_data(config):
    return (
        jnp.array([[0, 0, 0]]),  # coords
        jnp.array([[0, 0, 0]]),  # velocities
        jnp.array([0.1]),  # metallicity
        jnp.array([1.0]),  # mass
        jnp.array([1.0]),  # age
    )


@pytest.fixture
def setup_environment(monkeypatch):
    # Monkeypatch the necessary data functions to return dummy data
    monkeypatch.setattr("rubix.core.pipeline.get_rubix_data", dummy_get_rubix_data)


# Dummy user configuration
dummy_user_config = {
    "pipeline": {"name": "calc_ifu"},
    "logger": {
        "log_level": "DEBUG",
        "log_file_path": None,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    },
    "simulation": {
        "name": "IllustrisTNG",
        "args": {
            "path": "data/galaxy-id-14.hdf5",
        },
    },
    "output_path": "output",
    "telescope": {"name": "MUSE"},
    "cosmology": {"name": "PLANCK15"},
    "galaxy": {"dist_z": 0.1},
    "ssp": {
        "template": {"name": "BruzualCharlot2003"},
    },
}


def test_rubix_pipeline_not_implemented(setup_environment):
    config = {"pipeline": {"name": "dummy"}}
    with pytest.raises(
        ValueError, match="Pipeline dummy not found in the configuration"
    ):
        pipeline = RubixPipeline(user_config=config)  # noqa


def test_rubix_pipeline_gradient_not_implemented(setup_environment):
    pipeline = RubixPipeline(user_config=dummy_user_config)
    with pytest.raises(
        NotImplementedError, match="Gradient calculation is not implemented yet"
    ):
        pipeline.gradient()


def test_rubix_pipeline_run(setup_environment):
    pipeline = RubixPipeline(user_config=dummy_user_config)
    output = pipeline.run()

    # Check if output is as expected
    assert "coords" in output
    assert "velocities" in output
    assert "metallicity" in output
    assert "mass" in output
    assert "age" in output
    assert "spectra" in output

    assert isinstance(pipeline.telescope, BaseTelescope)
    assert isinstance(pipeline.ssp, SSPGrid)
