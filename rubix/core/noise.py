import jax.numpy as jnp
import jax
from rubix.telescope.noise.noise import (
    calculate_noise_cube,
    SUPPORTED_NOISE_DISTRIBUTIONS,
)

from rubix.logger import get_logger


def get_apply_noise(config: dict):

    if "noise" not in config["telescope"]:
        raise ValueError("Noise information not provided in telescope config")

    if "signal_to_noise" not in config["telescope"]["noise"]:
        raise ValueError("Signal to noise information not provided in noise config")

    if "noise_distribution" not in config["telescope"]["noise"]:
        raise ValueError(
            f"Noise distribution not provided in noise config. Currently supported distributions are: {SUPPORTED_NOISE_DISTRIBUTIONS}"
        )

    # Get the signal to noise ratio
    signal_to_noise = config["telescope"]["noise"]["signal_to_noise"]

    # Get the noise distribution
    noise_distribution = config["telescope"]["noise"]["noise_distribution"]

    logger = get_logger()

    def apply_noise(inputs: dict[str, jax.Array]) -> dict[str, jax.Array]:
        logger.info(
            f"Applying noise to datacube with signal to noise ratio: {signal_to_noise} and noise distribution: {noise_distribution}"
        )
        datacube = inputs["datacube"]

        # Define S2n for each spaxel
        S2N = jnp.ones(datacube.shape[:2]) * signal_to_noise

        # Calculate the noise cube
        noise_cube = calculate_noise_cube(
            datacube, S2N, noise_distribution=noise_distribution
        )

        # Add noise to the datacube
        inputs["datacube"] += noise_cube
        return inputs

    return apply_noise
