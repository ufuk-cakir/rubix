from typing import Dict
import jax
from rubix.logger import get_logger
from rubix.galaxy.alignment import rotate_galaxy as rotate_galaxy_core


def get_galaxy_rotation(config: dict):
    # Check if rotation information is provided under galaxy config
    if "rotation" not in config["galaxy"]:
        raise ValueError("Rotation information not provided in galaxy config")

    logger = get_logger(config.get("logger", None))
    # Check if type is provided
    if "type" in config["galaxy"]["rotation"]:
        # Check if type is valid: face-on or edge-on
        if config["galaxy"]["rotation"]["type"] not in ["face-on", "edge-on"]:
            raise ValueError("Invalid type provided in rotation information")

        # if type is face on, alpha = beta = gamma = 0
        # if type is edge on, alpha = 90, beta = gamma = 0
        if config["galaxy"]["rotation"]["type"] == "face-on":
            logger.debug("Roataion Type found: Face-on")
            alpha = 0
            beta = 0
            gamma = 0

        else:
            # type is edge-on
            logger.debug("Roataion Type found: edge-on")
            alpha = 90
            beta = 0
            gamma = 0

    else:
        # If type is not provided, then alpha, beta, gamma should be set
        # Check if alpha, beta, gamma are provided
        for key in ["alpha", "beta", "gamma"]:
            if key not in config["galaxy"]["rotation"]:
                raise ValueError(f"{key} not provided in rotation information")

        # Get the rotation angles from the user config
        alpha = config["galaxy"]["rotation"]["alpha"]
        beta = config["galaxy"]["rotation"]["beta"]
        gamma = config["galaxy"]["rotation"]["gamma"]

    def rotate_galaxy(inputs: dict[str, jax.Array], type: str = "face-on"):
        logger.info(f"Rotating galaxy with alpha={alpha}, beta={beta}, gamma={gamma}")

        # Get the inputs
        coords = inputs["coords"]
        velocities = inputs["velocities"]
        masses = inputs["mass"]
        halfmass_radius = inputs["halfmassrad_stars"]

        # Rotate the galaxy
        coords, velocities = rotate_galaxy_core(
            positions=coords,
            velocities=velocities,
            masses=masses,
            halfmass_radius=halfmass_radius,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )

        # Update the inputs
        inputs["coords"] = coords
        inputs["velocities"] = velocities
        return inputs

    return rotate_galaxy
