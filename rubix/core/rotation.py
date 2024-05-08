from typing import Dict
import jax


def get_galaxy_rotation(config: dict):
    def rotate_galaxy(inputs: dict[str, jax.Array], type:str = "face-on"):
        print("rotating galaxy: ", type)
        coords = inputs["coords"]
        velocities = inputs["velocities"]
        inputs["coords"] = coords
        inputs["velocities"] = velocities
        return inputs

    return rotate_galaxy
