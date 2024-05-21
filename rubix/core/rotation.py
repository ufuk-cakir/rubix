from typing import Dict
import jax


# TODO: implement the real rotation function
# this is currently only a placeholder
def get_galaxy_rotation(config: dict):
    def rotate_galaxy(inputs: dict[str, jax.Array], type: str = "face-on"):
        # TODO: I guess this breaks the functional programming paradigm?
        print("rotating galaxy: ", type)
        coords = inputs["coords"]
        velocities = inputs["velocities"]
        inputs["coords"] = coords
        inputs["velocities"] = velocities
        return inputs

    return rotate_galaxy
