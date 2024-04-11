"""This file defines the file structure of the data we expect """

from dataclasses import dataclass, fields
from jaxtyping import Float, Int, Array


@dataclass
class Stars:
    coords: Float[Array, "n_stars 3"]
    mass: Float[Array, " n_stars"]
    metallicity: Float[Array, " n_stars"]
    mass: Float[Array, " n_stars"]
    velocity: Float[Array, " n_stars 3"]
    age: Float[Array, " n_stars"]
    # [RBX-12] TODO: Check if we need particle ids?
    # particleIDs: Int[Array, " n_stars"]


@dataclass
class Gas:
    pass
    # TODO: for now only focus on stars particles
