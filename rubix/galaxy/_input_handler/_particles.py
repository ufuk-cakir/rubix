'''This file defines the file structure of the data we expect '''

from dataclasses import dataclass, fields
from jaxtyping import Float, Int, Array

@dataclass
class Stars:
    coordinates: Float[Array, "n_stars 3"]
    masses: Float[Array, " n_stars"]
    metallicity: Float[Array, " n_stars"]
    masses: Float[Array, " n_stars"]
    particleIDs: Int[Array, " n_stars"]
    velocities: Float[Array, " n_stars 3"]
    age : Float[Array, " n_stars"]
    
@dataclass
class Gas:
    pass
    # TODO: for now only focus on stars particles