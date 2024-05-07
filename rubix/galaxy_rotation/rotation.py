from rubix import logger
from jaxtyping import Array, Float
import jax.numpy as jnp

def moment_of_inertia_tensor(positions, masses, half_light_radius):
    """Calculate the moment of inertia tensor for a given set of positions and masses within the half-light radius.
       Assumes the galaxy is already centered.
       
    Parameters
    ----------
    positions : jnp.ndarray
        The positions of the particles.
    masses : jnp.ndarray
        The masses of the particles.
    half_light_radius : float
        The half-light radius of the galaxy.

    Returns
    -------
    jnp.ndarray
        The moment of inertia tensor.
       """
    
    distances = jnp.sqrt(jnp.sum(positions**2, axis=1))  # Direct calculation since positions are already centered
    
    # Mask to consider only particles within the half-light radius
    within_half_light_radius = distances <= half_light_radius
    
    # Filter positions and masses
    filtered_positions = positions[within_half_light_radius]
    filtered_masses = masses[within_half_light_radius]

    I = jnp.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                I = I.at[i, j].set(jnp.sum(filtered_masses * jnp.sum(filtered_positions**2, axis=1) - filtered_masses * filtered_positions[:, i]**2))
            else:
                I = I.at[i, j].set(-jnp.sum(filtered_masses * filtered_positions[:, i] * filtered_positions[:, j]))
    return I


def rotation_matrices_from_inertia_tensor(I):
    """Calculate 3x3 rotation matrix by diagonalization of the moment of inertia tensor.
    
    Parameters
    ----------
    I : jnp.ndarray
        The moment of inertia tensor.
        
    Returns
    -------
    jnp.ndarray
        The rotation matrix.
    """
    
    eigen_values, eigen_vectors = jnp.linalg.eigh(I)
    order = jnp.argsort(eigen_values)
    rotation_matrix = eigen_vectors[:, order]
    return rotation_matrix


def apply_rotation(positions, rotation_matrix):
    """Apply a rotation matrix to a set of positions.
    
    Parameters
    ----------
    positions : jnp.ndarray
        The positions of the particles.
        
    rotation_matrix : jnp.ndarray
        The rotation matrix.
    
    Returns
    -------
    jnp.ndarray
        The rotated positions.
    """
    
    return jnp.dot(positions, rotation_matrix)

