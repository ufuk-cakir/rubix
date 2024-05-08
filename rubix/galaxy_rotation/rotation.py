from rubix import logger
from jaxtyping import Array, Float
import jax.numpy as jnp

def moment_of_inertia_tensor(positions, masses, halfmass_radius):
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
    within_halfmass_radius = distances <= halfmass_radius
    
    # Filter positions and masses
    filtered_positions = positions[within_halfmass_radius]
    filtered_masses = masses[within_halfmass_radius]

    I = jnp.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                I = I.at[i, j].set(jnp.sum(filtered_masses * jnp.sum(filtered_positions**2, axis=1) - filtered_masses * filtered_positions[:, i]**2))
            else:
                I = I.at[i, j].set(-jnp.sum(filtered_masses * filtered_positions[:, i] * filtered_positions[:, j]))
    return I


def rotation_matrix_from_inertia_tensor(I):
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


def apply_init_rotation(positions, rotation_matrix):
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


def euler_rotation_matrix(alpha, beta, gamma):
    """Create a 3x3 rotation matrix given Euler angles (in degrees)
    
    Parameters
    ----------
    alpha : float
        Rotation around the x-axis
    
    beta : float
        Rotation around the y-axis

    gamma : float
        Rotation around the z-axis

    Returns
    -------
    np.ndarray
        The rotation matrix.
    """

    alpha = alpha/180*jnp.pi
    beta = beta/180*jnp.pi
    gamma = gamma/180*jnp.pi
    
    # Rotation around the x-axis
    R_x = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(alpha), -jnp.sin(alpha)],
        [0, jnp.sin(alpha), jnp.cos(alpha)]
    ])
    
    # Rotation around the y-axis (pitch)
    R_y = jnp.array([
        [jnp.cos(beta), 0, jnp.sin(beta)],
        [0, 1, 0],
        [-jnp.sin(beta), 0, jnp.cos(beta)]
    ])
    
    # Rotation around the z-axis (yaw)
    R_z = jnp.array([
        [jnp.cos(gamma), -jnp.sin(gamma), 0],
        [jnp.sin(gamma), jnp.cos(gamma), 0],
        [0, 0, 1]
    ])
    
    # Combine the rotations by matrix multiplication: R = R_z * R_y * R_x
    R = jnp.dot(R_z, jnp.dot(R_y, R_x))
    
    return R

def apply_rotation(positions, alpha, beta, gamma):
    """Apply a rotation to a set of positions given Euler angles.
    
    Parameters
    ----------
    positions : jnp.ndarray
        The positions of the particles.
        
    alpha : float
        Rotation around the x-axis
    
    beta : float
        Rotation around the y-axis

    gamma : float
        Rotation around the z-axis
    
    Returns
    -------
    jnp.ndarray
        The rotated positions.
    """
    
    R = euler_rotation_matrix(alpha, beta, gamma)
    return jnp.dot(positions, R)
