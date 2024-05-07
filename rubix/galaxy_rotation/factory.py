from .rotation import moment_of_inertia_tensor
from .rotation import rotation_matrix_from_inertia_tensor
from .rotation import apply_init_rotation
from .rotation import euler_rotation_matrix
from .rotation import apply_rotation

def GalaxyRotationFactory(positions, velocities, masses, halfmass_radius, alpha, beta, gamma):
    """Orientate the galaxy by applying a rotation matrix to the positions of the particles.
    
    Parameters
    ----------
    positions : jnp.ndarray
        The positions of the particles.

    velocities : jnp.ndarray
        The velocities of the particles.

    masses : jnp.ndarray
        The masses of the particles.
    
    halflight_radius : float
        The half-light radius of the galaxy.
    
    alpha : float
        The Euler angle alpha (in degrees), x-axis.
    
    beta : float
        The Euler angle beta (in degrees), y-axis.
    
    gamma : float
        The Euler angle gamma (in degrees), z-axis.
    
    Returns
    -------
    jnp.ndarray
        The rotated positions aqnd velocities.
    """
    
    I = moment_of_inertia_tensor(positions, masses, halfmass_radius)
    R = rotation_matrix_from_inertia_tensor(I)
    pos_rot = apply_init_rotation(positions, R)
    vel_rot = apply_init_rotation(velocities, R)
    pos_final = apply_rotation(pos_rot, alpha, beta, gamma)
    vel_final = apply_rotation(vel_rot, alpha, beta, gamma)

    return pos_final, vel_final
