"""Helper functions for alignment tasks.


Some of the helper function in this module were taken from Kate Harborne's
SimSpin code.
"""

import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


def center_particles(rubixdata, key):
    """Center the stellar particles around the galaxy center.

    Parameters
    ----------
    particle_coordinates : jnp.ndarray
        The coordinates of the particles.
    particle_velocities : jnp.ndarray
        The velocities of the particles.
    galaxy_center : jnp.ndarray
        The center of the galaxy.

    Returns
    -------
    jnp.ndarray
        The new coordinates of the particles.
    jnp.ndarray
        The new velocities of the particles.
    """
    if key == "stars":
        particle_coordinates = rubixdata.stars.coords
        particle_velocities = rubixdata.stars.velocity
    elif key == "gas":
        particle_coordinates = rubixdata.gas.coords
        particle_velocities = rubixdata.gas.velocity
    galaxy_center = rubixdata.galaxy.center

    # Check if Center is within bounds
    check_bounds = (
        (galaxy_center[0] >= jnp.min(particle_coordinates[:, 0]))
        & (galaxy_center[0] <= jnp.max(particle_coordinates[:, 0]))
        & (galaxy_center[1] >= jnp.min(particle_coordinates[:, 1]))
        & (galaxy_center[1] <= jnp.max(particle_coordinates[:, 1]))
        & (galaxy_center[2] >= jnp.min(particle_coordinates[:, 2]))
        & (galaxy_center[2] <= jnp.max(particle_coordinates[:, 2]))
    )

    if not check_bounds:
        raise ValueError("Center is not within the bounds of the galaxy")

    # Calculate Central Velocity from median velocities within 10kpc of center
    mask = jnp.linalg.norm(particle_coordinates - galaxy_center, axis=1) < 10
    # TODO this should be a median
    central_velocity = jnp.median(particle_velocities[mask], axis=0)

    if key == "stars":
        rubixdata.stars.coords = particle_coordinates - galaxy_center
        rubixdata.stars.velocity = particle_velocities - central_velocity
    elif key == "gas":
        rubixdata.gas.coords = particle_coordinates - galaxy_center
        rubixdata.gas.velocity = particle_velocities - central_velocity

    return rubixdata


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

    distances = jnp.sqrt(
        jnp.sum(positions**2, axis=1)
    )  # Direct calculation since positions are already centered

    within_halfmass_radius = distances <= halfmass_radius

    # Ensure within_halfmass_radius is concrete
    concrete_indices = jnp.where(
        within_halfmass_radius, size=within_halfmass_radius.shape[0]
    )[0]

    filtered_positions = positions[concrete_indices]
    filtered_masses = masses[concrete_indices]

    I = jnp.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j:
                I = I.at[i, j].set(
                    jnp.sum(
                        filtered_masses * jnp.sum(filtered_positions**2, axis=1)
                        - filtered_masses * filtered_positions[:, i] ** 2
                    )
                )
            else:
                I = I.at[i, j].set(
                    -jnp.sum(
                        filtered_masses
                        * filtered_positions[:, i]
                        * filtered_positions[:, j]
                    )
                )
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

    # alpha = alpha/180*jnp.pi
    # beta = beta/180*jnp.pi
    # gamma = gamma/180*jnp.pi

    # Rotation around the x-axis
    # R_x = jnp.array([
    #    [1, 0, 0],
    #    [0, jnp.cos(alpha), -jnp.sin(alpha)],
    #    [0, jnp.sin(alpha), jnp.cos(alpha)]
    # ])
    R_x = Rotation.from_euler("x", alpha, degrees=True)

    # Rotation around the y-axis (pitch)
    # R_y = jnp.array([
    #    [jnp.cos(beta), 0, jnp.sin(beta)],
    #    [0, 1, 0],
    #    [-jnp.sin(beta), 0, jnp.cos(beta)]
    # ])
    R_y = Rotation.from_euler("y", beta, degrees=True)

    # Rotation around the z-axis (yaw)
    # R_z = jnp.array([
    #    [jnp.cos(gamma), -jnp.sin(gamma), 0],
    #    [jnp.sin(gamma), jnp.cos(gamma), 0],
    #    [0, 0, 1]
    # ])
    R_z = Rotation.from_euler("z", gamma, degrees=True)

    # Combine the rotations by matrix multiplication: R = R_z * R_y * R_x
    R = R_z * R_y * R_x

    return R.as_matrix()


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


def rotate_galaxy(positions, velocities, masses, halfmass_radius, alpha, beta, gamma):
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
