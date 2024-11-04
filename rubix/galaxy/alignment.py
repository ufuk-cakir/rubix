import jax.numpy as jnp
from jaxtyping import Float, Array
from typing import Tuple
from jax.scipy.spatial.transform import Rotation

from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker


# @jaxtyped(typechecker=typechecker)
def center_particles(rubixdata: object, key) -> object:
    """
    Center the stellar particles around the galaxy center.

    Args:
        rubixdata (object): The RubixData object.
        key (str): The key to the particle data.
        stellar_coordinates (jnp.ndarray): The coordinates of the particles.
        stellar_velocities (jnp.ndarray): The velocities of the particles.
        galaxy_center (jnp.ndarray): The center of the galaxy.

    Returns:
        The RubixData object with the centered particles, which contain of a new set
        of coordinates and velocities as jnp.ndarray.

    Example
    -------
    >>> from rubix.galaxy.alignment import center_particles
    >>> rubixdata = center_particles(rubixdata, "stars")
    """
    if key == "stars":
        stellar_coordinates = rubixdata.stars.coords
        stellar_velocities = rubixdata.stars.velocity
    elif key == "gas":
        stellar_coordinates = rubixdata.gas.coords
        stellar_velocities = rubixdata.gas.velocity
    galaxy_center = rubixdata.galaxy.center

    # Check if Center is within bounds
    check_bounds = (
        (galaxy_center[0] >= jnp.min(stellar_coordinates[:, 0]))
        & (galaxy_center[0] <= jnp.max(stellar_coordinates[:, 0]))
        & (galaxy_center[1] >= jnp.min(stellar_coordinates[:, 1]))
        & (galaxy_center[1] <= jnp.max(stellar_coordinates[:, 1]))
        & (galaxy_center[2] >= jnp.min(stellar_coordinates[:, 2]))
        & (galaxy_center[2] <= jnp.max(stellar_coordinates[:, 2]))
    )

    if not check_bounds:
        raise ValueError("Center is not within the bounds of the galaxy")

    # Calculate Central Velocity from median velocities within 10kpc of center
    mask = jnp.linalg.norm(stellar_coordinates - galaxy_center, axis=1) < 10
    # TODO this should be a median
    central_velocity = jnp.median(stellar_velocities[mask], axis=0)

    if key == "stars":
        rubixdata.stars.coords = stellar_coordinates - galaxy_center
        rubixdata.stars.velocity = stellar_velocities - central_velocity
    elif key == "gas":
        rubixdata.gas.coords = stellar_coordinates - galaxy_center
        rubixdata.gas.velocity = stellar_velocities - central_velocity

    return rubixdata


# @jaxtyped(typechecker=typechecker)
def moment_of_inertia_tensor(
    positions: jnp.ndarray, masses: jnp.ndarray, halfmass_radius: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the moment of inertia tensor for a given set of positions and masses within the half-light radius.
    Assumes the galaxy is already centered.

    Args:
        positions (jnp.ndarray): The positions of the particles.
        masses (jnp.ndarray): The masses of the particles.
        half_light_radius (float): The half-light radius of the galaxy.

    Returns:
        The moment of inertia tensor as a jnp.ndarray.

    Example
    -------
    >>> from rubix.galaxy.alignment import moment_of_inertia_tensor
    >>> I = moment_of_inertia_tensor(rubixdata.stars.coords, rubixdata.stars.mass, rubixdata.galaxy.half_light_radius)
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


# @jaxtyped(typechecker=typechecker)
def rotation_matrix_from_inertia_tensor(I: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate 3x3 rotation matrix by diagonalization of the moment of inertia tensor.

    Args:
        I (jnp.ndarray): The moment of inertia tensor.

    Returns:
        The rotation matrix as a jnp.ndarray.
    """

    eigen_values, eigen_vectors = jnp.linalg.eigh(I)
    order = jnp.argsort(eigen_values)
    rotation_matrix = eigen_vectors[:, order]
    return rotation_matrix


# @jaxtyped(typechecker=typechecker)
def apply_init_rotation(
    positions: jnp.ndarray, rotation_matrix: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply a rotation matrix to a set of positions.

    Args:
        positions (jnp.ndarray): The positions of the particles.
        rotation_matrix (jnp.ndarray): The rotation matrix.

    Returns:
        The rotated positions as a jnp.ndarray.
    """

    return jnp.dot(positions, rotation_matrix)


# @jaxtyped(typechecker=typechecker)
# def euler_rotation_matrix(alpha: Float[jnp.ndarray, ""], beta: Float[jnp.ndarray, ""], gamma: Float[jnp.ndarray, ""]) -> Float[jnp.ndarray, "3 3"]:
def euler_rotation_matrix(alpha, beta, gamma):
    """
    Create a 3x3 rotation matrix given Euler angles (in degrees)

    Args:
        alpha (float): Rotation around the x-axis in degrees
        beta (float): Rotation around the y-axis in degrees
        gamma (float): Rotation around the z-axis in degrees

    Returns:
        The rotation matrix as a jnp.ndarray.
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


# @jaxtyped(typechecker=typechecker)
def apply_rotation(positions, alpha, beta, gamma):
    """
    Apply a rotation to a set of positions given Euler angles.

    Args:
        positions (jnp.ndarray): The positions of the particles.
        alpha (float): Rotation around the x-axis in degrees
        beta (float): Rotation around the y-axis in degrees
        gamma (float): Rotation around the z-axis in degrees

    Returns:
        The rotated positions as a jnp.ndarray.
    """

    R = euler_rotation_matrix(alpha, beta, gamma)
    return jnp.dot(positions, R)


# @jaxtyped(typechecker=typechecker)
def rotate_galaxy(positions, velocities, masses, halfmass_radius, alpha, beta, gamma):
    """
    Orientate the galaxy by applying a rotation matrix to the positions of the particles.

    Args:
        positions (jnp.ndarray): The positions of the particles.
        velocities (jnp.ndarray): The velocities of the particles.
        masses (jnp.ndarray): The masses of the particles.
        halfmass_radius (float): The half-mass radius of the galaxy.
        alpha (float): Rotation around the x-axis in degrees
        beta (float): Rotation around the y-axis in degrees
        gamma (float): Rotation around the z-axis in degrees

    Returns:
        The rotated positions and velocities as a jnp.ndarray.
    """

    I = moment_of_inertia_tensor(positions, masses, halfmass_radius)
    R = rotation_matrix_from_inertia_tensor(I)
    pos_rot = apply_init_rotation(positions, R)
    vel_rot = apply_init_rotation(velocities, R)
    pos_final = apply_rotation(pos_rot, alpha, beta, gamma)
    vel_final = apply_rotation(vel_rot, alpha, beta, gamma)

    return pos_final, vel_final
