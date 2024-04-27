"""Helper functions for alignment tasks.


Some of the helper function in this module were taken from Kate Harborne's
SimSpin code.
"""

import jax
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402


def _ellipsoid_tensor(x, y, z, mass, p, q):
    # Source: SimSpin from Kate Harborne

    ellip_radius = jnp.sqrt((x**2) + ((y / p) ** 2) + ((z / q) ** 2))
    M = jnp.zeros((3, 3))

    weighted_mass = mass / ellip_radius
    weighted_mass = weighted_mass.value
    x = x.value
    y = y.value
    z = z.value
    M = M.at[0, 0].set(jnp.sum(weighted_mass * x**2))
    M = M.at[0, 1].set(jnp.sum(weighted_mass * x * y))
    M = M.at[1, 0].set(M[0, 1])
    M = M.at[0, 2].set(jnp.sum(weighted_mass * x * z))
    M = M.at[2, 0].set(M[0, 2])
    M = M.at[1, 1].set(jnp.sum(weighted_mass * y**2))
    M = M.at[1, 2].set(jnp.sum(weighted_mass * y * z))
    M = M.at[2, 1].set(M[1, 2])
    M = M.at[2, 2].set(jnp.sum(weighted_mass * z**2))
    return M


def _ellipsoid_ratios_p_q(x, y, z, mass, p, q) -> dict:
    # Source: SimSpin from Kate Harborne
    M = _ellipsoid_tensor(x, y, z, mass, p, q)
    # TODO units not supported for eigh?
    eig_values, eig_vectors = jax.linalg.eigh(M.value)  # type: ignore
    p = jnp.sqrt(eig_values[1] / eig_values[0])  # type: ignore
    q = jnp.sqrt(eig_values[2] / eig_values[0])  # type: ignore
    y_axis = eig_vectors[:, 1]
    z_axis = eig_vectors[:, 2]

    return {
        "eigenvalues": eig_values,
        "p": p,
        "q": q,
        "y_axis": y_axis,
        "z_axis": z_axis,
        "ellipsoid_tensor": M,
    }


def measure_pq(
    x,
    y,
    z,
    mass,
    half_mass_radius,
    abort_count=50,
):
    """Function to measure the p and q values of a galaxy.

    The p and q values are the ratios of the semi-major and semi-minor axes
    of an ellipsoid that best fits the galaxy. The function iteratively
    calculates the p and q values until convergence.

    This is a python implementation from Kate Harborne's SimSpin code.

    """

    # Initial assumptions for a sphere
    a = b = c = 1.0
    p = b / a
    q = c / a
    temp_p = []
    temp_q = []

    # Main loop to find p and q
    cnt = 0
    while True:
        # Get the indices that are within the half mass radius
        indices = (x**2 + (y / p) ** 2 + (z / q) ** 2) < half_mass_radius**2
        hm_x, hm_y, hm_z, hm_mass = x[indices], y[indices], z[indices], mass[indices]

        fit_ellip = _ellipsoid_ratios_p_q(hm_x, hm_y, hm_z, hm_mass, p, q)
        temp_p.append(fit_ellip["p"])
        temp_q.append(fit_ellip["q"])

        # Convergence or stability check
        if cnt >= 10:

            last_10_p = jnp.array(temp_p[-10:])
            diff_p = jnp.abs(jnp.diff(last_10_p))
            # If the change in p and q values is less than 0.01 for the last 10 iterations
            if jnp.all(diff_p < 0.01) and jnp.all(diff_p < 0.01):
                
                break
            """
            if np.all(np.abs(np.diff(last_10_p)) < 0.01) and np.all(
                np.abs(np.diff(last_10_p)) < 0.01
            ):
                break"""

        if cnt >= abort_count:
            # If the function does not converge within the abort count
         
            break

        p, q = temp_p[-1], temp_q[-1]
        cnt += 1

    final_p = np.mean(temp_p[-6:])
    final_q = np.mean(temp_q[-6:])

    return {"p": final_p, "q": final_q}


def _additional_rotation_matrix(theta):
    # Create a rotation matrix for rotation by theta about the z-axis
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )


def _rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2"""
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    R_extra = _additional_rotation_matrix(np.pi / 4)
    # Compose the rotations (original then extra)
    # TODO T-24 Check why we need this?
    return np.dot(rotation_matrix, R_extra)


def _rotate_galaxy(coords, velocities, z_axis):
    # Target axis for alignment, typically the z-axis for face-on
    target_axis = jnp.array([0, 1, 0])

    # Calculate the rotation matrix to align z_axis with the target axis
    R = _rotation_matrix_from_vectors(z_axis, target_axis)

    # Rotate positions
    rotated_positions = jnp.matmul(R, coords.T).T  # type: ignore

    # Rotate velocities
    rotated_velocities = jnp.matmul(R, velocities.T).T  # type: ignore

    # Update galaxy data with rotated positions and velocities

    return rotated_positions, rotated_velocities


def face_on_rotation(coords, velocities, masses, half_mass_rad):
    half_mass_mask = jnp.sum(coords**2, axis=1) < half_mass_rad**2
    # half_mass_indices = jnp.where(jnp.sum(coords**2, axis=1) < half_mass_rad**2)[0]
    half_mass_coords = coords[half_mass_mask]
    half_mass = masses[half_mass_mask]
    pq = measure_pq(
        x=half_mass_coords[:, 0],
        y=half_mass_coords[:, 1],
        z=half_mass_coords[:, 2],
        mass=half_mass,
        half_mass_radius=half_mass_rad,
    )
    ellipsoid_ratios = _ellipsoid_ratios_p_q(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mass=masses,
        p=pq["p"],
        q=pq["q"],
    )
    coords_rot, velocities_rot = _rotate_galaxy(
        coords,
        velocities,
        ellipsoid_ratios["z_axis"],
    )

    return coords_rot, velocities_rot


def center_galaxy(stellar_coordinates, stellar_velocities, galaxy_center):

    # gas_coordinates = self.gas.coordinates
    # gas_velocities = self.gas.velocities

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
    mask = jnp.linalg.vector_norm(stellar_coordinates - galaxy_center, axis=1) < 10
    # TODO this should be a median
    central_velocity = jnp.median(stellar_velocities[mask], axis=0)

    new_stellar_coordinates = stellar_coordinates - galaxy_center
    new_stellar_velocities = stellar_velocities - central_velocity
    return new_stellar_coordinates, new_stellar_velocities
    # Centering the galaxy
    # for part_type in ["gas", "stars"]:
    #     part = getattr(self, part_type)
    #     # Center Coordinates and velocities
    #     part.coordinates = part.coordinates - center
    #     part.velocities = part.velocities - central_velocity
