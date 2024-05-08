import pytest
from rubix.galaxy_rotation.rotation import moment_of_inertia_tensor, rotation_matrix_from_inertia_tensor, apply_init_rotation, euler_rotation_matrix, apply_rotation
from rubix.galaxy_rotation.factory import GalaxyRotationFactory
import jax.numpy as jnp

def test_moment_of_inertia_tensor():
    """Test the moment_of_inertia_tensor function."""
    
    # Example positions and masses
    positions = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    masses = jnp.array([1.0, 1.0, 1.0])
    halfmass_radius = 2.0

    expected_tensor = jnp.array([
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0]
    ])

    result_tensor = moment_of_inertia_tensor(positions, masses, halfmass_radius)

    assert result_tensor.all() == expected_tensor.all(), f"Test failed. Got other tensor than expected."


def test_rotation_matrix_from_inertia_tensor():
    # Example inertia tensor: simple diagonal tensor with distinct values
    I = jnp.diag(jnp.array([1.0, 2.0, 3.0]))

    rotation_matrix = rotation_matrix_from_inertia_tensor(I)

    # Expected result
    # For a diagonal inertia tensor with distinct values, the rotation matrix
    # should effectively be the identity matrix because the eigenvectors of a
    # diagonal matrix are the standard basis vectors, assuming eigenvalues are
    # already sorted.
    expected_rotation_matrix = jnp.eye(3)

    assert rotation_matrix.all() == expected_rotation_matrix.all(), f"Test failed. Got other rotation matrix than expected."


def test_apply_init_rotation():
    # Example positions
    positions = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Example rotation matrix (90-degree rotation around the z-axis)
    rotation_matrix = jnp.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Expected rotated positions
    expected_rotated_positions = jnp.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    # Apply rotation using the function
    rotated_positions = apply_init_rotation(positions, rotation_matrix)

    # Verify the function output matches expected values
    assert rotated_positions.all() == expected_rotated_positions.all(), f"Test failed. Initial rotation was not successful."


def test_euler_rotation_matrix():
    alpha = 90
    beta = 0
    gamma = 0

    expected_rotation_matrix = jnp.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    # Get the rotation matrix from the function
    result_rotation_matrix = euler_rotation_matrix(alpha, beta, gamma)

    # Check if the result matches the expected output
    assert result_rotation_matrix.all() == expected_rotation_matrix.all(), f"Test failed. Expected other rotation matrix."


def test_apply_rotation():
    positions = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0] 
    ])

    alpha = 90
    beta = 0
    gamma = 0

    rotation_matrix = jnp.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    # Calculate the expected rotated positions
    expected_rotated_positions = jnp.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])

    # Apply the rotation using the apply_rotation function
    result_rotated_positions = apply_rotation(positions, alpha, beta, gamma)

    # Verify that the result matches the expected rotated positions
    assert result_rotated_positions.all() == expected_rotated_positions.all(), f"Test failed. Expected other positions."


def test_GalaxyRotationFactory():

    # Example positions, velocities, and masses
    positions = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    velocities = jnp.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0]
    ])
    masses = jnp.array([1.0, 1.0, 1.0])
    halfmass_radius = 2.0

    alpha = 30
    beta = 45
    gamma = 60

    rotated_positions, rotated_velocities = GalaxyRotationFactory(positions, velocities, masses, halfmass_radius, alpha, beta, gamma)

    assert rotated_positions.shape == positions.shape
    assert rotated_velocities.shape == velocities.shape

