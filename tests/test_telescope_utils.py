from rubix.telescope.utils import assign_particles_to_pixel_positions
import jax
import jax.numpy as jnp

# enfrce that jax uses cpu only

# Global flag to set a specific platform, must be used at startup.
jax.config.update("jax_platform_name", "cpu")


def test_assign_particles_to_pixel_positions():
    # Define a small set of coordinates and bin edges
    coords = jnp.array([[0.5, 1.5], [2.5, 3.5]])
    spatial_bin_edges = jnp.array([0, 1, 2, 3, 4])

    # Expected output for these inputs
    expected_output = jnp.array([4, 14])  # Expected indices in the flattened array

    # Call the function with the test inputs
    output = assign_particles_to_pixel_positions(coords, spatial_bin_edges)

    # Assert that the output is as expected
    assert jnp.all(
        output == expected_output
    ), "The function failed to assign particles to pixel positions correctly."
