import pytest
import numpy as np
from rubix.galaxy.alignment import center_particles


def test_center_galaxy_center_not_in_bounds():
    stellar_coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    stellar_velocities = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    center = np.array([0, 0, 0])

    with pytest.raises(
        ValueError, match="Center is not within the bounds of the galaxy"
    ):
        center_particles(stellar_coordinates, stellar_velocities, center)  # type: ignore


def test_center_galaxy_sucessful():
    stellar_coordinates = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    stellar_velocities = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    center = np.array([4, 5, 6])
    new_stellar_coordinates, new_stellar_velocities = center_particles(
        stellar_coordinates, stellar_velocities, center  # type: ignore
    )
    assert np.all(new_stellar_coordinates == stellar_coordinates - center)
    assert np.all(
        new_stellar_velocities
        == stellar_velocities - np.median(stellar_velocities, axis=0)
    )
