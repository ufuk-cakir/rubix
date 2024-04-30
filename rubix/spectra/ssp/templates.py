""" This module contains the supported templates for the SSP grid. """

from .base import BaseGrid


class BruzualCharlot2003(BaseGrid):
    """
    Bruzual & Charlot 2003 SSP grid.
    """

    def __init__(self):
        super().__init__()

    def get_metallicity_grid(self):
        """
        Return the metallicity grid.
        """
        pass

    def get_age_grid(self):
        """
        Return the age grid.
        """
        pass
