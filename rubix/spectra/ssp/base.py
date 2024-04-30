from abc import ABC, abstractmethod


class BaseGrid(ABC):
    """
    Base class for all SSP grids.
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_metallicity_grid(self):
        """
        Return the metallicity grid.
        """
        pass

    @abstractmethod
    def get_age_grid(self):
        """
        Return the age grid.
        """
        pass

    def lookup(self, mass, metallicity, age):
        """
        Return the SSP for the given metallicity and age, weigthed by the mass.
        """
        pass
