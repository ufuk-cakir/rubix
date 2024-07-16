class Particles:
    """
    Mixin class to handle subsetting of dataclasses

    Methods:
        apply_subset(indices): Applies subsetting to all fields of the dataclass.
            Each field is updated to only contain elements at the specified indices.
            It gets a random subset of data for speed reason and testing.

    Example usage:
        @dataclass
        class MyData(SubsetMixin):
            field1: List[int]
            field2: List[str]

        data = MyData(field1=[1, 2, 3], field2=['a', 'b', 'c'])
        data.apply_subset([0, 2])  # data now contains elements at indices 0 and 2
    """
    def apply_subset(self, indices):
        """
        Applies subsetting to all fields of the dataclass.

        Parameters:
            indices: List[int]
                The indices to keep in each field of the dataclass
        """
        for field_name, value in self.__dataclass_fields__.items():
            current_value = getattr(self, field_name)
            if current_value is not None:
                setattr(self, field_name, current_value[indices])


def create_dynamic_dataclass(name, fields):
    """
    Create a dataclass dynamically based on the provided fields, all of which are optional and default to None.
    Each field is of type Optional[jnp.ndarray], that it can hold a JAX numpy array or None.
    It inherits from SubsetMixin to allow subsetting of the dataclass using apply_subset method

    Parameters:
        name: str
            The name of the dataclass
        fields: List[str]
            The names of the fields to include in the dataclass

    Returns:
        type
            The dynamically created dataclass
    
    Example usage:
        MyDynamicData = create_dynamic_dataclass('MyDynamicData', ['field1', 'field2'])
        instance = MyDynamicData()
        instance.field1 = jnp.array([1, 2, 3])
        instance.apply_subset([0, 2])  # Applies subsetting to all fields
    """
    annotations = {field_name: Optional[jnp.ndarray] for field_name in fields}
    # Include SubsetMixin in the bases
    return make_dataclass(name, [(field_name, annotation, field(default=None)) for field_name, annotation in annotations.items()], bases=(Particles,))


@dataclass
class Galaxy:
    # Galaxy class definition here
    pass

@dataclass
class StarsData:
    # StarsData class definition here
    pass

@dataclass
class GasData:
    # GasData class definition here
    pass

@dataclass
class RubixData(Particles):
    """
    This class is used to store the Rubix data in a structured format.
    It is constructed in a dynamic way based on the configuration file.

    galaxy:
        Contains general information about the galaxy
        redshift, galaxy center, halfmassrad
    stars:
        Contains information about the stars
    gas:
        Contains information about the gas
    """
    def __init__(self, galaxy: Optional[Galaxy] = None, stars: Optional[StarsData] = None, gas: Optional[GasData] = None):
        self.galaxy = galaxy
        self.stars = stars
        self.gas = gas

    galaxy: Optional[Galaxy] = None
    stars: Optional[StarsData] = None
    gas: Optional[GasData] = None