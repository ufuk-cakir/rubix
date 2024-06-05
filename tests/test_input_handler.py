import pytest
from rubix.galaxy import BaseHandler
import h5py
from rubix import config

class ConcreteInputHandler(BaseHandler):
    def get_particle_data(self):
        # Mock particle data that satisfies the requirements
        return {
            "stars": {
                "coords": [1, 2, 3],
                "mass": [4, 5, 6],
                "metallicity": [0.1, 0.2, 0.3],
                "velocity": [7, 8, 9],
                "age": [10, 11, 12],
            },
            "gas": {
                "coords": [1, 2, 3],
                "density": [4, 5, 6],
                "mass": [4, 5, 6],
                "metallicity": [0.1, 0.2, 0.3],
                "hsml": [7, 8, 9],
                "sfr": [10, 11, 12],
                "internal_energy": [0.1, 0.2, 0.3],
                "velocity": [7, 8, 9],
                "electron_abundance": [0.1, 0.2, 0.3],
                "metals": [1, 2, 3],
            }
        }

    def get_galaxy_data(self):
        # Mock galaxy data that satisfies the requirements
        return {"redshift": 0.5, "center": [1, 2, 3], "halfmassrad_stars": 1.5}

    def get_simulation_metadata(self):
        # Mock simulation metadata (can be empty if no specific checks are required)
        return {"TIME": 0, "NAME": "TNG50-1", "SUBHALO_ID": 0}

    def get_units(self):
        # Mock units that satisfy the requirements
        return {
            "galaxy": config["BaseHandler"]["galaxy"],
            "stars": config["BaseHandler"]["particles"]["stars"],
            "gas": config["BaseHandler"]["particles"]["gas"],
        }


@pytest.fixture
def input_handler(tmp_path):
    handler = ConcreteInputHandler()
    return handler


def test_convert_to_rubix_creates_file(input_handler, tmp_path):
    input_handler.to_rubix(output_path=tmp_path)
    assert (tmp_path / "rubix_galaxy.h5").exists()
    # load the file and check if the groups and datasets are created as expected


def test_convert_to_rubix_structure(input_handler, tmp_path):
    input_handler.to_rubix(tmp_path)

    with h5py.File(tmp_path / "rubix_galaxy.h5", "r") as f:
        print(f.keys())
        assert "meta" in f
        assert "galaxy" in f
        assert "particles" in f
        assert "redshift" in f["galaxy"]  # type: ignore
        assert "center" in f["galaxy"]  # type: ignore
        assert "halfmassrad_stars" in f["galaxy"]  # type: ignore
        assert "stars" in f["particles"]  # type: ignore
        assert "coords" in f["particles/stars"]  # type: ignore
        assert "mass" in f["particles/stars"]  # type: ignore
        assert "metallicity" in f["particles/stars"]  # type: ignore
        assert "velocity" in f["particles/stars"]  # type: ignore
        assert "age" in f["particles/stars"]  # type: ignore
        assert "gas" in f["particles"]  # type: ignore
        assert "coords" in f["particles/gas"]  # type: ignore
        assert "density" in f["particles/gas"]  # type: ignore
        assert "mass" in f["particles/gas"]  # type: ignore
        assert "metallicity" in f["particles/gas"]  # type: ignore
        assert "hsml" in f["particles/gas"]  # type: ignore
        assert "sfr" in f["particles/gas"]  # type: ignore
        assert "internal_energy" in f["particles/gas"]  # type: ignore
        assert "velocity" in f["particles/gas"]  # type: ignore
        assert "electron_abundance" in f["particles/gas"]  # type: ignore
        assert "metals" in f["particles/gas"]  # type: ignore


def test_convert_to_rubix_correct_values(input_handler, tmp_path):
    input_handler.to_rubix(tmp_path)

    with h5py.File(tmp_path / "rubix_galaxy.h5", "r") as f:
        assert f["galaxy/redshift"][()] == 0.5  # type: ignore
        assert (f["galaxy/center"][()] == [1, 2, 3]).all()  # type: ignore
        assert f["galaxy/halfmassrad_stars"][()] == 1.5  # type: ignore
        assert (f["particles/stars/coords"][()] == [1, 2, 3]).all()  # type: ignore
        assert (f["particles/stars/mass"][()] == [4, 5, 6]).all()  # type: ignore
        assert (f["particles/stars/metallicity"][()] == [0.1, 0.2, 0.3]).all()  # type: ignore
        assert (f["particles/stars/velocity"][()] == [7, 8, 9]).all()  # type: ignore
        assert (f["particles/stars/age"][()] == [10, 11, 12]).all()  # type: ignore
        assert (f["particles/gas/coords"][()] == [1, 2, 3]).all()
        assert (f["particles/gas/density"][()] == [4, 5, 6]).all()
        assert (f["particles/gas/mass"][()] == [4, 5, 6]).all()
        assert (f["particles/gas/metallicity"][()] == [0.1, 0.2, 0.3]).all()
        assert (f["particles/gas/hsml"][()] == [7, 8, 9]).all()
        assert (f["particles/gas/sfr"][()] == [10, 11, 12]).all()
        assert (f["particles/gas/internal_energy"][()] == [0.1, 0.2, 0.3]).all()
        assert (f["particles/gas/velocity"][()] == [7, 8, 9]).all()
        assert (f["particles/gas/electron_abundance"][()] == [0.1, 0.2, 0.3]).all()
        assert (f["particles/gas/metals"][()] == [1, 2, 3]).all()
        assert f["meta/TIME"][()] == 0  # type: ignore
        assert f["meta/NAME"][()] == b"TNG50-1"  # type: ignore
        assert f["meta/SUBHALO_ID"][()] == 0  # type: ignore


def test_units_are_correct(input_handler):
    units = input_handler.get_units()
    assert units == {
        "galaxy": config["BaseHandler"]["galaxy"],
        "stars": config["BaseHandler"]["particles"]["stars"],
        "gas": config["BaseHandler"]["particles"]["gas"],
    }


def test_rubix_file_has_correct_units(input_handler, tmp_path):
    input_handler.to_rubix(tmp_path)

    # get the units from the rubix config
    from rubix import config 
    config = config["BaseHandler"]
    with h5py.File(tmp_path / "rubix_galaxy.h5", "r") as f:
        assert f["galaxy/redshift"].attrs["unit"] == config["galaxy"]["redshift"]
        assert f["galaxy/center"].attrs["unit"] == config["galaxy"]["center"]
        assert f["galaxy/halfmassrad_stars"].attrs["unit"] == config["galaxy"]["halfmassrad_stars"]
        assert f["particles/stars/coords"].attrs["unit"] == config["particles"]["stars"]["coords"]
        assert f["particles/stars/mass"].attrs["unit"] == config["particles"]["stars"]["mass"]
        assert f["particles/stars/metallicity"].attrs["unit"] == config["particles"]["stars"]["metallicity"]
        assert f["particles/stars/velocity"].attrs["unit"] == config["particles"]["stars"]["velocity"]
        assert f["particles/stars/age"].attrs["unit"] == config["particles"]["stars"]["age"]
        assert f["particles/gas/coords"].attrs["unit"] == config["particles"]["gas"]["coords"]
        assert f["particles/gas/density"].attrs["unit"] == config["particles"]["gas"]["density"]
        assert f["particles/gas/mass"].attrs["unit"] == config["particles"]["gas"]["mass"]
        assert f["particles/gas/metallicity"].attrs["unit"] == config["particles"]["gas"]["metallicity"]
        assert f["particles/gas/hsml"].attrs["unit"] == config["particles"]["gas"]["hsml"]
        assert f["particles/gas/sfr"].attrs["unit"] == config["particles"]["gas"]["sfr"]
        assert f["particles/gas/internal_energy"].attrs["unit"] == config["particles"]["gas"]["internal_energy"]
        assert f["particles/gas/velocity"].attrs["unit"] == config["particles"]["gas"]["velocity"]
        assert f["particles/gas/electron_abundance"].attrs["unit"] == config["particles"]["gas"]["electron_abundance"]
        assert f["particles/gas/metals"].attrs["unit"] == config["particles"]["gas"]["metals"]


def test_missing_galaxy_field_error(input_handler):
    with pytest.raises(ValueError) as excinfo:
        # Manually remove a required galaxy field
        galaxy_data = input_handler.get_galaxy_data()
        del galaxy_data["redshift"]
        input_handler._check_galaxy_data(galaxy_data, input_handler.get_units())
    assert "Missing field redshift in galaxy data" in str(excinfo.value)



def test_galaxy_field_unit_info_missing_error(input_handler):
    with pytest.raises(ValueError) as excinfo:
        # Manually change a unit to an unsupported one
        units = input_handler.get_units()
        galaxy_data = input_handler.get_galaxy_data()
        galaxy_data["unsupported_field"] = 1
        input_handler._check_galaxy_data(galaxy_data, units)
    assert "Units for unsupported_field not found in units" in str(excinfo.value)


def test_missing_particle_type_error(input_handler):
    with pytest.raises(ValueError) as excinfo:
        # Remove a required particle type
        particle_data = input_handler.get_particle_data()
        del particle_data["stars"]
        input_handler._check_particle_data(particle_data, input_handler.get_units())
    assert "Missing particle type stars in particle data" in str(excinfo.value)


def test_missing_particle_type_error_gas(input_handler):
    with pytest.raises(ValueError) as excinfo:
        # Remove a required particle type
        particle_data = input_handler.get_particle_data()
        del particle_data["gas"]
        input_handler._check_particle_data(particle_data, input_handler.get_units())
    assert "Missing particle type gas in particle data" in str(excinfo.value)


def test_missing_particle_field_error(input_handler):
    with pytest.raises(ValueError) as excinfo:
        # Remove a required field from a particle type
        particle_data = input_handler.get_particle_data()
        del particle_data["stars"]["coords"]
        input_handler._check_particle_data(particle_data, input_handler.get_units())
    assert "Missing field coords in particle data for particle type stars" in str(
        excinfo.value
    )
    

def test_missing_particle_field_error_gas(input_handler):
    with pytest.raises(ValueError) as excinfo:
        # Remove a required field from a particle type
        particle_data = input_handler.get_particle_data()
        del particle_data["gas"]["coords"]
        input_handler._check_particle_data(particle_data, input_handler.get_units())
    assert "Missing field coords in particle data for particle type gas" in str(
        excinfo.value
    )


def test_particle_field_unit_info_missing_error(input_handler):
    with pytest.raises(ValueError) as excinfo:
        # Manually change a unit to an unsupported one
        units = input_handler.get_units()
        particle_data = input_handler.get_particle_data()
        particle_data["stars"]["unsupported_field"] = 1
        input_handler._check_particle_data(particle_data, units)
    assert "Units for unsupported_field not found in units" in str(excinfo.value)


def test_particle_field_unit_info_missing_error_gas(input_handler):
    with pytest.raises(ValueError) as excinfo:
        # Manually change a unit to an unsupported one
        units = input_handler.get_units()
        particle_data = input_handler.get_particle_data()
        particle_data["gas"]["unsupported_field"] = 1
        input_handler._check_particle_data(particle_data, units)
    assert "Units for unsupported_field not found in units" in str(excinfo.value)