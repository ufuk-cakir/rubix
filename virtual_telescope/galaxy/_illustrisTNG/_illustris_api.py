import requests
import os
import h5py


class IllustrisAPI:
    '''This class is used to load data from the Illustris API.
    
    It loads both subhalo data and particle data from a given simulation, snapshot, and subhalo ID.
    '''
    DATAPATH = "./tempdata"
    URL = "http://www.tng-project.org/api/"
    DEFAULT_FIELDS = {
        "PartType0": [
            "Coordinates",
            "Density",
            "Masses",
            "ParticleIDs",
            "GFM_Metallicity",
            "SubfindHsml",
            "StarFormationRate",
            "InternalEnergy",
            "Velocities",
            "ElectronAbundance",
            "GFM_Metals",
        ],
        "PartType4": [
            "Coordinates",
            "GFM_InitialMass",
            "Masses",
            "ParticleIDs",
            "GFM_Metallicity",
            "GFM_StellarFormationTime",
            "Velocities",
        ],
    }

    def __init__(
        self,
        api_key,
        particle_type="stars",
        simulation="TNG50-1",
        snapshot=99,
    ):
        """Illustris API class.

        Class to load data from the Illustris API.

        Parameters
        ----------
        api_key : str
            API key for the Illustris API.
        particle_type : str
            Particle type to load. Default is "stars".
        simulation : str
            Simulation to load from. Default is "TNG100-1".
        snapshot : int
            Snapshot to load from. Default is 99.
        """

        if api_key is None:
            raise ValueError("Please set the API key.")

        self.headers = {"api-key": api_key}
        self.particle_type = particle_type
        self.snapshot = snapshot
        self.simulation = simulation
        self.baseURL = f"{self.URL}{self.simulation}/snapshots/{self.snapshot}"




    
    def _get(self, path, params=None, name=None):
        """Get data from the Illustris API.

        Parameters
        ----------
        path : str
            Path to load from.
        params : dict
            Parameters to pass to the API.
        name : str
            Name to save the file as. If None, the name will be taken from the content-disposition header.
        Returns
        -------
        r : requests object
            The requests object.

        """

        os.makedirs(self.DATAPATH, exist_ok=True)
        try:
            r = requests.get(path, params=params, headers=self.headers)
            # raise exception if response code is not HTTP SUCCESS (200)
            r.raise_for_status()
        except requests.exceptions.HTTPError as err:
            raise ValueError(err)
        
        # Check if the response is empty -- do I even need this?
        if r.headers.get("content-type") is None:
            raise ValueError("Response is empty.")
        
        if r.headers["content-type"] == "application/json":
            return r.json()  # parse json responses automatically
        if "content-disposition" in r.headers:
            filename = (
                r.headers["content-disposition"].split("filename=")[1]
                if name is None
                else name
            )
            file_path = os.path.join(self.DATAPATH, f"{filename}.hdf5")
            with open(file_path, "wb") as f:
                f.write(r.content)
            return filename  # return the filename string
        return r

    def get_subhalo(self, id):
        """Get subhalo data from the Illustris API.

        Returns the subhalo data for the given subhalo ID.

        Parameters
        ----------
        id : int
            Subhalo ID to load.
        Returns
        -------
        r : dict
            The subhalo data.

        """

        return self._get(f"{self.baseURL}/subhalos/{id}")

    def _load_hdf5(self, filename):
        """Load HDF5 file.

        Loads the HDF5 file with the given filename.

        Parameters
        ----------
        filename : str
            Filename to load.
        Returns
        -------
        returndict : dict
            Dictionary containing the data from the HDF5 file.
        """
        # Check if filename ends with .hdf5
        if filename.endswith(".hdf5"):
            filename = filename[:-5]

        returndict = dict()
        file_path = os.path.join(self.DATAPATH, f"{filename}.hdf5")
        with h5py.File(file_path, "r") as f:
            for type in f.keys():
                if type == "Header":
                    continue
                if type.startswith("PartType"):
                    for fields in f[type].keys():
                        returndict[fields] = f[type][fields][()]

        return returndict

    def get_particle_data(self, id, particle_type, fields=DEFAULT_FIELDS):
        """Get particle data from the Illustris API.

        Returns the particle data for the given subhalo ID.
        Parameters
        ----------
        id : int
            Subhalo ID to load.
        fields : str or list
            Fields to load. If a string, the fields should be comma-separated.

        Returns
        -------
        data : dict
            Dictionary containing the particle data in the given fields.
        """
        # Get fields in the right format
        if isinstance(fields, str):
            fields = [fields]
        fields = ",".join(fields)

        url = f"{self.baseURL}/subhalos/{id}/cutout.hdf5?{particle_type}={fields}"
        self._get(url, name="cutout")
        data = self._load_hdf5("cutout")
        return data

    def load_galaxy(self, id, verbose=False):
        if verbose:
            print(f"Getting data for subhalo {id}")

        # Get Fields in the right format
        fields_gas = self.DEFAULT_FIELDS["PartType0"]
        fields_stars = self.DEFAULT_FIELDS["PartType4"]
        fields_gas = ",".join(fields_gas)
        fields_stars = ",".join(fields_stars)
        url = f"{self.baseURL}/subhalos/{id}/cutout.hdf5?gas={fields_gas}&stars={fields_stars}"
        self._get(url, name=f"galaxy-id-{id}")
        subhalo_data = self.get_subhalo(id)
        self._append_subhalo_data(subhalo_data, id)
        data = self._load_hdf5(filename=f"galaxy-id-{id}")
        return data
    def _append_subhalo_data(self, subhalo_data, id):
        # Append subhalo data to the HDF5 file
        file_path = os.path.join(self.DATAPATH, f"galaxy-id-{id}.hdf5")
        with h5py.File(file_path, "a") as f:
            f.create_group("SubhaloData")
            for key in subhalo_data.keys():
                if type(subhalo_data[key]) == dict:
                    continue
                f["SubhaloData"].create_dataset(key, data=subhalo_data[key])