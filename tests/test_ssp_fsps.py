import pytest
import numpy as np
from unittest.mock import patch
from rubix.spectra.ssp.grid import SSPGrid
import sys

# Mock the fsps.StellarPopulation class
class MockFSPS:
    class StellarPopulation:
        def __init__(self, zcontinuous=0, **kwargs):
            self.zlegend = np.array([0.001, 0.01, 0.1])
            self.log_age = np.array([9.0, 9.1, 9.2])

        def get_spectrum(self):
            return (
                np.array([4000, 4100, 4200]),
                np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
            )

sys.modules['fsps'] = MockFSPS()

def test_retrieve_ssp_data_from_fsps_no_fsps():
    # Test when python-fsps is not installed
    from rubix.spectra.ssp.fsps_grid import retrieve_ssp_data_from_fsps
    with patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", False):
        with pytest.raises(AssertionError):
            result = retrieve_ssp_data_from_fsps()
            assert result is None


def test_retrieve_ssp_data_from_fsps():
    with patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True):
        #, \
        #patch.dict(sys.modules, {"fsps": MockFSPS()}):
        print("trying to import fsps")
        import fsps
        print(fsps)
        #from rubix.spectra.ssp.fsps_grid import retrieve_ssp_data_from_fsps
        import rubix.spectra.ssp.fsps_grid  
        # Call the function
        result = rubix.spectra.ssp.fsps_grid.retrieve_ssp_data_from_fsps()
        mock_sp_instance = MockFSPS.StellarPopulation()

        # Check the returned SSPGrid object
        assert isinstance(result, SSPGrid)
        assert np.allclose(result.metallicity, np.log10(mock_sp_instance.zlegend))
        assert np.allclose(result.age, mock_sp_instance.log_age - 9.0)
        assert np.allclose(result.wavelength, np.array([4000, 4100, 4200]))
        assert np.allclose(
            result.flux,
            np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
        )


def test_retrieve_ssp_data_from_fsps_with_kwargs():
    with patch.dict('sys.modules', {'fsps': MockFSPS()}), \
            patch("rubix.spectra.ssp.fsps_grid.HAS_FSPS", True):
        
        from rubix.spectra.ssp.fsps_grid import retrieve_ssp_data_from_fsps

        # Call the function with additional keyword arguments
        result = retrieve_ssp_data_from_fsps(add_neb_emission=False, imf_type=1)
        mock_sp_instance = MockFSPS.StellarPopulation()

        # Check the returned SSPGrid object
        assert isinstance(result, SSPGrid)
        assert np.allclose(result.metallicity, np.log10(mock_sp_instance.zlegend))
        assert np.allclose(result.age, mock_sp_instance.log_age - 9.0)
        assert np.allclose(result.wavelength, np.array([4000, 4100, 4200]))
        assert np.allclose(
            result.flux,
            np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]),
        )