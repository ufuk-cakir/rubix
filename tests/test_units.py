from rubix.units import Zsun
import astropy.units as u

def test_zsun_unit():
    assert str(Zsun) == "Zsun"
    assert u.Unit("Zsun") == Zsun