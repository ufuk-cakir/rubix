from jax import jit
from jax.lax import scan


# Source: https://github.com/ArgonneCPAC/dsps/blob/b81bac59e545e2d68ccf698faba078d87cfa2dd8/dsps/utils.py#L247C1-L256C1
@jit
def _cumtrapz_scan_func(carryover, el):
    b, fb = el
    a, fa, cumtrapz = carryover
    cumtrapz = cumtrapz + (b - a) * (fb + fa) / 2.0
    carryover = b, fb, cumtrapz
    accumulated = cumtrapz
    return carryover, accumulated


# Source: https://github.com/ArgonneCPAC/dsps/blob/b81bac59e545e2d68ccf698faba078d87cfa2dd8/dsps/utils.py#L278C1-L298C1
@jit
def trapz(xarr, yarr):
    """Trapezoidal integral

    Parameters
    ----------
    xarr : ndarray, shape (n, )

    yarr : ndarray, shape (n, )

    Returns
    -------
    result : float

    """
    res_init = xarr[0], yarr[0], 0.0
    scan_data = xarr, yarr
    cumtrapz = scan(_cumtrapz_scan_func, res_init, scan_data)[1]
    return cumtrapz[-1]
