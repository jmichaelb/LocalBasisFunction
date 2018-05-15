import scipy.interpolate as interp
import numpy as np

import loadSpline as ls
from functools import reduce
from operator import mul
from itertools import chain


def evalMultivariateSpline(spd, x):
    """ Performs recursive evaluation of b-spline for the given independent values
    For now, assumes 1-D spline (y for each n-D x is scalar)
    x and spd['coefs'] must have the same number of dimensions

    :param x:   a Numpy n-D array with the points at which
    :param spd: a b-spline definition as given by loadSpline.getSplineDict
    :return:    a Numpy n-D array with the same shape as x, with the y values for the spline
    """
    dimCt = spd['number'].size          # size of dim coeffs
    y = spd['coefs']
    for di in range(spd['number'].size - 1,-1,-1):
        xi = x[di]
        # wrap xi if necessary
        if not isinstance(xi,np.ndarray):
            xi = np.array(xi)
        tck = getNextSpline(di, dimCt, spd, y)
        y = np.array(interp.splev(xi, tck))
    # need to rearrange back to original order and shape
    return getNextSpline(-1,dimCt,spd,y)[1]


def getNextSpline(dimIdx, dimCt, spd, coefs):
    """ Get the tck (knots, coefs, and degree) for the next spline to be evaluated

    :param dimIdx:  The zero-based index of the dimension being evaluated
    :param dimCt:   The number of dimensions
    :param spd:     The spline being evaluated
    :param coefs:   The coefficients for this dimension - for this outermost dim, this is just the coefs from the spline
                    Otherwise, this is the output from the last call to interp.splev
    :return: The [t,c,k] spline representation for the next dimension (see splev documentation)
    """
    li = dimCt - 1
    if li != dimIdx:
        coefs = np.moveaxis(coefs, li, 0)
    t = spd['knots'][dimIdx]
    k = spd['order'][dimIdx] - 1
    return [t,coefs,k]













