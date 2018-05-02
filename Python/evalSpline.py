import scipy.interpolate as interp
import numpy as np

import loadSpline as ls
from functools import reduce
from operator import mul





def evalMultivariateSpline(x, spd):
    """ Performs recursive evaluation of b-spline for the given independent values
    For now, assumes 1-D spline (y for each n-D x is scalar)

    :param x:   a Numpy n-D array with the same number of dimensions as spd['coefs']
    :param spd: a b-spline definition as given by loadSpline.getSplineDict
    :return:    a Numpy n-D array with the same shape as x, with the y values for the spline
    """
    y = np.array([])
    for di in range(spd['number'].size - 1,0):
        xi = x[di]
        if not isinstance(xi,np.ndarray):
            xi = np.array(xi)
        tck = getNextSpline(di, spd, coefs=y)
        y = np.array(interp.splev(xi, tck))
        # rearrange in original order
    y = y.reshape([d.size for d in x])
    return y


def getNextSpline(dimIdx, spd, x, coefs=np.array([])):
    """ Get the tck (knots, coefs, and degree) for the next spline to be evaluated

    :param dimIdx:  The zero-based index of the dimension being evaluated
    :param spd:     The original spline dictionary, with number, knots, order, and coefs
    :param x:       The x values over which the spline is being evaluated
    :param coefs:   The coefficients for this dimension - for the highest index dimension, use the default.
                    Otherwise, this is the output from the last call to interp.splev
    :return: The [t,c,k] spline representation for the next dimension (see splev documentation)
    """
    if coefs.size == 0:
        coefs = spd['coefs']
    # output cols correspond to control pts for this dimension
    r = spd['number'][dimIdx]
    # rearrange the dimensions so the next dimension is first and everything else is pushed right
    # output rows correspond to other dimensions
    # for dimensions yet to be handled, use number of control points
    # for dimensions already handled, use size of x for that var
    d = int(spd['number'][0:dimIdx].prod() * reduce(mul, [d.size for d in x[dimIdx+1:]], 1))
    t = spd['knots'][dimIdx]
    c = coefs.reshape(d,r,order='F')    # TODO: is forced col-by-col ordering necessary?
    k = spd['order'][dimIdx] - 1
    return [t,c,k]

def getDSizes(dimIdx, spd, x):
    """ Get the proper size for each dimension given the current dimIdx

    :param dimIdx:  the index of the dimension worked on
    :param spd:     the original spline definition
    :param x:       the x values over which the spline is being evaluated
    """
    return np.concatenate((spd['number'][0:dimIdx], np.array([d.size for d in x[dimIdx:]])))




# r = spd['number'][2]
# d = spd['number'][0:2].prod()
# t = spd['knots'][2]
# c = spd['coefs'].reshape(d,r,order='F')
# k = spd['order'][2] - 1
# x = 2.1
#
# c2 = np.array(interp.splev(x, [t,c,k]))
#
# r = spd['number'][1]
# d = spd['number'][0:1].prod()
# t = spd['knots'][1]
# k = spd['order'][1] - 1
# c = c2.reshape(d,r,order='F')
# x = 285
#
# c3 = np.array(interp.splev(x,[t,c,k]))
#
# r = spd['number'][0]
# d = spd['number'][0:0].prod()
# t = spd['knots'][0]
# k = spd['order'][0] - 1
# c = c3.reshape(d,r,order='F')
# x = -.95


# y = np.array(interp.splev(x,[t,c,k]))


# splineFile = '/Users/pennyespinoza/iSchool/BrownLab/SolutionThermodynamics/appSS_BSpline.mat'
# rawsp = ls.loadMatSpline(splineFile)
# spd = ls.getSplineDict(rawsp)
# P = np.log10(np.logspace(-1,3.7,20))
# T = np.linspace(250,800,10)
# X = np.linspace(0,3,3)
# x = np.array([P,T,X])
# # x = np.array([-.95,285,2.1])
# y = evalMultivariateSpline(x,spd)
# print y
