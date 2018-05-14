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
    y = np.array([])
    for di in range(spd['number'].size - 1,-1,-1):
        xi = x[di]
        # wrap xi if necessary
        if not isinstance(xi,np.ndarray):
            xi = np.array(xi)
        tck = getNextSpline(di, spd, x, y)
        y = np.array(interp.splev(xi, tck))
    # need to rearrange back to original order and shape
    return rearrangeCoefs(y,-1,spd,x)


def getNextSpline(dimIdx, spd, x, coefs):
    """ Get the tck (knots, coefs, and degree) for the next spline to be evaluated

    :param dimIdx:  The zero-based index of the dimension being evaluated
    :param spd:     The original spline dictionary, with number, knots, order, and coefs
    :param x:       The x values over which the spline is being evaluated
    :param coefs:   The coefficients for this dimension - for the highest index dimension, use the default.
                    Otherwise, this is the output from the last call to interp.splev
                    For the highest order index, coefs.ndim will be the total number of dimensions
                    For other indices, coefs.ndim will be 2 - in these cases. assume that
                    dimIdx is alone in the second dimension and
                    the other dimensions are arrayed in order in the first dimension
    :return: The [t,c,k] spline representation for the next dimension (see splev documentation)
    """
    if coefs.size == 0:
        coefs = spd['coefs']
    t = spd['knots'][dimIdx]
    c = rearrangeCoefs(coefs,dimIdx,spd,x)
    k = spd['order'][dimIdx] - 1
    return [t,c,k]

def rearrangeCoefs(coefs,dimIdx,spd,x):
    dS = getDSizes(dimIdx, spd, x)  # the expected size for each dim in original order
    if dimIdx < 0:
        destIdx = 0          # move this dim to the beginning
        uSh = getDShape(0, dS, False)
        gSh = dS
    else:
        destIdx = -1         # move this dim to the end
        # coefs will be coming in untouched from the last iteration or raw directly from the spline (for the outermost dim)
        #       the previous dimIdx will always be last and the others will be in their original order
        #       (e.g. for 3 dims with dimIdx = 1, first will be the size of dim 0, then size of dim 2, then size of dim 1)
        uSh = getDShape(dimIdx, dS, False)  # the sizes of the dimensions in the order they were passed in for ungrouping
        gSh = getDShape(dimIdx, dS, True)  # the sizes of the dimensions in the new order for regrouping
    # you need to ungroup them first so you have the proper number of dimensions before you move them back around
    # then move the current dim to the end
    # then group all but the current dimension
    # TODO: is forced col-by-col ordering necessary?
    return np.moveaxis(coefs.reshape(uSh, order='F'), dimIdx, destIdx).reshape(gSh, order='F')

def getDSizes(dimIdx, spd, x):
    """ Get the proper size for each dimension given the current dimIdx

    :param dimIdx:  the index of the dimension worked on
    :param spd:     the original spline definition
    :param x:       the x values over which the spline is being evaluated
    """
    return np.concatenate((spd['number'][0:dimIdx+1], np.array([d.size for d in x[dimIdx+1:]],spd['number'].dtype)))

def getDShape(dimIdx, dSizes, group):
    """ Get the reshape

    :param dimIdx:  the index of the current dimension
    :param dSizes:  a Numpy array containing the size of the dimensions for this iteration
    :param group:   a Boolean indicating whether all dimensions other than the dimIdx should be grouped
    :return:        a tuple of ints with the indices of the dimensions
    """
    c = dSizes[dimIdx]
    if group:
        r = int(dSizes.prod() / dSizes[dimIdx]) # product of all dims other than the current one
        return (r,c)
    else:
        r = dSizes[[x for x in range(0,dSizes.size) if x!=dimIdx]]
        # TODO: test performance on this?  s/b small enough to not matter much
        return tuple([x for x in chain(r,[c])])


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
# y = es.evalMultivariateSpline(spd,x)

