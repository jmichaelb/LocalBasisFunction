import scipy.io as sio
import numpy as np

def loadMatSpline(splineFile, splineVar=None):
    """Loads a spline from .mat format file

    :param splineFile: full or relative path to Matlab file
    :param splineVar: variable to load from splineFile
    :return: a dict with the spline representation
    """
    contents = {var[0]: {'shape': var[1], 'class': var[2]} for var in sio.whosmat(splineFile)}
    if splineVar is None:
        if not len(contents) == 1:
            # TODO: throw error indicating we don't know which var to load
            pass
        else:
            splineVar = next(iter(contents))
    if not splineVar in contents or not contents[splineVar]['class']=='struct':
        # TODO: throw an error indicating splineVar is missing or has wrong type
        pass
    out = sio.loadmat(splineFile,chars_as_strings=True,variable_names=[splineVar])[splineVar]
    # out = h5py.File(splineFile,'r')[splineVar]
    validateSpline(out)
    return out


def validateSpline(matSp):
    """Checks the spline representation to make sure it has all the expected fields for evaluation
    Throws an error if anything is missing (one at a time)

    :param matSp: a dict representing the spline
    """
    # need to have all fields indicated in bSplineFieldType if form == B-
    # need to have same shape for knots, number, order
    # number of axes in coefs should be same as size of axis 2 for knots/number/order
    # size of each axis in coefs should equal values in number
    pass

def getSplineDict(matSp):
    # TODO change this to use a UnivariateSpline object?
    return {
        'form':     matSp[0][0][0][0],
        'knots':    np.array([kd[0] for kd in matSp[0][0][1][0]]),
        'number':   matSp[0][0][2][0],
        'order':    matSp[0][0][3][0],
        'dim':      matSp[0][0][4][0],
        'coefs':    matSp[0][0][5]
    }

