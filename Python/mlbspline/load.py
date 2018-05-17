from scipy.io import whosmat, loadmat
import numpy as np


def loadSpline(splineFile, splineVar=None):
    """Loads a spline from .mat format file

    :param splineFile: full or relative path to Matlab file
    :param splineVar: variable to load from splineFile
    :return: a dict with the spline representation
    """
    contents = {var[0]: {'shape': var[1], 'class': var[2]} for var in whosmat(splineFile)}
    if splineVar is None:
        if not len(contents) == 1:
            # TODO: throw error indicating we don't know which var to load
            pass
        else:
            splineVar = next(iter(contents))
    if splineVar not in contents or not contents[splineVar]['class'] == 'struct':
        # TODO: throw an error indicating splineVar is missing or has wrong type
        pass
    raw = loadmat(splineFile,chars_as_strings=True,variable_names=[splineVar])[splineVar]
    # out = h5py.File(splineFile,'r')[splineVar]
    spd = getSplineDict(raw)
    validateSpline(spd)
    return spd


def getSplineDict(matSp):
    return {
        'form':     matSp[0][0][0][0],
        'knots':    np.array([kd[0] for kd in matSp[0][0][1][0]]),
        'number':   matSp[0][0][2][0],
        'order':    matSp[0][0][3][0],
        'dim':      matSp[0][0][4][0],
        'coefs':    matSp[0][0][5]
    }


def validateSpline(spd):
    """Checks the spline representation to make sure it has all the expected fields for evaluation
    Throws an error if anything is missing (one at a time)

    :param spd: a dict (output from getSplineDict) representing the spline
    """
    # currently only supports b-splines
    if spd['form'] != 'B-':
        raise ValueError('These functions currently only support b-splines.')
    # currently only supports 1-D output
    if spd['dim'] != 1:
        raise ValueError('These functions currently only support 1-D values.')
    # need to have same size for knots, number, order
    if spd['number'].shape != spd['knots'].shape or spd['number'].shape != spd['order'].shape:
        raise ValueError('The spline''s knots, number, and order are inconsistent.')
    # number of dims in coefs should be same as size of number
    if spd['number'].size != spd['coefs'].ndim:
        raise ValueError('The spline''s coefficients are inconsistent with its number.')
    # each dim of coefs should have as many members as indicated by number
    if not (np.array(spd['coefs'].shape) == spd['number']).all():
        raise ValueError('At least one of the spline''s coefficients doesn''t match the corresponding number.')
    return






