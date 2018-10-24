#from scipy.io import whosmat, loadmat
from hdf5storage import loadmat
import numpy as np

# TODO: support a spline with dim > 1 (low priority)


def loadSpline(splineFile, splineVar=None):
    """Loads a spline from .mat format file

    :param splineFile: full or relative path to Matlab file
    :param splineVar: variable to load from splineFile
    :return: a dict with the spline representation
    """
    f = loadmat(splineFile, variable_names = None if splineVar is None else [splineVar], chars_as_strings=True)
    contents = [k for k in f.keys() if not k.startswith('__')]
    if splineVar is None:
        if len(contents) == 1:
            splineVar = contents[0]
        else:
            raise ValueError('The splineFile contains multiple variables: ' + ', '.join(contents) +
                             'Please provide the appropriate splineVar.')
    elif splineVar not in contents :
        raise ValueError('The specified splineVar cannot be found in the specified file.  It only contains ' +
                         'variables ' + ', '.join(contents))
    raw = f[splineVar]
    spd = getSplineDict(raw)
    validateSpline(spd)
    return spd


def _dataset2str(ds):
    return ''.join(chr(c) for c in ds)


def getSplineDict(matSp):
    # TODO: try to find better way to identify the Matlab version of the file
    v = 7.0 if matSp[0][0].dtype.names == matSp[0].dtype.names else 7.3
    spl = matSp[0][0] if v == 7.0 else matSp[0]

    out = {
        'form':     matSp[0]['form'][0][0],  # only on this one can the same slicing be used for both 7 and 7.3
        'knots':    np.array([kd[0] for kd in spl['knots'][0]]),
        'number':   spl['number'][0].astype(int),
        'order':    spl['order'][0].astype(int),
        'dim':      spl['dim'][0].astype(int),
        'coefs':    spl['coefs']
    }
    # note: if dim = 1 and coefs has dimensions [1 number], the coefs should be reshaped
    # e.g. number is [50,20] but coefs has shape (1,50,20)
    if out['dim'].size == 1 and out['dim'][0] == 1 and \
            np.array_equal(np.concatenate((out['dim'], out['number'])), np.array(out['coefs'].shape)):
        # don't squeeze in case other dimensions also singleton
        out['coefs'] = out['coefs'][0,] # just select the single value from the dimension with size 1
    return out


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
    # each dim of knots should have be of length dictated by corresponding number + order
    if not np.array(spd['number'] + spd['order'] == [k.size for k in spd['knots']]).all():
        raise ValueError('The number of knots does not correspond to the number and order for at least one variable.')
    return






