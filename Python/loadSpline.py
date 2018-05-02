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
    return {
        'form':     matSp[0][0][0][0],
        'knots':    np.array([kd[0] for kd in matSp[0][0][1][0]]),
        'number':   matSp[0][0][2][0],
        'order':    matSp[0][0][3][0],
        'dim':      matSp[0][0][4][0],
        'coefs':    matSp[0][0][5]
    }


def convertSpline(matSp):
    """Convert an imported Matlab spline into format expected by scipy

    :param matSp: the imported Matlab spline - expects import format determined by loadMatSpline
    :return:
    """
    spd = getSplineDict(matSp)
    indVars = spd['knots'].size
    # [tx, ty, c, kx, ky]
    # where tx, ty are the knots for the first two dimensions
    # c is the coefficients in a single vector (the product of the dimensions given by order
    # kx, ky are the degrees (NOT ORDER!) of the x and y dimensions, respectively
    return np.array([spd['knots'][0], spd['knots'][1], spd['coefs'].reshape(spd['number'].prod(),), spd['order'][0]-1, spd['order'][1]-1])

def getPartialSpline(spd, fdIdx, dims):
    """For a spline with more than 2 dimensions, split it to evaluate in 2-D for fixed values of other dimensions
    For a 3-variate spline, prep a spline for the first and second dimensions for the third dimension
    using fdIdx = [~,~,<val>] and dims=(0,1)

    :param spd: A dict containing the full spline, with knots, coefs, order, and number keys
    :param fdIdx: A list of floats containing the values indices of fixed dimensions
    :param dims: A 2-tuple containing the zero-based dimensions to be evaluated.
                Values of fdIdx will be ignored for these dimensions, as all values will be included.
    :return: [tx, ty, c, kx, ky]
                where tx and ty are the knots for the dimensions to be evaluated,
                c is the coefficients in a single vector, and
                kx and ky are the degrees of the x and y dimensions
    """
    indVars = spd['number'].size
    # assume spd is a valid spline in format from convertSpline
    knots = [kd[0] for kd in spd['knots'][0]]
    knotRanges = [(min(k), max(k)) for k in knots]

    # confirm that fdIdx length is same as number length
    if length(fdIdx) != length(spd['number']):
        # TODO: throw error that too many values are provided for fdIdx
        pass
    end

    # confirm that the fdIdx values are within the range of knots for the dimension
    # at the same time, determine the knot idx to use for each

    for i in range(0, length(fdIdx)-1):
        if i not in dims:
            if knotRanges(i)(0) <= fdIdx(i) <= knotRanges(i)(1):
                # valAbove = [fdIdx(i) > for k in knots(i)]
                pass
            else:
                # TODO: throw error than requested value is not within knot range
                pass
            end
        end


    #








# spline fields and types
bSplineFieldType = {
    'form': {'dtype': 'U'}, # unicode data - don't care about the byte order or size
    'knots': {'dtype': 'O','a1Size':1},
    'number': {'dtype':'uint8','a1Size':1},
    'order': {'dtype':'uint8','a1Size':1},
    'dim': {'dtype':'uint8','a1Size':1,'a2Size':1},
    'coefs':{'dtype':'f',}
}