from collections import namedtuple
from pprint import pformat
from inspect import signature

import numpy as np
from numpy.lib.scimath import sqrt

from mlbspline.eval import evalMultivarSpline

def evalSolutionGibbs(gibbsSp, x, M=0, *tdqSpec):
    # TODO: add Va, Cpa
    # TODO: check and document units for all measures
    # TODO: make it easier to add a new measure
    # TODO: make fn flexible enough to handle gibbsSp in different units (low priority)
    """ Calculates thermodynamic quantities for solutions based on a spline giving Gibbs energy
    This only supports single-solute solutions.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Warning: units must be as specified here because some conversions are hardcoded into this function.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    For developers: to add a new thermodynamic quantity, all of the following should be done:
    NOTE: new quantities cannot be named P, T, or X, as those symbols are reserved for the input (x)
    - add the measure spec to getSupportedMeasures - read comments in that fn and in _getMeasureSpec for details
    - create a short function to calculate the measure based on other values such as gibbsSp, x, xm, derivs, tdqout, etc.
        the procedure should be named with 'eval' + the FULL name for the measure - NOT the symbol / tdq flag
        record the symbol / tdq flag as the return value in comments for the function
        add only parameters required to calculate the measure
        Be consistent with parameter names used in other functions - hoping to call these automatically soon
        see evalWaterChemicalPotential function (or any other existing function) for one example
    - add the measure to the section of this function that sets values in the output value (tdqout)
        make sure the code follows the code that calculates other measures on which it depends
        and precedes code calculating measures that depend on it (if adding multiple measures at once)
    - update the comments for param *tdqSpec with the name of the measure and its units
        be sure to add it to the correct section of the comments (PT vs PTX spline, other parameters required, etc)
        or create a new section if one is warranted

    :param gibbsSp: A B-spline  (in format given by loadSpline.loadMatSpline) for giving Gibbs energy (J/mg)
                    with dimensions pressure (MPa), temperature (K), and (optionally) molality, IN THAT ORDER.
                    If molality is not provided, this function assumes that it is calculating
                    thermodynamic properties for pure water.
    :param x:       a Numpy ndarray of ndarrays with the points at which gibbsSp should be evaluated
                    the number and index of dimensions must be same as in the spline (x.size == gibbsSp['number'].size)
                    and each dimension must be sorted (low to high)
    :param M:       float with molecular weight of solute (kg/mol)
    :param tdqSpec: iterable indicating the thermodynamic quantities to be calculated
                    elements can be either strings showing the names or the TDQ objects from getSupportedMeasures
                    If not provided, this function will default to calculating the quantities in defTdq2 for a PT spline,
                    and to the union of quantities in defTdq2 and defTdqSpec3 for a PTX spline.
                    Args can be any of the following strings, each representing a thermodynamic quantity
                    that can be calculated based on a Gibbs energy spline.
                    Any other args provided will result in an error.
                        G           returns Gibbs energy in J/mg
                        rho         returns density in kg/m^3
                        vel         returns sound speed in m/s
                        Cp          returns isobaric specific heat in J/kg/K
                        alpha       returns thermal expansivity in 1/K
                        U           returns internal energy
                        H           returns enthalpy
                        S           returns entropy
                        Kt          returns isothermal bulk modulus
                        Kp          returns bulk modulus pressure derivative
                        Ks          returns isotropic bulk modulus
                        -------------------------------------------- below this line, require PTX spline and non-zero M
                        mus         returns solute chemical potential
                        muw         returns water chemical potential
                        Vm          ??
                        Cpm         ??
    :return:        a named tuple with the requested thermodynamic values
                    as named properties matching the measures requested in the *tdq parameter of this function
                    the x value is also included as tuple for reference
                    note that 0 values for concentration (x[iX]) will be replaced by eps
                    see function adjustConcentrations for details
    """
    dimCt = gibbsSp['number'].size
    tdqSpec = expandTdqSpec(tdqSpec, dimCt)

    _checkNecessaryDataForTDQ(M, dimCt, tdqSpec)

    # # prep for calculating apparent values
    # if dimCt == 3:
    #     # TODO: Do we really need to remove the 0ish concentrations if any were added?
    #     origXCt = len(x[iX])
    #     x[iX] = adjustConcentrations(x[iX])

    tdqout = createThermodynamicQuantitiesObj(dimCt, tdqSpec, x)

    derivs = getDerivatives(gibbsSp, x, dimCt, tdqSpec)
    xm = _getxm(tdqSpec, x)

    # generate thermodynamic quantities
    tdqNames = [t.name for t in tdqSpec]
    if 'f' in tdqNames:
        # TODO: document what f is
        tdqout.f = 1 + M * x[iX]
    if 'G' in tdqNames:
        tdqout.G = evalGibbs(gibbsSp, x)
    if 'Cp' in tdqNames:
        tdqout.Cp = evalSpecificHeat(xm, derivs)
    if 'S' in tdqNames:
        tdqout.S = evalEntropy(derivs)
    if 'vel' in tdqNames:
        tdqout.vel = evalSoundSpeed(derivs)
    if 'rho' in tdqNames:
        tdqout.rho = evalDensity(derivs)
    if 'Kt' in tdqNames:
        tdqout.Kt = evalIsothermalBulkModulus(derivs)
    if 'Kp' in tdqNames:
        tdqout.Kp = evalBulkModulusWrtPressure(derivs)
    if 'Vm' in tdqNames:
        # TODO: document what Vm is and add a fn for evaluating it
        tdqout.Vm = (M * derivs.d1P) + (tdqout.f * derivs.dPX)
    # these need to be done secondarily because they rely on previously generated quantities
    if 'Ks' in tdqNames:
        tdqout.Ks = evalIsotropicBulkModulus(tdqout)
    if 'alpha' in tdqNames:
        tdqout.alpha = evalThermalExpansivity(derivs, tdqout)
    if 'U' in tdqNames:
        tdqout.U = evalInternalEnergy(xm, tdqout)
    if 'mus' in tdqNames:
        tdqout.mus = evalSoluteChemicalPotential(M, derivs, tdqout)
    if 'muw' in tdqNames:
        tdqout.muw = evalWaterChemicalPotential(xm, derivs, tdqout)
    if 'Cpm' in tdqNames:
        # TODO: document what Cpm is and add a fn for evaluating it
        tdqout.Cpm = M * tdqout.Cp - tdqout.f * derivs.d2T1X * xm[iT]
    # these depend on secondary quantities
    if 'H' in tdqNames:
        tdqout.H = evalEnthalpy(xm, tdqout)

    return tdqout


def _checkNecessaryDataForTDQ(M, dimCt, tdqSpec):
    # TODO: issue warning if dim values fall (too far?) outside the knot sequence of gibbsSp?
    """ Checks error conditions before performing any calculations, throwing an error if anything doesn't match up
      - Ensures only supported measures are requested
      - Ensures necessary data is available for requested measures

    :param M:       molecular weight of solvent
    :param dimCt:   if spline being evaluated is a PT spline, this will be 2.  if a PTX spline, this will be 3
                    no other values are valid
    :param tdqSpec: iterable of thermodynamic quantities to be calculated (TDQ objects from getSupportedMeasures)

    """
    # check for completely unsupported measures so no one has to evaluate twice because of a typo
    if not tdqSpec.issubset(measures):
        raise ValueError('One or more unsupported measures have been requested: ' +
                         pformat([t.name for t in tdqSpec.difference(measures)]))

    # make sure that spline has 3 dims if quantities using concentration are requested
    reqX = [t for t in tdqSpec if t.reqX]
    if dimCt == 2 and reqX:
        raise ValueError('You cannot calculate ' + pformat([t.name for t in reqX]) + ' with a spline that does not '
                         'include concentration. Remove the measures and all their dependencies, or supply a spline '
                         'that includes concentration.')
    # make sure that M is provided if any tdqs that require M are requested
    reqM = [t for t in tdqSpec if t.reqM]
    if M == 0 and reqM:
        raise ValueError('You cannot calculate ' + pformat([t.name for t in reqM]) + ' without providing molecular '
                         'weight.  Remove the measures and all their dependencies, or provide a non-zero value '
                         'for the M parameter.')
    return


def expandTdqSpec(tdqSpec, dimCt):
    # TODO: add robust set of tests for this
    """ add dependencies for requested thermodynamic quantities or sets default tdqSpec if none provided
    derivatives are handled separately - see getDerivatives

    :param tdqSpec: an iterable with the thermodynamic quantities to be returned
    :param dimCt:   the number of dimensions - used if setting the default tdqSpec
    :return:        a set of TDQ namedtuples that includes those specied by the tdqSpec parameter and all its dependencies
    """
    if len(tdqSpec) == 0:
        tdqSpec = set(defTdqSpec2)      # make mutable
        if dimCt == 3:
            tdqSpec |= set(defTdqSpec3)
    else:
        # add thermodynamic quantities on which requested ones depend
        tdqSpec = _addTDQDependencies(tdqSpec)

    return tdqSpec


def createThermodynamicQuantitiesObj(dimCt, tdqSpec, x):
    svn = set(['P', 'T'] + ([] if dimCt == 2 else ['X']))
    # TODO: include derivs in output?  fnGval returns d1P/d2P/d3P, but no others
    out = namedtuple('ThermodynamicQuantities', {t.name for t in tdqSpec} & svn)
    # include input in the output so you always know the conditions for the thermodynamic quantities
    for i in range(0,dimCt):
        if i == iP:     out.P = x[i]
        if i == iT:     out.T = x[i]
        if i == iX:     out.X = x[i]
    return out


# def adjustConcentrations(X):
#     # TODO: properly document why eps is added in and determine whether to return amended X or original X (just to avoid divide by 0 error in Cpa and Va calcs?)
#     eps = np.finfo(type(X[0])).eps  # get the lowest positive value that can be distinguished from 0
#     # prepend the list of concentrations with a zero if there isn't already one there
#     out = X
#     # TODO: fix this when some calculation is added that uses it (Cpa or Va, possibly other quantities)
#     # out = X if X[0] == 0 else np.concatenate((np.array([0]), X))
#     out[out == 0] = eps  # add eps to zero concentrations to avoid divide by zero errors
#     return out


def _createGibbsDerivativesObj(tdqSpec):
    return namedtuple('GibbsDerivatives', {d for t in tdqSpec if t.reqDerivs for d in t.reqDerivs})


def getDerivatives(gibbsSp, x, dimCt, tdqSpec):
    out = _createGibbsDerivativesObj(tdqSpec)
    reqderiv = lambda d: [t for t in tdqSpec if d in t.reqDerivs]
    if reqderiv('d1P'):
        out.d1P = evalMultivarSpline(gibbsSp, x, [1 if i == iP else defDer for i in range(0, dimCt)])
    if reqderiv('d1T'):
        out.d1T = evalMultivarSpline(gibbsSp, x, [1 if i == iT else defDer for i in range(0, dimCt)])
    if reqderiv('d1X'):
        out.d1X = evalMultivarSpline(gibbsSp, x, [1 if i == iX else defDer for i in range(0, dimCt)])
    if reqderiv('dPT'):
        out.dPT = evalMultivarSpline(gibbsSp, x, [1 if (i == iP or i == iT) else defDer for i in range(0, dimCt)])
    if reqderiv('dPX'):
        out.dPX = evalMultivarSpline(gibbsSp, x, [1 if (i == iP or i == iX) else defDer for i in range(0, dimCt)])
    if reqderiv('d2P'):
        out.d2P = evalMultivarSpline(gibbsSp, x, [2 if i == iP else defDer for i in range(0, dimCt)])
    if reqderiv('d2T'):
        out.d2T = evalMultivarSpline(gibbsSp, x, [2 if i == iT else defDer for i in range(0, dimCt)])
    if reqderiv('d2T1X'):
        out.d2T1X = evalMultivarSpline(gibbsSp, x,
                                             [2 if i == iT else (1 if i == iX else defDer) for i in range(0, dimCt)])
    if reqderiv('d3P'):
        out.d3P = evalMultivarSpline(gibbsSp, x, [3 if i == iP else defDer for i in range(0, dimCt)])
    return out


def _getxm(tdqSpec, x):
    if [t for t in tdqSpec if t.reqGrid]:
        return np.meshgrid(*x.tolist(), indexing='ij')    # grid the dimensions of x
    else:
        return []


def evalGibbs(gibbsSp, x):
    """
    :return: G
    """
    return evalMultivarSpline(gibbsSp, x)


def evalSpecificHeat(xm, derivs):
    """
    :return:        Cp
    """
    return -1 * derivs.d2T * xm[iT]


def evalEntropy(derivs):
    """
    :return:        S
    """
    return -1 * derivs.d1T


def evalSoundSpeed(derivs):
    """
    :return:        vel
    """
    # MPa-Pa units conversion cancels
    return np.real(sqrt(np.power(derivs.d1P, 2) / (np.power(derivs.dPT, 2) / derivs.d2T - derivs.d2P)))


def evalDensity(derivs):
    """
    :return:        rho
    """
    return 1e6 * np.power(derivs.d1P, -1)   # 1e6 for MPa to Pa


def evalIsothermalBulkModulus(derivs):
    """
    :return:        Kt
    """
    return -1 * derivs.d1P / derivs.d2P


def evalBulkModulusWrtPressure(derivs):
    """
    :return:        Kp
    """
    return derivs.d1P * np.power(derivs.d2P, -2) * derivs.d3P - 1


def evalIsotropicBulkModulus(tdq):
    """
    :param tdq:     the ThermodynamicQuantities namedtuple that stores already-calculated values
    :return:        Ks
    """
    return tdq.rho * np.power(tdq.vel, 2) / 1e6


def evalThermalExpansivity(derivs, tdq):
    """
    :param tdq:     the ThermodynamicQuantities namedtuple that stores already-calculated values
    :return:        alpha
    """
    return 1e-6 * derivs.dPT * tdq.rho  #  1e6 for MPa to Pa


def evalInternalEnergy(xm, tdq):
    """
    :param tdq:     the ThermodynamicQuantities namedtuple that stores already-calculated values
    :param xm:      gridded dimensions over which spline is being evaluated
    :return:        U
    """
    return tdq.G - 1e6 * xm[iP] / tdq.rho + xm[iT] * tdq.S


def evalSoluteChemicalPotential(M, derivs, tdq):
    """
    :param M:       the molecular weight of the solvent
    :param tdq:     the ThermodynamicQuantities namedtuple that stores already-calculated values
    :return:        mus
    """
    return M * tdq.G + tdq.f * derivs.d1X


def evalWaterChemicalPotential(xm, derivs, tdq):
    """
    :param xm:      gridded dimensions over which spline is being evaluated
    :param tdq:     the ThermodynamicQuantities namedtuple that stores already-calculated values
    :return:        muw
    """
    return (tdq.G / nw) - (1 / nw * tdq.f * xm[iX] * derivs.d1X)


def evalEnthalpy(xm, tdq):
    """
    :param xm:      gridded dimensions over which spline is being evaluated
    :param tdq:     the ThermodynamicQuantities namedtuple that stores already-calculated values
    :return:        H
    """
    return tdq.U - xm[iT] * tdq.S


def _getTDQSpec(name, reqX=False, reqM=False, reqGrid=False, reqDerivs=[], reqTDQ=[]):
    """ Returns a TDQ namedtuple indicating what is required to calculate this particular thermodynamic quantity (tdq)
    :param name:        the name / symbol of the tdq (e.g., G, rho, alpha, muw)
    :param reqX:        True if DIRECTLY calculating the tdq requires concentration.  False otherwise
    :param reqM:        True if DIRECTLY calculating the tdq requires molecular weight.  False otherwise
    :param reqGrid:     True if DIRECTLY calculating the tdq requires you to grid the PT[X] values.  False otherwise
    :param reqDerivs:   A list of derivatives needed to DIRECTLY calculate the tdq
                        (e.g. for tdq Kp, this would be ['d1P','dP','d3P'] - see fn evalBulkModulusWrtPressure)
                        See getDerivatives for a full list of derivatives that can be calculated
    :param reqTDQ:      A list of other thermodynamic quantities needed to DIRECTLY calculate the tdq
                        (e.g. for tdq U, this would be ['G','rho'] - see fn evalInternalEnergy)
    :return:            a namedtuple giving the spec for the tdq
    """
    # TODO: probably better to write a full class so the equality method can check by name
    # TODO: figure out how to add the function associated with a tdq and automatically call it based on the spec
    out = namedtuple('TDQ', signature(_getTDQSpec).parameters.keys())
    # TODO: figure out how to do this in a meta way like the previous line
    out.name = name
    out.reqX=reqX
    out.reqM=reqM
    out.reqGrid=reqGrid
    out.reqDerivs=reqDerivs
    out.reqTDQ=reqTDQ
    return out

def getSupportedThermodynamicQuantities():
    """ When adding new tdqs, you don't need to worry about adding them in any particular order -
        dependencies are handled elsewhere
        See also the comments for _getTDQSpec

    :return: the full set of thermodynamic quantities supported by this module
    """
    return frozenset([
        _getTDQSpec('G'),  # it's true - G needs nothing but the spline
        _getTDQSpec('rho', reqDerivs=['d1P']),
        _getTDQSpec('vel', reqDerivs=['d1P', 'dPT', 'd2T', 'd2P']),
        _getTDQSpec('Cp', reqGrid=True, reqDerivs=['d2T']),
        _getTDQSpec('alpha', reqDerivs=['dPT'], reqTDQ=['rho']),
        _getTDQSpec('U', reqGrid=True, reqTDQ=['G', 'rho', 'S']),
        _getTDQSpec('H', reqGrid=True, reqTDQ=['U', 'S']),
        _getTDQSpec('S', reqDerivs=['d1T']),
        _getTDQSpec('Kt', reqDerivs=['d1P', 'd2P']),
        _getTDQSpec('Kp', reqDerivs=['d1P', 'd2P', 'd3P']),
        _getTDQSpec('Ks', reqTDQ=['rho', 'vel']),
        _getTDQSpec('f', reqX=True, reqM=True),
        _getTDQSpec('mus', reqM=True, reqDerivs=['d1X'], reqTDQ=['f', 'G']),
        _getTDQSpec('muw', reqGrid=True, reqDerivs=['d1X'], reqTDQ=['f', 'G']),
        _getTDQSpec('Vm', reqM=True, reqDerivs=['d1P', 'dPX'], reqTDQ=['f']),
        _getTDQSpec('Cpm', reqM=True, reqGrid=True, reqDerivs=['d2T1X'], reqTDQ=['Cp', 'f'])
            ])

def _addTDQDependencies(origTDQs):
    """ recursively adds thermodynamic quantities on which origTDQs are dependent
    This does not handle derivative dependencies - see getDerivatives

    :param origTDQs:    an iterable of TDQs to check for dependencies
                        elements can be either the string names of measures or the TDQ objects
    :return:            set of TDQ objects including origTDQs and any TDQs on which they are
                        directly or indirectly dependent
    """
    otnames = origTDQs if isinstance(next(iter(origTDQs)), str) else {m.name for m in origTDQs}
    while True:
        # get TDQs dependent on TDQs in otnames
        reqot = {m.name for m in measures for t in m.reqTDQ if t in otnames}
        # if every tdq in reqot is already in otnames, stop
        if reqot.issubset(otnames):
            break
        else:
            # add the new list of tdqs and repeat the loop to add more nested dependencies
            otnames = otnames.union(reqot)
    return set([m for m in measures if m.name in otnames])

def _setDefaultTdqSpecs():
    # concentration required if the flag says it does or if it uses a concentration derivative
    xreq = {m for m in measures if m.reqX or [d for d in m.reqDerivs if 'X' in d]}
    xreq = [r.name for r in _addTDQDependencies(xreq)]
    defspec3 = frozenset([m for m in measures if m.name in xreq])
    return measures.difference(defspec3), defspec3




#########################################
## Constants
#########################################
# TODO: Change to 18.01528? has a 10 oom effect on ML - Python calcs for muw (but relative still only ~1e-6)
nw = 1000/18.0152       # moles/kg for pure water
# dimension indices
iP = 0; iT = 1; iX = 2
# default derivative
defDer = 0
measures = getSupportedThermodynamicQuantities()
defTdqSpec2, defTdqSpec3 = _setDefaultTdqSpecs()














