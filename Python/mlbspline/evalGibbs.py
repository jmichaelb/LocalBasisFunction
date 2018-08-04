from collections import namedtuple
from pprint import pformat
from inspect import signature, currentframe, getargvalues
from time import time
from datetime import datetime

import numpy as np
from numpy.lib.scimath import sqrt

from mlbspline.eval import evalMultivarSpline


# TODO: probably should move this to a different namespace
def evalSolutionGibbs(gibbsSp, PTX, M=0, verbose=False, *tdqSpec):
    # TODO: add Va, Cpa
    # TODO: check and document units for all measures
    # TODO: add logging? possibly in stead of verbose
    """ Calculates thermodynamic quantities for solutions based on a spline giving Gibbs energy
    This only supports single-solute solutions.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Warning: units must be as specified here because some conversions are hardcoded into this function.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    For developers: to add a new thermodynamic quantity (TDQ), all of the following should be done:
    NOTE: new quantities cannot be named P, T, or X, as those symbols are reserved for the input
    You will need to read and understand the comments for getSupportedMeasures and _getTDQSpec
    - create a short function to calculate the measure based on other values
        such as gibbsSp, PTX, xm, derivs, tdqout, etc.
        the procedure should be named with 'eval' + the FULL name for the measure - NOT the symbol / tdq flag
        record the symbol / tdq flag as the return value in comments for the function
        add only parameters required to calculate the measure
        Be consistent with parameter names used in other functions or use the parm* parameters of _getTDQSpec
        If you end up with an as-yet unused parameter, add it to _getTDQSpec (defaulting to OFF)
         AND to the evaluation section of this function
    - add the measure spec to getSupportedThermodynamicQuantities -
        when the comments say DIRECTLY, they mean only consider something a requirement if it is used in
        the function built in the previous step
        dependencies (including nested dependencies) will be handed through the reqDerivs and reqTDQ parameters
    - update the comments for param *tdqSpec with the name of the measure and its units
        be sure to add it to the correct section of the comments (PT vs PTX spline, other parameters required, etc)
        or create a new section if one is warranted

    :param gibbsSp: A B-spline  (in format given by loadSpline.loadMatSpline) for giving Gibbs energy (J/mg)
                    with dimensions pressure (MPa), temperature (K), and (optionally) molality (mol/kg), IN THAT ORDER.
                    If molality is not provided, this function assumes that it is calculating
                    thermodynamic properties for pure water.
    :param PTX:     a Numpy ndarray of ndarrays with the points at which gibbsSp should be evaluated
                    the number of dimensions must be same as in the spline (PTX.size == gibbsSp['number'].size)
                    each of the inner ndarrays represents one of pressure (P), temperature (T), or molality (X)
                    and must be in the same order and units described in the notes for the gibbsSp parameter
                    Additionally, each dimension must be sorted from low to high values.
    :param M:       float with molecular weight of solute (kg/mol)
    :param verbose: boolean indicating whether to print status updates, warnings, etc.
    :param tdqSpec: iterable indicating the thermodynamic quantities to be calculated
                    elements can be either strings showing the names or the TDQ objects from getSupportedMeasures
                    If not provided, this function will calculate the quantities in defTdq2 for a PT spline,
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
                        Kp          returns pressure derivatives of isothermal bulk modulus
                        Ks          returns isotropic bulk modulus
                        V           returns unit volume in m^3/kg
                        -------------------------------------------- below this line, require PTX spline and non-zero M
                        mus         returns solute chemical potential
                        muw         returns water chemical potential
                        Vm          returns partial molar volume
                        Cpm         returns partial molar heat capacity
                        Cpa         returns apparent heat capacity
                        Va          returns apparent volume
    :return:        a named tuple with the requested thermodynamic quantities
                    as named properties matching the measures requested in the *tdq parameter of this function
                    the output will also include P, T, and X (if provided) properties
    """
    dimCt = gibbsSp['number'].size
    tdqSpec = expandTdqSpec(tdqSpec, dimCt)
    _checkNecessaryDataForTDQ(M, dimCt, tdqSpec)

    tdqout = createThermodynamicQuantitiesObj(dimCt, tdqSpec, PTX)
    derivs = getDerivatives(gibbsSp, PTX, dimCt, tdqSpec, verbose)
    xm = _getxm(tdqSpec, PTX)

    # calculate thermodynamic quantities and store in appropriate fields in tdqout
    comptdq = set()     # list of completed tdqs
    while len(comptdq) < len(tdqSpec):
        # get tdqs that either have empty reqTDQ or all of those tdqs have already been calculated
        # but that are not in comptdq themselves (don't recalculate anything)
        tdqsToEval = tuple(t for t in tdqSpec if t.name not in comptdq and (not t.reqTDQ or not t.reqTDQ.difference(comptdq)))
        for t in tdqsToEval:
            # build args for the calcFn
            args = dict()
            if t.reqDerivs: args[t.parmderivs] = derivs
            if t.reqGrid:   args[t.parmgrid] = xm
            if t.reqM:      args[t.parmM] = M
            if t.reqTDQ:    args[t.parmtdq] = tdqout
            if t.reqSpline: args[t.parmspline] = gibbsSp
            if t.reqPTX:    args[t.parmptx] = PTX
            start = time()
            setattr(tdqout, t.name, t.calcFn(**args))
            end = time()
            if verbose: _printTiming('tdq '+t.name, start, end)
            comptdq.add(t.name)

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


# TODO: add robust set of tests for this
def expandTdqSpec(tdqSpec, dimCt):
    """ add dependencies for requested thermodynamic quantities or sets default tdqSpec if none provided
    derivatives are handled separately - see getDerivatives

    :param tdqSpec: an iterable with the names of thermodynamic quantities to be evaluated
    :param dimCt:   the number of dimensions - used if setting the default tdqSpec
    :return:        an immutable iterable of TDQ namedtuples that includes those specified by tdqSpec
                    and all their dependencies
    """
    # check for completely unsupported measures so no one has to evaluate twice because of a typo
    if not set(tdqSpec).issubset(measurenames):
        raise ValueError('One or more unsupported measures have been requested: ' +
                         pformat([t.name for t in tdqSpec.difference(measures)]))

    if len(tdqSpec) == 0:
        tdqSpec = list(defTdqSpec2)      # make mutable
        if dimCt == 3:
            tdqSpec.extend(defTdqSpec3)
    else:
        # add thermodynamic quantities on which requested ones depend
        tdqSpec = _addTDQDependencies(tdqSpec)

    return tdqSpec


def createThermodynamicQuantitiesObj(dimCt, tdqSpec, PTX):
    flds = {t.name for t in tdqSpec} | set(['P', 'T'] + ([] if dimCt == 2 else ['X']))
    TDQ = type('ThermodynamicQuantities', (object,), {i: None for i in flds})
    out = TDQ()
    # include input in the output so you always know the conditions for the thermodynamic quantities
    for i in range(0,dimCt):
        if i == iP:     out.P = PTX[i]
        if i == iT:     out.T = PTX[i]
        if i == iX:     out.X = PTX[i]
    return out


def _createGibbsDerivativesClass(tdqSpec):
    flds = {d for t in tdqSpec if t.reqDerivs for d in t.reqDerivs}
    return type('GibbsDerivatives', (object,), {d:None for t in tdqSpec if t.reqDerivs for d in t.reqDerivs})


def _buildDerivDirective(derivSpec, dimCt):
    out = [defDer] * dimCt
    if derivSpec.wrtP: out[iP] = derivSpec.wrtP
    if derivSpec.wrtT: out[iT] = derivSpec.wrtT
    if derivSpec.wrtX: out[iX] = derivSpec.wrtX
    return out


def getDerivatives(gibbsSp, PTX, dimCt, tdqSpec, verbose=False):
    GibbsDerivs = _createGibbsDerivativesClass(tdqSpec)
    out = GibbsDerivs()
    reqderivs = {d for t in tdqSpec for d in t.reqDerivs}   # get set of derivative names that are needed
    getDerivSpec = lambda dn: next(d for d in derivatives if d.name == dn)
    for rd in reqderivs:
        derivDirective = _buildDerivDirective(getDerivSpec(rd), dimCt)
        start = time()
        setattr(out, rd, evalMultivarSpline(gibbsSp, PTX, derivDirective))
        end = time()
        if verbose: _printTiming('deriv '+rd, start, end)
    return out


def _getxm(tdqSpec, PTX):
    if [t for t in tdqSpec if t.reqGrid]:
        return np.meshgrid(*PTX.tolist(), indexing='ij')    # grid the dimensions of PTX
    else:
        return []


def evalVolWaterInVolSolutionConversion(M, PTX):
    """ Conversion factor for how much of the volume of 1 kg of solution is really just the water
    :return:    f
    """
    return 1 + M * PTX[iX]


def evalGibbs(gibbsSp, PTX):
    """
    :return: G
    """
    return evalMultivarSpline(gibbsSp, PTX)


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


def evalIsothermalBulkModulusWrtPressure(derivs):
    """
    :return:        Kp (K')
    """
    return derivs.d1P * np.power(derivs.d2P, -2) * derivs.d3P - 1


def evalIsotropicBulkModulus(tdq):
    """
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


def evalPartialMolarVolume(M, derivs, tdq):
    """ Slope at a point of the V v. X graph
    :return:        Vm
    """
    return (M * derivs.d1P) + (tdq.f * derivs.dPX)


def evalPartialMolarHeatCapacity(M, tdq, derivs, xm):
    """
    :return:    Cpm
    """
    return M * tdq.Cp - tdq.f * derivs.d2T1X * xm[iT]


def evalVolume(tdq):
    """
    :return:    V
    """
    return np.power(tdq.rho, -1)


# TODO: implement once key questions are answered
def evalApparentSpecificHeat(PTX, xm, tdq):
    """
    :return:    Cpa
    """
    # zeroXIdx = PTX[iX][PTX[iX]==0]      # returns empty array if nothing there
    # zeroXCp = np.squeeze(tdq.Cp[:,:,zeroXIdx])
    # return tdq.Cp * tdq.f -
    #Cpa=(Cp.*f -repmat(squeeze(Cp(:,:,1)),1,1,length(m)))./mm;
    return None


# TODO: implement once key questions are answered
def evalApparentVolume(PTX, xm, tdq):
    """ slope of a chord between pure water and a concentration on a V v. X graph
    :return:    Va
    """
    # Va=1e6*(V.*f - repmat(squeeze(V(:,:,1)),1,1,length(m)))./mm;
    return None


def _getTDQSpec(name, calcFn, reqX=False, reqM=False, parmM='M', reqGrid=False, parmgrid='xm', reqDerivs=[],
                parmderivs='derivs', reqTDQ=[], parmtdq='tdq', reqSpline=False, parmspline='gibbsSp', reqPTX=False,
                parmptx='PTX'):
    """ Builds a TDQSpec namedtuple indicating what is required to calculate this particular thermodynamic quantity
    :param name:        the name / symbol of the tdq (e.g., G, rho, alpha, muw)
    :param calcFn:      the name of the function used to calculate the tdq
    :param reqX:        True if DIRECTLY calculating the tdq requires concentration.  False otherwise
    :param reqM:        True if DIRECTLY calculating the tdq requires molecular weight.  False otherwise
    :param parmM:       the name of the parameter of calcFn used to pass in M if reqM
    :param reqGrid:     True if DIRECTLY calculating the tdq requires you to grid the PT[X] values.  False otherwise
    :param parmgrid:    the name of the parameter of calcFn used to pass in the gridded input dimensions if reqGrid
    :param reqDerivs:   A list of derivatives needed to DIRECTLY calculate the tdq
                        (e.g. for tdq Kp, this would be ['d1P','dP','d3P'] - see fn evalIsothermalBulkModulusWrtPressure)
                        See getDerivatives for a full list of derivatives that can be calculated
    :param parmderivs:  the name of the parameter of calcFn used to pass in the pre-calculated derivatives if reqDerivs
    :param reqTDQ:      A list of other thermodynamic quantities needed to DIRECTLY calculate the tdq
                        (e.g. for tdq U, this would be ['G','rho'] - see fn evalInternalEnergy)
                        Note: recursive dependencies are not supported
    :param parmtdq:     the name of the parameter of calcFn used to pass in the tdqout if reqTDQ
    :param reqSpline:   If True, calcFn needs the spline definition
    :param parmspline:  the name of the parameter of calcFn used to pass in the spline def if reqSpline
    :param reqPTX:      If True, calcFn needs the original dimension input (parm PTX in evalSolutionGibbs) to run
    :param parmptx:     the name of the parameter of calcFn used to pass in the original input if reqPTX
    :return:            a namedtuple giving the spec for the tdq
    """
    reqTDQ = set(reqTDQ); reqDerivs = set(reqDerivs)    # make these sets to enforce uniqueness and immutability
    if name in reqTDQ:
        raise ValueError('Recursive dependencies are not supported.  Amend the ' + name + ' TDQSpec accordingly')
    if reqDerivs.difference(derivativenames):
        raise ValueError('One or more derivatives are not supported. Amend the ' + name + ' TDQSpec accordingly.' +
                         'Supported derivatives are '+pformat(derivativenames))
    arginfo = getargvalues(currentframe())
    flds = arginfo.args
    if not reqM:        flds.remove('parmM')
    if not reqGrid:     flds.remove('parmgrid')
    if not reqDerivs:   flds.remove('parmderivs')
    if not reqTDQ:      flds.remove('parmtdq')
    if not reqSpline:   flds.remove('parmspline')
    if not reqPTX:      flds.remove('parmptx')
    vals = [arginfo.locals[v] for v in arginfo.locals if v in flds]

    tdqspec = namedtuple('TDQSpec', flds, defaults=vals)
    return tdqspec()


def getSupportedThermodynamicQuantities():
    """ When adding new tdqs, you don't need to worry about adding them in any particular order -
        dependencies are handled elsewhere
        See also the comments for _getTDQSpec and evalSolutionGibbs

    :return: immutable iterable with the the full set of specs for thermodynamic quantities supported by this module
    """
    out = tuple([
        _getTDQSpec('G', evalGibbs, reqSpline=True, reqPTX=True),
        _getTDQSpec('rho', evalDensity, reqDerivs=['d1P']),
        _getTDQSpec('vel', evalSoundSpeed, reqDerivs=['d1P', 'dPT', 'd2T', 'd2P']),
        _getTDQSpec('Cp', evalSpecificHeat, reqGrid=True, reqDerivs=['d2T']),
        _getTDQSpec('alpha', evalThermalExpansivity, reqDerivs=['dPT'], reqTDQ=['rho']),
        _getTDQSpec('U', evalInternalEnergy, reqGrid=True, reqTDQ=['G', 'rho', 'S']),
        _getTDQSpec('H', evalEnthalpy, reqGrid=True, reqTDQ=['U', 'S']),
        _getTDQSpec('S', evalEntropy, reqDerivs=['d1T']),
        _getTDQSpec('Kt', evalIsothermalBulkModulus, reqDerivs=['d1P', 'd2P']),
        _getTDQSpec('Kp', evalIsothermalBulkModulusWrtPressure, reqDerivs=['d1P', 'd2P', 'd3P']),
        _getTDQSpec('Ks', evalIsotropicBulkModulus, reqTDQ=['rho', 'vel']),
        _getTDQSpec('f', evalVolWaterInVolSolutionConversion, reqX=True, reqM=True, reqPTX=True),
        _getTDQSpec('mus', evalSoluteChemicalPotential, reqM=True, reqDerivs=['d1X'], reqTDQ=['f', 'G']),
        _getTDQSpec('muw', evalWaterChemicalPotential, reqX=True, reqGrid=True, reqDerivs=['d1X'], reqTDQ=['f', 'G']),
        _getTDQSpec('Vm', evalPartialMolarVolume, reqM=True, reqDerivs=['d1P', 'dPX'], reqTDQ=['f']),
        _getTDQSpec('Cpm', evalPartialMolarHeatCapacity, reqM=True, reqGrid=True, reqDerivs=['d2T1X'],
                    reqTDQ=['Cp', 'f']),
        _getTDQSpec('V', evalVolume, reqTDQ=['rho']),
        _getTDQSpec('Cpa', evalApparentSpecificHeat, reqX=True, reqGrid=True, reqTDQ=['Cp', 'f'], reqPTX=True),
        _getTDQSpec('Va', evalApparentVolume, reqX=True, reqGrid=True, reqTDQ=['V', 'f'], reqPTX=True)
            ])
    # check that all reqTDQs are represented in the list
    outnames = frozenset([t.name for t in out])
    unrecognizedDependencies = [(tdq.name, dep) for tdq in out for dep in tdq.reqTDQ if dep not in outnames]
    if unrecognizedDependencies:
        raise ValueError('One or more measures depend on an unrecognized TDQ: '+pformat(unrecognizedDependencies))
    return out, outnames


def getSupportedDerivatives():
    """ Define a name for a derivative of Gibbs energy
    and the derivative order with respect to P, T, and X required to calculate it
    (if not provided, all default to defDer)
    These can then be used in a TDQSpec

    :return:    an immutable iterable containing a list of supported derivatives
    """
    return tuple([
        derivSpec('d1P', wrtP=1),
        derivSpec('d1T', wrtT=1),
        derivSpec('d1X', wrtX=1),
        derivSpec('dPT', wrtP=1, wrtT=1),
        derivSpec('dPX', wrtP=1, wrtX=1),
        derivSpec('d2P', wrtP=2),
        derivSpec('d2T', wrtT=2),
        derivSpec('d2T1X', wrtT=2, wrtX=1),
        derivSpec('d3P', wrtP=3)
    ])


def _addTDQDependencies(origTDQs):
    """ recursively adds thermodynamic quantities on which origTDQs are dependent
    This does not handle derivative dependencies - see getDerivatives

    :param origTDQs:    an iterable of TDQs to check for dependencies
                        elements can be either the string names of measures or the TDQ objects
    :return:            immutable iterable of TDQ objects including origTDQs and any TDQs on which they are
                        directly or indirectly dependent
    """
    otnames = set(origTDQs) if isinstance(next(iter(origTDQs)), str) else {m.name for m in origTDQs}
    while True:
        # get TDQs dependent on TDQs in otnames
        reqot = {t for m in measures if m.name in otnames for t in m.reqTDQ}
        # if every tdq in reqot is already in otnames, stop
        if reqot.issubset(otnames):
            break
        else:
            # add the new list of tdqs and repeat the loop to add more nested dependencies
            otnames = otnames.union(reqot)
    return tuple([m for m in measures if m.name in otnames])


def _setDefaultTdqSpecs():
    # concentration required if the flag says it does or if it uses a concentration derivative
    xreq = [m for m in measures if m.reqX or [d for d in m.reqDerivs if 'X' in d]]
    xreq = [r.name for r in _addTDQDependencies(xreq)]
    # return values should be immutable - use tuples as TDQSpec namedtuples are not hashable so set is not usable
    return tuple([m for m in measures if m.name not in xreq]), tuple([m for m in measures if m.name in xreq])


def _printTiming(calcdesc, start, end):
    endDT = datetime.fromtimestamp(end)
    print(endDT.strftime('%H:%M:%S.%f'), ':\t', calcdesc,'took',str(end-start),'seconds to calculate')



#########################################
## Constants
#########################################
# TODO: Change to 18.01528? has a 10 oom effect on ML - Python calcs for muw (but relative still only ~1e-6)
nw = 1000/18.0152       # moles/kg for pure water
iP = 0; iT = 1; iX = 2  # dimension indices
defDer = 0              # default derivative

derivSpec = namedtuple('DerivativeSpec', ['name', 'wrtP', 'wrtT', 'wrtX'], defaults=[None, defDer, defDer, defDer])
derivatives = getSupportedDerivatives()
derivativenames = {d.name for d in derivatives}

measures, measurenames = getSupportedThermodynamicQuantities()
defTdqSpec2, defTdqSpec3 = _setDefaultTdqSpecs()
















