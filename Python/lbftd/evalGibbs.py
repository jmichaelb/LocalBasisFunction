from collections import namedtuple
from datetime import datetime
from inspect import currentframe, getargvalues
from pprint import pformat
from time import time
from warnings import warn

import numpy as np
from numpy.lib.scimath import sqrt
from psutil import virtual_memory

from mlbspline.eval import evalMultivarSpline


def evalSolutionGibbs(gibbsSp, PTX, MWv=18.01528e-3, MWu=None, failOnExtrapolate=True, verbose=False, *tdvSpec):
    # TODO: add logging? possibly in stead of verbose
    """ Calculates thermodynamic variables for solutions based on a spline giving Gibbs energy
    This currently only supports single-solute solutions.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Warning: units must be as specified here because some conversions are hardcoded into this function.
        With the exception of pressure, units are SI.  Pressure is in MPa rather than Pa.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    For developers: to add a new thermodynamic variable (TDV), all of the following should be done:
    NOTE: new variables cannot be named PTX, as that symbols are reserved for the input
    You will need to read and understand the comments for getSupportedMeasures and _getTDVSpec
    - create a short function to calculate the measure based on other values
        such as gibbsSp, PTX, gPTX, derivs, tdvout, etc.
        the procedure should be named with 'eval' + the FULL name for the measure - NOT the symbol / tdv flag
        record the symbol / tdv flag as the return value in comments for the function
        add only parameters required to calculate the measure
        Be consistent with parameter names used in other functions or use the parm* parameters of _getTDVSpec
        If you end up with an as-yet unused parameter, add it to _getTDVSpec (defaulting to OFF)
        AND to the evaluation section of this function
    - add the measure spec to getSupportedThermodynamicVariables -
        when the comments say DIRECTLY, they mean only consider something a requirement if it is used in
        the function built in the previous step
        dependencies (including nested dependencies) will be handed through the reqDerivs and reqTDV parameters
    - update the comments for param *tdvSpec with the name of the measure and its units
        be sure to add it to the correct section of the comments (PT vs PTX spline, other parameters required, etc)
        or create a new section if one is warranted

    :param gibbsSp: A B-spline  (in format given by loadSpline.loadMatSpline) for giving Gibbs energy (J/mg)
                    with dimensions pressure (MPa), temperature (K), and (optionally) molality (mol/kg), IN THAT ORDER.
                    If molality is not provided, this function assumes that it is calculating
                    thermodynamic properties for a pure substance.
    :param PTX:     a Numpy ndarray of ndarrays with the points at which gibbsSp should be evaluated
                    the number of dimensions must be same as in the spline (PTX.size == gibbsSp['number'].size)
                    each of the inner ndarrays represents one of pressure (P), temperature (T), or molality (X)
                    and must be in the same order and units described in the notes for the gibbsSp parameter
                    Additionally, each dimension must be sorted from low to high values.
    :param MWv:     float with molecular weight of solvent (kg/mol).  Defaults to molecular weight of water
    :param MWu:     float with molecular weight of solute (kg/mol).
    :param failOnExtrapolate:   True if you want an error to appear if PTX includes values that fall outside the knot
                    sequence of gibbsSp.  If False, throws a warning rather than an error, and
                    proceeds with the calculation.
    :param verbose: boolean indicating whether to print status updates, warnings, etc.
    :param tdvSpec: iterable indicating the thermodynamic variables to be calculated
                    elements can be either strings showing the names or the TDV objects from getSupportedMeasures
                    If not provided, this function will calculate the variables in defTDV2 for a PT spline,
                    and to the union of variables in defTDV2 and defTDVSpec3 for a PTX spline.
                    Args can be any of the following strings, each representing a thermodynamic quantity
                    that can be calculated based on a Gibbs energy spline.
                    Any other args provided will result in an error.
                        G           returns Gibbs energy in J/kg
                        rho         returns density in kg/m^3
                        vel         returns sound speed in m/s
                        Cp          returns isobaric specific heat in J/kg/K
                        Cv          returns isochoric specific heat in J/kg/K
                        alpha       returns thermal expansivity in 1/K
                        U           returns internal energy in J/kg
                        H           returns enthalpy in J/kg
                        S           returns entropy in J/kg/K
                        Kt          returns isothermal bulk modulus in MPa
                        Kp          returns pressure derivatives of isothermal bulk modulus (dimensionless)
                        Ks          returns isotropic bulk modulus in MPa
                        V           returns unit volume in m^3/kg
                        -------------------------------------------- below this line, require PTX spline and non-zero M
                        mus         returns solute chemical potential in J/mol
                        muw         returns solvent chemical potential in J/mol
                        Vm          returns partial molar volume in m^3/mol
                        Cpm         returns partial molar heat capacity in J/kg/K/mol
                        Cpa         returns apparent heat capacity J/Kg/K/mol
                        Va          returns apparent volume m^3/mol
    :return:        a named tuple with the requested thermodynamic variables as named properties
                    matching the statevars requested in the *tdvSpec parameter of this function
                    the output will also include P, T, and X (if provided) properties
    """
    dimCt = gibbsSp['number'].size
    origSpec = tdvSpec
    tdvSpec = expandTDVSpec(tdvSpec, dimCt)
    addedTDVs = [s.name for s in tdvSpec if s.name not in origSpec]
    if origSpec and addedTDVs:  # the original spec was not empty and more tdvs were added
        print('NOTE: The requested thermodynamic variables depend on the following variables, which will be '+
              'included as properties of the output object: '+pformat(addedTDVs))
    _checkInputs(gibbsSp, MWu, dimCt, tdvSpec, PTX, failOnExtrapolate)

    tdvout = createThermodynamicStatesObj(dimCt, tdvSpec, PTX)
    derivs = getDerivatives(gibbsSp, tdvout.PTX, dimCt, tdvSpec, verbose)
    gPTX = _getGriddedPTX(tdvSpec, tdvout.PTX, verbose)
    f = _getVolSolventInVolSlnConversion(MWu, tdvout.PTX)

    # calculate thermodynamic variables and store in appropriate fields in tdvout
    completedTDVs = set()     # list of completed tdvs
    while len(completedTDVs) < len(tdvSpec):
        # get tdvs that either have empty reqTDV or all of those tdvs have already been calculated
        # but that are not in completedTDVs themselves (don't recalculate anything)
        tdvsToEval = tuple(t for t in tdvSpec if t.name not in completedTDVs and (not t.reqTDV or not t.reqTDV.difference(completedTDVs)))
        for t in tdvsToEval:
            # build args for the calcFn
            args = dict()
            if t.reqDerivs: args[t.parmderivs] = derivs
            if t.reqGrid:   args[t.parmgrid] = gPTX
            if t.reqMWv:    args[t.parmMWv] = MWv
            if t.reqMWu:    args[t.parmMWu] = MWu
            if t.reqTDV:    args[t.parmtdv] = tdvout
            if t.reqSpline: args[t.parmspline] = gibbsSp
            if t.reqPTX:    args[t.parmptx] = tdvout.PTX    # use the PTX from tdvout, which may have 0 X added
            if t.reqF:      args[t.parmf] = f
            start = time()
            setattr(tdvout, t.name, t.calcFn(**args))       # calculate the value and set it in the output
            end = time()
            if verbose: _printTiming('tdv '+t.name, start, end)
            completedTDVs.add(t.name)

    _remove0X(tdvout, PTX)

    return tdvout


def _checkInputs(gibbsSp, MWu, dimCt, tdvSpec, PTX, failOnExtrapolate):
    """ Checks error conditions before performing any calculations, throwing an error if anything doesn't match up
      - Ensures that a PTX spline includes 0 concentrations
      - Ensures necessary data is available for requested statevars
      - Throws a warning (or error if failOnExtrapolate=True) if spline is being evaluated on values
        outside the range of its knots
      - Warns the user if the requested output would exceed vmWarningFactor times the amount of virtual memory
    """
    knotranges = [(k[0], k[-1]) for k in gibbsSp['knots']]
    # if a PTX spline, make sure the X dimension starts with 0
    if dimCt == 0 and knotranges[iX][0] != 0:
        raise ValueError('The PTX spline does not include 0 concentrations.')
    # make sure that spline has 3 dims if tdvs using concentration or f are requested
    reqX = [t for t in tdvSpec if t.reqX or t.reqF]
    if dimCt == 2 and reqX:
        raise ValueError('You cannot calculate ' + pformat([t.name for t in reqX]) + ' with a spline that does not ' +
                         'include concentration. Remove those statevars and all their dependencies, or supply a ' +
                         'spline that includes concentration.')
    # make sure that MWv is provided if any tdvs that require MWv are requested
    # make sure that MWu is provided if any tdvs that require MWu or f are requested
    reqMWu = [t for t in tdvSpec if t.reqMWu or t.reqF]
    if MWu == 0 and reqMWu:
        raise ValueError('You cannot calculate ' + pformat([t.name for t in reqMWu]) + ' without providing solute ' +
                         'molecular weight.  Remove those statevars and all their dependencies, or provide a ' +
                         'non-zero value for the M parameter.')
    # make sure that all the PTX values fall inside the knot ranges of the spline
    ptxranges = [(d[0],d[-1]) for d in PTX]         # since values are sorted, these are the min/max vals for each dim
    hasValsOutsideKnotRange = lambda kr, dr: dr[0] < kr[0] or dr[1] > kr[1]
    extrapolationDims =  [i for i in range(0,dimCt) if hasValsOutsideKnotRange(knotranges[i],ptxranges[i])]
    if extrapolationDims:
        msg = ' '.join(['Dimensions',pformat(['P' if d==iP else ('T' if d==iT else 'X') for d in extrapolationDims]),
                'contain values that fall outside the knot sequence for the given spline, ',
                'which will result in extrapolation, which may not produce meaningful values.'])
        if failOnExtrapolate:
            raise ValueError(msg)
        else:
            warn(msg)
    # warn the user if the calculation results will take more than some factor times total virtual memory
    outputSize = (len(tdvSpec) + len(PTX)) * np.prod([len(d) for d in PTX]) * floatSizeBytes
    if outputSize > virtual_memory().total * vmWarningFactor:
        warn('The projected output is more than {0} times the total virtual memory for this machine.'.format(vmWarningFactor))
    return


# TODO: add robust set of tests for this
def expandTDVSpec(tdvSpec, dimCt):
    """ add dependencies for requested thermodynamic variables or sets default tdvSpec if none provided
    derivatives are handled separately - see getDerivatives

    :param tdvSpec: an iterable with the names of thermodynamic variables to be evaluated
    :param dimCt:   the number of dimensions - used if setting the default tdvSpec
    :return:        an immutable iterable of TDV namedtuples that includes those specified by tdvSpec
                    and all their dependencies
    """
    # check for completely unsupported statevars so no one has to evaluate twice because of a typo
    if not set(tdvSpec).issubset(statevarnames):
        raise ValueError('One or more unsupported statevars have been requested: ' +
                         pformat([t.name for t in tdvSpec.difference(statevars)]))

    if len(tdvSpec) == 0:
        tdvSpec = list(defTDVSpec2)      # make mutable
        if dimCt == 3:
            tdvSpec.extend(defTDVSpec3)
    else:
        # add thermodynamic variables on which requested ones depend
        tdvSpec = _addTDVDependencies(tdvSpec)

    return tdvSpec


def createThermodynamicStatesObj(dimCt, tdvSpec, PTX):
    flds = {t.name for t in tdvSpec} | {'PTX'}
    # copy PTX so if you add a 0 concentration, you affect only the version in the output var
    # so later you can compare the original PTX to the one in tdvout.PTX to see if you need to remove the 0 X
    TDS = type('ThermodynamicStates', (object,), {f: (np.copy(PTX) if f == 'PTX' else None) for f in flds})
    out = TDS()
    # prepend a 0 concentration if one is needed by any of the quantities being calculated
    if _needs0X(dimCt, PTX, tdvSpec):
        out.PTX[iX] = np.insert(PTX[iX], 0, 0)
    return out


def _needs0X(dimCt, PTX, tdvSpec):
    """ If necessary, add a 0 value to the concentration dimension

    :return:    True if 0 needs to be added
    """
    return dimCt == 3 and PTX[iX][0] != 0 and any([t for t in tdvSpec if t.req0X])


def _remove0X(tdvout, origPTX):
    """ If a 0 concentration was added, take it back out.
    NOTE: This method changes the value of tdvout without returning it.

    :param tdvout:  The object with calculated thermodynamic properties
    :param origPTX: The original input
    """
    if tdvout.PTX.size == 3 and tdvout.PTX[iX].size != origPTX[iX].size:
        tdvout.PTX[iX] = np.delete(tdvout.PTX[iX], 0, 0)
        # go through all calculated values and remove the first item from the X dimension
        # TODO: figure out why PTX doesn't show up in vars
        for p,v in vars(tdvout).items():
            slc = [slice(None)] * len(tdvout.shape)
            slc[iX] = slice(1,None)
            setattr(tdvout, p, v[tuple(slc)])


def _createGibbsDerivativesClass(tdvSpec):
    flds = {d for t in tdvSpec if t.reqDerivs for d in t.reqDerivs}
    return type('GibbsDerivatives', (object,), {d: None for t in tdvSpec if t.reqDerivs for d in t.reqDerivs})


def _buildDerivDirective(derivSpec, dimCt):
    """ Gets a list of the derivatives for relevant dimensions
    """
    out = [defDer] * dimCt
    if derivSpec.wrtP: out[iP] = derivSpec.wrtP
    if derivSpec.wrtT: out[iT] = derivSpec.wrtT
    if derivSpec.wrtX: out[iX] = derivSpec.wrtX
    return out


def getDerivatives(gibbsSp, PTX, dimCt, tdvSpec, verbose=False):
    """

    :param gibbsSp:     The Gibbs energy spline
    :param PTX:         The pressure, temperature[, molality] points at which the spine should be evaluated
    :param dimCt:       2 if a PT spline, 3 if a PTX spline
    :param tdvSpec:     The expanded TDVSpec describing the thermodynamic variables to be calculated
    :param verbose:     True to output timings
    :return:            Gibbs energy derivatives necessary to calculate the tdvs listed in the tdvSpec
    """
    GibbsDerivs = _createGibbsDerivativesClass(tdvSpec)
    out = GibbsDerivs()
    reqderivs = {d for t in tdvSpec for d in t.reqDerivs}   # get set of derivative names that are needed
    getDerivSpec = lambda dn: next(d for d in derivatives if d.name == dn)
    for rd in reqderivs:
        derivDirective = _buildDerivDirective(getDerivSpec(rd), dimCt)
        start = time()
        setattr(out, rd, evalMultivarSpline(gibbsSp, PTX, derivDirective))
        end = time()
        if verbose: _printTiming('deriv '+rd, start, end)
    return out


def _getGriddedPTX(tdvSpec, PTX, verbose=False):
    if any([t.reqGrid for t in tdvSpec]):
        start = time()
        out = np.meshgrid(*PTX.tolist(), indexing='ij')    # grid the dimensions of PTX
        end = time()
        if verbose: _printTiming('grid', start, end)
    else:
        out = []
    return out


def _getVolSolventInVolSlnConversion(MWu, PTX):
    """ Dimensionless conversion factor for how much of the volume of 1 kg of solution is really just the solvent
    :return:    f
    """
    return 1 + MWu * PTX[iX]


def evalGibbs(gibbsSp, PTX):
    """
    :return: G
    """
    return evalMultivarSpline(gibbsSp, PTX)


def evalIsobaricSpecificHeat(gPTX, derivs):
    """
    :return:        Cp
    """
    return -1 * derivs.d2T * gPTX[iT]


def evalIsochoricSpecificHeat(tdv, gPTX, derivs):
    """

    :return:    Cv
    """
    #Cv= Cp + Tm.*dPT.^2./d2P;
    return tdv.Cp + gPTX[iT] * np.power(derivs.dPT, 2) / derivs.d2P


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


def evalIsotropicBulkModulus(tdv):
    """
    :return:        Ks
    """
    return tdv.rho * np.power(tdv.vel, 2) / 1e6 #  1e6 for MPa conversion


def evalThermalExpansivity(derivs, tdv):
    """
    :return:        alpha
    """
    return 1e-6 * derivs.dPT * tdv.rho  #  1e-6 for MPa conversion


def evalInternalEnergy(gPTX, tdv):
    """
    :param gPTX:    gridded dimensions over which spline is being evaluated
    :return:        U
    """
    return tdv.G - 1e6 * gPTX[iP] / tdv.rho + gPTX[iT] * tdv.S  #  1e6 for MPa conversion


def evalSoluteChemicalPotential(MWu, f, derivs, tdv):
    """
    :return:        mus
    """
    return MWu * tdv.G + f * derivs.d1X


def evalSolventChemicalPotential(MWv, f, gPTX, derivs, tdv):
    """
    :return:        muw
    """
    return (tdv.G * MWv) - (MWv * f * gPTX[iX] * derivs.d1X)


def evalEnthalpy(gPTX, tdv):
    """
    :return:        H
    """
    return tdv.U - gPTX[iT] * tdv.S


def evalPartialMolarVolume(MWu, f, derivs):
    """ Slope at a point of the V v. X graph
    :return:        Vm
    """
    return (MWu * derivs.d1P) + (f * derivs.dPX)


def evalPartialMolarHeatCapacity(MWu, f, tdv, derivs, gPTX):
    """
    :return:    Cpm
    """
    return MWu * tdv.Cp - f * derivs.d2T1X * gPTX[iT]


def evalVolume(tdv):
    """
    :return:    V
    """
    return np.power(tdv.rho, -1)


def evalApparentSpecificHeat(f, gPTX, tdv):
    """
    :return:    Cpa
    """
    return (tdv.Cp * f - _get0XTdv(tdv, 'Cp')) / _getDividableBy(gPTX[iX])


def evalApparentVolume(f, gPTX, tdv):
    """ slope of a chord between pure solvent and a concentration on a V v. X graph
    :return:    Va
    """
    return 1e6 * (tdv.V * f - _get0XTdv(tdv, 'V')) / _getDividableBy(gPTX[iX]) # 1e6 for MPa conversion


def _get0XTdv(tdv, prop):
    slc = [slice(None)] * (len(tdv.PTX))
    slc[iX] = slice(0,1)
    return getattr(tdv, prop)[tuple(slc)]


def _getDividableBy(inp):
    eps = np.finfo(inp.dtype).eps
    return np.where(inp != 0, inp, eps)


def _getTDVSpec(name, calcFn, reqX=False, reqMWv=False, parmMWv='MWv', reqMWu=False, parmMWu='MWu',
                reqGrid=False, parmgrid='gPTX', reqF=False, parmf='f',
                reqDerivs=[], parmderivs='derivs', reqTDV=[], parmtdv='tdv', reqSpline=False,
                parmspline='gibbsSp', reqPTX=False, parmptx='PTX', req0X=False):
    """ Builds a TDVSpec namedtuple indicating what is required to calculate this particular thermodynamic variable
    :param name:        the name / symbol of the tdv (e.g., G, rho, alpha, muw)
    :param calcFn:      the name of the function used to calculate the tdv
    :param reqX:        True if DIRECTLY calculating the tdv requires concentration.  False otherwise
    :param reqMWv:      True if DIRECTLY calculating the tdv requires solvent molecular weight.  False otherwise
    :param parmMWv:     the name of the parameter of calcFn used to pass in solvent molecular weight if reqMWv
    :param reqMWu:      True if DIRECTLY calculating the tdv requires solute molecular weight.  False otherwise
    :param parmMWu:     the name of the parameter of calcFn used to pass in solute molecular weight if reqMWu
    :param reqGrid:     True if DIRECTLY calculating the tdv requires you to grid the PT[X] values.  False otherwise
    :param parmgrid:    the name of the parameter of calcFn used to pass in the gridded input dimensions if reqGrid
    :param reqF:        True if DIRECTLY calculating the tdv requires the conversion factor that gives the volume
                        of solvent in a unit volume of solution
    :param parmf:       the name of the parameter of calcFn used to pass in the vol solvent to vol solution conversion
                        factor
    :param reqDerivs:   A list of derivatives needed to DIRECTLY calculate the tdv
                        (e.g. for tdv Kp, this would be ['d1P','dP','d3P']
                        - see fn evalIsothermalBulkModulusWrtPressure)
                        See getDerivatives for a full list of derivatives that can be calculated
    :param parmderivs:  the name of the parameter of calcFn used to pass in the pre-calculated derivatives if reqDerivs
    :param reqTDV:      A list of other thermodynamic variables needed to DIRECTLY calculate the tdv
                        (e.g. for tdv U, this would be ['G','rho'] - see fn evalInternalEnergy)
                        Note: recursive dependencies are not supported
    :param parmtdv:     the name of the parameter of calcFn used to pass in the tdvout if reqTDV
    :param reqSpline:   If True, calcFn needs the spline definition
    :param parmspline:  the name of the parameter of calcFn used to pass in the spline def if reqSpline
    :param reqPTX:      If True, calcFn needs the original dimension input (parm PTX in evalSolutionGibbs) to run
    :param parmptx:     the name of the parameter of calcFn used to pass in the original input if reqPTX
    :param req0X:       if True, calcFn needs the 0 concentration for calculating apparent values
    :return:            a namedtuple giving the spec for the tdv
    """
    reqTDV = set(reqTDV); reqDerivs = set(reqDerivs)    # make these sets to enforce uniqueness and immutability
    if name in reqTDV:
        raise ValueError('Recursive dependencies are not supported.  Amend the ' + name + ' TDVSpec accordingly')
    if reqDerivs.difference(derivativenames):
        raise ValueError('One or more derivatives are not supported. Amend the ' + name + ' TDVSpec accordingly.' +
                         'Supported derivatives are '+pformat(derivativenames))
    arginfo = getargvalues(currentframe())
    # build properties of the TDVSpec dynamically
    flds = arginfo.args
    if not reqMWv:      flds.remove('parmMWv')
    if not reqMWu:      flds.remove('parmMWu')
    if not reqGrid:     flds.remove('parmgrid')
    if not reqDerivs:   flds.remove('parmderivs')
    if not reqTDV:      flds.remove('parmtdv')
    if not reqSpline:   flds.remove('parmspline')
    if not reqPTX:      flds.remove('parmptx')
    vals = {f: v for f, v in arginfo.locals.items() if f in flds}

    tdvspec = namedtuple('TDVSpec', flds)
    return tdvspec(**vals)


def getSupportedThermodynamicVariables():
    """ When adding new tdvs, you don't need to worry about adding them in any particular order -
        dependencies are handled elsewhere
        See also the comments for _getTDVSpec and evalSolutionGibbs

    :return: immutable iterable with the the full set of specs for thermodynamic variables supported by this module
    """
    out = tuple([
        _getTDVSpec('G', evalGibbs, reqSpline=True, reqPTX=True),
        _getTDVSpec('rho', evalDensity, reqDerivs=['d1P']),
        _getTDVSpec('vel', evalSoundSpeed, reqDerivs=['d1P', 'dPT', 'd2T', 'd2P']),
        _getTDVSpec('Cp', evalIsobaricSpecificHeat, reqGrid=True, reqDerivs=['d2T']),
        _getTDVSpec('Cv', evalIsochoricSpecificHeat, reqGrid=True, reqDerivs=['dPT', 'd2P'], reqTDV=['Cp']),
        _getTDVSpec('alpha', evalThermalExpansivity, reqDerivs=['dPT'], reqTDV=['rho']),
        _getTDVSpec('U', evalInternalEnergy, reqGrid=True, reqTDV=['G', 'rho', 'S']),
        _getTDVSpec('H', evalEnthalpy, reqGrid=True, reqTDV=['U', 'S']),
        _getTDVSpec('S', evalEntropy, reqDerivs=['d1T']),
        _getTDVSpec('Kt', evalIsothermalBulkModulus, reqDerivs=['d1P', 'd2P']),
        _getTDVSpec('Kp', evalIsothermalBulkModulusWrtPressure, reqDerivs=['d1P', 'd2P', 'd3P']),
        _getTDVSpec('Ks', evalIsotropicBulkModulus, reqTDV=['rho', 'vel']),
        _getTDVSpec('mus', evalSoluteChemicalPotential, reqMWu=True, reqF=True, reqDerivs=['d1X'], reqTDV=['G']),
        _getTDVSpec('muw', evalSolventChemicalPotential, reqX=True, reqMWv=True, reqGrid=True, reqF=True,
                    reqDerivs=['d1X'], reqTDV=['G']),
        _getTDVSpec('Vm', evalPartialMolarVolume, reqMWu=True, reqF=True, reqDerivs=['d1P', 'dPX']),
        _getTDVSpec('Cpm', evalPartialMolarHeatCapacity, reqMWu=True, reqGrid=True, reqF=True, reqDerivs=['d2T1X'],
                    reqTDV=['Cp']),
        _getTDVSpec('V', evalVolume, reqTDV=['rho']),
        _getTDVSpec('Cpa', evalApparentSpecificHeat, reqX=True, reqGrid=True, reqF=True, reqTDV=['Cp'], req0X=True),
        _getTDVSpec('Va', evalApparentVolume, reqX=True, reqGrid=True, reqF=True, reqTDV=['V'], req0X=True)
            ])
    # check that all reqTDVs are represented in the list
    outnames = frozenset([t.name for t in out])
    unrecognizedDependencies = [(tdv.name, dep) for tdv in out for dep in tdv.reqTDV if dep not in outnames]
    if unrecognizedDependencies:
        raise ValueError('One or more statevars depend on an unrecognized TDV: '+pformat(unrecognizedDependencies))
    return out, outnames


def getSupportedDerivatives():
    """ Define a name for a derivative of Gibbs energy
    and the derivative order with respect to P, T, and X required to calculate it
    (if not provided, all default to defDer)
    These can then be used in a TDVSpec

    :return:    an immutable iterable containing a list of supported derivatives
    """
    return tuple([
        derivSpec('d1P', wrtP=1, wrtT=defDer, wrtX=defDer),
        derivSpec('d1T', wrtP=defDer, wrtT=1, wrtX=defDer),
        derivSpec('d1X', wrtP=defDer, wrtT=defDer, wrtX=1),
        derivSpec('dPT', wrtP=1, wrtT=1, wrtX=defDer),
        derivSpec('dPX', wrtP=1, wrtT=defDer, wrtX=1),
        derivSpec('d2P', wrtP=2, wrtT=defDer, wrtX=defDer),
        derivSpec('d2T', wrtP=defDer, wrtT=2, wrtX=defDer),
        derivSpec('d2T1X', wrtP=defDer, wrtT=2, wrtX=1),
        derivSpec('d3P', wrtP=3,wrtT=defDer, wrtX=defDer)
    ])


def _addTDVDependencies(origTDVs):
    """ recursively adds thermodynamic variables on which origTDVs are dependent
    This does not handle derivative dependencies - see getDerivatives

    :param origTDVs:    an iterable of TDVs to check for dependencies
                        elements can be either the string names of statevars or the TDV objects
    :return:            immutable iterable of TDV objects including origTDVs and any TDVs on which they are
                        directly or indirectly dependent
    """
    otdvnames = set(origTDVs) if isinstance(next(iter(origTDVs)), str) else {m.name for m in origTDVs}
    while True:
        # get TDVs dependent on TDVs in otdvnames
        reqot = {t for m in statevars if m.name in otdvnames for t in m.reqTDV}
        # if every tdv in reqot is already in otdvnames, stop
        if reqot.issubset(otdvnames):
            break
        else:
            # add the new list of tdvs and repeat the loop to add more nested dependencies
            otdvnames = otdvnames.union(reqot)
    return tuple([m for m in statevars if m.name in otdvnames])


def _setDefaultTDVSpecs():
    # concentration required if the flag says it does or if it uses a concentration derivative
    xreq = [m for m in statevars if m.reqX or [d for d in m.reqDerivs if 'X' in d]]
    xreq = [r.name for r in _addTDVDependencies(xreq)]
    # return values should be immutable - use tuples as TDVSpec namedtuples are not hashable so set is not usable
    return tuple([m for m in statevars if m.name not in xreq]), tuple([m for m in statevars if m.name in xreq])


def _printTiming(calcdesc, start, end):
    endDT = datetime.fromtimestamp(end)
    print(endDT.strftime('%H:%M:%S.%f'), ':\t', calcdesc,'took',str(end-start),'seconds to calculate')



#########################################
## Constants
#########################################
iP = 0; iT = 1; iX = 2      # dimension indices
defDer = 0                  # default derivative
vmWarningFactor = 2         # warn the user when size of output would exceed vmWarningFactor times total virtual memory
floatSizeBytes = int(np.finfo(float).bits / 8)

derivSpec = namedtuple('DerivativeSpec', ['name', 'wrtP', 'wrtT', 'wrtX'])
derivatives = getSupportedDerivatives()
derivativenames = {d.name for d in derivatives}

statevars, statevarnames = getSupportedThermodynamicVariables()
defTDVSpec2, defTDVSpec3 = _setDefaultTDVSpecs()
















