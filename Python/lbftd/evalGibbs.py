from datetime import datetime
from pprint import pformat
from time import time
from warnings import warn

import numpy as np
from psutil import virtual_memory

from lbftd import statevars
from lbftd.statevars import iT, iP, iX
from mlbspline.eval import evalMultivarSpline


def evalSolutionGibbs(gibbsSp, PTX, *tdvSpec, MWv=18.01528e-3, MWu=None, failOnExtrapolate=True, verbose=False):
    # TODO: add logging? possibly in lieu of verbose
    """ Calculates thermodynamic variables for solutions based on a spline giving Gibbs energy
    This currently only supports single-solute solutions.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Warning: units must be as specified here because some conversions are hardcoded into this function.
        With the exception of pressure, units are SI.  Pressure is in MPa rather than Pa.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    :param gibbsSp: A B-spline  (in format given by loadSpline.loadMatSpline) for giving Gibbs energy (J/mg)
                    with dimensions pressure (MPa), temperature (K), and (optionally) molality (mol/kg), IN THAT ORDER.
                    If molality is not provided, this function assumes that it is calculating
                    thermodynamic properties for a pure substance.
    :param PTX:     a Numpy ndarray of ndarrays with the points at which gibbsSp should be evaluated
                    the number of dimensions must be same as in the spline (PTX.size == gibbsSp['number'].size)
                    each of the inner ndarrays represents one of pressure (P), temperature (T), or molality (X)
                    and must be in the same order and units described in the notes for the gibbsSp parameter
                    Additionally, each dimension must be sorted from low to high values.
    :param MWv:     float with molecular weight of solvent (kg/mol).
                    Defaults to molecular weight of water (7 sig figs)
    :param MWu:     float with molecular weight of solute (kg/mol).
    :param failOnExtrapolate:   True if you want an error to appear if PTX includes values that fall outside the knot
                    sequence of gibbsSp.  If False, throws a warning rather than an error, and
                    proceeds with the calculation.
    :param verbose: boolean indicating whether to print status updates, warnings, etc.
    :param tdvSpec: iterable indicating the thermodynamic variables to be calculated
                    elements can be either strings showing the names (full list at statevars.statevarnames)
                    or TDV objects from statevars.statevars.
                    If not provided, this function will calculate the variables in statevars.tdvsPTOnly for a PT spline,
                    and those in statevars.statevars for a PTX spline.
                    Any other args provided will result in an error.
                    See the README for units.
    :return:        a named tuple with the requested thermodynamic variables as named properties
                    matching the statevars requested in the *tdvSpec parameter of this function
                    the output will also include P, T, and X (if provided) properties
    """
    dimCt = gibbsSp['number'].size
    # expand spec to add dependencies (or set to default spec if no spec given)
    origSpec = tdvSpec
    tdvSpec = statevars.expandTDVSpec(tdvSpec, dimCt)
    addedTDVs = [s.name for s in tdvSpec if s.name not in origSpec]
    if origSpec and addedTDVs:  # the original spec was not empty and more tdvs were added
        print('NOTE: The requested thermodynamic variables depend on the following variables, which will be '+
              'included as properties of the output object: '+pformat(addedTDVs))
    _checkInputs(gibbsSp, dimCt, tdvSpec, PTX, MWv, MWu, failOnExtrapolate)

    tdvout = createThermodynamicStatesObj(dimCt, tdvSpec, PTX)
    derivs = getDerivatives(gibbsSp, tdvout.PTX, dimCt, tdvSpec, verbose)
    gPTX = _getGriddedPTX(tdvSpec, tdvout.PTX, verbose) if any([tdv.reqGrid for tdv in tdvSpec]) else None
    f = _getVolSolventInVolSlnConversion(MWu, tdvout.PTX) if any([tdv.reqF for tdv in tdvSpec]) else None

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


def _checkInputs(gibbsSp, dimCt, tdvSpec, PTX, MWv, MWu, failOnExtrapolate):
    """ Checks error conditions before performing any calculations, throwing an error if anything doesn't match up
      - Ensures necessary data is available for requested statevars (check req parameters for statevars)
      - Throws a warning (or error if failOnExtrapolate=True) if spline is being evaluated on values
        outside the range of its knots
      - Warns the user if the requested output would exceed vmWarningFactor times the amount of virtual memory

      Note that the mlbspline.eval module performs additional checks (like dimension count mismatches
    """
    knotranges = [(k[0], k[-1]) for k in gibbsSp['knots']]
    # if a PTX spline, make sure the spline's concentration dimension starts with 0 if any tdv has req0X=True
    # Note that the evalGibbs functions elsewhere handle the case where PTX does not include 0 concentration.
    req0X = [t for t in tdvSpec if t.req0X]
    if dimCt == 3 and req0X and knotranges[iX][0] != 0:
            raise ValueError('You cannot calculate ' + pformat([t.name for t in req0X]) + ' with a spline that does ' +
                             'not include 0 concentration. Remove those statevars and all their dependencies, or ' +
                             'supply a spline that includes 0 concentration.')
    # make sure that spline has 3 dims if tdvs using concentration or f are requested
    reqF = [t for t in tdvSpec if t.reqF]
    reqX = [t for t in tdvSpec if t.reqX]
    if dimCt == 2 and (reqX or reqF):
        raise ValueError('You cannot calculate ' + pformat(set([t.name for t in (reqF + reqX)])) + ' with a spline ' +
                         'that does not include concentration. Remove those statevars and all their dependencies, ' +
                         'or supply a spline that includes concentration.')
    # make sure that solvent molecular weight is provided if any tdvs that require MWv are requested
    reqMWv = [t for t in tdvSpec if t.reqMWv]
    if (MWv == 0 or not MWv) and reqMWv:
        raise ValueError('You cannot calculate ' + pformat([t.name for t in reqMWv]) + ' without ' +
                         'providing solvent molecular weight.  Remove those statevars and all their dependencies, or ' +
                         'provide a valid value for the MWv parameter.')
    # make sure that solute molecular weight is provided if any tdvs that require MWu or f are requested
    reqMWu = [t for t in tdvSpec if t.reqMWu]
    if (MWu == 0 or not MWu) and (reqMWu or reqF):
        raise ValueError('You cannot calculate ' + pformat([t.name for t in set(reqMWu + reqF)]) + ' without ' +
                         'providing solute molecular weight.  Remove those statevars and all their dependencies, or ' +
                         'provide a valid value for the MWu parameter.')
    # make sure that all the PTX values fall inside the knot ranges of the spline
    ptxranges = [(d[0], d[-1]) for d in PTX]         # since values are sorted, these are the min/max vals for each dim
    hasValsOutsideKnotRange = lambda kr, dr: dr[0] < kr[0] or dr[1] > kr[1]
    extrapolationDims = [i for i in range(0, dimCt) if hasValsOutsideKnotRange(knotranges[i], ptxranges[i])]
    if extrapolationDims:
        msg = ' '.join(['Dimensions',pformat({'P' if d == iP else ('T' if d == iT else 'X') for d in extrapolationDims}),
                'contain values that fall outside the knot sequence for the given spline,',
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
    return dimCt == 3 and PTX[iX][0] != 0 and any([t.req0X for t in tdvSpec])


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
            slc[iX] = slice(1, None)
            setattr(tdvout, p, v[tuple(slc)])


def _createGibbsDerivativesClass(tdvSpec):
    flds = {d for t in tdvSpec if t.reqDerivs for d in t.reqDerivs}
    return type('GibbsDerivatives', (object,), {d: None for t in tdvSpec if t.reqDerivs for d in t.reqDerivs})


def _buildDerivDirective(derivSpec, dimCt):
    """ Gets a list of the derivatives for relevant dimensions
    """
    out = [statevars.defDer] * dimCt
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
    getDerivSpec = lambda dn: next(d for d in statevars.derivatives if d.name == dn)
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


def _printTiming(calcdesc, start, end):
    endDT = datetime.fromtimestamp(end)
    print(endDT.strftime('%H:%M:%S.%f'), ':\t', calcdesc,'took',str(end-start),'seconds to calculate')


#########################################
## Constants
#########################################
vmWarningFactor = 2         # warn the user when size of output would exceed vmWarningFactor times total virtual memory
floatSizeBytes = int(np.finfo(float).bits / 8)
















