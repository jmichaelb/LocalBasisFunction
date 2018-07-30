from collections import namedtuple
from inspect import signature

import numpy as np

from mlbspline import *

# moles/kg for pure water
nw=1000/18.0152
# dimension indices
iP = 0; iT = 1; iX = 2
# default derivative
defDer = 0


def evalSolutionGibbs(gibbsSp, x, M=0, rG=True, rrho=True, rvel=True, rCp=True, ralpha=True, rU=True, rH=True, rS=True,
                      rKt=True, rKp=True, rKs=True, rmus=True, rmuw=True, rVm=True, rCpm=True):
    # TODO: convert all the bools to **kwargs
    # TODO: check/document units for all measures
    # TODO: make fn flexible enough to handle gibbsSp in different units (low priority)
    """ Calculates thermodynamic quantities for solutions based on a spline giving Gibbs energy
    This only supports for single-solute solutions.
    Warning: units must be as specified here because some conversions are hardcoded into this function.

    :param gibbsSp: A B-spline  (in format given by loadSpline.loadMatSpline) for giving Gibbs energy (J/mg)
                    with dimensions pressure (MPa), temperature (K), and (optionally) molality, IN THAT ORDER.
                    If molality is not provided, this function assumes that it is calculating
                    thermodynamic properties for pure water.
    :param x:       a Numpy ndarray of ndarrays with the points at which gibbsSp should be evaluated
                    the number and index of dimensions must be same as in the spline (x.size == gibbsSp['number'].size)
                    and each dimension must be sorted (low to high)
    :param M:       float with molecular weight of solute (kg/mol)
    :param rG:      boolean indicating whether to return Gibbs energy values
    :param rrho:    boolean indicating whether to return density (kg/m^3)
    :param rvel:    boolean indicating whether to return sound speed (m/s)
    :param rCp:     boolean indicating whether to return isobaric specific heat (J/kg/K)
    :param ralpha:  boolean indicating whether to return thermal expansivity (1/K)
    :param rU:      boolean indicating whether to return internal energy
    :param rH:      boolean indicating whether to return enthalpy
    :param rS:      boolean indicating whether to return entropy
    :param rKt:     boolean indicating whether to return isothermal bulk modulus
    :param rKp:     boolean indicating whether to return bulk modulus pressure derivative
    :param rKs:     boolean indicating whether to return isotropic bulk modulus
    :param rmus:    boolean indicating whether to return solute chemical potential (dG/dm)
    :param rmuw:    boolean indicating whether to return water chemical potential
    :param rVm:     boolean indicating whether to return ????
    :param rCpm:    boolean indicating whether to return ????
    :param rCpa:    boolean indicating whether to return ????
    :param rVa:     boolean indicating whether to return ????
    :return:        a named tuple with the requested thermodynamic values
                    as named properties matching the parameter names of this function
                    the x value is also included as tuple for reference
    """
    dimCt = gibbsSp['number'].size

    # TODO: issue warning if dim values fall (too far?) outside the knot sequence of gibbsSp?

    # make sure that spline has 3 dims if quantities using concentration are requested
    if dimCt == 2 and (rmus or rmuw or rVm or rCpm or rCpa or rVa):
        raise ValueError("You cannot generate mus, muw, or Vm with a spline that does not include concentration.")

    # check that M is provided if a thermodynamic quantity that uses it is calculated
    if M == 0 and (rmus or rmuw or rVm or rCpm or rCpa or rVa):
        raise ValueError("Molecular weight (M) must be provided for mus, muw, or Vm to be generated - either turn off the flag or provide the weight.")

    # prep for calculating apparent values
    if dimCt == 3:
        origXCt = length(x[iX])
        x[iX] = adjustConcentrations(x[iX])

    # reset flags for thermodynamic quantities that depend on other ones
    # first look at the ternary quantities
    if rH:      rU = True; rS = True
    # then look at secondary ones
    if rKs:     rrho = True; rvel = True
    if ralpha:  rrho = True
    if rU:      rG = True;  rrho = True; rS = True
    if rmus:    rG = True
    if rmuw:    rG = True
    if rCpm:    rCp = True
    if rCpa:    rCp = True

    if rmus or rmuw or vVm:
        f = 1 + M * x[iX]

    # TODO: fix this with along with kwargs - there has GOT to be a better way than this
    # generate list of quantities to generate based on r parameters
    tdq = [k[1:] for k, v in locals().items()
           if (k[0] == 'r' and v and k in signature(evalSolutionGibbs).parameters.keys())]
    # create the output object and load it with the PT[X] values (use the original values for concentration)
    out = createThermodynamicQuantitiesObj(dimCt, tdq, np.array([x[i] if i != iX else OX for i in range(0, dimCt)]))

    # generate derivative values as required by requested thermodynamic quantities
    if rG:
        G = eval.evalMultivarSpline(gibbsSp,x)
    if rrho or rvel or rKt or rKp or rVm:
        d1P = eval.evalMultivarSpline(gibbsSp, x, [1 if i == iP else defDer for i in range(0, dimCt)])
    if rS:
        d1T = eval.evalMultivarSpline(gibbsSp, x, [1 if i == iT else defDer for i in range(0, dimCt)])
    if rmus:
        d1X = eval.evalMultivarSpline(gibbsSp, x, [1 if i == iX else defDer for i in range(0, dimCt)])
    if rvel or ralpha:
        dPT = eval.evalMultivarSpline(gibbsSp, x, [1 if (i == iP or i == iT) else defDer for i in range(0, dimCt)])
    if rVm:
        dPX = eval.evalMultivarSpline(gibbsSp, x, [1 if (i == iP or i == iX) else defDer for i in range(0, dimCt)])
    if rvel or rKt or rKp:
        d2P = eval.evalMultivarSpline(gibbsSp, x, [2 if i == iP else defDer for i in range(0, dimCt)])
    if rCp or rvel:
        d2T = eval.evalMultivarSpline(gibbsSp, x, [2 if i == iT else defDer for i in range(0, dimCt)])
    if rCpm:
        d2T1X = eval.eval.evalMultivarSpline(gibbsSp, x,
                                             [2 if i == iT else (1 if i == iX else defDer) for i in range(0, dimCt)])
    if rKp:
        d3P = eval.evalMultivarSpline(gibbsSp, x, [3 if i == iP else defDer for i in range(0, dimCt)])
    if rCp or rU:
        xm = np.meshgrid(*x.tolist(), indexing='ij')    # grid the dimensions of x

    # generate thermodynamic quantities
    if 'G' in tdq:
        out.G = G
    if 'Cp' in tdq:
        out.Cp = -1 * d2T * xm[iT]
    if 'S' in tdq:
        out.S = -1 * d1T
    if 'vel' in tdq:
        out.vel = np.real(np.sqrt(np.power(d1P,2) / (np.power(dPT,2) / d2T - d2P))) # MPa-Pa units conversion cancels
    if 'rho' in tdq:
        out.rho = 1e6 * np.power(d1P, -1)   # 1e6 for MPa to Pa
    if 'Kt' in tdq:
        out.Kt = -1 * d1P / d2P
    if 'Kp' in tdq:
        out.Kp = d1P * np.power(d2P,-2) * d3P - 1
    if 'Vm' in tdq:
        out.Vm = (M * d1P) + (f * dPX)

    # these need to be done secondarily because they rely on previously generated quantities
    if 'Ks' in tdq:
        out.Ks = out.rho * np.power(out.vel,2) / 1e6
    if 'alpha' in tdq:
        out.alpha = 1e-6 * dPT * out.rho  #  1e6 for MPa to Pa
    if 'U' in tdq:
        out.U = out.G - 1e6 * xm[iP] / out.rho + xm[iT] * out.S
    if 'mus' in tdq:
        out.mus = M * out.G + f * d1X # mus=M*G + f.*dGdm
    if 'muw' in tdq:
        out.muw = (out.G / nw) - (1 / nw * f * x[iX] * d1X)
    if 'Cpm' in tdq:
        out.Cpm = M * out.Cp - f * d2T1X * x[iT]

    # these depend on secondary quantities
    if 'H' in tdq:
        out.H = out.U - xm[iT] * out.S

    return out


def createThermodynamicQuantitiesObj(dimCt, tdq, x):
    svn = ['P', 'T']
    if (dimCt == 3):
        svn = svn + ['X']
    out = namedtuple('ThermodynamicQuantities', tdq + svn)
    # include input in the output so you always know the conditions for the thermodynamic quantities
    for i in range(0,dimCt):
        if i == iP:
            out.P = x[i]
        if i == iT:
            out.T = x[i]
        if i == iX:
            out.X = x[i]
    return out


def adjustConcentrations(X):
    eps = np.finfo(type(X[0])).eps  # get the lowest positive value that can be distinguished from 0
    # prepend the list of concentrations with a zero if there isn't already one there
    out = X if X[0] == 0 else np.concatenate((np.array([0]), X))
    out[out == 0] = eps  # add eps to zero concentrations to avoid divide by zero errors
    return out







