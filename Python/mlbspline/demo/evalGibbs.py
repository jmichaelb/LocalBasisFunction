from collections import namedtuple

import numpy as np

from mlbspline import *

# moles/kg for pure water
nw=1000/18.0152
# dimension indices
iP=0;
iT=1;
im=2;
# default derivative
defDer = 0;


def evalSolutionGibbs(gibbsSp, x, M=0, rG=True, rrho=True, rvel=True, rCp=True, ralpha=True, rU=True, rH=True, rS=True, rKt=True, rKp=True, rKs=True, rmus=True, rmuw=True):
    # TODO: convert all the bools to **kwargs
    """ Calculates thermodynamic quantities for solutions based on a spline giving Gibbs energy
    This only supports for single-solute solutions.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !! WARNING: you must provide x values in units that are commensurate with the units of gibbsSp.
    !! Typically, gibbsSp should be give Gibbs energy in units of J/mg --
    !!  then M should be in kg/mol,
    !!      x as follows: P in MPa, T in K, and (if used) m in molality
    !!  and output values will be:
    !!      G in J/mg (obviously), rho in kg/m^3, vel in m/s, Cp in J/kg/K, alpha in 1/K, mu in dG/dm
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    :param gibbsSp: A B-spline for giving Gibbs energy (in format given by loadSpline.loadMatSpline)
                    with dimensions pressure, temperature, and (optionally) molality, in that order.
                    If molality is not provided, this function assumes that it is calculating
                    thermodynamic properties for pure water.
    :param x:       a Numpy n-D array with the points at which gibbsSp should be evaluated
                    the number of dimensions must be same as in the spline (x.size == gibbsSp['number'].size)
                    and the dimensions must be in the same order as in gibbsSp
    :param M:       float with molecular weight of solute
    :param rG:      boolean indicating whether to return Gibbs energy values
    :param rrho:    boolean indicating whether to return density
    :param rvel:    boolean indicating whether to return sound speed
    :param rCp:     boolean indicating whether to return isobaric specific heat
    :param ralpha:  boolean indicating whether to return thermal expansivity
    :param rU:      boolean indicating whether to return internal energy
    :param rH:      boolean indicating whether to return enthalpy
    :param rS:      boolean indicating whether to return entropy
    :param rKt:     boolean indicating whether to return isothermal bulk modulus
    :param rKp:     boolean indicating whether to return bulk modulus pressure derivative
    :param rKs:     boolean indicating whether to return isotropic bulk modulus
    :param rmus:    boolean indicating whether to return solute chemical potential
    :param rmuw:    boolean indicating whether to return water chemical potential
    :return:        a named tuple with the requested thermodynamic values
                    as named properties matching the parameter names of this function
                    the x value is also included as tuple for reference
    """
    dimCt = spd['number'].size

    # check that M is provided if a TD quantity that uses it is calculated

    # generate derivative values as required by requested thermodynamic quantities
    if(rG):
        G = eval.evalMultivarSpline(gibbsSp,x)
    if(rrho or rvel or rKt or rKp):
        d1P = eval.evalMultivarSpline(gibbsSp,x,[1 if i == iP else defDer for i in range(0,dimCt)])
    if(rS):
        d1T = eval.evalMultivarSpline(gibbsSp,x,[1 if i == iT else defDer for i in range(0,dimCt)])
    if(rCp):
        d2T = eval.evalMultivarSpline(gibbsSp,x,[2 if i == iT else defDer for i in range(0,dimCt)])
    if(rCp):
        xm = np.meshgrid(*x.tolist(), indexing='ij') # grid the dimensions of x

    # generate themodynamic quantities
    rho = 1e6 * np.power(d1P, -1)   # 1e6 for MPa to Pa
    S = -1 * d1T
    Cp = -1 * d2T * xm[iT];





