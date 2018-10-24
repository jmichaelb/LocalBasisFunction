from mlbspline import load

def loadGibbsSpline(splineFile, splineVar=None):
    """Loads a spline from .mat format file
    A Gibbs energy spline must contain the following:
      - sp is the main spline itself.  It must be a 2D or 3D spline as outlined in mlbspline.eval
      - Go is a Gibbs spline in T only (assuming pure solvent and 1 bar).
            only required if calculating apparent thermodynamic variables
      - nu is the number of ions in solution.  All values must be positive integers.
            For multi-solute solutions, the value for each species must be in teh same order as in MW
            such that nu[i] corresponds to MW[i + 1]
            Required only if calculating
      - MW is the molecular weight of each species in the solution in kg/mol,
            such that MW[0] is the molecular weight of the solvent
            and MW[1] is the molecular weight of solute A (corresponding to nu[0]),
                MW[2] is the molecular weight of solute B (corresponding to nu[1]),
                etc.

    :param splineFile:  full or relative path to Matlab file
    :param splineVar:   variable to load from splineFile.
                        If not provided, the splineFile must contain exactly one variable
    :return:            a dict with the spline representation
    """
    raw = load._stripNestingToFields(load._getRaw(splineFile, splineVar))
    sp = load.getSplineDict(load._stripNestingToValue(raw['sp']))
    load.validateSpline(sp)
    Go = load.getSplineDict(load._stripNestingToValue(raw['Go']))
    load.validateSpline(Go)
    spd = {
        'sp':   sp,
        'Go':   Go,
        'nu':   load._stripNestingToValue(raw['nu']),
        'MW':   load._stripNestingToValue(raw['MW']) # kg/mol
    }
    return spd

