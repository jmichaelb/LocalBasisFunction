import warnings, unittest as ut
import numpy as np
import scipy.io as sio
from mlbspline import load
from lbftd import loadGibbs as lg, evalGibbs as eg


class TestEvalGibbsMLSpline(ut.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
    def tearDown(sThermodyelf):
        pass
    def test_evalgibbs_puresubst_allmeasures(self):
        gsp = lg.loadGibbsSpline('gsp_puresubstance.mat')
        P = np.arange(0, 3001, 200)
        T = np.arange(0, 401, 50)
        out = eg.evalSolutionGibbs(gsp['sp'], np.array([P, T]), MWv=gsp['MW'][0])
        mlout = load._stripNestingToFields(sio.loadmat('gsp2d_out.mat')['gsp2d_out'])
        valErrs = ''
        # check all values and output just one error for all of them
        for f in vars(out).keys():
            outfield = getattr(out, f)
            if f not in mlout.dtype.fields:
                warnings.warn('Matlab output is missing field ' + f)
            else:
                mloutfield = mlout[f]
                self.assertEqual(outfield.shape, mloutfield.shape, f+' output not the same shape as MatLab output')
                if not (np.allclose(outfield, mloutfield, rtol=relTolerance, atol=0)  # check both abs and rel differences
                        and np.allclose(outfield, mloutfield, rtol=0, atol=absTolerance)):
                    absDiffs = np.absolute(outfield - mloutfield)
                    relDiffs = absDiffs / np.absolute(mloutfield)
                    valErrs = valErrs + 'Output for '+f+' has absolute differences as large as '+str(np.max(absDiffs)) +\
                              ' and relative differences as large as '+str(np.max(relDiffs))+'.\n'
        if valErrs:
            self.fail(valErrs)


relTolerance = 5e-9
absTolerance = 1e-6

if __name__ == '__main__':
    ut.main()