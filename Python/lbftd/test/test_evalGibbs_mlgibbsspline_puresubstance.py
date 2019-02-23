import warnings, unittest as ut
import numpy as np
import scipy.io as sio
from mlbspline import load
from lbftd import loadGibbs as lg, evalGibbs as eg


class TestEvalGibbsPureSubstance(ut.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
        self.spline = lg.loadGibbsSpline('gsp_puresubstance.mat')
        self.mlout = load._stripNestingToFields(sio.loadmat('gsp2d_out.mat')['gsp2d_out'])
        self.P = np.arange(0, 3001, 200).astype(float)
        self.T = np.arange(0, 401, 50).astype(float)
    def tearDown(sThermodyelf):
        pass
    def test_evalgibbs_puresubstance_allmeasures(self):
        out = eg.evalSolutionGibbs(self.spline['sp'], np.array([self.P, self.T]), MWv=self.spline['MW'][0])
        valErrs = ''
        # check all values and output just one error for all of them
        for tdv in vars(out).keys():
            outfield = getattr(out, tdv)
            if tdv not in self.mlout.dtype.fields:
                warnings.warn('Matlab output does not include tdv ' + tdv)
            else:
                mloutfield = self.mlout[tdv]
                self.assertEqual(outfield.shape, mloutfield.shape, tdv+' output not the same shape as MatLab output')
                if not (np.allclose(outfield, mloutfield, rtol=relTolerance, atol=0)  # check both abs and rel differences
                        and np.allclose(outfield, mloutfield, rtol=0, atol=absTolerance)):
                    absDiffs = np.absolute(outfield - mloutfield)
                    relDiffs = absDiffs / np.absolute(mloutfield)
                    valErrs = valErrs + 'Output for '+tdv+' has absolute differences as large as '+str(np.max(absDiffs)) +\
                              ' and relative differences as large as '+str(np.max(relDiffs))+'.\n'
        if valErrs:
            self.fail(valErrs)
    def test_evalgibbs_puresubstance_singlepoint(self):
        pidx = 0; tidx = 0
        out = eg.evalSolutionGibbs(self.spline['sp'], (self.P[pidx], self.T[tidx]), MWv=self.spline['MW'][0])
        valErrs = ''
        # check all values and output just one error for all of them
        for tdv in vars(out).keys():
            outfield = getattr(out, tdv)
            self.assertEqual(1, outfield.size, 'Output for '+tdv+' has too many values')
            if tdv not in self.mlout.dtype.fields:
                warnings.warn('Matlab output does not include tdv ' + tdv)
            else:
                mloutfield = self.mlout[tdv][pidx][tidx]
                self.assertEqual(1, outfield.size, tdv + ' output not the same shape as MatLab output')
                if not (np.allclose(outfield, mloutfield, rtol=relTolerance,
                                    atol=0)  # check both abs and rel differences
                        and np.allclose(outfield, mloutfield, rtol=0, atol=absTolerance)):
                    absDiffs = np.absolute(outfield - mloutfield)
                    relDiffs = absDiffs / np.absolute(mloutfield)
                    valErrs = valErrs + 'Output for ' + tdv + ' has absolute differences as large as ' + str(
                        np.max(absDiffs)) + \
                              ' and relative differences as large as ' + str(np.max(relDiffs)) + '.\n'
        if valErrs:
            self.fail(valErrs)



relTolerance = 5e-9
absTolerance = 1e-6

if __name__ == '__main__':
    ut.main()