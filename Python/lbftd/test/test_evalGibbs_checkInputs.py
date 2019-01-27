import warnings, unittest as ut
import numpy as np
from lbftd import loadGibbs as lg, evalGibbs as eg


class TestEvalGibbsCheckInputs(ut.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
    def tearDown(self):
        pass
    #########################################
    ## 2d spline _checkInputs tests
    #########################################
    # ignore req0X checks for 2d spline
    # no tdv yet defined requires X but not f
    # no tdv defined that requires solvent molecular weight for a pure substance
    # no tdv yet defined that requires MWu but not f or X
    def test_evalgibbs_2d_reqFOnly(self):
        gsp = lg.loadGibbsSpline('gsp_puresubstance.mat')
        P = np.arange(0, 3001, 100)
        T = np.arange(0, 401, 50)
        with self.assertRaisesRegex(ValueError, 'You cannot calculate {\'mus\'} with a spline ' +
                         'that does not include concentration. Remove those statevars and all their dependencies, ' +
                         'or supply a spline that includes concentration.'):
            eg.evalSolutionGibbs(gsp['sp'], np.array([P, T]), 'mus', MWu=1)
    def test_evalgibbs_2d_reqFandX(self):
        gsp = lg.loadGibbsSpline('gsp_puresubstance.mat')
        P = np.arange(0, 3001, 100)
        T = np.arange(0, 401, 50)
        with self.assertRaisesRegex(ValueError, 'You cannot calculate {\'muw\'} with a spline ' +
                         'that does not include concentration. Remove those statevars and all their dependencies, ' +
                         'or supply a spline that includes concentration.'):
            eg.evalSolutionGibbs(gsp['sp'], np.array([P, T]), 'muw', MWu=1)
    def test_evalgibbs_2d_extrapPOnly(self):
        gsp = lg.loadGibbsSpline('gsp_puresubstance.mat')
        P = np.arange(0, 5001, 100)
        T = np.arange(0, 401, 50)
        with self.assertRaisesRegex(ValueError, 'Dimensions {\'P\'} ' +
                'contain values that fall outside the knot sequence for the given spline, '+
                'which will result in extrapolation, which may not produce meaningful values.'):
            eg.evalSolutionGibbs(gsp['sp'], np.array([P, T]), 'G', failOnExtrapolate=True)
    def test_evalgibbs_2d_extrapTOnly(self):
        gsp = lg.loadGibbsSpline('gsp_puresubstance.mat')
        P = np.arange(0, 3001, 100)
        T = np.arange(0, 601, 50)
        with self.assertRaisesRegex(ValueError, 'Dimensions {\'T\'} ' +
                'contain values that fall outside the knot sequence for the given spline, '+
                'which will result in extrapolation, which may not produce meaningful values.'):
            eg.evalSolutionGibbs(gsp['sp'], np.array([P, T]), 'G', failOnExtrapolate=True)
    def test_evalgibbs_2d_extrapAllDims(self):
        gsp = lg.loadGibbsSpline('gsp_puresubstance.mat')
        P = np.arange(0, 5001, 100)
        T = np.arange(0, 601, 50)
        # no order imposed on list of failed dimensions so accommodate either order
        with self.assertRaisesRegex(ValueError, 'Dimensions {\'[PT]\', \'[PT]\'} ' +
                'contain values that fall outside the knot sequence for the given spline, '+
                'which will result in extrapolation, which may not produce meaningful values.'):
            eg.evalSolutionGibbs(gsp['sp'], np.array([P, T]), 'G', failOnExtrapolate=True)







if __name__ == '__main__':
    ut.main()