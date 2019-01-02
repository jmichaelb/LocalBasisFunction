import warnings, unittest as ut
import numpy as np
import scipy.io as sio
from lbftd import loadGibbs


class TestEval1DSpline(ut.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
    def tearDown(self):
        pass
    def test2dgibbsload_ScalarGo(self):
        gsp7 = loadGibbs.loadGibbsSpline('gsp_1scalarGo.mat')
        # just do a spot check of a field in the spline since it calls load.getSplineDict which is tested elsewhere
        self.assertTrue(np.array_equal(gsp7['sp']['number'], np.array([29, 20, 14])))
        self.assertTrue(np.array_equal(gsp7['MW'], np.array([0.01801528, 0.05844])))


if __name__ == '__main__':
    ut.main()
