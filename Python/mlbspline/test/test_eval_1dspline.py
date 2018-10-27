import warnings, unittest as ut
import numpy as np
import scipy.io as sio
from mlbspline import load, eval


class TestEval1DSpline(ut.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
    def tearDown(self):
        pass
    def test1dsplineeval(self):
        sp = load.loadSpline('spline1d_v7.mat')
        x = np.empty(1, np.object)              # sadly, must do this so numpy won't nest or unnest against your will
        x[0] = np.arange(240, 501, 20)          # in Matlab, a = 240:20:500;
        out = eval.evalMultivarSpline(sp, x)
        # TODO: shouldn't have to squeeze
        mlout = sio.loadmat('spline1d_out.mat')['sp1d_out'].squeeze()
        self.assertEqual(out.shape, mlout.shape, 'shapes not equal')
        self.assertTrue(np.allclose(out, mlout, rtol=0, atol=1e-11), 'output not within absolute tolerances')
        self.assertTrue(np.allclose(out, mlout, rtol=1e-12, atol=0), 'output not within relative tolerances')

if __name__ == '__main__':
    ut.main()




