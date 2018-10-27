import warnings, unittest as ut
import numpy as np
import scipy.io as sio
from mlbspline import load, eval


class TestEval2DSpline(ut.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
    def tearDown(self):
        pass
    def test2dsplineeval(self):
        sp = load.loadSpline('spline2d_v7.mat')
        x = np.empty(2, np.object)
        x[0] = np.logspace(-1,3.7,50)
        x[1] = np.linspace(250,800,20)
        out = eval.evalMultivarSpline(sp, x)
        mlout = sio.loadmat('spline2d_out.mat')['sp2d_out']
        self.assertEqual(out.shape, mlout.shape, 'shapes not equal')
        self.assertTrue(np.allclose(out, mlout, rtol=0, atol=1e-10), 'output not within absolute tolerances')
        self.assertTrue(np.allclose(out, mlout, rtol=1e-10, atol=0), 'output not within relative tolerances')

if __name__ == '__main__':
    ut.main()
