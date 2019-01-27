import warnings, unittest as ut
import numpy as np
import scipy.io as sio
from mlbspline import load, eval


class TestEval3DSpline(ut.TestCase):
    def setUp(self):
        warnings.simplefilter('ignore', category=ImportWarning)
    def tearDown(self):
        pass
    def test2dsplineeval(self):
        sp = load.loadSpline('spline3d_v7.mat')
        x = np.empty(3, np.object)
        x[0] = np.arange(.1, 8001, 10)
        x[1] = np.arange(239, 502, 5)
        x[2] = np.arange(0, 8.1, .5)
        out = eval.evalMultivarSpline(sp, x)
        mlout = sio.loadmat('spline3d_out.mat')['sp3d_out']
        self.assertEqual(out.shape, mlout.shape, 'shapes not equal')
        # unfortunately, the three rounds of estimations that go on with a 3d spline result in more error
        # TODO: see if anything can be done to reduce this error
        self.assertTrue(np.allclose(out, mlout, rtol=0, atol=1e-8), 'output not within absolute tolerances')
        self.assertTrue(np.allclose(out, mlout, rtol=1e-10, atol=0), 'output not within relative tolerances')

if __name__ == '__main__':
    ut.main()
