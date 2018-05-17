from scipy.io import loadmat
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import get_current_fig_manager

from mlbspline import *

# A spline giving volume of pure water
# dimensions of spline are pressure (GPa) and temperature (T)
# ice VII Mie-Grüneisen equation of state based on X-Ray diffraction data from Bezacier et al. (2014)

splineFile = 'iceVII_EOS.mat'
spd = load.loadSpline(splineFile)
volML = loadmat('iceVIIvol.mat')['vol']

P = np.logspace(0,np.log10(20),50)
T = np.linspace(0,550,100)
x = np.array([P,T])
y = eval.evalMultivarSpline(spd,x)

maxDiff = abs((y - volML)).max()
print('The maximum difference between the Matlab calculated spline values and those calculated by ' +
      'evalMultiVarSpline is ' + str(maxDiff))

fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
P, T = np.meshgrid(P,T)
ax.plot_surface(P,T,y.T)
fm = get_current_fig_manager()
fm.show()

input("Press Enter to close")