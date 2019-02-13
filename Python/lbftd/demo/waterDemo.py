from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from lbftd import statevars, loadGibbs as lg, evalGibbs as eg

# fname = askopenfilename(defaultextension='.mat', title='Choose the spline to evaluate')
water_spline = lg.loadGibbsSpline('water_demo_spline.mat')

P = np.linspace(0.1, 1500, num=200)
T = np.linspace(240, 500, num=200)
# evaluate the spline at the requested P(ressure, in MPa) and T(emperature, in K)
# requested thermodynamic state variables:
# - rho: density in kg m^-3
# - Cp: isobaric specific heat in J kg^-1 K^-1
# - Kt: isothermal bulk modulus in MPa
# - alpha:  thermal expansivity in K-1
tdstate = eg.evalSolutionGibbs(water_spline['sp'], np.array([P, T]), 'rho', 'Cp', 'Kt', 'alpha')


fig = plt.figure()
pP, pT = np.meshgrid(P, T)

rho_ax = fig.add_subplot(221, projection='3d')
rho_ax.set_xlabel('Pressure ($MPa$)',labelpad=20)
rho_ax.set_ylabel('Temperature ($K$)',labelpad=20)
rho_ax.set_zlabel('Density ($kg/m^3$)',labelpad=20)
rho_surf = rho_ax.plot_surface(pP, pT, tdstate.rho)
rho_ax.invert_yaxis()

cp_ax = fig.add_subplot(222, projection='3d')
cp_ax.set_xlabel('Pressure ($MPa$)',labelpad=20)
cp_ax.set_ylabel('Temperature ($K$)',labelpad=20)
cp_ax.set_zlabel('Specific Heat ($J/kg \cdot K$)',labelpad=20)
rho_surf = cp_ax.plot_surface(pP, pT, tdstate.Cp)

kt_ax = fig.add_subplot(223, projection='3d')
kt_ax.set_xlabel('Pressure ($MPa$)',labelpad=20)
kt_ax.set_ylabel('Temperature ($K$)',labelpad=20)
kt_ax.set_zlabel('Isothermal Bulk Modulus ($MPa$)',labelpad=20)
kt_surf = kt_ax.plot_surface(pP, pT, tdstate.Kt)

alpha_ax = fig.add_subplot(224, projection='3d')
alpha_ax.set_xlabel('Pressure ($MPa$)',labelpad=20)
alpha_ax.set_ylabel('Temperature ($K$)',labelpad=20)
alpha_ax.set_zlabel('Thermal Expansivity ($K^{-1}$)',labelpad=20)
alpha_surf = alpha_ax.plot_surface(pP, pT, tdstate.alpha)

plt.show()









