import numpy as np
import astropy.units as u
import scipy.stats

np.random.seed(4)

V = 1e6 * u.m**3 # coalescence cell volume
n0 = 100 / u.cm**3 # initial number density of droplets
N = int(1.5e4) # initial number of super-droplets
dt = 0.01 * u.s
NT = int(3e6)

multiplicity = (n0*V/N * np.ones(N)).si.astype(int)
r_average = 30.531 * u.micron
v_average = 4 * np.pi / 3 * r_average ** 3
exponential_distribution = scipy.stats.expon(0, v_average.to(u.m**3))

volumes = exponential_distribution.rvs(N) * u.m**3
radii = (3 * volumes / (4 * np.pi))**(1/3)
density_solute = 1 * u.g / u.m**3
masses = volumes * density_solute
E_jk = 2
new_run = True
