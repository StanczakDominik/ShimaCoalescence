import numpy as np
import astropy.units as u
import scipy.stats

np.random.seed(4)

V = 1e6 * u.m**3 # coalescence cell volume
n0 = 2**23 / u.m**3 # initial number density of droplets
N = int(2**17) # initial number of super-droplets
dt = (0.1 * u.s).si.value
NT = int(40000)
density_solute = 1 * u.g / u.m**3
E_jk = 0.5

radii_min_plot = 1e-7
radii_max_plot = 1e-2

multiplicity = (n0*V/N * np.ones(N)).si.value.astype(int)
r_average = 30.531 * u.micron
v_average = 4 * np.pi / 3 * r_average ** 3
exponential_distribution = scipy.stats.expon(0, v_average.to(u.m**3))

volumes = (exponential_distribution.rvs(N) * u.m**3).astype(np.float32)
radii = ((3 * volumes / (4 * np.pi))**(1/3)).si.value
masses = (volumes * density_solute).si.value
V = V.si.value
new_run = True
