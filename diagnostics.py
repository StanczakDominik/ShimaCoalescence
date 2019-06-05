import numpy as np
import numba
import astropy.units as u

def droplet_number_density(multiplicity, V):
    return np.sum(multiplicity) / V

def precipitation_rate(multiplicity, radius, V):
    return np.pi / 6.0 / V * np.sum(multiplicity * (2 * radius)**3 * terminal_velocity(radius))

def radar_reflectivity_factor(multiplicity, radius, V, z0=1*u.mm**6*u.mm**-3):
    z = np.sum(multiplicity * (2 * radius)**6) / V
    Z = 10 * np.log10(z/z0)
    return Z

@numba.vectorize(['float32(float32, float32)',
                  'float64(float64, float64)',
                  ],
                 target='parallel')
def W_estimator(Y, sigma):
    return np.exp(-Y**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def g_estimator(multiplicity, radii, masses, V, logRadiiPlot, sigma0=0.62):
    # TODO rewrite via numba
    sigma = sigma0 * radii.size**(-1/5)
    logRadii = np.log(radii)
    argW = logRadiiPlot.reshape(1, logRadiiPlot.size) - logRadii[:, np.newaxis]
    W = W_estimator(argW, sigma)
    return np.sum((multiplicity * masses)[:, np.newaxis] * W, axis=0) / V
