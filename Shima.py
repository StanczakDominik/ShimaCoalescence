import numpy as np
import pandas
from astropy import units as u
from astropy import constants
import tqdm
import matplotlib.pyplot as plt
import seaborn
from astropy.visualization import quantity_support
import math
from IPython.display import display
import numba

drizzle_cutoff = 100 * u.um
rain_cutoff = 1 * u.mm

eta = (1.81e-5 * u.kg / (u.m * u.s)).si.value    # fluid dynamic viscosity
rho_F = (1.2754 * u.kg / u.m**3).si.value  # fluid density - assumed IUPAC for dry (!) air
rho_b = (997 * u.kg / u.m**3).si.value  # body density - assumed water
g = constants.g0.si.value
c_1 = 0.0902
delta_0 = 9.06
C_0 = 0.29
# @numba.njit

@numba.vectorize('float32(float32)', target='parallel')
def spherical_terminal_velocity(radius):
    nu = eta / rho_F
    D = 2 * radius            # maximum dimension of the body - diameter
    vb = 4/3 * np.pi * radius ** 3  # body volume - droplet assumed spherical
    Area = np.pi * radius ** 2   # cross sectional area
    X = 2 * vb * (rho_b - rho_F) * g * D**2 / (Area * rho_F * nu**2)

    parenthesis = 1 + c_1 * X**0.5
    b_re = 0.5 * c_1 * X ** 0.5 * (parenthesis**0.5 -1)**-1 * parenthesis**-0.5
    a_re = (delta_0**2 / 4) * (parenthesis ** 0.5 -1)**2 / X ** b_re
    A = a_re * nu ** (1 - 2 * b_re) * (4 * rho_b * g / (3 * rho_F))**b_re
    B = 3 * b_re - 1
    
    velocities = A * D ** B
    return velocities

def pairwise_probabilities(multiplicities, radii, dt, V, E_jk):
    N = radii.size
    pairs = np.random.permutation((N//2)*2)
    permuted_multiplicities = multiplicities[pairs]
    permuted_radii = radii[pairs]
    terminal_velocities = spherical_terminal_velocity(permuted_radii)

    max_multiplicities = np.max(np.vstack((permuted_multiplicities[::2],
                                           permuted_multiplicities[1::2])), axis=0)

    P_pairs = E_jk * np.pi * (permuted_radii[::2] + permuted_radii[1::2])**2 * dt / V * \
        np.abs(terminal_velocities[::2] - terminal_velocities[1::2])

    fixed_probabilities = max_multiplicities * P_pairs * N * (N-1) / (2 * int(N/2))
    random_numbers = np.random.random(fixed_probabilities.size)

    gamma = np.floor(fixed_probabilities)
    floor_diff = fixed_probabilities - gamma
    coalescing_pairs_alternate = random_numbers < floor_diff
    gamma[coalescing_pairs_alternate] += 1

    coalescing_pairs = gamma > 0
    coalescing_pair_indices = pairs.reshape(int(N/2), 2)[coalescing_pairs]
    return coalescing_pair_indices, gamma[coalescing_pairs]

@numba.njit
def simple_coalescence(multiplicity, radii, masses, coalescing_pairs, gamma):
    N_pairs = coalescing_pairs.shape[0]
    for i in range(N_pairs):
        j, k = coalescing_pairs[i]
        pair_gamma = gamma[i]
        if multiplicity[j] < multiplicity[k]:
            j, k = k, j  # swap indices
        ej = multiplicity[j]
        ek = multiplicity[k]
        gamma_eff = min((pair_gamma, int(ej/ek)))

        if ej > gamma_eff * ek:
            multiplicity[j] = ej - gamma_eff * ek
            # multiplicity[k] = unchanged
            # radii[j] = unchanged
            radii[k] = (gamma_eff * radii[j]**3 + radii[k]**3)**(1/3)
            # masses[j] = unchanged
            masses[k] += gamma_eff * masses[j]
            # x unchanged on both counts
        elif ej == gamma_eff * ek:    # TODO float comparison?
            multiplicity[j] = math.floor(ek/2)
            multiplicity[k] -= multiplicity[j]
            radii[j] = radii[k] = (gamma_eff * radii[j]**3 + radii[k]**3)**(1/3)
            masses[j] = masses[k] = gamma_eff * masses[j] + masses[k]
            # x unchanged on both counts
        else:
            raise ValueError("wut?")


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


def simulation(multiplicity, radii, masses, NT, V, E_jk, dt, radii_min_plot = 1e-7, radii_max_plot = 1e-2):
    tables = []
    multiplicity = multiplicity.copy()
    radii = radii.copy()
    masses = masses.copy()
    diagnostics = []
    progressbar = tqdm.trange(NT)

    waiting_for_drizzle = True
    waiting_for_rain = True
    logRadiiPlot = np.log(np.logspace(np.log10(radii_min_plot), np.log10(radii_max_plot), 2000))
    for i in progressbar:
        # simulation loop
        N = radii.size # this can change dynamically
        coalescing_pair_indices, gamma = pairwise_probabilities(multiplicity, radii, dt, V, E_jk)
        if len(coalescing_pair_indices) > 0:
            simple_coalescence(multiplicity, radii, masses, coalescing_pair_indices, gamma)

        removed_particles = multiplicity == 0
        multiplicity = multiplicity[~removed_particles]
        radii = radii[~removed_particles]
        masses = masses[~removed_particles]

        # diagnostics
        if waiting_for_drizzle and (radii.max() > drizzle_cutoff.si.value):
            progressbar.write(f"radii max: {radii.max():.2e} m, drizzle achieved at iteration {i}")
            waiting_for_drizzle = False
        if waiting_for_rain and (radii.max() > rain_cutoff.si.value):
            progressbar.write(f"radii max: {radii.max():.2e} m, rain achieved at iteration {i}")
            waiting_for_rain = False

        if (i % 10) == 0:
                
            current_diagnostics = {
                "t": i * dt,
                "N superdroplets [1]":N,
                "max radius [m]":np.max(radii),
                # "mult_dtype": multiplicity.dtype,
                # "radii_dtype": radii.dtype,
                # "masses_dtype": masses.dtype,
            }
            
            diagnostics.append(dict(**current_diagnostics,
                **{
                    "i": i,
                    "Droplet number density [m^-3]":droplet_number_density(multiplicity, V),
                    "median radius [m]":np.median(radii),
                    "mean_radius":radii.mean(),
                    "std_radius":radii.std(),
                }
                                    ))
            progressbar.set_postfix(**current_diagnostics)
        if (i % 600) == 0:
            logRestim = g_estimator(multiplicity, radii, masses, V, logRadiiPlot)
            if N == 1:
                progress.bar(f"One superdroplet remaining at {i}; there's no more interesting dynamics to be had so I'm shutting this down")
                break
            tables.append({"g": (logRadiiPlot, logRestim),
                           "i": i, "t": i * dt})
    return diagnostics, tables

from config import *
def main(plot = False):

    if new_run:
        diagnostics, tables = simulation(multiplicity, radii, masses, NT, V, E_jk=E_jk, dt = dt)
        df = pandas.DataFrame(diagnostics)
        df.to_json("shima2.json")
    else:
        df = pandas.read_json("shima2.json")

    plotted = ["N superdroplets [1]",
               "Droplet number density [m^-3]",
               "median radius [m]",
               "max radius [m]",
               ]
    units = [1, u.m**-3, u.m, u.m]
    fig, axes = plt.subplots(len(plotted), sharex=True)
    # this is not sorted for some reason
    for col, unit, ax in zip(plotted, units, axes):
        ax.semilogy(df.t, df[col] * unit, ".", label=col)
        ax.set_title(col)
        ax.set_xlim(df.t.min(), df.t.max())
        if "radius" in col:
            # ax.fill_between(df.t,
            #                 (df[col] - df['std_radius']) * unit,
            #                 (df[col] + df['std_radius']) * unit,
            #                 alpha=0.5,
            #                 )
            ax.axhline(drizzle_cutoff.si.value, color="k", label="drizzle")
            ax.axhline(rain_cutoff.si.value, color="r", label="rain")
        ax.legend(loc='best')
        ax.set_ylabel(col)
    ax.set_xlabel("time [s]")
    fig.savefig("Shima2.png")
    try:
        tables
    except NameError:
        print("Can only plot radial distributions after fresh sim run")
    else:
        fig2, axis2 = plt.subplots()
        for T in tables:
            logR, logRestim = T['g']
            axis2.semilogx(np.exp(logR), logRestim, label=f"t = {T['t']:.1f} s")
        axis2.axvline(drizzle_cutoff.si.value, color="k", label="drizzle")
        axis2.axvline(rain_cutoff.si.value, color="r", label="rain")
        axis2.legend(loc='best')
        axis2.set_xlabel("radius [m]")
        axis2.set_title("Shima's radius kernel estimator")
        fig2.savefig("Shima2_radii.png")


    display(df.tail())
    if plot:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main(False)
