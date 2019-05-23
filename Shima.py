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

def terminal_velocity(radius, rho_a = None, rho_a0 = 1.2e-3 * u.g * u.cm**-3):
    if rho_a is None:
        rho_a = rho_a0
    """ Rogers (1976), page 2 of Terminal Velocities of Droplets and Crystals: Power Laws with Continuous Parameters over the Size Spectrum
    """
    k1 = 1.19e6 / u.cm / u.s
    k2 = 8e3 / u.s
    indices_turbulent = radius > 40 * u.micrometer
    indices_sqrt = radius > 600 * u.micrometer
    velocities = u.Quantity(np.empty(radius.size), unit=u.m/u.s)
    velocities[indices_turbulent] = k1 * radius[indices_turbulent] ** 2
    velocities[~indices_turbulent] = k2 * radius[~indices_turbulent]
    # TODO get some rho_a0, rho_a approximations)
    k3 = 2.2e3 * u.cm**0.5 / u.s * (rho_a0 / rho_a) ** 0.5
    velocities[indices_sqrt] = k3 * radius[indices_sqrt]**0.5
    return velocities

def better_terminal_velocity(radius,
                             eta = (1.81e-5 * u.kg / (u.m * u.s)).si.value,    # fluid dynamic viscosity
                             rho_F = (1.2754 * u.kg / u.m**3).si.value,  # fluid density - assumed IUPAC for dry (!) air
                             rho_b = (997 * u.kg / u.m**3).si.value,  # body density - assumed water
                             g = constants.g0.si.value,
                             c_1 = 0.0902,
                             delta_0 = 9.06,
                             C_0 = 0.29,
                             alpha = 0.524,
                             beta = 3,
                             sigma = 2,
                             gamma = 0.785,
                             ):
    nu = eta / rho_F          # fluid kinematic viscosity
    D = 2 * radius            # maximum dimension of the body - diameter
    vb = 4/3 * np.pi * radius ** 3  # body volume - droplet assumed spherical
    Area = np.pi * radius ** 2   # cross sectional area
    X = 2 * vb * (rho_b - rho_F) * g * D**2 / (Area * rho_F * nu**2)
    # phi = (delta_0**2 / 4) * ((1 + c_1 * X**0.5)**0.5 -1)**2
    # phiprime = ((1 + c_1 * X ** 0.5)**0.5 -1) * (1 + c_1 * X**0.5)**(-0.5) * X **(-0.5) / (2 * C_0**0.5)
    # b_re = X * phiprime / phi
    # a_re = phi / X ** b_re

    parenthesis = (1 + c_1 * X**0.5)
    b_re = 0.5 * c_1 * X ** 0.5 * (parenthesis**0.5 -1)**-1 * parenthesis**-0.5
    a_re = (delta_0**2 / 4) * (parenthesis ** 0.5 -1)**2 / X ** b_re
    A = a_re * nu ** (1 - 2 * b_re) * (2 * alpha * g / (rho_F * gamma))**b_re
    B = b_re * (beta - sigma + 2) - 1
    
    velocities = A * D ** B
    return velocities

# @profile
def spherical_terminal_velocity(radius,
                             eta = (1.81e-5 * u.kg / (u.m * u.s)).si.value,    # fluid dynamic viscosity
                             rho_F = (1.2754 * u.kg / u.m**3).si.value,  # fluid density - assumed IUPAC for dry (!) air
                             rho_b = (997 * u.kg / u.m**3).si.value,  # body density - assumed water
                             g = constants.g0.si.value,
                             c_1 = 0.0902,
                             delta_0 = 9.06,
                             C_0 = 0.29,
                             alpha = 0.524,
                             beta = 3,
                             sigma = 2,
                             gamma = 0.785,
                             ):
    radius = radius.si.value
    nu = (eta / rho_F)
    D = 2 * radius            # maximum dimension of the body - diameter
    vb = 4/3 * np.pi * radius ** 3  # body volume - droplet assumed spherical
    Area = np.pi * radius ** 2   # cross sectional area
    X = (2 * vb * (rho_b - rho_F) * g * D**2 / (Area * rho_F * nu**2))

    parenthesis = (1 + c_1 * X**0.5)
    b_re = 0.5 * c_1 * X ** 0.5 * (parenthesis**0.5 -1)**-1 * parenthesis**-0.5
    a_re = (delta_0**2 / 4) * (parenthesis ** 0.5 -1)**2 / X ** b_re
    A = a_re * nu ** (1 - 2 * b_re) * (4 * rho_b * g / (3 * rho_F))**b_re
    B = 3 * b_re - 1
    
    velocities = A * D ** B
    return velocities * u.m / u.s

# @profile
def pairwise_probabilities(multiplicities, radii, dt, V, E_jk):
    N = radii.size
    pairs = np.random.permutation(range((N//2)*2)).astype(int)
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

# @profile
def apply_coalescence(multiplicity_j, radius_j, multiplicity_k, radius_k):
    if multiplicity_j == multiplicity_k:
        new_multiplicity_j = int(math.floor(multiplicity_j/2))
        new_multiplicity_k = multiplicity_j - new_multiplicity_j
        new_radius_j = new_radius_k = (radius_j**3 + radius_k**3)**(1/3)
    elif multiplicity_j > multiplicity_k:
        new_multiplicity_j = multiplicity_j - multiplicity_k
        new_multiplicity_k = multiplicity_k
        new_radius_j = radius_j
        new_radius_k = (radius_j **3 + radius_k**3)**(1/3)
    elif multiplicity_j < multiplicity_k:
        new_multiplicity_k = multiplicity_k - multiplicity_j
        new_multiplicity_j = multiplicity_j
        new_radius_k = radius_k
        new_radius_j = (radius_k **3 + radius_j**3)**(1/3)
    else:
        raise ValueError("wat")
    return new_multiplicity_j, new_radius_j, new_multiplicity_k, new_radius_k

# @profile
@numba.njit
def simple_coalescence(multiplicity, radii, masses, coalescing_pairs, gamma):
    for i in range(len(coalescing_pairs)):
        j, k = coalescing_pairs[i]
        pair_gamma = gamma[i]
    # for (j, k), pair_gamma in zip(coalescing_pairs, gamma):
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
        elif ej == gamma_eff * ek:
            multiplicity[j] = math.floor(ek/2)
            multiplicity[k] -= multiplicity[j]
            radii[j] = radii[k] = (gamma_eff * radii[j]**3 + radii[k]**3)**(1/3)
            masses[j] = masses[k] = gamma_eff * masses[j] + masses[k]
            # x unchanged on both counts
        else:
            raise ValueError("wut?")


# @profile
def droplet_number_density(multiplicity, V):
    return np.sum(multiplicity) / V

def precipitation_rate(multiplicity, radius, V):
    return np.pi / 6.0 / V * np.sum(multiplicity * (2 * radius)**3 * terminal_velocity(radius))

def radar_reflectivity_factor(multiplicity, radius, V, z0=1*u.mm**6*u.mm**-3):
    z = np.sum(multiplicity * (2 * radius)**6) / V
    Z = 10 * np.log10(z/z0)
    return Z

# @profile
def W_estimator(Y, sigma):
    return np.exp(-Y**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

# @profile
def g_estimator(multiplicity, radii, masses, V, sigma0=0.62):
    sigma = sigma0 * radii.size**(-1/5)
    logRadii = np.log(radii.si.value)
    logRadiiPlot = np.linspace(logRadii.min() - 2, logRadii.max() + 2, 2000)
    argW = logRadiiPlot.reshape(1, logRadiiPlot.size) - logRadii[:, np.newaxis]
    W = W_estimator(argW, sigma)
    return logRadiiPlot, np.sum((multiplicity * masses)[:, np.newaxis] * W, axis=0) / V


# @profile
def simulation(multiplicity, radii, masses, NT, V, E_jk, dt = 0.01 * u.s):
    tables = []
    multiplicity = multiplicity.copy()
    radii = radii.copy()
    masses = masses.copy()
    diagnostics = []
    progressbar = tqdm.trange(NT)
    for i in progressbar:
        N = radii.size # this can change dynamically
        coalescing_pair_indices, gamma = pairwise_probabilities(multiplicity, radii, dt, V, E_jk)
        if len(coalescing_pair_indices) > 0:
            simple_coalescence(multiplicity, radii, masses, coalescing_pair_indices, gamma)

        removed_particles = multiplicity == 0
        multiplicity = multiplicity[~removed_particles]
        radii = radii[~removed_particles]
        masses = masses[~removed_particles]

        if (i % 100) == 0:
            current_diagnostics = {
                "t": i * dt.si.value,
                "N_superdroplets":N,
                "droplet_number_density":droplet_number_density(multiplicity, V).si.value,
                "median_radius":np.median(radii.si.value)
            }
            
            diagnostics.append(dict(**current_diagnostics,
                **{
                    "i": i,
                    "mean_radius":radii.si.value.mean(),
                }
                                    ))
            progressbar.set_postfix(**current_diagnostics)
        if (i % 10000) == 0:
            logR, logRestim = g_estimator(multiplicity, radii, masses, V)
            with quantity_support():
                fig2, axis2 = plt.subplots()
                axis2.semilogx(np.exp(logR), logRestim, label=f"t = {i * dt.si.value:.1f} s")
                axis2.axvline(drizzle_cutoff, color="k", label="mżawka - górna granica")
                axis2.axvline(rain_cutoff, color="r", label="deszcz - górna granica")
                axis2.legend(loc='best')
                fig2.savefig(f"{i:08d}_Shima2_radii.png")
                plt.close()
            if N == 1:
                break
                # tables.append({"g": (logR, logRestim)
                #                "i": i, "t": i * dt.si.value})
    return diagnostics, tables

if __name__ == "__main__":
    from config import *

    if new_run:
        diagnostics, tables = simulation(multiplicity, radii, masses, NT, V, E_jk=E_jk, dt = dt)
        df = pandas.DataFrame(diagnostics)
        df.to_json("shima2.json")
    else:
        df = pandas.read_json("shima2.json")

    plotted = ["N_superdroplets", "droplet_number_density", "mean_radius",
               # "median_radius",
               ]
    fig, axes = plt.subplots(len(plotted), sharex=True)
    # this is not sorted for some reason
    with quantity_support():
        for col, ax in zip(plotted, axes):
            ax.semilogy(df.t, df[col], "o", label=col)
            ax.set_title(col)
            ax.set_xlim(df.t.min(), df.t.max())
            if "radius" in col:
                ax.axhline(drizzle_cutoff, color="k", label="mżawka - górna granica")
                ax.axhline(rain_cutoff, color="r", label="deszcz - górna granica")
            ax.legend(loc='best')
        fig.savefig("Shima2.png")
        try:
            tables
            fig2, axis2 = plt.subplots()
            for T in tables:
                logR, logRestim = T['g']
                axis2.semilogx(np.exp(logR), logRestim, label=f"t = {T['t']:.1f} s")
            axis2.axvline(drizzle_cutoff, color="k", label="mżawka - górna granica")
            axis2.axvline(rain_cutoff, color="r", label="deszcz - górna granica")
            axis2.legend(loc='best')
            fig2.savefig("Shima2_radii.png")
        except NameError:
            pass


    display(df.tail())
    plt.close()


