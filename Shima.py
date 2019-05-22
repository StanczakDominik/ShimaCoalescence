import numpy as np
import pandas as pd
from astropy import units as u
from astropy import constants
from tqdm import tqdm
from scipy.stats import expon
import matplotlib.pyplot as plt
import seaborn
from astropy.visualization import quantity_support
import math
from IPython.display import display
import numba

# volumes = exponential_distribution.rvs(int(1e4)) * u.m**3
# with quantity_support():
#     seaborn.distplot(volumes)
# with quantity_support():
#     seaborn.distplot(volumes)

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
                             a_re = 1,
                             nu = 1,
                             b_re = 1,
                             rho_w = 1,
                             g = 9.81 * constants.g0,
                             rho_a = 1,
                             ):
    # TODO prześledzić 
    A = a_re * nu ** (1 - 2 * b_re) * (4 / 3 * rho_w * g / rho_a)**b_re
    B = 3 * b_re - 1
    
    velocities = A * (2 * radius) ** B
    velocities = None
    return velocities

def pairwise_probabilities(multiplicities, radii, dt, V, E_jk):
    N = radii.size
    pairs = np.random.permutation(range((N//2)*2))
    permuted_multiplicities = multiplicities[pairs]
    permuted_radii = radii[pairs]
    terminal_velocities = terminal_velocity(permuted_radii)
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

def apply_coalescence(multiplicity_j, radius_j, multiplicity_k, radius_k):
#     if multiplicity_j == multiplicity_k == 1:
#         raise NotImplementedError("Remove j from simulation") # TODO
#         new_multiplicity_j = new_radius_j = np.nan
#         new_
#     assert isinstance(multiplicity_j, int) & isinstance(multiplicity_k, int)
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



def vector_coalescence(multiplicity, radii, coalescing_pairs, gamma):
    # TODO put gamma
    breakpoint()
    coalescing_multiplicity = multiplicity[coalescing_pairs]
    indices_j = np.argmax(coalescing_multiplcitiy, axis=1)
    indices_k = np.argmin(coalescing_multiplicity, axis=1)
    multiplicity_j = coalescing_multiplicity[indices_j]
    multiplicity_k = coalescing_multiplicity.min(axis=1)
    coalescing_radii = radii[coalescing_pairs]
    gamma_effective = np.min(np.vstack((multiplicity_j / multiplicity_k, gamma)), axis=1)

    indices_a = multiplicity_j > gamma_effective * multiplicity_k
    assert not (multiplicity_j < gamma_effective * multiplicity_k).any()

    # multiplicities_j, multiplicities_k = coalescing_multiplicity_effective.T
    # radii_j, radii_k = coalescing_radii.T

    # new_radii = (radii_j**3 + radii_k**3)**(1/3)

    # indices_equal = multiplicities_j == multiplicities_k
    # indices_j = multiplicities_j > multiplicities_k
    # indices_k = multiplicities_j < multiplicities_k

    new_multiplicities_j = multiplicity_j[indices_a] - gamma_effective[indices_a] * multiplicity_k[indices_a]
    multiplicities_k[indices_equal] = multiplicities_j[indices_equal] - new_multiplicities_j
    multiplicities_j[indices_equal] = new_multiplicities_j
    radii_j[indices_equal] = radii_k[indices_equal] = new_radii[indices_equal]

    multiplicities_j[indices_j] -= multiplicities_k[indices_j]
    radii_k[indices_j] = new_radii[indices_j]

    multiplicities_k[indices_k] -= multiplicities_j[indices_k]
    radii_j[indices_k] = new_radii[indices_k]

    rewrite_indices_j, rewrite_indices_k = coalescing_pairs.T

    multiplicity[rewrite_indices_j] = multiplicities_j
    multiplicity[rewrite_indices_j] = multiplicities_j
    radii[rewrite_indices_j] = radii_j
    radii[rewrite_indices_k] = radii_k

    

def droplet_number_density(multiplicity, V):
    return np.sum(multiplicity) / V

def precipitation_rate(multiplicity, radius, V):
    return np.pi / 6.0 / V * np.sum(multiplicity * (2 * radius)**3 * terminal_velocity(radius))

def radar_reflectivity_factor(multiplicity, radius, V, z0=1*u.mm**6*u.mm**-3):
    z = np.sum(multiplicity * (2 * radius)**6) / V
    Z = 10 * np.log10(z/z0)
    return Z

def simulation(multiplicity, radii, masses, NT, V, E_jk, dt = 0.01 * u.s):
    tables = [{"multiplicity":multiplicity, "radii":radii}]
    multiplicity = multiplicity.copy()
    radii = radii.copy()
    masses = masses.copy()
    diagnostics = []
    for i in tqdm(range(NT)):
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
                "N_superdroplets":N,
                "num_coalesced":2 * len(coalescing_pair_indices),
                # "mean_probability":fixed_probabilities.mean(),
                "droplet_number_density":droplet_number_density(multiplicity, V).si.value,
                "mean_radius":radii.si.value.mean(),
                # "precipitation_rate":precipitation_rate(multiplicity, radii, V).si.value,
                # "radar_reflectivity_factor":radar_reflectivity_factor(multiplicity,radii,V).si.value,
            }
            diagnostics.append(current_diagnostics)
            # tables.append({
            #     "multiplicity":multiplicity.copy(),
            #     "radii": radii.copy()
            # })
            # tqdm.set_postfix(current_diagnostics)
    return diagnostics, tables

if __name__ == "__main__":
    np.random.seed(4)

    V = 1e6 * u.m**3 # coalescence cell volume
    n0 = 100 / u.cm**3 # initial number density of droplets
    N = int(1e5) # initial number of super-droplets
    dt = 0.01 * u.s
    NT = int(1e5)

    multiplicity = (n0*V/N * np.ones(N)).si.astype(int)
    # breakpoint()
    r_average = 30.531 * u.micron
    v_average = 4 * np.pi / 3 * r_average ** 3
    exponential_distribution = expon(0, v_average.to(u.m**3))

    volumes = exponential_distribution.rvs(N) * u.m**3
    radii = (3 * volumes / (4 * np.pi))**(1/3)
    density_solute = 1 * u.g / u.m**3
    masses = volumes * density_solute
    E_jk = 2

    # N = radii.size # this can change dynamically
    # indices_j, indices_k, P_jk = pairwise_probabilities(multiplicity, radii, dt, V, E_jk)
    # # indices_j: array([3615, 9180, 3601, ..., 1386, 5634, 3932]), (5000,)
    # # P_jk: <Quantity [7.46711490e-19, 9.23716135e-19, 3.39917593e-18, ...,
    # #       3.67174827e-18, 1.72423352e-18, 3.69688307e-18]>, (5000,)
    
    # # TODO filter out as much as possible on P_jk first

    # multiplicities_j = multiplicity[indices_j]
    # multiplicities_k = multiplicity[indices_k]

    # max_multiplicities = np.max(np.vstack((multiplicities_j, multiplicities_k)), axis=0)
    # # array([1.e+10, 1.e+10, 1.e+10, ..., 1.e+10, 1.e+10, 1.e+10]), (5000,)
    # fixed_probabilities = max_multiplicities * P_jk * N * (N-1) / (2 * int(N/2)) # (5000,)
    # random_numbers = np.random.random(fixed_probabilities.size)
    # coalescing_pairs = fixed_probabilities > random_numbers # (5000,) bool for every pair
    # # this could be simplified: turn output of pairwise_probabilities into a list of tuples, filtering on coalescing_pairs

    # sequential_multiplicity = multiplicity.copy()

    # """ list of tuples = filtered_pairwise_probabilities 
    # """
    # for j, k in zip(indices_j[coalescing_pairs], indices_k[coalescing_pairs]): # TODO rewrite this in array style
    #     sequential_multiplicity[j], _, sequential_multiplicity[k], _ = apply_coalescence(multiplicity[j], radii[j], multiplicity[k], radii[k])

    # coalescing_pair_indices = pairwise_probabilities(multiplicity, radii, dt, V, E_jk)
    # vector_coalescence(multiplicity, radii, coalescing_pair_indices)

    # diagnostics, tables = simulation(multiplicity, radii, masses, NT, V, E_jk=E_jk)
    # df = pd.DataFrame(diagnostics)
    # df.to_pickle("shima2.pickle")
    df = pd.read_pickle("shima2.pickle")
    fig, axes = plt.subplots(len(df.columns), sharex=True)
    for col, ax in zip(df.columns, axes):
        ax.semilogy(df[col], label=col)
        ax.set_title(col)
        ax.legend(loc='best')
        ax.set_xlim(0, len(df))
    display(df.tail())
    plt.savefig("Shima2.png")
    plt.show()


