import numpy as np
import pandas as pd
from astropy import units as u
from tqdm import tqdm
from scipy.stats import expon
import matplotlib.pyplot as plt
import seaborn
from astropy.visualization import quantity_support
from math import floor
from IPython.display import display

# volumes = exponential_distribution.rvs(int(1e4)) * u.m**3
# with quantity_support():
#     seaborn.distplot(volumes)
# with quantity_support():
#     seaborn.distplot(volumes)

# np.sum(radii < 40*u.micrometer) / len(radii)

# np.sum(radii > 600*u.micrometer) / len(radii)

def terminal_velocity(radius):
    """ Rogers (1976), page 2 of Terminal Velocities of Droplets and Crystals: Power Laws with Continuous Parameters over the Size Spectrum
    """
    k1 = 1.19e6 / u.cm / u.s
    k2 = 8e3 / u.s
    indices_turbulent = radius > 40 * u.micrometer
    velocities = u.Quantity(np.empty(radius.size), unit=u.m/u.s)
    velocities[indices_turbulent] = k1 * radius[indices_turbulent] ** 2
    velocities[~indices_turbulent] = k2 * radius[~indices_turbulent]
    return velocities

def pairwise_probabilities(multiplicities, radii, dt, V, E_jk):
    breakpoint()
    terminal_velocities = terminal_velocity(radii)
    N = radii.size
    pairs = np.random.permutation(range((N//2)*2)).reshape(int(N/2), 2)
    j_indices, k_indices = pairs.T
    P_pairs = E_jk * np.pi * (radii[j_indices] + radii[k_indices])**2 * dt / V * \
                     np.abs(terminal_velocities[j_indices] - terminal_velocities[k_indices])
    return j_indices, k_indices, P_pairs

def apply_coalescence(multiplicity_j, radius_j, multiplicity_k, radius_k):
#     if multiplicity_j == multiplicity_k == 1:
#         raise NotImplementedError("Remove j from simulation") # TODO
#         new_multiplicity_j = new_radius_j = np.nan
#         new_
#     assert isinstance(multiplicity_j, int) & isinstance(multiplicity_k, int)
    if multiplicity_j == multiplicity_k:
        new_multiplicity_j = int(floor(multiplicity_j/2))
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

def vector_coalescence(multiplicity, radii, j_indices, k_indices):
    # TODO try np.take
    multiplicity_j = multiplicity[j_indices]
    multiplicity_k = multiplicity[k_indices]
    equals = multiplicity_j == multiplicity_k
#     multiplicity[equals &]

def droplet_number_density(multiplicity, V):
    return np.sum(multiplicity) / V

def precipitation_rate(multiplicity, radius, V):
    return np.pi / 6.0 / V * np.sum(multiplicity * (2 * radius)**3 * terminal_velocity(radius))

def radar_reflectivity_factor(multiplicity, radius, V, z0=1*u.mm**6*u.mm**-3):
    z = np.sum(multiplicity * (2 * radius)**6) / V
    Z = 10 * np.log10(z/z0)
    return Z

def simulation(multiplicity, radii, NT, V, E_jk, dt = 0.01 * u.s):
    tables = [{"multiplicity":multiplicity, "radii":radii}]
    multiplicity = multiplicity.copy()
    radii = radii.copy()
    diagnostics = []
    for i in tqdm(range(NT)):
        N = radii.size # this can change dynamically
        indices_j, indices_k, P_jk = pairwise_probabilities(radii, dt, V, E_jk)
        multiplicities_j = multiplicity[indices_j]
        multiplicities_k = multiplicity[indices_k]
        max_multiplicities = np.max(np.vstack((multiplicities_j, multiplicities_k)), axis=0)
        fixed_probabilities = max_multiplicities * P_jk * N * (N-1) / (2 * int(N/2))
        random_numbers = np.random.random(fixed_probabilities.size)
        coalescing_pairs = fixed_probabilities > random_numbers
        num_coalesced = coalescing_pairs.sum()
        for j, k in zip(indices_j[coalescing_pairs], indices_k[coalescing_pairs]): # TODO rewrite this in array style
            multiplicity[j], radii[j], multiplicity[k], radii[k] = apply_coalescence(multiplicity[j], radii[j], multiplicity[k], radii[k])
        removed_particles = multiplicity == 0
        multiplicity = multiplicity[~removed_particles]
        radii = radii[~removed_particles]
        if (i % 10) == 0:
            diagnostics.append({
                "N_superdroplets":N,
                "num_coalesced":num_coalesced,
                # "mean_probability":fixed_probabilities.mean(),
                "droplet_number_density":droplet_number_density(multiplicity, V).si.value,
                "mean_radius":radii.si.value.mean(),
                # "precipitation_rate":precipitation_rate(multiplicity, radii, V).si.value,
                # "radar_reflectivity_factor":radar_reflectivity_factor(multiplicity,radii,V).si.value,
            })
            tables.append({
                "multiplicity":multiplicity.copy(),
                "radii": radii.copy()
            })
    return diagnostics, tables

if __name__ == "__main__":
    np.random.seed(0)

    V = 1e6 * u.m**3 # coalescence cell volume
    n0 = 100 / u.cm**3 # initial number density of droplets
    N = 10000   # initial number of super-droplets
    dt = 0.01 * u.s

    multiplicity = (n0*V/N * np.ones(N)).astype(int).si
    r_average = 30.531 * u.micron
    v_average = 4 * np.pi / 3 * r_average ** 3
    exponential_distribution = expon(0, v_average.to(u.m**3))

    volumes = exponential_distribution.rvs(N) * u.m**3
    radii = (3 * volumes / (4 * np.pi))**(1/3)
    E_jk = 0.5 # TODO

    breakpoint()
    N = radii.size # this can change dynamically
    indices_j, indices_k, P_jk = pairwise_probabilities(radii, dt, V, E_jk)
    # indices_j: array([3615, 9180, 3601, ..., 1386, 5634, 3932]), (5000,)
    # P_jk: <Quantity [7.46711490e-19, 9.23716135e-19, 3.39917593e-18, ...,
    #       3.67174827e-18, 1.72423352e-18, 3.69688307e-18]>, (5000,)
    
    # TODO filter out as much as possible on P_jk first

    multiplicities_j = multiplicity[indices_j]
    multiplicities_k = multiplicity[indices_k]

    max_multiplicities = np.max(np.vstack((multiplicities_j, multiplicities_k)), axis=0)
    # array([1.e+10, 1.e+10, 1.e+10, ..., 1.e+10, 1.e+10, 1.e+10]), (5000,)
    fixed_probabilities = max_multiplicities * P_jk * N * (N-1) / (2 * int(N/2)) # (5000,)
    random_numbers = np.random.random(fixed_probabilities.size)
    coalescing_pairs = fixed_probabilities > random_numbers # (5000,) bool for every pair
    # this could be simplified: turn output of pairwise_probabilities into a list of tuples, filtering on coalescing_pairs

    sequential_multiplicity = multiplicity.copy()

    """ list of tuples = filtered_pairwise_probabilities 
    """
    for j, k in zip(indices_j[coalescing_pairs], indices_k[coalescing_pairs]): # TODO rewrite this in array style
        sequential_multiplicity[j], _, sequential_multiplicity[k], _ = apply_coalescence(multiplicity[j], radii[j], multiplicity[k], radii[k])

    vector_multiplicity, vector_radius = vector_coalescence(multiplicity, radii, indices_j, indices_k)

    # diagnostics, tables = simulation(multiplicity, radii, int(50), V, E_jk=E_jk)
    # df = pd.DataFrame(diagnostics)
    # df.to_pickle("shima2.json")
    # fig, axes = plt.subplots(len(df.columns), sharex=True)
    # for col, ax in zip(df.columns, axes):
    #     ax.plot(df[col], label=col)
    #     ax.set_title(col)
    #     ax.legend(loc='best')
    #     ax.set_xlim(0, len(df))
    # display(df.tail())
    # plt.savefig("Shima2.png")
    # plt.show()


