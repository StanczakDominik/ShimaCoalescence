from Shima import spherical_terminal_velocity
import pandas
import matplotlib.pyplot as plt
from astropy import visualization, units as u

def test_terminal_velocity():
    data = pandas.read_csv("vel_vs_r.dat", names = ["radius", "velocity"], sep=r"\s+")
    data_radius = data['radius'].values * u.um
    data_velocity = data['velocity'].values * u.m / u.s
    my_velocity = spherical_terminal_velocity(data_radius)
    if True:
        with visualization.quantity_support():
            plt.plot(data_radius, data_velocity, label="Dane źródłowe")
            plt.plot(data_radius, my_velocity, label="Obliczone")
            plt.legend()
            plt.savefig("terminal_velocity.png")
            plt.show()
    assert u.allclose(data_velocity, my_velocity, rtol=4e-3)
