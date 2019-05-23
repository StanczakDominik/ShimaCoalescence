from Shima import spherical_terminal_velocity
import pandas
import matplotlib.pyplot as plt
from astropy import units as u

def test_terminal_velocity():
    data = pandas.read_csv("vel_vs_r.dat", names = ["radius", "velocity"], sep=r"\s+")
    data_radius = data['radius'].values * u.m
    data_velocity = data['velocity'].values * u.m / u.s
    my_velocity = spherical_terminal_velocity(data_radius)
    if False:
        plt.plot(data_radius, data_velocity)
        plt.plot(data_radius, my_velocity)
        plt.show()
    assert u.allclose(data_velocity, my_velocity)
