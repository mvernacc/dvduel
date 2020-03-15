import numpy as np
from matplotlib import pyplot as plt
import sim
import plotter

def main():
    scA_init = sim.SpacecraftState(
        pos=np.array([500., 0., 0.]), vel=np.array([-2e3, 100., 0.]),
        mass=10e3, elec_energy=1e9)
    scB_init = sim.SpacecraftState(
        pos=np.array([-500., 0., 0.]), vel=np.array([2e3, -100., 0.]),
        mass=10e3, elec_energy=1e9)
    simulator = sim.Simulator(scA_init, scB_init)
    dt = 1e-3

    for i in range(simulator.n_steps - 1):
        simulator.update(dt)

    fig, axes = plt.subplots()
    plotter.plot_pos_2d(axes, simulator.spacecraft_states_record)
    plt.show()

if __name__ == '__main__':
    main()
