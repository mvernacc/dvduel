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
        no_action = sim.SpacecraftAction(fire_railgun=None)
        actions = {'A': no_action, 'B': no_action}
        if i == 200:
            actions['A'] = sim.SpacecraftAction(
                fire_railgun=np.array([1000., 1000., 0.]))
        simulator.update(actions, dt)


    fig_traj, axes_traj = plt.subplots()
    plotter.plot_pos_2d(
        axes_traj, simulator.spacecraft_states_record, simulator.projectile_states_record)
    fig_resource, axes_resoure = plt.subplots(
        nrows=2, ncols=1, figsize=(6, 7))
    plotter.plot_resources(
        axes_resoure, simulator.spacecraft_states_record, simulator.time_record)
    fig_resource.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
