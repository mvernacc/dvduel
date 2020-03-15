import numpy as np
from matplotlib import pyplot as plt
import sim

colors = {'A': 'C0', 'B': 'C1'}

def plot_pos_2d(axes, spacecraft_states_record, projectile_states_record, dims=(0, 1)):
    for key, scsr in spacecraft_states_record.items():
        axes.plot(
            scsr.pos[:, dims[0]], scsr.pos[:, dims[1]],
            color=colors[key], label=f'Spacecraft {key:s}')
    for ip, psr in enumerate(projectile_states_record):
        axes.plot(
            psr.pos[:, dims[0]], psr.pos[:, dims[1]],
            color='black', label=f'Projectile {ip:d}')
    axes.set_xlabel(f'Position[{dims[0]}] [m]')
    axes.set_ylabel(f'Position[{dims[1]}] [m]')
    axes.set_title('Trajectories')
    axes.legend()

def plot_resources(axes, spacecraft_states_record, time_record):
    for key, scsr in spacecraft_states_record.items():
        axes[0].plot(
            1e3 * time_record, 1e-6 * scsr.elec_energy,
            color=colors[key], label=f'Spacecraft {key:s}')
    axes[0].set_xlabel('Time [ms]')
    axes[0].set_ylabel('Electric energy [MJ]')
    axes[0].legend()

    for key, scsr in spacecraft_states_record.items():
        axes[1].plot(
            1e3 * time_record, 1e-3 * scsr.mass,
            color=colors[key], label=f'Spacecraft {key:s}')
    axes[1].set_xlabel('Time [ms]')
    axes[1].set_ylabel('Mass [Mg]')
    axes[1].legend()
