import numpy as np
from matplotlib import pyplot as plt
import sim

def plot_pos_2d(axes, spacecraft_states_record, dims=(0, 1)):
    colors = {'A': 'C0', 'B': 'C1'}
    for key, scsr in spacecraft_states_record.items():
        axes.plot(
            scsr.pos[:, dims[0]], scsr.pos[:, dims[1]],
            color=colors[key], label=key) 
    axes.set_xlabel(f'Position[{dims[0]}] [m]')
    axes.set_ylabel(f'Position[{dims[1]}] [m]')
    axes.set_title('Trajectories')
    axes.legend()
