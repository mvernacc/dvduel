import numpy as np


class SpacecraftState:
    def __init__(self, pos, vel, mass, elec_energy):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.elec_energy = elec_energy

spacecraft_properties = {
    'radius': 5., # [units: meter]
}

class Simulator:
    def __init__(self, scA_init, scB_init, n_steps=1000):
        self.n_steps = n_steps
        self.spacecraft_states = {'A': scA_init, 'B': scB_init}
        self.spacecraft_states_record = {}
        for key, scs in self.spacecraft_states.items():
            self.spacecraft_states_record[key] = SpacecraftState(
                pos=np.full((n_steps, 3), np.nan),
                vel=np.full((n_steps, 3), np.nan),
                mass=np.full(n_steps, np.nan),
                elec_energy=np.full(n_steps, np.nan),
                )
        self.step_count = 0
        self.time = 0
        self.time_record = np.full(n_steps, np.nan)

    def update(self, dt):
        self.step_count += 1
        self.time += dt
        self.time_record[self.step_count] = self.time

        for key in self.spacecraft_states:
            # Simulate state forward in time
            # TODO something better than forward Euler
            self.spacecraft_states[key].pos += self.spacecraft_states[key].vel * dt

            # Record the positions
            self.spacecraft_states_record[key].pos[self.step_count] = self.spacecraft_states[key].pos
            self.spacecraft_states_record[key].vel[self.step_count] = self.spacecraft_states[key].vel
            self.spacecraft_states_record[key].mass[self.step_count] = self.spacecraft_states[key].mass
            self.spacecraft_states_record[key].elec_energy[self.step_count] = (
                self.spacecraft_states[key].elec_energy)
