import numpy as np
import copy

class SpacecraftState:
    def __init__(self, pos, vel, mass, elec_energy):
        self.pos = pos
        self.vel = vel
        self.mass = mass
        self.elec_energy = elec_energy

class SpacecraftAction:
    def __init__(self, fire_railgun):
        # Projectile launch velocity relative to spacecraft, inertial frame
        self.fire_railgun = fire_railgun

spacecraft_properties = {
    'radius': 5., # [units: meter]
    'railgun mass': 10., # projectile mass [units: kg]
    'railgun max speed': 5e3, # [units: m s^-1]
    'railgun efficiency': 0.5,
}

class ProjectileState:
    def __init__(self, pos, vel):
        self.pos = pos
        self.vel = vel


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
        self.projectile_states = []
        self.projectile_states_record = []

    def update(self, actions, dt):
        self.step_count += 1
        self.time += dt
        self.time_record[self.step_count] = self.time

        for key in self.spacecraft_states:
            if actions[key].fire_railgun is not None:
                mass_ratio = spacecraft_properties['railgun mass'] / self.spacecraft_states[key].mass
                self.spacecraft_states[key].mass -= spacecraft_properties['railgun mass']
                # Projectile launch velocity, relative to spacecraft, in inertial coordinates
                v_launch = actions[key].fire_railgun
                speed = np.linalg.norm(v_launch)
                if speed > spacecraft_properties['railgun max speed']:
                    v_launch *= spacecraft_properties['railgun max speed'] / speed
                # Kinetic energy of projectile, relative to launching spacecraft
                projectile_ke = 0.5 * spacecraft_properties['railgun mass'] * speed**2
                # Railgun energy consumption for this shot
                railgun_energy = projectile_ke / spacecraft_properties['railgun efficiency']
                if railgun_energy > self.spacecraft_states[key].elec_energy:
                    print(f'Spacecraft {key:s} insufficient energy for railgun shot at t = {self.time:.4f} s.')
                else:
                    # decrement spacecraft electric energy
                    self.spacecraft_states[key].elec_energy -= railgun_energy
                    # Solve for new spacecraft velocity after firing railgun
                    self.spacecraft_states[key].vel -= mass_ratio * v_launch
                    # Solve for projective velocity relative to inertial frame
                    v_projectile = v_launch + self.spacecraft_states[key].vel

                    # Add a projectile object
                    self.projectile_states.append(ProjectileState(
                        pos=copy.deepcopy(self.spacecraft_states[key].pos), vel=v_projectile))
                    self.projectile_states_record.append(ProjectileState(
                        pos=np.full((self.n_steps, 3), np.nan),
                        vel=np.full((self.n_steps, 3), np.nan),
                        ))
            # Simulate state forward in time
            # TODO something better than forward Euler
            self.spacecraft_states[key].pos += self.spacecraft_states[key].vel * dt

            # Record the positions
            self.spacecraft_states_record[key].pos[self.step_count] = self.spacecraft_states[key].pos
            self.spacecraft_states_record[key].vel[self.step_count] = self.spacecraft_states[key].vel
            self.spacecraft_states_record[key].mass[self.step_count] = self.spacecraft_states[key].mass
            self.spacecraft_states_record[key].elec_energy[self.step_count] = (
                self.spacecraft_states[key].elec_energy)

        for ip, _ in enumerate(self.projectile_states):
            # Simulate state forward in time
            self.projectile_states[ip].pos += self.projectile_states[ip].vel * dt
            # Record the positions
            self.projectile_states_record[ip].pos[self.step_count] = self.projectile_states[ip].pos
            self.projectile_states_record[ip].vel[self.step_count] = self.projectile_states[ip].vel
