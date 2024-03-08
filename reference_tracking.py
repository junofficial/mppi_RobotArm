import numpy as np
import pdb

from trajectory import TRAJECTORY
from quad import Quadrotor

import torch
from torch.autograd import Variable

from sys_params import SYS_PARAMS
params = SYS_PARAMS()


class Space(object):

    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.shape = self.low.shape

    def sample(self):
        return np.random.uniform(self.low, self.high)


class Tracking(object):

    def __init__(self, pid, sim_T, sim_dt):
        #
        self.pid = pid

        # simulation parameters ....
        self.sim_T = sim_T  # Episode length, seconds
        self.sim_dt = sim_dt  # simulation time step

        # Simulators, a quadrotor and a reference
        self.quad = Quadrotor(dt=self.sim_dt)
        self.traj = TRAJECTORY(self.sim_T, 'hover')

        # reset the environment
        self.t = 0
        self.reset()

    def seed(self, seed):
        np.random.seed(seed=seed)

    def reset(self,):
        self.t = 0
        # state for ODE
        self.quad_state = self.quad.reset()
        self.traj_state = self.traj.reset()

        # observation, can be part of the state, e.g., postion
        # or a cartesian representation of the state
        quad_obs = self.quad.get_pos()  # [x, y, z, r, p, y]
        traj_obs = self.traj.get_pos()  # [xd, yd, zd, rd, pd, yd]

        # obs = (quad_obs - traj_obs).tolist()
        obs = quad_obs.tolist() + traj_obs.tolist()

        return np.array(obs)

    def step(self, u=0):
        self.t += self.sim_dt
        opt_u = u

        # run the actual control command on the quadrotor
        self.quad_state = self.quad.run(opt_u)
        # simulate one step pendulum
        self.traj_state = self.traj.run(self.t)

        # update the observation.
        quad_obs = self.quad.get_pos()
        traj_obs = self.traj.get_pos()
        # obs = (quad_obs - traj_obs).tolist()
        obs = quad_obs.tolist() + traj_obs.tolist()
        info = {
            "quad_obs": quad_obs,
            "traj_obs": traj_obs,
            "opt_u": opt_u}
        done = False
        if self.t >= (self.sim_T-self.sim_dt):
            done = True

        reward = 0
        # pdb.set_trace()
        return np.array(obs), reward, done, info

    def close(self,):
        return True

    def render(self,):
        return False
