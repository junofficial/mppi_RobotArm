import numpy as np
import pdb

from trajectory import TRAJECTORY
from plant import Quadrotor, Dynamics, Quadrotor_v0

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
    def __init__(self, ctr, sim_T, sim_dt):
        self.ctr = ctr

        ## simulation parameters ....
        self.sim_T = sim_T  # Episode length, seconds
        self.sim_dt = sim_dt  # simulation time step

        ## Simulators, a quadrotor and a reference
        self.quad = Quadrotor_v0(dt=self.sim_dt)
        self.traj = TRAJECTORY(self.sim_T, "eight")

        ## reset the environment
        self.t = 0
        self.reset()

    def seed(self, seed):
        np.random.seed(seed=seed)

    def reset(
        self,
    ):
        self.t = 0
        ## state for ODE
        self.quad_state = self.quad.reset()
        self.traj_state = self.traj.reset()

        ## observation, can be part of the state, e.g., postion
        ## or a cartesian representation of the state
        quad_obs = self.quad.get_pos()  # [x, y, z]
        traj_obs = self.traj.get_pos()  # [xd, yd, zd, rd, pd, yd]

        obs = quad_obs.tolist() + traj_obs[:3].tolist()

        return np.array(obs)

    def step(self, u=0):
        self.t += self.sim_dt
        ## simulate one step reference
        self.traj_state = self.traj.run(self.t)

        ## create control
        # opt_u = u
        cont, pred_traj = self.ctr.solve(self.traj_state)
        opt_u = cont.squeeze(1)
        # opt_u = cont

        ## run the actual control command on the quadrotor
        self.quad_state = self.quad.run(opt_u)

        ## update the observation.
        quad_obs = self.quad.get_pos()
        traj_obs = self.traj.get_pos()
        # obs = (quad_obs - traj_obs).tolist()
        obs = quad_obs.tolist() + traj_obs[:3].tolist()
        info = {"quad_obs": quad_obs, "traj_obs": traj_obs, "opt_u": opt_u}
        done = False
        if self.t >= (self.sim_T - self.sim_dt):
            done = True

        reward = 0
        # pdb.set_trace()
        return np.array(obs), reward, done, info

    def close(
        self,
    ):
        return True

    def render(
        self,
    ):
        return False


class Tracking_LPV(object):
    def __init__(self, ctr, sim_T, sim_dt):
        self.crt = ctr

        # simulation parameters ....
        self.sim_T = sim_T  # Episode length, seconds
        self.sim_dt = sim_dt  # time step

        # Simulators, a plant and a reference
        self.sys = Dynamics(dt=self.sim_dt)
        self.trj = TRAJECTORY(self.sim_T, "eight")

        # reset the environment
        self.t = 0
        self.reset()

    def seed(self, seed):
        np.random.seed(seed=seed)

    def reset(
        self,
    ):
        self.t = 0
        # state for ODE
        self.sys_state = self.sys.reset()
        self.trj_state = self.trj.reset()

        # observation, can be part of the state, e.g., postion
        # or a cartesian representation of the state
        sys_obs = self.sys.get_pos()
        trj_obs = self.trj.get_pos()

        # obs = (quad_obs - traj_obs).tolist()
        obs = sys_obs.tolist() + trj_obs.tolist()

        return np.array(obs)

    def step(self):
        self.t += self.sim_dt

        # self.traj_state = self.traj.run(self.t)
        self.traj_state = np.array([1.0, 0.0]) if self.t < 10 else np.array([-1.0, 0.0])
        obs = self.sys_state.tolist() + self.traj_state.tolist()
        obs = np.array(obs)

        ## PYTHON MPC
        # action = self.crt.get_action(obs)

        ## CASADI MPC
        cont, pred_traj = self.crt.solve(self.traj_state)
        action = cont.squeeze(1)

        # run the actual control command on the quadrotor
        self.sys_state = self.sys.run(action)

        # update the observation.
        sys_obs = self.sys.get_pos()
        trj_obs = self.trj.get_pos()
        # obs = (quad_obs - traj_obs).tolist()
        # obs = quad_obs.tolist() + traj_obs.tolist()

        info = {
            "sys_state": self.sys_state,
            "traj_state": self.traj_state,
            "action": action,
        }
        done = False
        if self.t >= (self.sim_T - self.sim_dt):
            done = True

        reward = 0
        # pdb.set_trace()
        return np.array(obs), reward, done, info

    def close(
        self,
    ):
        return True

    def render(
        self,
    ):
        return False
