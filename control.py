from cvxpy import *
from scipy import sparse
import numpy as np
from sys_params import SYS_PARAMS
import pdb


class PID(object):
    """
        PID control for quadrotor         
    """

    def __init__(self, sim_T, sim_dt):
        self.params = SYS_PARAMS()
        self.sim_T = sim_T
        self.sim_dt = sim_dt
        # action dimensions (thrust, wx, wy, wz)
        self._u_dim = 4
        self.KpF = np.array([4.55e0, 4.55e0, 2.55e0])
        self.KdF = np.array([8.75e0, 8.75e0, 2.75e0])
        self.KpM = np.array([3e2, 3e2, 3e2])
        self.KdM = np.array([3e1, 3e1, 3e1])

        self.old_pos = np.zeros(6)
        self.old_pos_d = np.zeros(6)

    def get_action(self, obs):
        '''
        obs: np.array

        '''
        quad_pos = obs[:6]
        traj_pos = obs[6:]
        # pdb.set_trace()

        vx = (quad_pos[0] - self.old_pos[0]) / self.sim_dt
        vy = (quad_pos[1] - self.old_pos[1]) / self.sim_dt
        vz = (quad_pos[2] - self.old_pos[2]) / self.sim_dt
        vp = (quad_pos[3] - self.old_pos[3]) / self.sim_dt
        vq = (quad_pos[4] - self.old_pos[4]) / self.sim_dt
        vr = (quad_pos[5] - self.old_pos[5]) / self.sim_dt
        # pdb.set_trace()
        vx_d = (traj_pos[0] - self.old_pos_d[0]) / self.sim_dt
        vy_d = (traj_pos[1] - self.old_pos_d[1]) / self.sim_dt
        vz_d = (traj_pos[2] - self.old_pos_d[2]) / self.sim_dt
        vp_d = (traj_pos[3] - self.old_pos_d[3]) / self.sim_dt
        vq_d = (traj_pos[4] - self.old_pos_d[4]) / self.sim_dt
        vr_d = (traj_pos[5] - self.old_pos_d[5]) / self.sim_dt

        u1x = -self.KpF[0] * (quad_pos[0] - traj_pos[0]) - \
            self.KdF[0] * (vx - vx_d)
        u1y = -self.KpF[1] * (quad_pos[1] - traj_pos[1]) - \
            self.KdF[1] * (vy - vy_d)
        u1z = -self.KpF[2] * (quad_pos[2] - traj_pos[2]) - self.KdF[2] * \
            (vz - vz_d) + self.params['grav']

        p_d = np.arctan2(np.sin(traj_pos[5]) * np.cos(traj_pos[5]) *
                         u1x - np.cos(traj_pos[5]) * np.cos(traj_pos[5]) * u1y, u1z)
        q_d = np.arcsin((np.cos(traj_pos[5]) * np.cos(traj_pos[5]) *
                         u1x + np.sin(traj_pos[5]) * np.cos(traj_pos[5]) * u1y) / u1z)
        r_d = traj_pos[5]

        ang_d = np.array([p_d, q_d, r_d])
        rate_d = np.array([0, 0, 0])

        Fz = u1z / (np.cos(p_d) * np.cos(r_d))

        I = np.diag(
            [self.params['Ixx'], self.params['Iyy'], self.params['Izz']])
        Mr = np.dot(
            I, self.KdM * (rate_d - np.array([vp, vq, vr])) + self.KpM * (ang_d - quad_pos[3:6]))

        self.old_pos = np.array(quad_pos)
        self.old_pos_d = np.array(traj_pos)

        return np.hstack((Fz, Mr))


class MPC(object):
    """
        MPC for quadrotor         
    """

    def __init__(self, sim_T, sim_dt):
        self.params = SYS_PARAMS()
        self.sim_T = sim_T
        self.sim_dt = sim_dt
        self.N = 10  # Prediction horizon
        # action dimensions (thrust, wx, wy, wz)
        self.nx = 12
        self.nu = 4

        self.umin = np.array([-500.0, -500.0, -500.0, -500.0])
        self.umax = np.array([500.0, 500.0, 500.0, 500.0])
        # self.umin = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        # self.umax = np.array([np.inf, np.inf, np.inf, np.inf])
        self.xmin = np.array([-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi,
                              -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf])
        self.xmax = np.array([np.inf, np.inf, np.inf, np.pi, np.pi, np.pi,
                              np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
        # Objective function
        self.Q = sparse.diags(
            [10., 10., 10., 0.1, 0.1, 0.1, 0., 0., 10., 0., 0., 0.])
        self.QN = self.Q
        self.R = 0.001 * sparse.eye(4)
        self.Ad = np.array([
            [1., 0., 0., 0.01, 0., 0., 0, 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0.01, 0., 0., 0, 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.01, 0., 0., 0, 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0.0981, 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., -0.0981, 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.01, 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.01, 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.01],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,]
        ])
        self.Bd = np.array([
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [2.676*1e-4, 2.676*1e-4, 2.676*1e-4, 2.676*1e-4],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., -0.0064, 0., 0.0064],
            [0.0064, 0., -0.0064, 0.],
            [-6.757*1e-4, 6.757*1e-4, -6.757*1e-4, 6.757*1e-4]])
        self.ueq = [91.6418, 91.6418, 91.6418, 91.6418]

    def get_action(self, obs):
        '''
        obs: np.array

        '''
        quad_pos = obs[:6]  # [x, y, z, r, p, y]
        traj_pos = obs[6:]  # [xd, yd, zd, rd, pd, yd]
        # pdb.set_trace()

        x_init = np.array(quad_pos[:3].tolist() +
                          [0., 0., 0.] + quad_pos[3:6].tolist() + [0., 0., 0.])
        xr = np.array(traj_pos[:3].tolist() +
                      [0., 0., 0.] + traj_pos[3:6].tolist() + [0., 0., 0.])
        u = Variable((self.nu, self.N))
        x = Variable((self.nx, self.N + 1))
        objective = 0
        constraints = [x[:, 0] == x_init]
        for k in range(self.N):
            objective += quad_form(x[:, k] - xr, self.Q) + \
                quad_form(u[:, k], self.R)
            constraints += [x[:, k + 1] == self.Ad @
                            x[:, k] + self.Bd @ u[:, k]]
            constraints += [self.xmin <= x[:, k], x[:, k] <= self.xmax]
            constraints += [self.umin <= u[:, k], u[:, k] <= self.umax]
        objective += quad_form(x[:, self.N] - xr, self.QN)
        prob = Problem(Minimize(objective), constraints)

        prob.solve(solver=OSQP, warm_start=True)
        u_mpc = u[:, 0].value + self.ueq
        # pdb.set_trace()
        F_mpc = self.params['T2W'].dot(u_mpc)
        return F_mpc
