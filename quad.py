import numpy as np
from scipy.spatial.transform import Rotation as R
from sys_params import SYS_PARAMS
import pdb


class Quadrotor(object):
    #
    def __init__(self, dt):
        self._dt = dt

        self.params = SYS_PARAMS()
        self.s_dim = 12
        self.a_dim = 4
        #
        self._state = np.zeros(shape=self.s_dim)
        self._actions = np.zeros(shape=self.a_dim)

        # x, y, z, r, p, y, vx, vy, vz
        self.obs_low = np.array(
            [-10, -10, -10, -10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        self.obs_high = np.array(
            [10, 10, 10, 10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])
        #
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=self.s_dim)
        return self._state

    def run(self, action):
        """
        Apply the control command on the quadrotor and transits the system to the next state
        action: torque
        u: rotor speed
        """
        # rk4 int

        X = self._state

        # u = np.linalg.inv(self.params['T2W']).dot(action)
        # k1 = self._dt*self._f(X, u)
        # k2 = self._dt*self._f(X + 0.5*k1, u)
        # k3 = self._dt*self._f(X + 0.5*k2, u)
        # k4 = self._dt*self._f(X + k3, u)
        # X = X + (k1 + 2.0*(k2 + k3) + k4)/6.0

        u = np.linalg.inv(self.params['T2W']).dot(action)  # F->u
        X = X + self._dt*self._f(X, u)

        self._state = X
        return self._state

    def _f(self, x, u):
        """
        System dynamics: dx = f(x, u)
        """
        dxdt = np.zeros(12)
        dxdt[0] = x[3]
        dxdt[1] = x[4]
        dxdt[2] = x[5]

        total_thrust = self.params['b'] / self.params['mass'] * sum(u)
        dxdt[3] = (np.cos(x[6]) * np.sin(x[7]) * np.cos(x[8]) +
                   np.sin(x[6]) * np.sin(x[8])) * total_thrust
        dxdt[4] = (np.cos(x[6]) * np.sin(x[7]) * np.sin(x[8]) -
                   np.sin(x[6]) * np.cos(x[8])) * total_thrust
        dxdt[5] = -self.params['grav'] + \
            np.cos(x[6]) * np.cos(x[7]) * total_thrust

        dxdt[6] = x[9]
        dxdt[7] = x[10]
        dxdt[8] = x[11]

        dxdt[9] = self.params['Lxx'] * self.params['b'] / self.params['Ixx'] * \
            (u[3] - u[1]) + ((self.params['Iyy'] - self.params['Izz']) /
                             self.params['Ixx']) * x[9] * x[10]
        dxdt[10] = self.params['Lyy'] * self.params['b'] / self.params['Iyy'] * \
            (u[0] - u[2]) + ((self.params['Izz'] - self.params['Ixx']) /
                             self.params['Iyy']) * x[11] * x[9]
        dxdt[11] = self.params['d'] / self.params['Izz'] * (-u[0] + u[1] - u[2] + u[3]) + (
            (self.params['Ixx'] - self.params['Iyy']) / self.params['Izz']) * x[9] * x[10]

        return dxdt

    def get_state(self):
        """
        Get the vehicle's state
        """
        return self._state

    def get_pos(self):
        """
        get pose (x, y, z)
        """
        return np.hstack((self._state[:3], self._state[6:9]))
