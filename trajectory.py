import numpy as np
import pdb


class TRAJECTORY(object):
    '''
    return: np.array([xd, yd, zd, rd, pd, yd])
    '''

    def __init__(self, sim_time, trajName):
        self.sim_T = sim_time
        self.trajName = trajName

        self.s_dim = 6
        self._state = np.zeros(shape=self.s_dim)

        # x, y, z, r, p, y
        self.obs_low = np.array(
            [-np.inf, -np.inf, -np.inf, -np.pi, -np.pi, -np.pi])
        self.obs_low = np.array(
            [np.inf, np.inf, np.inf, np.pi, np.pi, np.pi])
        #
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=self.s_dim)
        return self._state

    def run(self, t):
        # pdb.set_trace()
        if self.trajName == 'cycloid':
            r = np.array([np.sqrt(t) * np.sin(t), np.sqrt(t)
                         * np.cos(t), 0.5 * t, 0, 0, 0])

        elif self.trajName == 'hover':
            T = self.sim_T
            if t <= T / 3:
                r = np.array([t, t, t, 0, 0, 0])
            else:
                r = np.array([T / 3, t, T / 3, 0, 0, 0])

        elif self.trajName == 'circle':
            r = np.array([np.sin(t), np.cos(t), t, 0, 0, 0])

        elif self.trajName == 'eight':
            T = self.sim_T
            radius = 2
            w = 6 * np.pi / T
            if t <= T / 3:
                rx = 0
                ry = 0
                rz = t
            elif t <= 2 * T / 3:
                rx = -radius * np.cos(w * (t - T / 3)) + radius
                ry = radius * np.sin(w * (t - T / 3))
                rz = T / 3
            else:
                rx = radius * np.cos(w * (t - T / 3)) - radius
                ry = radius * np.sin(w * (t - T / 3))

                rz = T / 3
            r = np.array([rx, ry, rz, 0, 0, 0])
        else:
            print('wrong trajectory')
            r = np.zeros(6)

        self._state = r
        return self._state

    def get_state(self):
        """
        Get the vehicle's state
        """
        return self._state

    def get_pos(self):
        """
        get pose (x, y, z)
        """
        return self._state
