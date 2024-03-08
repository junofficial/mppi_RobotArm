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
            [-10, -10, -10, -10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10]
        )
        self.obs_high = np.array(
            [10, 10, 10, 10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10]
        )
        #
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=self.s_dim)
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
        return np.hstack((self._state[:3], self._state[6:9]))

    def _f(self, x, u):
        """
        System dynamics: dx = f(x, u)
        """
        dxdt = np.zeros(12)
        dxdt[0] = x[3]
        dxdt[1] = x[4]
        dxdt[2] = x[5]

        total_thrust = self.params["b"] / self.params["mass"] * sum(u)
        dxdt[3] = (
            np.cos(x[6]) * np.sin(x[7]) * np.cos(x[8]) + np.sin(x[6]) * np.sin(x[8])
        ) * total_thrust
        dxdt[4] = (
            np.cos(x[6]) * np.sin(x[7]) * np.sin(x[8]) - np.sin(x[6]) * np.cos(x[8])
        ) * total_thrust
        dxdt[5] = -self.params["grav"] + np.cos(x[6]) * np.cos(x[7]) * total_thrust

        dxdt[6] = x[9]
        dxdt[7] = x[10]
        dxdt[8] = x[11]

        dxdt[9] = (
            self.params["Lxx"] * self.params["b"] / self.params["Ixx"] * (u[3] - u[1])
            + ((self.params["Iyy"] - self.params["Izz"]) / self.params["Ixx"])
            * x[9]
            * x[10]
        )
        dxdt[10] = (
            self.params["Lyy"] * self.params["b"] / self.params["Iyy"] * (u[0] - u[2])
            + ((self.params["Izz"] - self.params["Ixx"]) / self.params["Iyy"])
            * x[11]
            * x[9]
        )
        dxdt[11] = (
            self.params["d"] / self.params["Izz"] * (-u[0] + u[1] - u[2] + u[3])
            + ((self.params["Ixx"] - self.params["Iyy"]) / self.params["Izz"])
            * x[9]
            * x[10]
        )

        return dxdt

    def run(self, action):
        """
        Apply the control command on the quadrotor and transits the system to the next state
        action: torque
        u: rotor speed
        """
        # rk4 int

        X = self._state

        u = np.linalg.inv(self.params["T2W"]).dot(action)
        k1 = self._dt * self._f(X, u)
        k2 = self._dt * self._f(X + 0.5 * k1, u)
        k3 = self._dt * self._f(X + 0.5 * k2, u)
        k4 = self._dt * self._f(X + k3, u)
        X = X + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

        # u = np.linalg.inv(self.params["T2W"]).dot(action)  # F->u
        # X = X + self._dt * self._f(X, u)

        self._state = X
        return self._state


class Dynamics(object):
    def __init__(self, dt):
        super(Dynamics, self).__init__()
        self.params = SYS_PARAMS()
        self.s_dim = 2
        self.a_dim = 1
        self._dt = dt
        self.obs_low = None
        self.obs_high = None
        self._state = np.zeros(shape=self.s_dim)
        self._actions = np.zeros(shape=self.a_dim)
        self.reset()

    def reset(self):
        self._state = np.zeros(shape=self.s_dim)
        return self._state

    def get_state(self):
        return self._state

    def get_pos(self):
        return self._state

    def _f(self, x, u):
        dxdt = np.zeros(self.s_dim)
        A = np.array([[0, 1], [1, 1]])
        B = np.array([[0, 1]])
        dxdt = x @ A + u @ B
        return dxdt

    def run(self, action):
        x = self._state
        u = action
        # x = x + self._f(x, u) * self._dt

        k1 = self._dt * self._f(x, u)
        k2 = self._dt * self._f(x + 0.5 * k1, u)
        k3 = self._dt * self._f(x + 0.5 * k2, u)
        k4 = self._dt * self._f(x + k3, u)
        x = x + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

        self._state = x
        return self._state


class Quadrotor_v0(object):
    #
    def __init__(self, dt):
        self.s_dim = 10
        self.a_dim = 4
        #
        self._state = np.zeros(shape=self.s_dim)
        self._state[3] = 1.0
        #
        self._actions = np.zeros(shape=self.a_dim)

        #
        self._gz = 9.81
        self._dt = dt
        self._arm_l = 0.3  # m

        # Sampling range of the quadrotor's initial position
        self._xyz_dist = np.array(
            [[-3.0, -1.0], [-2.0, 2.0], [0.0, 2.5]]  # x  # y  # z
        )
        # Sampling range of the quadrotor's initial velocity
        self._vxyz_dist = np.array(
            [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]  # vx  # vy  # vz
        )

        # x, y, z, r, p, y, vx, vy, vz
        self.obs_low = np.array([-10, -10, -10, -np.pi, -np.pi, -np.pi, -10, -10, -10])
        self.obs_high = np.array([10, 10, 10, np.pi, np.pi, np.pi, 10, 10, 10])
        #
        self.reset()
        # self._t = 0.0

    def reset(self):
        self._state = np.zeros(shape=self.s_dim)
        self._state[3] = 1.0
        #
        # initialize position, randomly
        self._state[0] = np.random.uniform(
            low=self._xyz_dist[0, 0], high=self._xyz_dist[0, 1]
        )
        self._state[1] = np.random.uniform(
            low=self._xyz_dist[1, 0], high=self._xyz_dist[1, 1]
        )
        self._state[2] = np.random.uniform(
            low=self._xyz_dist[2, 0], high=self._xyz_dist[2, 1]
        )

        # initialize rotation, randomly
        quad_quat0 = np.random.uniform(low=0.0, high=1, size=4)
        # normalize the quaternion
        self._state[3:7] = quad_quat0 / np.linalg.norm(quad_quat0)

        # initialize velocity, randomly
        self._state[7] = np.random.uniform(
            low=self._vxyz_dist[0, 0], high=self._vxyz_dist[0, 1]
        )
        self._state[8] = np.random.uniform(
            low=self._vxyz_dist[1, 0], high=self._vxyz_dist[1, 1]
        )
        self._state[9] = np.random.uniform(
            low=self._vxyz_dist[2, 0], high=self._vxyz_dist[2, 1]
        )
        #
        return self._state

    def run(self, action):
        """
        Apply the control command on the quadrotor and transits the system to the next state
        """
        # rk4 int
        M = 4
        DT = self._dt / M
        #
        X = self._state
        for i in range(M):
            k1 = DT * self._f(X, action)
            k2 = DT * self._f(X + 0.5 * k1, action)
            k3 = DT * self._f(X + 0.5 * k2, action)
            k4 = DT * self._f(X + k3, action)
            #
            X = X + (k1 + 2.0 * (k2 + k3) + k4) / 6.0
        #
        self._state = X
        return self._state

    def _f(self, state, action):
        """
        System dynamics: ds = f(x, u)
        """
        thrust, wx, wy, wz = action
        #
        dstate = np.zeros(shape=self.s_dim)

        dstate[:3] = state[7:10]

        qw, qx, qy, qz = self.get_quaternion()

        dstate[3] = 0.5 * (-wx * qx - wy * qy - wz * qz)
        dstate[4] = 0.5 * (wx * qw + wz * qy - wy * qz)
        dstate[5] = 0.5 * (wy * qw - wz * qx + wx * qz)
        dstate[6] = 0.5 * (wz * qw + wy * qx - wx * qy)

        dstate[7] = 2 * (qw * qy + qx * qz) * thrust
        dstate[8] = 2 * (qy * qz - qw * qx) * thrust
        dstate[9] = (qw * qw - qx * qx - qy * qy + qz * qz) * thrust - self._gz

        return dstate

    def set_state(self, state):
        """
        Set the vehicle's state
        """
        self._state = state

    def get_state(self):
        """
        Get the vehicle's state
        """
        return self._state

    def get_cartesian_state(self):
        """
        Get the Full state in Cartesian coordinates
        """
        cartesian_state = np.zeros(shape=9)
        cartesian_state[0:3] = self.get_position()
        cartesian_state[3:6] = self.get_euler()
        cartesian_state[6:9] = self.get_velocity()
        return cartesian_state

    def get_pos(
        self,
    ):
        """
        Retrieve Position
        """
        return self._state[:3]

    def get_velocity(
        self,
    ):
        """
        Retrieve Linear Velocity
        """
        return self._state[7:10]

    def get_quaternion(
        self,
    ):
        """
        Retrieve Quaternion
        """
        quat = np.zeros(4)
        quat = self._state[3:7]
        quat = quat / np.linalg.norm(quat)
        return quat

    def get_euler(
        self,
    ):
        """
        Retrieve Euler Angles of the Vehicle
        """
        quat = self.get_quaternion()
        euler = self._quatToEuler(quat)
        return euler

    def get_axes(self):
        """
        Get the 3 axes (x, y, z) in world frame (for visualization only)
        """
        # axes in body frame
        b_x = np.array([self._arm_l, 0, 0])
        b_y = np.array([0, self._arm_l, 0])
        b_z = np.array([0, 0, -self._arm_l])

        # rotation matrix
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()

        # axes in body frame
        w_x = rot_matrix @ b_x + quad_center
        w_y = rot_matrix @ b_y + quad_center
        w_z = rot_matrix @ b_z + quad_center
        return [w_x, w_y, w_z]

    def get_motor_pos(self):
        """
        Get the 4 motor poses in world frame (for visualization only)
        """
        # motor position in body frame
        b_motor1 = np.array([np.sqrt(self._arm_l / 2), np.sqrt(self._arm_l / 2), 0])
        b_motor2 = np.array([-np.sqrt(self._arm_l / 2), np.sqrt(self._arm_l / 2), 0])
        b_motor3 = np.array([-np.sqrt(self._arm_l / 2), -np.sqrt(self._arm_l / 2), 0])
        b_motor4 = np.array([np.sqrt(self._arm_l / 2), -np.sqrt(self._arm_l / 2), 0])
        #
        rot_matrix = R.from_quat(self.get_quaternion()).as_matrix()
        quad_center = self.get_position()

        # motor position in world frame
        w_motor1 = rot_matrix @ b_motor1 + quad_center
        w_motor2 = rot_matrix @ b_motor2 + quad_center
        w_motor3 = rot_matrix @ b_motor3 + quad_center
        w_motor4 = rot_matrix @ b_motor4 + quad_center
        return [w_motor1, w_motor2, w_motor3, w_motor4]

    @staticmethod
    def _quatToEuler(quat):
        """
        Convert Quaternion to Euler Angles
        """
        quat_w, quat_x, quat_y, quat_z = quat[0], quat[1], quat[2], quat[3]
        euler_x = np.arctan2(
            2 * quat_w * quat_x + 2 * quat_y * quat_z,
            quat_w * quat_w - quat_x * quat_x - quat_y * quat_y + quat_z * quat_z,
        )
        euler_y = -np.arcsin(2 * quat_x * quat_z - 2 * quat_w * quat_y)
        euler_z = np.arctan2(
            2 * quat_w * quat_z + 2 * quat_x * quat_y,
            quat_w * quat_w + quat_x * quat_x - quat_y * quat_y - quat_z * quat_z,
        )
        return [euler_x, euler_y, euler_z]
