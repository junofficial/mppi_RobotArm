import pdb
import numpy as np
from os import system

from cvxpy import *
from scipy import sparse

import casadi as ca

from sys_params import SYS_PARAMS

class MPPIControllerForPathTracking():
    def __init__(
            self,
            delta_t: float = 0.05,
            wheel_base: float = 2.5, # [m]
            max_steer_abs: float = 0.523, # [rad]
            max_accel_abs: float = 2.000, # [m/s^2]
            ref_path: np.ndarray = np.array([[0.0, 0.0, 0.0, 1.0], [10.0, 0.0, 0.0, 1.0]]),
            horizon_step_T: int = 30,
            number_of_samples_K: int = 1000,
            param_exploration: float = 0.0,
            param_lambda: float = 50.0,
            param_alpha: float = 1.0,
            sigma: np.ndarray = np.array([[0.5, 0.0], [0.0, 0.1]]), 
            stage_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            terminal_cost_weight: np.ndarray = np.array([50.0, 50.0, 1.0, 20.0]), # weight for [x, y, yaw, v]
            visualize_optimal_traj = True,  # if True, optimal trajectory is visualized
            visualze_sampled_trajs = False, # if True, sampled trajectories are visualized
    ) -> None:
        """initialize mppi controller for path-tracking"""
        # mppi parameters
        self.dim_x = 4 # dimension of system state vector
        self.dim_u = 2 # dimension of control input vector
        self.T = horizon_step_T # prediction horizon
        self.K = number_of_samples_K # number of sample trajectories
        self.param_exploration = param_exploration  # constant parameter of mppi
        self.param_lambda = param_lambda  # constant parameter of mppi
        self.param_alpha = param_alpha # constant parameter of mppi
        self.param_gamma = self.param_lambda * (1.0 - (self.param_alpha))  # constant parameter of mppi
        self.Sigma = sigma # deviation of noise
        self.stage_cost_weight = stage_cost_weight
        self.terminal_cost_weight = terminal_cost_weight
        self.visualize_optimal_traj = visualize_optimal_traj
        self.visualze_sampled_trajs = visualze_sampled_trajs

        # vehicle parameters
        self.delta_t = delta_t #[s]
        self.wheel_base = wheel_base#[m]
        self.max_steer_abs = max_steer_abs # [rad]
        self.max_accel_abs = max_accel_abs # [m/s^2]
        self.ref_path = ref_path

        # mppi variables
        self.u_prev = np.zeros((self.T, self.dim_u))

        # ref_path info
        self.prev_waypoints_idx = 0

    def calc_control_input(self, observed_x: np.ndarray) -> Tuple[float, np.ndarray]:
        """calculate optimal control input"""
        # load privious control input sequence
        u = self.u_prev

        # set initial x value from observation
        x0 = observed_x

        # get the waypoint closest to current vehicle position 
        self._get_nearest_waypoint(x0[0], x0[1], update_prev_idx=True)
        if self.prev_waypoints_idx >= self.ref_path.shape[0]-1:
            print("[ERROR] Reached the end of the reference path.")
            raise IndexError

        # prepare buffer
        S = np.zeros((self.K)) # state cost list

        # sample noise
        epsilon = self._calc_epsilon(self.Sigma, self.K, self.T, self.dim_u) # size is self.K x self.T

        # prepare buffer of sampled control input sequence
        v = np.zeros((self.K, self.T, self.dim_u)) # control input sequence with noise

        # loop for 0 ~ K-1 samples
        for k in range(self.K):         

            # set initial(t=0) state x i.e. observed state of the vehicle
            x = x0

            # loop for time step t = 1 ~ T
            for t in range(1, self.T+1):

                # get control input with noise
                if k < (1.0-self.param_exploration)*self.K:
                    v[k, t-1] = u[t-1] + epsilon[k, t-1] # sampling for exploitation
                else:
                    v[k, t-1] = epsilon[k, t-1] # sampling for exploration

                # update x
                x = self._F(x, self._g(v[k, t-1]))

                # add stage cost
                S[k] += self._c(x) + self.param_gamma * u[t-1].T @ np.linalg.inv(self.Sigma) @ v[k, t-1]

            # add terminal cost
            S[k] += self._phi(x)

        # compute information theoretic weights for each sample
        w = self._compute_weights(S)

        # calculate w_k * epsilon_k
        w_epsilon = np.zeros((self.T, self.dim_u))
        for t in range(self.T): # loop for time step t = 0 ~ T-1
            for k in range(self.K):
                w_epsilon[t] += w[k] * epsilon[k, t]

        # apply moving average filter for smoothing input sequence
        w_epsilon = self._moving_average_filter(xx=w_epsilon, window_size=10)

        # update control input sequence
        u += w_epsilon

        # calculate optimal trajectory
        optimal_traj = np.zeros((self.T, self.dim_x))
        if self.visualize_optimal_traj:
            x = x0
            for t in range(self.T):
                x = self._F(x, self._g(u[t-1]))
                optimal_traj[t] = x

        # calculate sampled trajectories
        sampled_traj_list = np.zeros((self.K, self.T, self.dim_x))
        sorted_idx = np.argsort(S) # sort samples by state cost, 0th is the best sample
        if self.visualze_sampled_trajs:
            for k in sorted_idx:
                x = x0
                for t in range(self.T):
                    x = self._F(x, self._g(v[k, t-1]))
                    sampled_traj_list[k, t] = x

        # update privious control input sequence (shift 1 step to the left)
        self.u_prev[:-1] = u[1:]
        self.u_prev[-1] = u[-1]

        # return optimal control input and input sequence
        return u[0], u, optimal_traj, sampled_traj_list

    def _calc_epsilon(self, sigma: np.ndarray, size_sample: int, size_time_step: int, size_dim_u: int) -> np.ndarray:
        """sample epsilon"""
        # check if sigma row size == sigma col size == size_dim_u and size_dim_u > 0
        if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != size_dim_u or size_dim_u < 1:
            print("[ERROR] sigma must be a square matrix with the size of size_dim_u.")
            raise ValueError

        # sample epsilon
        mu = np.zeros((size_dim_u)) # set average as a zero vector
        epsilon = np.random.multivariate_normal(mu, sigma, (size_sample, size_time_step))
        return epsilon

    def _g(self, v: np.ndarray) -> float:
        """clamp input"""
        # limit control inputs
        v[0] = np.clip(v[0], -self.max_steer_abs, self.max_steer_abs) # limit steering input
        v[1] = np.clip(v[1], -self.max_accel_abs, self.max_accel_abs) # limit acceleraiton input
        return v

    def _c(self, x_t: np.ndarray) -> float:
        """calculate stage cost"""
        # parse x_t
        x, y, yaw, v = x_t
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate stage cost
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        stage_cost = self.stage_cost_weight[0]*(x-ref_x)**2 + self.stage_cost_weight[1]*(y-ref_y)**2 + \
                     self.stage_cost_weight[2]*(yaw-ref_yaw)**2 + self.stage_cost_weight[3]*(v-ref_v)**2
        return stage_cost

    def _phi(self, x_T: np.ndarray) -> float:
        """calculate terminal cost"""
        # parse x_T
        x, y, yaw, v = x_T
        yaw = ((yaw + 2.0*np.pi) % (2.0*np.pi)) # normalize theta to [0, 2*pi]

        # calculate terminal cost
        _, ref_x, ref_y, ref_yaw, ref_v = self._get_nearest_waypoint(x, y)
        terminal_cost = self.terminal_cost_weight[0]*(x-ref_x)**2 + self.terminal_cost_weight[1]*(y-ref_y)**2 + \
                        self.terminal_cost_weight[2]*(yaw-ref_yaw)**2 + self.terminal_cost_weight[3]*(v-ref_v)**2
        return terminal_cost

    def _get_nearest_waypoint(self, x: float, y: float, update_prev_idx: bool = False):
        """search the closest waypoint to the vehicle on the reference path"""

        SEARCH_IDX_LEN = 200 # [points] forward search range
        prev_idx = self.prev_waypoints_idx
        dx = [x - ref_x for ref_x in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 0]]
        dy = [y - ref_y for ref_y in self.ref_path[prev_idx:(prev_idx + SEARCH_IDX_LEN), 1]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        min_d = min(d)
        nearest_idx = d.index(min_d) + prev_idx

        # get reference values of the nearest waypoint
        ref_x = self.ref_path[nearest_idx,0]
        ref_y = self.ref_path[nearest_idx,1]
        ref_yaw = self.ref_path[nearest_idx,2]
        ref_v = self.ref_path[nearest_idx,3]

        # update nearest waypoint index if necessary
        if update_prev_idx:
            self.prev_waypoints_idx = nearest_idx 

        return nearest_idx, ref_x, ref_y, ref_yaw, ref_v

    def Arm_Dynamic(self, q, dq, u):
    
        dt = self.delta_t
        M11 = m1 * lc1 ** 2 + l1 + m2 * \
            (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(q[1])) + l2
        M22 = m2 * lc2 ** 2 + l2
        M12 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
        M21 = m2 * l1 * lc2 * np.cos(q[1]) + m2 * lc2 ** 2 + l2
        M = np.array([[M11, M12], [M21, M22]])
        h = m2 * l1 * lc2 * np.sin(q[1])
        g1 = m1 * lc1 * g * np.cos(q[0]) + m2 * g * \
            (lc2 * np.cos(q[0] + q[1]) + l1 * np.cos(q[0]))
        g2 = m2 * lc2 * g * np.cos(q[0] + q[1])
        G = np.array([g1, g2])
        C = np.array([[-h * dq[1], -h * dq[0] - h * dq[1]], [h * dq[0], 0]])
        ddq = np.linalg.inv(M).dot(u - C.dot(dq) - G)
        dq = dq + ddq * dt
        q = q + dq * dt
        

        return q, dq

    def _F(self, x_t: np.ndarray, v_t: np.ndarray) -> np.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        x, y, yaw, v = x_t
        steer, accel = v_t

        # prepare params
        l = self.wheel_base
        dt = self.delta_t

        # update state variables
        new_x = x + v * np.cos(yaw) * dt
        new_y = y + v * np.sin(yaw) * dt
        new_yaw = yaw + v / l * np.tan(steer) * dt
        new_v = v + accel * dt

        # return updated state
        x_t_plus_1 = np.array([new_x, new_y, new_yaw, new_v])
        return x_t_plus_1

    def _compute_weights(self, S: np.ndarray) -> np.ndarray:
        """compute weights for each sample"""
        # prepare buffer
        w = np.zeros((self.K))

        # calculate rho
        rho = S.min()

        # calculate eta
        eta = 0.0
        for k in range(self.K):
            eta += np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )

        # calculate weight
        for k in range(self.K):
            w[k] = (1.0 / eta) * np.exp( (-1.0/self.param_lambda) * (S[k]-rho) )
        return w

    def _moving_average_filter(self, xx: np.ndarray, window_size: int) -> np.ndarray:
        """apply moving average filter for smoothing input sequence
        Ref. https://zenn.dev/bluepost/articles/1b7b580ab54e95
        """
        b = np.ones(window_size)/window_size
        dim = xx.shape[1]
        xx_mean = np.zeros(xx.shape)

        for d in range(dim):
            xx_mean[:,d] = np.convolve(xx[:,d], b, mode="same")
            n_conv = math.ceil(window_size/2)
            xx_mean[0,d] *= window_size/n_conv
            for i in range(1, n_conv):
                xx_mean[i,d] *= window_size/(i+n_conv)
                xx_mean[-i,d] *= window_size/(i + n_conv - (window_size % 2)) 
        return xx_mean


class PID(object):
    """
    PID control for quadrotor
    """
    def __init__(self, dt):
        self.params = SYS_PARAMS()
        self.dt = dt
        # action dimensions (thrust, wx, wy, wz)
        self._u_dim = 4
        
        # 피드백 게인은 위치 오차를 줄이고 원하는 위치에 QUADROTOR를 안정적으로 유지
        self.KpF = np.array([4.55e0, 4.55e0, 2.55e0])
        # 미분 게인은 속도 오차를 줄이고 원하는 속도로 QUADROTOR를 안정적으로 유지
        self.KdF = np.array([8.75e0, 8.75e0, 2.75e0])
        # 각도 오차를 줄이기 위한 피드백 게인
        self.KpM = np.array([3e2, 3e2, 3e2])
        # 각속도 오차를 줄이기 위한 미분 게인
        self.KdM = np.array([3e1, 3e1, 3e1])
        # 이전 QUADROTOR 위치와 목표 위치를 저장하는 배열을 초기화
        self.old_pos = np.zeros(6) # 이전 위치
        self.old_pos_d = np.zeros(6) #이전 DESIRED한 위치

    def get_action(self, obs): # 현재상태(OBS)를 입력으로 받아 행동을 반환
        """
        obs: np.array

        """
        quad_pos = obs[:6] # OBS를 통한 현재 위치
        traj_pos = obs[6:] # OBS를 통한 목표 위치
        # pdb.set_trace()
   
        # 이전의 POS와 현재의 POS를 빼서 시간 간격으로 나눔(속도를 구함)
        vx = (quad_pos[0] - self.old_pos[0]) / self.dt
        vy = (quad_pos[1] - self.old_pos[1]) / self.dt
        vz = (quad_pos[2] - self.old_pos[2]) / self.dt
        vp = (quad_pos[3] - self.old_pos[3]) / self.dt
        vq = (quad_pos[4] - self.old_pos[4]) / self.dt
        vr = (quad_pos[5] - self.old_pos[5]) / self.dt
        # pdb.set_trace()
        
        # 이전의 DESIRED POS와 현재의 DESIRED POS를 빼서 시간 간격으로 나눔(속도를 구함)
        vx_d = (traj_pos[0] - self.old_pos_d[0]) / self.dt
        vy_d = (traj_pos[1] - self.old_pos_d[1]) / self.dt
        vz_d = (traj_pos[2] - self.old_pos_d[2]) / self.dt
        vp_d = (traj_pos[3] - self.old_pos_d[3]) / self.dt
        vq_d = (traj_pos[4] - self.old_pos_d[4]) / self.dt
        vr_d = (traj_pos[5] - self.old_pos_d[5]) / self.dt

        # X,Y,Z 축 방향의 제어 입력 
        u1x = -self.KpF[0] * (quad_pos[0] - traj_pos[0]) - self.KdF[0] * (vx - vx_d)
        u1y = -self.KpF[1] * (quad_pos[1] - traj_pos[1]) - self.KdF[1] * (vy - vy_d)
        u1z = (
            -self.KpF[2] * (quad_pos[2] - traj_pos[2])
            - self.KdF[2] * (vz - vz_d)
            + self.params["grav"]
        )
        
        # ROLL 속도 DESIRED값
        p_d = np.arctan2(
            np.sin(traj_pos[5]) * np.cos(traj_pos[5]) * u1x
            - np.cos(traj_pos[5]) * np.cos(traj_pos[5]) * u1y,
            u1z,
        )
        
        # PITCH 속도 DESIRED값
        q_d = np.arcsin(
            (
                np.cos(traj_pos[5]) * np.cos(traj_pos[5]) * u1x
                + np.sin(traj_pos[5]) * np.cos(traj_pos[5]) * u1y
            )
            / u1z
        )
        # YAW 속도 DESIRED값
        r_d = traj_pos[5]
    
        # 목표하는 ROLL, PITCH, YAW를 ANGLE_DESIRED로 NUMPY배열 생성
        ang_d = np.array([p_d, q_d, r_d])
        rate_d = np.array([0, 0, 0])

        # Z축 방향의 힘을 계산
        Fz = u1z / (np.cos(p_d) * np.cos(r_d))
        
        # 관성 모멘트 행렬을 생성. 
        I = np.diag([self.params["Ixx"], self.params["Iyy"], self.params["Izz"]])
        
        # 모멘트 계산
        Mr = np.dot(
            I,
            self.KdM * (rate_d - np.array([vp, vq, vr]))
            + self.KpM * (ang_d - quad_pos[3:6]),
        )

        # 
        self.old_pos = np.array(quad_pos)
        self.old_pos_d = np.array(traj_pos)

        # z축 방향의 힘과 모멘트를 리턴
        return np.hstack((Fz, Mr))




class MPC_PY(object):
    """TODO, Maybe discrete update"""

    def __init__(self, mpc_T, dt):
        self.mpc_T = mpc_T
        self.dt = dt
        self._N = int(self.mpc_T / self.dt)
        
        # 상태 x와 입력 u의 크기 
        self.nx = 2
        self.nu = 1

        # x와 u의 최대 최소값 설정
        self.umin = np.array([-5.0])
        self.umax = np.array([5.0])
        # self.umin = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        # self.umax = np.array([np.inf, np.inf, np.inf, np.inf])
        self.xmin = np.array([-np.inf, -np.inf])
        self.xmax = np.array(
            [
                np.inf,
                np.inf,
            ]
        )

        # 목적 함수(제어 입력과 상태 변수의 차이를 최소화하기 위함)
        # Q : 상태 변수의 가중치 행렬, 대각 행렬로써 상태 변수가 목표값과의 차이에 대해 얼마나 중요한지 보임
        self.Q = sparse.diags([10.0, 10.0])
        # 마지막 단계의 가중치 행렬
        self.QN = self.Q
        # R : 제어 입력의 가중치 행렬
        self.R = 1.0 * sparse.eye(self.nu)
        # 이산화된 상태 공간 모델의 상태 및 입력 행렬을 설정
        self.Ad = np.array([[0.0, 1.0], [1.0, 1.0]])
        self.Bd = np.array([[0.0], [1.0]])

    def get_action(self, obs):
    
        # 시스템 상태 변수와 trj 목표 상태 변수를 받음 
        sys_pos = obs[:2]
        trj_pos = obs[2:]
        
        # 초기 상태 변수와 목표 상태 변수 값을 설정
        x_init = np.array([0.0, 0.0])
        xr = np.array(trj_pos.tolist())
        
        # 최적 제어 문제를 해결하기 위한 것으로 (입력의 수, 입력의 길이)
        u = Variable((self.nu, self._N))
        x = Variable((self.nx, self._N + 1))
        
        # 목적함수 및 제약 조건 초기화(reset)
        objective = 0
        constraints = [x[:, 0] == x_init]

        # MPC 돌아가는 부분
        for k in range(self._N):
            # quad_form : 행렬과 벡터사이의 제곱 항을 계산하는 데 사용
            # 목적함수를 계산, 상태변수와 제어 입력의 차이에 대한 가중치를 사용하여 정의
            objective += quad_form(x[:, k] - xr, self.Q) + quad_form(u[:, k], self.R)
            
            # 각 단계에서 다음 상태 변수를 현재 상태 변수와 제어 입력을 사용하여 예측
            constraints += [x[:, k + 1] == self.Ad @ x[:, k] + self.Bd @ u[:, k]]
            # 상태변수 및 제어 입력이 허용 범위내에 있어야 하는 조건
            constraints += [self.xmin <= x[:, k], x[:, k] <= self.xmax]
            constraints += [self.umin <= u[:, k], u[:, k] <= self.umax]
        # 최종 목적 함수 및 제약조건 설정하여 최적화 문제를 정의
        objective += quad_form(x[:, self._N] - xr, self.QN)
        prob = Problem(Minimize(objective), constraints)
        # 최적화 문제를 해결하고, MPC를 통해 계산된 제어 입력을 반환
        prob.solve(solver=OSQP, warm_start=True)
        u_mpc = u[:, 0].value
        # pdb.set_trace()
        return u_mpc


class MPC_CA(object):
    def __init__(self, mpc_T, dt, so_path="./nmpc.so"):
        """
        Nonlinear MPC for quadrotor control
        """
        self.so_path = so_path

        # Time constant
        self.mpc_T = mpc_T
        self._dt = dt
        self._N = int(self.mpc_T / self._dt)

        # constants
        self._u_min = -20.0
        self._u_max = 20.0

        self._Q_x = np.diag([10, 0.1])
        self._Q_u = np.diag([0.1])

        # initial state and control action
        self._sys_s0 = [0.0, 0.0]
        self._sys_u0 = [0.0]

        self._initDynamics()

    def _initDynamics(
        self,
    ):
        # state symbolic variables
        px, vx = ca.SX.sym("px"), ca.SX.sym("vx")
        self._x = ca.vertcat(px, vx)
        self.nx = self._x.numel()

        pu = ca.SX.sym("pu")
        self._u = ca.vertcat(pu)
        self.nu = self._u.numel()

        # x_dot = ca.vertcat(vx, px + vx)
        # self.f = ca.Function("f", [self._x, self._u], [x_dot], ["x", "u"], ["ode"])

        A = ca.DM([[0, 1], [1, 1]])
        B = ca.DM([[0], [1]])
        RHS = A @ self._x + B @ self._u
        self.f = ca.Function("f", [self._x, self._u], [RHS], ["x", "u"], ["ode"])

        # loss function
        # placeholder for the quadratic cost function
        Delta_s = ca.SX.sym("Delta_s", self.nx)
        Delta_u = ca.SX.sym("Delta_u", self.nu)

        cost_s = Delta_s.T @ self._Q_x @ Delta_s
        cost_u = Delta_u.T @ self._Q_u @ Delta_u

        f_cost_s = ca.Function("cost_s", [Delta_s], [cost_s])
        f_cost_u = ca.Function("cost_u", [Delta_u], [cost_u])

        # Non-linear Optimization
        self.nlp_w = []  # nlp variables
        self.nlp_w0 = []  # initial guess of nlp variables
        self.lbw = []  # lower bound of the variables, lbw <= nlp_x
        self.ubw = []  # upper bound of the variables, nlp_x <= ubw

        self.mpc_obj = 0  # objective
        self.nlp_g = []  # constraint functions
        self.lbg = []  # lower bound of constrait functions, lbg < g
        self.ubg = []  # upper bound of constrait functions, g < ubg

        u_min = [self._u_min]
        u_max = [self._u_max]
        x_bound = ca.inf
        x_min = [-x_bound for _ in range(self.nx)]
        x_max = [+x_bound for _ in range(self.nx)]

        g_min = [0 for _ in range(self.nx)]
        g_max = [0 for _ in range(self.nx)]

        # P = ca.SX.sym("P", self._s_dim + (self._s_dim + 3) * self._N + self._s_dim)
        P = ca.SX.sym("P", self.nx + self.nx)
        X = ca.SX.sym("X", self.nx, self._N + 1)
        U = ca.SX.sym("U", self.nu, self._N)

        F = self.sys_dynamics(self._dt)
        fMap = F.map(self._N, "openmp")  # parallel
        X_next = fMap(X[:, : self._N], U)

        # "Lift" initial conditions
        self.nlp_w += [X[:, 0]]
        self.nlp_w0 += self._sys_s0
        self.lbw += x_min
        self.ubw += x_max

        # # starting point.
        self.nlp_g += [X[:, 0] - P[0 : self.nx]]
        self.lbg += g_min
        self.ubg += g_max

        for k in range(self._N):
            self.nlp_w += [U[:, k]]
            self.nlp_w0 += self._sys_u0
            self.lbw += u_min
            self.ubw += u_max

            delta_s_k = X[:, k + 1] - P[self.nx :]
            cost_s_k = f_cost_s(delta_s_k)
            delta_u_k = U[:, k]
            cost_u_k = f_cost_u(delta_u_k)

            self.mpc_obj = self.mpc_obj + cost_s_k + cost_u_k

            # New NLP variable for state at end of interval
            self.nlp_w += [X[:, k + 1]]
            self.nlp_w0 += self._sys_s0
            self.lbw += x_min
            self.ubw += x_max

            # Add equality constraint
            self.nlp_g += [X_next[:, k] - X[:, k + 1]]
            self.lbg += g_min
            self.ubg += g_max

        # nlp objective
        nlp_dict = {
            "f": self.mpc_obj,
            "x": ca.vertcat(*self.nlp_w),
            "p": P,
            "g": ca.vertcat(*self.nlp_g),
        }

        ipopt_options = {
            "verbose": False,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-4,
            "ipopt.max_iter": 100,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.print_level": 0,
            "print_time": False,
        }

        self.solver = ca.nlpsol("solver", "ipopt", nlp_dict, ipopt_options)

    def solve(self, ref_states):
        system_state = ca.vertcat(self.nlp_w0[:2])
        traj_state = ca.vertcat(ref_states[:2])

        self.sol = self.solver(
            x0=self.nlp_w0,
            lbx=self.lbw,
            ubx=self.ubw,
            lbg=self.lbg,
            ubg=self.ubg,
            # p=ca.vertcat(self.nlp_w0[:2], ref_states),
            p=ca.vertcat(system_state, traj_state),
        )

        sol_x0 = self.sol["x"].full()
        opt_u = sol_x0[self.nx : self.nx + self.nu]

        # Warm initialization
        self.nlp_w0 = list(sol_x0[self.nx + self.nu : 2 * (self.nx + self.nu)]) + list(
            sol_x0[self.nx + self.nu :]
        )

        x0_array = np.reshape(sol_x0[: -self.nx], newshape=(-1, self.nx + self.nu))

        # return optimal action, and a sequence of predicted optimal trajectory.
        return opt_u, x0_array

    def sys_dynamics(self, dt):
        X0 = ca.SX.sym("X", self.nx)
        U = ca.SX.sym("U", self.nu)

        X = X0
        k1 = dt * self.f(X, U)
        k2 = dt * self.f(X + 0.5 * k1, U)
        k3 = dt * self.f(X + 0.5 * k2, U)
        k4 = dt * self.f(X + k3, U)

        X = X + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        F = ca.Function("F", [X0, U], [X])
        return F



